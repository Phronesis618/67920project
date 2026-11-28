import copy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from context_aux import (
    ContextBuffer,
    ContextCallback,
    ContextLabels,
    ContextPolicy,
    ContextPPO,
    context_target,
    prediction_loss,
)
from context_experiment import convert_parent
from red_gym_env_v2_nuzlocke import RedGymEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


class TinyPixels(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "screens": gym.spaces.Box(0, 255, (3, 72, 80), np.uint8),
            "recent_actions": gym.spaces.MultiDiscrete([7, 7, 7]),
        }
    )
    action_space = gym.spaces.Discrete(7)

    def obs(self):
        return {
            "screens": np.full((3, 72, 80), self.step_count % 255, np.uint8),
            "recent_actions": np.array([self.step_count % 7, 0, 0]),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self.obs(), {}

    def step(self, action):
        labels = np.array([self.step_count % 3, self.step_count % 2, 0.5, 1], np.float32)
        self.step_count += 1
        return (
            self.obs(),
            float(action == 4),
            self.step_count == 5,
            False,
            {
                "context_target_before_action": labels,
            },
        )


def test_labels_pre_action_and_never_observation(fake_pyboy, config):
    config.update(observation_mode="pixels", include_explore_map=False)
    env = ContextLabels(RedGymEnv(config))
    try:
        obs, _ = env.reset()
        assert set(obs) == {"screens", "recent_actions"}
        before = context_target(env)

        def poison(game):
            game.memory[0xD16B + 4] = 8

        env.unwrapped.pyboy.transitions[1] = poison
        obs, _, _, _, info = env.step(4)
        np.testing.assert_array_equal(info["context_target_before_action"], before)
        assert before[1] == 0 and context_target(env)[1] == 1
        assert set(obs) == {"screens", "recent_actions"}
    finally:
        env.close()


def test_buffer_targets_follow_shuffle_padding_and_resets():
    torch.set_num_threads(1)
    b = ContextBuffer(
        4, TinyPixels.observation_space, TinyPixels.action_space, (4, 1, 2, 8), n_envs=2
    )
    z = torch.zeros(1, 2, 8)
    for i in range(4):
        markers = np.array([i, i + 20])
        obs = {
            "screens": np.broadcast_to(markers[:, None, None, None], (2, 3, 72, 80)),
            "recent_actions": np.zeros((2, 3)),
        }
        b.pending_targets = np.stack([markers] * 4, axis=1)
        b.add(
            obs,
            np.zeros((2, 1)),
            np.ones(2),
            np.array([i == 0, i in (0, 2)]),
            torch.zeros(2),
            torch.zeros(2),
            lstm_states=RNNStates((z, z), (z, z)),
        )
    for _ in range(3):
        count = 0
        for sample in b.get(5):
            mask = sample.mask.bool()
            torch.testing.assert_close(
                sample.context_targets[mask, 0],
                sample.observations["screens"][mask, 0, 0, 0].float(),
            )
            count += int(mask.sum())
        assert count == 8


def test_loss_masks_padding_and_aux_gradient_isolation():
    torch.set_num_threads(1)
    policy = ContextPolicy(TinyPixels.observation_space, TinyPixels.action_space, lambda _: 0.001)
    obs = {"screens": torch.zeros(3, 3, 72, 80), "recent_actions": torch.zeros(3, 3).long()}
    z = torch.zeros(1, 1, 256)
    states = RNNStates((z, z), (z, z))
    targets = torch.tensor([[0, 0, 0.5, 1], [2, 1, 0.2, 1], [-1, 0, 0, 0.0]])
    mask = torch.tensor([True, True, False])
    for detached in [True, False]:
        policy.zero_grad()
        *_, logits = policy.evaluate_with_context(
            obs, torch.zeros(3).long(), states, torch.tensor([1.0, 0.0, 0.0]), detached
        )
        loss = prediction_loss(logits, targets, mask)
        torch.testing.assert_close(loss, prediction_loss(logits[:2], targets[:2], mask[:2]))
        loss.backward()
        grad = policy.lstm_actor.weight_ih_l0.grad
        assert (grad is None) if detached else (grad is not None and grad.abs().sum() > 0)
        assert policy.context_head.weight.grad.abs().sum() > 0
        assert policy.action_net.weight.grad is None


def test_detached_control_matches_upstream_update_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    base_env, control_env = DummyVecEnv([TinyPixels]), DummyVecEnv([TinyPixels])
    try:
        base = RecurrentPPO(
            "MultiInputLstmPolicy",
            base_env,
            n_steps=8,
            batch_size=8,
            n_epochs=2,
            seed=5,
            device="cpu",
        )
        base.learn(8)  # Populate Adam moments before conversion.
        control = convert_parent(base, control_env, seed=5, backprop=False, device="cpu")
        before = copy.deepcopy(base.policy.state_dict())
        for key, value in before.items():
            torch.testing.assert_close(control.policy.state_dict()[key], value, rtol=0, atol=0)
        for key, value in base.policy.optimizer.state_dict()["state"].items():
            for field, expected in value.items():
                torch.testing.assert_close(
                    control.policy.optimizer.state_dict()["state"][key][field], expected
                )
        # A new segment starts from reset in both arms.
        base._last_obs = None
        base._last_lstm_states = RNNStates(
            tuple(torch.zeros_like(t) for t in base._last_lstm_states.pi),
            tuple(torch.zeros_like(t) for t in base._last_lstm_states.vf),
        )
        set_random_seed(11)
        base.learn(16, reset_num_timesteps=False)
        set_random_seed(11)
        control.learn(16, reset_num_timesteps=False, callback=ContextCallback())
        # Includes auto-reset transitions; each label belongs to its source image.
        buffer = control.rollout_buffer
        np.testing.assert_array_equal(
            buffer.context_targets[:, 0], buffer.observations["screens"][:, 0, 0, 0] % 3
        )
        for key, value in base.policy.state_dict().items():
            torch.testing.assert_close(
                control.policy.state_dict()[key], value, rtol=1e-6, atol=1e-7
            )
        assert any(not torch.equal(before[k], v) for k, v in base.policy.state_dict().items())
        control.save(tmp_path / "model")
        reloaded = ContextPPO.load(tmp_path / "model", env=control_env, device="cpu")
        inference = RecurrentPPO.load(tmp_path / "model", device="cpu")
        obs, _ = TinyPixels().reset()
        for m in (control, reloaded, inference):
            action, _ = m.predict(obs, deterministic=True)
            assert action == control.predict(obs, deterministic=True)[0]
        reloaded.learn(8, reset_num_timesteps=False, callback=ContextCallback())
    finally:
        base_env.close()
        control_env.close()


def test_batched_history_warmup_matches_stepwise_memory(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from context_benchmark import warm_history

    torch.set_num_threads(1)
    env = TinyPixels()
    model = ContextPPO("MultiInputLstmPolicy", env, n_steps=8, batch_size=8, device="cpu")
    history = []
    env.reset()
    for i in range(9):
        env.step_count = i
        history.append(env.obs())
    np.savez_compressed(
        tmp_path / "history.npz",
        **{key: np.stack([obs[key] for obs in history]) for key in history[0]},
    )
    state = None
    for i, obs in enumerate(history):
        _, state = model.predict(
            obs, state=state, episode_start=np.array([i == 0]), deterministic=True
        )
    warmed = warm_history(model, tmp_path / "history.npz", chunk=4)
    for got, expected in zip(warmed, state):
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-5)
