"""The memory target set inside the context-prediction PPO (context_aux, target set "memory")."""

import numpy as np
import torch
from context_aux import (
    TARGET_WIDTH,
    ContextCallback,
    ContextPolicy,
    ContextPPO,
    MemoryLabels,
    context_layout,
    memory_prediction_loss,
)
from context_experiment import convert_parent
from lstm_memory import HEAD_DIM, HEAD_SLICES, TARGET_NAMES, TARGETS, memory_target
from red_gym_env_v2_nuzlocke import RedGymEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv
from test_context_aux import TinyPixels

N = len(TARGETS)


class MemoryPixels(TinyPixels):
    """TinyPixels whose labels have the memory layout: N values, then N validity flags."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        values = np.zeros(N, np.float32)
        values[TARGET_NAMES.index("poisoned")] = self.step_count % 2
        values[TARGET_NAMES.index("battle_type")] = self.step_count % 3
        values[TARGET_NAMES.index("party_hp")] = 0.5
        valid = np.zeros(N, np.float32)
        for name in ("poisoned", "battle_type", "party_hp"):
            valid[TARGET_NAMES.index(name)] = 1
        info["context_target_before_action"] = np.concatenate([values, valid])
        return obs, reward, terminated, truncated, info


def test_memory_labels_are_read_before_the_action(fake_pyboy, config):
    env = MemoryLabels(RedGymEnv(dict(config, max_steps=4)))
    try:
        env.reset(seed=0)
        expected = memory_target(env.unwrapped, env.last_wild)
        env.unwrapped.pyboy.transitions = {24: lambda p: p.set_mon(hp=5)}
        info = env.step(4)[4]
        target = info["context_target_before_action"]
        assert target.shape == (TARGET_WIDTH["memory"],)
        np.testing.assert_array_equal(target, expected)  # HP before the hit, not after
        assert target[TARGET_NAMES.index("party_hp")] == 1.0
    finally:
        env.close()


def test_memory_loss_masks_and_reaches_lstm_and_cnn_only_in_the_auxiliary_arm():
    torch.set_num_threads(1)
    policy = ContextPolicy(
        TinyPixels.observation_space,
        TinyPixels.action_space,
        lambda _: 0.001,
        context_targets="memory",
    )
    assert policy.context_head.out_features == HEAD_DIM
    obs = {"screens": torch.zeros(3, 3, 72, 80), "recent_actions": torch.zeros(3, 3).long()}
    z = torch.zeros(1, 1, 256)
    states = RNNStates((z, z), (z, z))
    targets = torch.zeros(3, 2 * N)
    poisoned = TARGET_NAMES.index("poisoned")
    targets[:, poisoned] = torch.tensor([0.0, 1.0, 1.0])
    targets[:, N + poisoned] = 1
    padding = torch.tensor([True, True, False])
    for detached in (True, False):
        policy.zero_grad()
        *_, logits = policy.evaluate_with_context(
            obs, torch.zeros(3).long(), states, torch.tensor([1.0, 0.0, 0.0]), detached
        )
        loss = memory_prediction_loss(logits, targets, padding)
        torch.testing.assert_close(
            loss, memory_prediction_loss(logits[:2], targets[:2], padding[:2])
        )
        loss.backward()
        lstm = policy.lstm_actor.weight_ih_l0.grad
        cnn = policy.features_extractor.extractors["screens"].cnn[0].weight.grad
        if detached:
            assert lstm is None and cnn is None
        else:
            assert lstm.abs().sum() > 0 and cnn is not None
        assert policy.context_head.weight.grad[HEAD_SLICES["poisoned"]].abs().sum() > 0
        assert policy.action_net.weight.grad is None


def test_memory_arm_trains_saves_and_reloads(tmp_path):
    torch.set_num_threads(1)
    base_env, env = DummyVecEnv([TinyPixels]), DummyVecEnv([MemoryPixels])
    try:
        base = RecurrentPPO(
            "MultiInputLstmPolicy", base_env, n_steps=8, batch_size=8, seed=3, device="cpu"
        )
        model = convert_parent(base, env, seed=3, backprop=True, device="cpu", targets="memory")
        assert model.context_targets == "memory"
        head = model.policy.context_head.weight.detach().clone()
        cnn = model.policy.features_extractor.extractors["screens"].cnn[0].weight.detach().clone()
        model.learn(16, reset_num_timesteps=False, callback=ContextCallback())
        assert model.rollout_buffer.context_targets.shape[-1] == TARGET_WIDTH["memory"]
        assert not torch.equal(head, model.policy.context_head.weight)
        assert not torch.equal(
            cnn, model.policy.features_extractor.extractors["screens"].cnn[0].weight
        )
        model.save(tmp_path / "model")
        reloaded = ContextPPO.load(tmp_path / "model", env=env, device="cpu")
        assert reloaded.context_targets == "memory"
        assert reloaded.policy.context_head.out_features == HEAD_DIM
        obs, _ = TinyPixels().reset()
        action, _ = RecurrentPPO.load(tmp_path / "model", device="cpu").predict(
            obs, deterministic=True
        )
        assert action == model.predict(obs, deterministic=True)[0]
    finally:
        base_env.close()
        env.close()


def test_memory_predictions_feed_the_pilot_diagnostics_in_its_layout():
    logits = torch.zeros(2, HEAD_DIM)
    logits[:, HEAD_SLICES["battle_type"]] = torch.tensor([0.0, 0.0, 4.0])
    logits[:, HEAD_SLICES["poisoned"]] = 2.0
    logits[:, HEAD_SLICES["party_hp"]] = 0.25
    logits[:, HEAD_SLICES["has_antidote"]] = -3.0
    layout = context_layout(logits)
    assert layout.shape == (2, 6)
    assert layout[0, :3].argmax() == 2 and layout[0, 3] == 2.0 and layout[0, 5] == -3.0
    torch.testing.assert_close(layout[:, 4].sigmoid(), torch.full((2,), 0.25))
