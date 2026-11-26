import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from experiment import ALGORITHMS
from nuzlocke_rules import PartyMon
from poison_curriculum import CureSignals, MatchedCombinedExtractor, PoisonEnv, load_suite
from red_gym_env_v2_nuzlocke import RedGymEnv

MON = PartyMon((10, 20), 177, 18, 22, 6, 8)


@pytest.mark.parametrize("lag", [0, 1, 2, -1, -2])
def test_cure_matches_inventory_across_frame_boundary(lag):
    signal = CureSignals([MON], 3, 0)
    for frame in range(1, 5):
        clear_at, spent_at = (1, 1 + lag) if lag >= 0 else (1 - lag, 1)
        signal.observe(
            [replace(MON, status=0 if frame >= clear_at else 8)],
            2 if frame >= spent_at else 3,
            0,
            set(),
        )
    assert len(signal.events) == 1
    assert signal.events[0]["identity"] == MON.identity
    assert signal.antidotes_spent == 1


@pytest.mark.parametrize(
    "after,count,battle,dead",
    [
        (replace(MON, hp=0, status=0), 2, 0, {MON.identity}),  # faint clears status
        (replace(MON, hp=22, status=0), 2, 0, set()),  # Center/healing is not antidote
        (replace(MON, status=0), 3, 0, set()),  # no consumed antidote
        (MON, 2, 0, set()),  # discarded item, still poisoned
        (replace(MON, status=0), 2, 1, set()),  # in-battle use is outside task scope
        (replace(MON, status=0), 2, 0, {MON.identity}),  # previously dead/revived
    ],
)
def test_false_cures_are_rejected(after, count, battle, dead):
    signal = CureSignals([MON], 3, battle)
    signal.observe([after], count, battle, dead)
    assert signal.events == []


def test_healthy_or_unrelated_old_item_spend_is_not_success():
    healthy = replace(MON, status=0)
    signal = CureSignals([healthy], 3, 0)
    signal.observe([healthy], 2, 0, set())
    assert not signal.events
    signal = CureSignals([MON], 3, 0)
    for _ in range(4):
        signal.observe([MON], 2, 0, set())
    signal.observe([healthy], 2, 0, set())
    assert not signal.events


@pytest.fixture
def poison_config(fake_pyboy, config, tmp_path, monkeypatch):
    original = fake_pyboy.load_state

    def load(self, file):
        original(self, file)
        self.memory[0xD16F] = 8
        self.memory[0xD31D:0xD321] = bytes([1, 11, 3, 255])

    monkeypatch.setattr(fake_pyboy, "load_state", load)
    state = Path(config["init_state"])
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "poison",
                        "split": "train",
                        "state_file": state.name,
                        "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                        "initial_used_regions": [12, 51],
                        "dead": [],
                        "ram_edited": False,
                        "poisoned": True,
                        "antidotes": 3,
                    }
                ]
            }
        )
    )
    return dict(config, observation_mode="pixels", task="cure", scenario_suite=str(suite))


def test_hash_and_initial_rule_history(poison_config):
    env = PoisonEnv(poison_config)
    try:
        obs, _ = env.reset(seed=3)
        assert set(obs) == {"screens", "recent_actions"}
        assert env.tracker.used_regions == {12, 51}
    finally:
        env.close()
    Path(poison_config["init_state"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_suite(poison_config["scenario_suite"], "train")


def test_reward_terminates_once_and_death_takes_precedence(poison_config):
    env = PoisonEnv(poison_config)
    try:
        env.reset()

        def cure(p):
            p.memory[0xD16F] = 0
            p.memory[0xD31F] = 2

        env.pyboy.transitions = {1: cure}
        _, reward, terminated, truncated, info = env.step(4)
        assert reward == pytest.approx(0.999)
        assert terminated and not truncated and info["is_success"]
        assert info["end_reason"] == "antidote_cure"
        with pytest.raises(RuntimeError, match="reset"):
            env.step(4)
        env.reset()
        env.pyboy.transitions = {1: cure, 2: lambda p: p.set_mon(hp=0)}
        _, reward, terminated, _, info = env.step(4)
        assert terminated and not info["is_success"] and reward < -1
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["ppo", "lstm"])
def test_equal_parameter_counts_and_exact_half_resolution_weight_transfer(
    fake_pyboy, config, tmp_path, algorithm
):
    torch.set_num_threads(1)
    environments = [
        RedGymEnv(dict(config, observation_mode="pixels", screen_scale=s)) for s in (2, 2, 1)
    ]
    models = []
    try:
        for index, env in enumerate(environments):
            kwargs = {} if index == 0 else {"features_extractor_class": MatchedCombinedExtractor}
            models.append(
                ALGORITHMS[algorithm](
                    "MultiInputPolicy" if algorithm == "ppo" else "MultiInputLstmPolicy",
                    env,
                    n_steps=8,
                    batch_size=8,
                    n_epochs=1,
                    policy_kwargs=kwargs,
                    device="cpu",
                    seed=1,
                )
            )
        for model in models[1:]:
            model.policy.load_state_dict(models[0].policy.state_dict(), strict=True)
        assert len({sum(p.numel() for p in m.policy.parameters()) for m in models}) == 1
        obs, _ = environments[0].reset(seed=1)
        rng = np.random.default_rng(2)
        obs["screens"] = rng.integers(0, 256, obs["screens"].shape, dtype=np.uint8)
        tensor = models[0].policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            before = models[0].policy.extract_features(tensor)
            after = models[1].policy.extract_features(tensor)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        for index, model in enumerate(models[1:]):
            model.learn(16)
            path = tmp_path / f"{algorithm}-{index}.zip"
            model.save(path)
            loaded = ALGORITHMS[algorithm].load(path, env=environments[index + 1])
            assert loaded.num_timesteps == 16
            obs, _ = environments[index + 1].reset()
            assert environments[index + 1].observation_space.contains(obs)
            action, _ = loaded.predict(obs)
            assert environments[index + 1].action_space.contains(action)
    finally:
        for env in environments:
            env.close()
