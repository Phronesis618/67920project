import io
import pickle

import numpy as np
import pytest
from failure_restart import FailureRestart, capture, restore
from pyboy.utils import WindowEvent
from red_gym_env_v2_nuzlocke import RedGymEnv


@pytest.fixture
def snapshot_pyboy(fake_pyboy, monkeypatch):
    original = fake_pyboy.load_state

    def save(self, f):
        f.write(b"TEST" + pickle.dumps((self.memory, self.frames)))

    def load(self, f):
        raw = f.read()
        if raw.startswith(b"TEST"):
            self.memory, self.frames = pickle.loads(raw[4:])
        else:
            original(self, io.BytesIO(raw))

    monkeypatch.setattr(fake_pyboy, "save_state", save, raising=False)
    monkeypatch.setattr(fake_pyboy, "load_state", load)


def test_roundtrip_preserves_rules_rewards_and_observation_history(snapshot_pyboy, config):
    env = RedGymEnv(dict(config, observation_mode="pixels", max_steps=100, navigation_reward=True))
    try:
        env.reset(seed=7)
        env.tracker.dead.add((999, 999))
        env.tracker.used_regions.add(51)
        env.tracker.encounter_catchable = False
        env.tracker.encounter_region = 51
        obs, *_ = env.step(0)
        env.pyboy.events = [WindowEvent(WindowEvent.RELEASE_ARROW_DOWN)]
        saved = capture(env, obs)
        expected = env.step(4)
        env.reset(seed=9)
        env.pyboy.events = [WindowEvent(WindowEvent.PRESS_BUTTON_START)]
        restored = restore(env, saved)
        assert [int(e) for e in env.pyboy.events] == [WindowEvent.RELEASE_ARROW_DOWN]
        assert env.tracker.dead == {(999, 999)}
        assert 51 in env.tracker.used_regions
        assert not env.tracker.encounter_catchable and env.tracker.encounter_region == 51
        for key in obs:
            np.testing.assert_array_equal(restored[key], obs[key])
        actual = env.step(4)
        for key in actual[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        assert actual[1:] == expected[1:]
    finally:
        env.close()


@pytest.mark.parametrize("failure", ["wipe", "time_limit", "rule_violation"])
def test_failure_bank_and_remaining_budget(snapshot_pyboy, config, tmp_path, failure):
    base = RedGymEnv(dict(config, observation_mode="pixels", max_steps=6))
    env = FailureRestart(base, tmp_path / failure, probability=1, stride=1, window=4, min_lead=2)
    try:
        env.reset(seed=1)
        if failure == "wipe":
            base.pyboy.transitions = {120: lambda p: p.set_mon(hp=0)}
        elif failure == "rule_violation":
            base.pyboy.transitions = {
                120: lambda p: (p.set_mon(level=15), p.memory.__setitem__(0xD057, 2))
            }
        while True:
            _, _, done, trunc, info = env.step(0)
            if done or trunc:
                break
        assert info["end_reason"] == failure
        assert env.bank and all(2 <= r["lead_actions"] <= 4 for r in env.bank)
        obs, info = env.reset(seed=2)
        assert info["training_reset"] == "failure"
        assert base.step_count == info["restart_original_step"] > 0
        assert base.max_steps == 6  # no new full episode allowance
        assert not base.tracker.terminated
        assert set(obs) == {"screens", "recent_actions"}
        assert base.episode_return == 0
    finally:
        env.close()


def test_success_is_not_promoted_and_terminal_capture_rejected(snapshot_pyboy, config, tmp_path):
    base = RedGymEnv(config)
    env = FailureRestart(base, tmp_path / "bank", probability=1, stride=1, min_lead=1)
    try:
        env.reset()
        base.pyboy.transitions = {48: lambda p: p.memory.__setitem__(0xD356, 1)}
        env.step(4)
        obs, _, done, _, info = env.step(4)
        assert done and info["is_success"] and not env.bank
        with pytest.raises(ValueError, match="terminal"):
            capture(base, obs)
    finally:
        env.close()


def test_ordinary_arm_exact_reward_equivalence_and_bounded_bank(snapshot_pyboy, config, tmp_path):
    ordinary = RedGymEnv(config)
    wrapped = FailureRestart(
        RedGymEnv(config),
        tmp_path / "bank",
        probability=0,
        stride=1,
        min_lead=1,
        window=3,
        max_failures=2,
        anchors=2,
    )
    try:
        for episode in range(4):
            ordinary.reset(seed=episode)
            _, info = wrapped.reset(seed=episode)
            assert info["training_reset"] == "original"
            for action in [0, 1, 2, 4]:
                expected = ordinary.step(action)
                actual = wrapped.step(action)
                assert actual[1:4] == expected[1:4]
                assert all(actual[4][k] == v for k, v in expected[4].items())
                for key in expected[0]:
                    np.testing.assert_array_equal(actual[0][key], expected[0][key])
        assert len(wrapped.bank) <= 4
        assert len(list((tmp_path / "bank").glob("*.pkl.gz"))) == len(wrapped.bank)
        row = wrapped.bank[0]
        (tmp_path / "bank" / row["file"]).write_bytes(b"corrupt")
        # Restrict selection to the corrupted entry to verify failure is explicit.
        wrapped.bank = [row]
        with pytest.raises(ValueError, match="hash mismatch"):
            wrapped.reset(options={"failure_restart": True})
    finally:
        ordinary.close()
        wrapped.close()
