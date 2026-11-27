import json

import numpy as np
import pytest
from red_gym_env_v2_nuzlocke import RedGymEnv
from route_replay import RouteReplay


def finish(env):
    while True:
        result = env.step(4)
        if result[2] or result[3]:
            return result


def test_transition_anchors_reach_before_tail_and_preserve_healthy_entry(
    snapshot_pyboy, config, tmp_path
):
    cfg = dict(config, observation_mode="pixels", max_steps=14)
    source = RouteReplay(RedGymEnv(cfg), tmp_path / "route", min_lead=2, window=4, stride=1)
    reader = RouteReplay(RedGymEnv(cfg), tmp_path / "replay", sources=[source.directory])
    try:
        source.reset(seed=1)
        source.env.pyboy.transitions = {72: lambda p: p.memory.__setitem__(0xD057, 2)}
        finish(source)
        trainer = [row for row in source.bank if row["battle"] == 2]
        assert len(trainer) == 1 and trainer[0]["original_step"] == 3
        assert trainer[0]["lead_actions"] == 11  # outside the old 1024-equivalent tail
        assert all(row["source_reset"] == "original" for row in source.bank)
        # Constrain selection to the trainer entry and check actual restored state.
        (source.directory / "bank.json").write_text(json.dumps(trainer))
        obs, info = reader.reset(seed=2)
        assert set(obs) == {"screens", "recent_actions"}
        assert info["training_reset"] == "failure" and info["restart_original_step"] == 3
        assert reader.env.read_m(0xD057) == 2
        assert reader.env.read_party_mons()[0].hp == 20
        result = finish(reader)
        assert result[4]["fresh_actions"] == 11 and result[4]["end_reason"] == "time_limit"
        assert not reader.bank and not (reader.directory / "bank.json").exists()
    finally:
        source.close()
        reader.close()


@pytest.mark.parametrize("mode", ["tail", "transition"])
def test_route_equivalence_and_equal_worker_action_share(snapshot_pyboy, config, tmp_path, mode):
    cfg = dict(config, observation_mode="pixels", max_steps=8)
    plain = RedGymEnv(cfg)
    route = RouteReplay(RedGymEnv(cfg), tmp_path / "route", mode=mode, stride=1, min_lead=1)
    replay = RouteReplay(RedGymEnv(cfg), tmp_path / "replay", sources=[route.directory], mode=mode)
    try:
        _, empty = replay.reset()
        assert empty["training_reset"] == "original"  # empty-bank warmup
        for episode in range(3):
            plain.reset(seed=episode)
            route.reset(seed=episode)
            if episode:
                replay.reset(seed=episode)
            for _ in range(8):
                expected, actual = plain.step(4), route.step(4)
                assert actual[1:4] == expected[1:4]
                assert all(actual[4][k] == v for k, v in expected[4].items())
                for key in expected[0]:
                    np.testing.assert_array_equal(expected[0][key], actual[0][key])
                r = replay.step(4)
                if r[2] or r[3]:
                    replay.reset()
        rs, ps = route.replay_stats(), replay.replay_stats()
        assert rs["actions"] == {"original": 24}
        assert sum(ps["actions"].values()) == 24 and ps["actions"]["replay"] > 0
        assert ps["failures_published"] == 0
    finally:
        plain.close()
        route.close()
        replay.close()


def test_tail_window_bounded_index_immutable_blobs_and_hash_check(snapshot_pyboy, config, tmp_path):
    route = RouteReplay(
        RedGymEnv(dict(config, max_steps=10)),
        tmp_path / "route",
        mode="tail",
        stride=1,
        window=4,
        min_lead=2,
        max_failures=1,
    )
    reader = RouteReplay(RedGymEnv(config), tmp_path / "reader", sources=[route.directory])
    try:
        route.reset()
        finish(route)
        old = list(route.bank)
        route.reset()
        finish(route)
        assert all(2 <= row["lead_actions"] <= 4 for row in route.bank)
        assert all(row["failure"] == 2 for row in route.bank)
        assert all((route.directory / row["file"]).exists() for row in old)
        row = route.bank[0]
        (route.directory / "bank.json").write_text(json.dumps([row]))
        (route.directory / row["file"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="hash mismatch"):
            reader.reset()
    finally:
        route.close()
        reader.close()


def test_success_never_published_and_map_transitions_are_captured(snapshot_pyboy, config, tmp_path):
    route = RouteReplay(RedGymEnv(dict(config, max_steps=6)), tmp_path / "route", min_lead=1)
    try:
        route.reset()
        route.env.pyboy.transitions = {
            48: lambda p: p.set_map(1),
            120: lambda p: p.memory.__setitem__(0xD356, 1),
        }
        result = finish(route)
        assert result[4]["is_success"] and not route.bank
        assert len(route.transitions[0]) >= 2
    finally:
        route.close()
