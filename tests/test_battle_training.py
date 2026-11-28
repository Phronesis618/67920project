import gzip
import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest
from battle_training import (
    ENEMY_HP,
    ENEMY_MAX_HP,
    ENEMY_SPECIES,
    PRESETS,
    BattleReplay,
    BattleReward,
    BattleRewardWeights,
    catch_in_progress,
    snapshot_config_sha,
    verify_rom,
)
from failure_restart import capture
from red_gym_env_v2_nuzlocke import RedGymEnv

from scripts import build_battle_bank

ONLY = dict.fromkeys(
    (
        "trainer_damage",
        "trainer_ko",
        "trainer_win",
        "wild_damage",
        "wild_ko",
        "wild_win",
        "level",
        "hp_potential",
        "trainer_run_rejected",
        "overtime",
    ),
    0.0,
)


def weights(**kw):
    return BattleRewardWeights(**dict(ONLY, **kw))


def enemy(p, species, hp, max_hp=20):
    p.memory[ENEMY_SPECIES] = species
    p.memory[ENEMY_HP : ENEMY_HP + 2] = hp.to_bytes(2, "big")
    p.memory[ENEMY_MAX_HP : ENEMY_MAX_HP + 2] = max_hp.to_bytes(2, "big")


def battle(p, kind):
    p.memory[0xD057] = kind


def script(env, steps):
    """Apply step i's RAM edit at that action's last frame, then step; return terms."""
    p = env.unwrapped.pyboy
    p.transitions = {24 * (i + 1): f for i, f in enumerate(steps) if f is not None}
    return [env.step(4)[4]["battle_reward"] for _ in steps]


def total(terms, key):
    return sum(t.get(key, 0.0) for t in terms)


def test_trainer_damage_ko_switch_and_win(fake_pyboy, config):
    env = BattleReward(
        RedGymEnv(dict(config, max_steps=8)),
        weights(trainer_damage=1, trainer_ko=0.5, trainer_win=2),
        gamma=1.0,
    )
    try:
        env.reset(seed=0)
        terms = script(
            env,
            [
                # Battle entry shows a stale damaged wild mon: never rewarded.
                lambda p: (battle(p, 2), enemy(p, 99, 3, 9)),
                lambda p: enemy(p, 10, 20),  # trainer's first mon loads
                lambda p: enemy(p, 10, 10),
                lambda p: enemy(p, 10, 0),
                lambda p: enemy(p, 11, 20),  # switch-in at full HP
                lambda p: enemy(p, 11, 0),
                lambda p: battle(p, 0),
            ],
        )
        assert terms[0] == {} and terms[1] == {} and terms[4] == {}
        assert terms[2] == {"trainer_damage": 0.5}
        assert terms[3] == {"trainer_damage": 0.5, "trainer_ko": 0.5}
        assert terms[5] == {"trainer_damage": 1.0, "trainer_ko": 0.5}
        assert terms[6] == {"trainer_win": 2.0}
        assert env.totals["trainer_wins"] == 1 and env.totals["trainer_kos"] == 2
    finally:
        env.close()


def test_reward_is_base_plus_battle_terms(fake_pyboy, config):
    cfg = dict(config, max_steps=4)
    plain = RedGymEnv(cfg)
    env = BattleReward(RedGymEnv(cfg), weights(trainer_damage=1), gamma=1.0)
    try:
        plain.reset(seed=0)
        env.reset(seed=0)
        edits = {24: lambda p: (battle(p, 2), enemy(p, 5, 20)), 48: lambda p: enemy(p, 5, 5)}
        plain.pyboy.transitions = dict(edits)
        env.unwrapped.pyboy.transitions = dict(edits)
        for _ in range(3):
            base = plain.step(4)
            out = env.step(4)
            assert out[1] == pytest.approx(base[1] + sum(out[4]["battle_reward"].values()))
            assert out[2:4] == base[2:4]
        assert env.totals["reward_trainer_damage"] == pytest.approx(0.75)
    finally:
        plain.close()
        env.close()


def test_wild_run_is_not_a_win_and_wild_terms_stop_at_level_cap(fake_pyboy, config):
    w = weights(wild_damage=1, wild_ko=1, wild_win=1)
    env = BattleReward(RedGymEnv(dict(config, max_steps=8)), w, gamma=1.0)
    try:
        env.reset(seed=0)
        terms = script(
            env,
            [
                lambda p: (p.set_mon(level=6), battle(p, 1), enemy(p, 7, 20)),
                lambda p: enemy(p, 7, 15),
                lambda p: battle(p, 0),  # ran away with enemy HP left
            ],
        )
        assert terms[1] == {"wild_damage": 0.25} and terms[2] == {}
        assert env.totals["wild_battles"] == 1 and not env.totals["wild_wins"]

        env.reset(seed=0)  # fake default party level 14 == level cap
        terms = script(
            env,
            [
                lambda p: (battle(p, 1), enemy(p, 7, 20)),
                lambda p: enemy(p, 7, 0),
                lambda p: battle(p, 0),
            ],
        )
        assert terms == [{}, {}, {}]
    finally:
        env.close()


def test_level_reward_counts_only_up_to_the_cap(fake_pyboy, config):
    env = BattleReward(RedGymEnv(dict(config, max_steps=8)), weights(level=1), gamma=1.0)
    try:
        env.reset(seed=0)
        terms = script(
            env,
            [
                lambda p: p.set_mon(level=12),  # an edit, not a level-up: no reward
                lambda p: p.set_mon(level=13),
                lambda p: p.set_mon(level=16),
                lambda p: p.set_mon(level=17),
            ],
        )
        assert [t.get("level", 0) for t in terms] == [0, 1, 1, 0]
    finally:
        env.close()


def test_hp_potential_cycle_cancels_and_wipe_drops_potential(fake_pyboy, config):
    env = BattleReward(RedGymEnv(dict(config, max_steps=8)), weights(hp_potential=1), gamma=1.0)
    try:
        env.reset(seed=0)
        terms = script(env, [lambda p: p.set_mon(hp=10), lambda p: p.set_mon(hp=20)])
        assert [t["hp"] for t in terms] == [pytest.approx(-0.5), pytest.approx(0.5)]
        env.reset(seed=0)
        env.unwrapped.pyboy.transitions = {24: lambda p: p.set_mon(hp=0)}
        _, _, terminated, _, info = env.step(4)
        assert terminated and info["end_reason"] == "wipe"
        assert info["battle_reward"]["hp"] == pytest.approx(-1.0)
    finally:
        env.close()


def test_rejected_trainer_run_and_overtime_penalties(fake_pyboy, config):
    w = weights(trainer_run_rejected=0.05, overtime=0.01, overtime_after=1)
    env = BattleReward(RedGymEnv(dict(config, max_steps=8)), w, gamma=1.0)
    try:
        env.reset(seed=0)
        env.unwrapped.pyboy.transitions = {24: lambda p: (battle(p, 2), enemy(p, 5, 20))}
        assert env.step(4)[4]["battle_reward"] == {}  # entry
        assert env.step(4)[4]["battle_reward"] == {}  # battle action 1
        env._hook("trainer_run_rejected")
        env._hook("trainer_run_rejected")
        terms = env.step(4)[4]["battle_reward"]
        assert terms == {"trainer_run_rejected": pytest.approx(-0.1), "overtime": -0.01}
        env.unwrapped.pyboy.memory[0xD057] = 1
        env._hook("trainer_run_rejected")  # only counted in trainer battles
        assert env.pending_rejected == 0
    finally:
        env.close()


def test_outcome_preset_win_scaled_by_hp_flee_zero_loss_minus_ten(fake_pyboy, config):
    env = BattleReward(RedGymEnv(dict(config, max_steps=8)), PRESETS["outcome"], gamma=1.0)
    try:
        env.reset(seed=0)
        terms = script(
            env,
            [
                lambda p: (battle(p, 2), enemy(p, 5, 20)),
                lambda p: (enemy(p, 5, 0), p.set_mon(hp=5)),  # damage/KO not rewarded
                lambda p: battle(p, 0),
            ],
        )
        assert terms == [{}, {}, {"win_hp": pytest.approx(2.5)}]  # 10 x 5/20 HP left

        env.reset(seed=0)
        terms = script(
            env,
            [
                lambda p: (p.set_mon(level=6), battle(p, 1), enemy(p, 7, 20)),
                lambda p: enemy(p, 7, 4),
                lambda p: battle(p, 0),  # fled
            ],
        )
        assert terms == [{}, {}, {}]

        env.reset(seed=0)
        env.unwrapped.pyboy.transitions = {
            24: lambda p: (battle(p, 2), enemy(p, 5, 20)),
            48: lambda p: p.set_mon(hp=0),
        }
        assert env.step(4)[4]["battle_reward"] == {}
        _, _, terminated, _, info = env.step(4)
        assert terminated and info["end_reason"] == "wipe"
        assert info["battle_reward"] == {"loss": -10.0}
        assert env.totals["trainer_losses"] == 1

        env.reset(seed=0)  # fainting outside battle (walking poison) is not a battle loss
        env.unwrapped.pyboy.transitions = {24: lambda p: p.set_mon(hp=0)}
        _, _, terminated, _, info = env.step(4)
        assert terminated and info["battle_reward"] == {}
    finally:
        env.close()


def test_rom_verification_rejects_other_roms(tmp_path):
    rom = tmp_path / "other.gb"
    rom.write_bytes(bytes(1 << 20))
    with pytest.raises(ValueError, match="verified Red ROM"):
        verify_rom(rom)


def make_bank(env, directory, kinds, config, lead=1):
    directory.mkdir()
    rows = []
    for i, kind in enumerate(kinds):
        env.reset(seed=i)
        battle(env.unwrapped.pyboy, kind)
        enemy(env.unwrapped.pyboy, 3, 20)
        for _ in range(lead):
            obs = env.step(4)[0]
        blob = capture(env.unwrapped, obs)
        name = f"snap-{i}.pkl.gz"
        (directory / name).write_bytes(blob)
        rows.append(
            {"file": name, "sha256": hashlib.sha256(blob).hexdigest(), "battle": kind, "map": 0}
        )
    (directory / "bank.json").write_text(json.dumps(rows))
    sha = snapshot_config_sha(config, "rom", "state")
    (directory / "bank_manifest.json").write_text(json.dumps({"config_sha256": sha}))
    return rows, sha


def test_battle_replay_starts_in_battle_and_ends_with_it(snapshot_pyboy, config, tmp_path):
    cfg = dict(config, observation_mode="pixels", max_steps=50)
    source = RedGymEnv(cfg)
    try:
        _, sha = make_bank(source, tmp_path / "bank", [1, 2], cfg)
    finally:
        source.close()
    replay = BattleReplay(RedGymEnv(cfg), tmp_path / "bank", trainer_share=1.0, config_sha=sha)
    env = BattleReward(replay, weights(trainer_win=2), gamma=1.0)
    try:
        obs, info = env.reset(seed=3)
        assert set(obs) == {"screens", "recent_actions"}
        assert info["training_reset"] == "battle" and info["battle_source"]["battle"] == 2
        assert env.unwrapped.read_m(0xD057) == 2 and env.battle == 2
        env.unwrapped.pyboy.memory[ENEMY_HP : ENEMY_HP + 2] = bytes(2)  # enemy faints
        env.step(4)
        env.unwrapped.pyboy.memory[0xD057] = 0
        _, _, terminated, truncated, info = env.step(4)
        assert truncated and not terminated and info["end_reason"] == "battle_over"
        assert info["battle_reward"] == {"trainer_win": 2.0}
        assert env.battle_stats()["replay"]["end_battle_over"] == 1
    finally:
        env.close()


def test_battle_replay_timeout_share_and_integrity(snapshot_pyboy, config, tmp_path):
    cfg = dict(config, max_steps=50)
    source = RedGymEnv(cfg)
    try:
        rows, _ = make_bank(source, tmp_path / "bank", [1, 1, 2], cfg)
    finally:
        source.close()
    env = BattleReplay(RedGymEnv(cfg), tmp_path / "bank", trainer_share=0.0, max_battle_actions=3)
    try:
        for _ in range(5):
            assert env.reset()[1]["battle_source"]["battle"] == 1
        results = [env.step(4) for _ in range(3)]
        assert [r[3] for r in results] == [False, False, True]
        assert results[-1][4]["end_reason"] == "battle_timeout"
        with pytest.raises(ValueError, match="different environment config"):
            BattleReplay(RedGymEnv(cfg), tmp_path / "bank", config_sha="other")
        (tmp_path / "bank" / rows[0]["file"]).write_bytes(b"corrupt")
        (tmp_path / "bank" / rows[1]["file"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="hash mismatch"):
            env.reset()
    finally:
        env.close()


def test_late_snapshot_gets_its_whole_battle_not_the_old_remaining_horizon(
    snapshot_pyboy, config, tmp_path
):
    cfg = dict(config, max_steps=4)
    source = RedGymEnv(cfg)
    try:
        make_bank(source, tmp_path / "bank", [2], cfg, lead=3)  # 1 action left
    finally:
        source.close()
    env = BattleReplay(RedGymEnv(cfg), tmp_path / "bank", trainer_share=1.0, max_battle_actions=3)
    try:
        assert env.reset()[1]["restart_original_step"] == 3
        results = [env.step(4) for _ in range(3)]
        assert [r[3] for r in results] == [False, False, True]
        assert results[-1][4]["end_reason"] == "battle_timeout"
        assert results[-1][4]["episode_steps"] == 3
    finally:
        env.close()


def test_snapshot_config_sha_ignores_reward_and_path_keys(config):
    base = snapshot_config_sha(config, "r", "s")
    moved = dict(config, session_path="/elsewhere", death_penalty=3, violation_penalty=1)
    assert snapshot_config_sha(moved, "r", "s") == base
    assert snapshot_config_sha(dict(config, max_steps=99), "r", "s") != base
    assert snapshot_config_sha(config, "r", "other") != base


def fake_blob(tag, catchable=False, used=()):
    tracker = SimpleNamespace(encounter_catchable=catchable, used_regions=set(used), region_map={})
    return gzip.compress(pickle.dumps({"tag": tag, "attributes": {"tracker": tracker}}))


def write_run(root, name, config, rows_by_worker):
    run = root / name
    (run).mkdir(parents=True)
    manifest = {"config": config, "rom_sha1": "r", "state_sha1": "s"}
    (run / "manifest.json").write_text(json.dumps(manifest, default=str))
    for worker, rows in rows_by_worker.items():
        d = run / "failure-banks" / f"worker-{worker}"
        d.mkdir(parents=True)
        out = []
        for i, (battle_type, map_id, tag, *extra) in enumerate(rows):
            name = f"failure-{i}.pkl.gz"
            blob = fake_blob(tag, *extra)
            (d / name).write_bytes(blob)
            out.append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "battle": battle_type,
                    "map": map_id,
                    "mode": "transition",
                    "original_step": i,
                    "lead_actions": 100,
                    "end_reason": "time_limit",
                }
            )
        (d / "bank.json").write_text(json.dumps(out))
    return run


def test_bank_builder_filters_groups_dedups_and_checks(config, tmp_path):
    a = write_run(
        tmp_path,
        "a",
        config,
        {0: [(0, 12, "overworld"), (1, 12, "w1"), (1, 12, "w2"), (1, 12, "w3")]},
    )
    b = write_run(tmp_path, "b", config, {0: [(2, 51, "t1"), (1, 12, "w1")]})
    out = tmp_path / "bank"
    build_battle_bank.main([str(a), str(b), "--output", str(out), "--per-group", "2"])
    bank = json.loads((out / "bank.json").read_text())
    manifest = json.loads((out / "bank_manifest.json").read_text())
    assert sorted((r["battle"], r["map"]) for r in bank) == [(1, 12), (1, 12), (2, 51)]
    assert len({r["sha256"] for r in bank}) == 3
    assert manifest["config_sha256"] == snapshot_config_sha(config, "r", "s")
    for row in bank:
        blob = (out / row["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row["sha256"]

    other = write_run(tmp_path, "c", dict(config, max_steps=99), {0: [(2, 51, "t9")]})
    with pytest.raises(SystemExit):
        build_battle_bank.main([str(a), str(other), "--output", str(tmp_path / "mixed")])
    (a / "failure-banks/worker-0/failure-1.pkl.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Hash mismatch"):
        build_battle_bank.main([str(a), "--output", str(tmp_path / "bad"), "--per-group", "9"])


def test_bank_builder_balances_legal_and_illegal_catches(config, tmp_path):
    rows = [(1, 13, f"illegal{i}", False) for i in range(6)]
    rows += [(1, 13, "legal0", True), (1, 13, "legal1", True), (2, 51, "trainer")]
    run = write_run(tmp_path, "a", config, {0: rows})
    out = tmp_path / "bank"
    build_battle_bank.main([str(run), "--output", str(out), "--per-group", "2"])
    bank = json.loads((out / "bank.json").read_text())
    assert sorted(str(r["catchable"]) for r in bank) == ["False", "False", "None", "True", "True"]
    counts = json.loads((out / "bank_manifest.json").read_text())["counts"]
    assert counts == {
        "battle1/map13/battle_entry/catchable": 2,
        "battle1/map13/battle_entry/illegal": 2,
        "battle2/map51/battle_entry": 1,
    }


def test_memory_consistent_bank_keeps_only_starts_a_fresh_lstm_can_judge(config, tmp_path):
    cfg = dict(config, initial_used_regions=[12])
    rows = [
        (1, 13, "route2-illegal", False),  # an earlier Route 2 encounter the LSTM never saw
        (1, 13, "route2-first", True),
        (1, 12, "route1", False),  # Route 1 is used in the starting save
        (0, 13, "route2-entry", False, ()),
        (0, 13, "route2-after-encounter", False, (13,)),
        (0, 1, "town", False, ()),  # not an --area-maps map
        (2, 51, "trainer"),
    ]
    run = write_run(tmp_path, "a", cfg, {0: rows})
    out = tmp_path / "bank"
    build_battle_bank.main(
        [str(run), "--output", str(out), "--area-maps", "13", "--memory-consistent"]
    )
    bank = json.loads((out / "bank.json").read_text())
    kept = sorted((r["battle"], r["map"], r["catchable"], r["area_fresh"]) for r in bank)
    assert kept == [
        (0, 13, None, True),
        (1, 12, False, None),
        (1, 13, True, None),
        (2, 51, None, None),
    ]


def test_legal_catch_rewarded_and_illegal_catch_is_a_violation(fake_pyboy, config):
    env = BattleReward(RedGymEnv(dict(config, max_steps=8)), PRESETS["outcome"], gamma=1.0)
    try:
        env.reset(seed=0)
        terms = script(
            env,
            [
                lambda p: (battle(p, 1), enemy(p, 7, 20)),  # first encounter here: legal
                lambda p: (p.set_mon(slot=1, identity=(7, 7)), p.memory.__setitem__(0xD163, 2)),
                lambda p: battle(p, 0),
                lambda p: (battle(p, 1), enemy(p, 8, 20)),  # second encounter: illegal
                lambda p: (p.set_mon(slot=2, identity=(8, 8)), p.memory.__setitem__(0xD163, 3)),
            ],
        )
        assert terms[1] == {"legal_catch": 5.0} and terms[2] == {}
        assert terms[4] == {}
        assert env.totals["legal_catches"] == 1 and env.totals["illegal_catches"] == 1
        assert env.unwrapped.tracker.violations == {"illegal_catch"}
    finally:
        env.close()


def make_area_bank(env, directory, config):
    directory.mkdir()
    env.reset(seed=0)
    blob = capture(env, env.step(4)[0])
    rows = [{"file": "area.pkl.gz", "sha256": hashlib.sha256(blob).hexdigest(), "battle": 0}]
    rows[0]["map"] = 0
    (directory / "area.pkl.gz").write_bytes(blob)
    (directory / "bank.json").write_text(json.dumps(rows))
    sha = snapshot_config_sha(config, "rom", "state")
    (directory / "bank_manifest.json").write_text(json.dumps({"config_sha256": sha}))


def test_area_start_runs_through_battles_and_ends_on_leaving_the_map(
    snapshot_pyboy, config, tmp_path
):
    cfg = dict(config, max_steps=50)
    source = RedGymEnv(cfg)
    try:
        make_area_bank(source, tmp_path / "bank", cfg)
    finally:
        source.close()
    env = BattleReplay(
        RedGymEnv(cfg), tmp_path / "bank", area_share=1.0, max_battle_actions=3, max_area_actions=9
    )
    try:
        assert env.reset()[1]["training_reset"] == "area"
        p = env.unwrapped.pyboy
        battle(p, 1)
        assert not env.step(4)[3]  # a battle inside the area keeps the episode going
        battle(p, 0)
        assert not env.step(4)[3]
        p.set_map(1)
        _, _, terminated, truncated, info = env.step(4)
        assert truncated and not terminated and info["end_reason"] == "left_area"

        env.reset()
        battle(env.unwrapped.pyboy, 2)
        results = [env.step(4) for _ in range(3)]
        assert results[-1][3] and results[-1][4]["end_reason"] == "battle_timeout"
        assert results[-1][1] < results[0][1] - 0.5  # timeout penalty applied

        env.reset()
        results = [env.step(4) for _ in range(9)]
        assert [r[3] for r in results].index(True) == 8
        assert results[-1][4]["end_reason"] == "area_timeout"
    finally:
        env.close()


def test_catch_in_progress_detects_a_catch_that_needs_no_input(snapshot_pyboy, config):
    env = RedGymEnv(dict(config, max_steps=50))
    try:
        env.reset(seed=0)
        battle(env.pyboy, 1)
        blob = capture(env, env.step(4)[0])  # restored emulator resumes at frame 24
        assert not catch_in_progress(env, blob, actions=4)
        env.pyboy.transitions = {72: lambda p: p.memory.__setitem__(0xD163, 2)}
        assert catch_in_progress(env, blob, actions=4)
    finally:
        env.close()


def test_freeze_memory_keeps_cnn_and_actor_lstm_and_trains_the_rest():
    import torch
    from battle_experiment import freeze_memory
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from test_context_aux import TinyPixels

    torch.set_num_threads(1)
    env = DummyVecEnv([TinyPixels])
    try:
        model = RecurrentPPO(
            "MultiInputLstmPolicy", env, n_steps=8, batch_size=8, seed=0, device="cpu"
        )
        frozen = freeze_memory(model.policy)
        assert any(n.startswith("lstm_actor.") for n in frozen)
        assert any(n.startswith("features_extractor.") for n in frozen)
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        model.learn(16)
        after = model.policy.state_dict()
        for key, value in before.items():
            same = torch.equal(value, after[key])
            if key.startswith(("lstm_actor.", "features_extractor.", "pi_features_extractor.")):
                assert same, key
        assert not torch.equal(before["action_net.weight"], after["action_net.weight"])
        assert not torch.equal(
            before["lstm_critic.weight_ih_l0"], after["lstm_critic.weight_ih_l0"]
        )
    finally:
        env.close()
