import gzip
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from lstm_memory import (
    HEAD_DIM,
    HEAD_SLICES,
    NAMES,
    TARGET_NAMES,
    MemoryModel,
    label_loss,
    read_labels,
    targets_from,
)
from memory_addresses import BATTLE_TYPE_ADDRESS
from red_gym_env_v2_nuzlocke import RedGymEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts import build_memory_dataset, train_memory

CATCH = NAMES.index("encounter_catchable")
ENEMY = NAMES.index("enemy_hp")


def test_catchable_label_follows_the_area_and_the_encounter(fake_pyboy, config):
    env = RedGymEnv(dict(config, max_steps=8))
    try:
        env.reset(seed=0)
        labels, mask = read_labels(env, 0)
        assert mask[CATCH] == 0  # map 0 has no wild encounters
        env.pyboy.set_map(12, 3, 3)
        env.step(4)
        labels, mask = read_labels(env, 0)
        assert mask[CATCH] == 1 and labels[CATCH] == 1 and mask[ENEMY] == 0
        env.pyboy.memory[0xD057] = 1  # first encounter on Route 1
        env.pyboy.memory[0xCFF5] = 20
        env.step(4)
        labels, mask = read_labels(env, 0)
        assert labels[CATCH] == 1 and mask[ENEMY] == 1
        env.pyboy.memory[0xD057] = 0
        env.step(4)
        labels, mask = read_labels(env, 0)  # the area is now used
        assert mask[CATCH] == 1 and labels[CATCH] == 0
    finally:
        env.close()


def test_label_loss_masks_items_and_balances_rare_positives():
    n = len(TARGET_NAMES)
    prediction, targets, mask = torch.zeros(4, HEAD_DIM), torch.ones(4, n), torch.zeros(4, n)
    mask[:2, CATCH] = 1
    loss, per_target = label_loss(prediction, targets, mask)
    assert loss.item() == pytest.approx(np.log(2), rel=1e-5)  # BCE at logit 0
    assert per_target[CATCH].item() == pytest.approx(np.log(2), rel=1e-5)
    assert np.isnan(per_target[ENEMY].item())  # absent targets are left out of the mean
    weight = torch.ones(n)
    weight[CATCH] = 3.0
    weighted, _ = label_loss(prediction, targets, mask, weight)
    assert weighted.item() == pytest.approx(3 * np.log(2), rel=1e-5)


def test_battle_type_is_a_class_balanced_three_way_target():
    n, battle = len(TARGET_NAMES), TARGET_NAMES.index("battle_type")
    targets, mask = torch.zeros(4, n), torch.zeros(4, n)
    targets[:, battle] = torch.tensor([0.0, 0.0, 0.0, 2.0])  # three overworld, one trainer
    mask[:, battle] = 1
    prediction = torch.zeros(4, HEAD_DIM)
    prediction[3, HEAD_SLICES["battle_type"]] = torch.tensor([0.0, 0.0, 5.0])
    loss, _ = label_loss(prediction, targets, mask)
    # Each class present counts equally: the confident trainer row is not outvoted.
    overworld = np.log(3)
    trainer = -np.log(np.exp(5) / (2 + np.exp(5)))
    assert loss.item() == pytest.approx((overworld + trainer) / 2, rel=1e-5)


def test_targets_add_battle_type_and_antidote_availability():
    labels = np.zeros((3, len(NAMES)), np.float32)
    mask = np.ones((3, len(NAMES)), np.uint8)
    labels[1, NAMES.index("antidotes")] = 0.4
    targets, target_mask = targets_from(labels, mask, np.array([0, 2, 0xFF], np.uint8))
    battle, antidote = TARGET_NAMES.index("battle_type"), TARGET_NAMES.index("has_antidote")
    assert targets[:, battle].tolist() == [0, 2, -1] and target_mask[:, battle].tolist() == [
        1,
        1,
        0,
    ]
    assert targets[:, antidote].tolist() == [0, 1, 0]
    assert targets.shape == (3, len(TARGET_NAMES))


def test_control_standardises_and_memory_loads_the_policy_lstm():
    mean, std = torch.full((4,), 2.0), torch.full((4,), 4.0)
    control = MemoryModel(4, recurrent=False, mean=mean, std=std)
    out, state = control(torch.full((1, 1, 4), 6.0))
    assert state is None
    torch.testing.assert_close(out, control.head(torch.ones(1, 1, 4)))  # (6 - 2) / 4
    assert out.shape[-1] == HEAD_DIM
    policy_lstm = torch.nn.LSTM(4, 8)
    memory = MemoryModel(4, hidden=8)
    memory.load_policy_lstm(policy_lstm)
    for key, value in policy_lstm.state_dict().items():
        assert torch.equal(value, memory.lstm.state_dict()[key])


def write_trace_dir(env, directory, actions_by_episode, config):
    """Play episodes in order like diagnose_forest and record its per-action trace."""
    directory.mkdir(parents=True)
    records = []
    for index, actions in enumerate(actions_by_episode):
        env.reset(seed=6000 + index)
        rows = []
        for step, action in enumerate(actions, 1):
            _, reward, terminated, truncated, info = env.step(action)
            party = [asdict(m) for m in env.read_party_mons()]
            rows.append(
                {
                    "step": step,
                    "action": build_memory_dataset.BUTTONS[action],
                    "reward": reward,
                    "position": list(env.get_game_coords()),
                    "battle": env.read_m(BATTLE_TYPE_ADDRESS),
                    "party": json.loads(json.dumps(party)),
                }
            )
            if terminated or truncated:
                break
        with gzip.open(directory / f"episode-{index:02d}-actions.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        records.append({"evaluation_seed": 6000 + index, "info": info})
    summary = {"config": {k: v for k, v in config.items() if k != "session_path"}}
    summary["records"] = records
    (directory / "summary.json").write_text(json.dumps(summary))


def test_replay_builds_verified_episodes_and_rejects_divergence(fake_pyboy, config, tmp_path):
    cfg = dict(config, observation_mode="pixels", max_steps=5, session_path=tmp_path / "rec")
    env = RedGymEnv(cfg)
    try:
        write_trace_dir(env, tmp_path / "diag", [[4, 5, 0, 1, 2], [3, 3, 4, 6, 4]], cfg)
    finally:
        env.close()
    plain = DummyVecEnv([lambda: RedGymEnv(dict(cfg, session_path=tmp_path / "p"))])
    RecurrentPPO("MultiInputLstmPolicy", plain, n_steps=8, batch_size=8).save(tmp_path / "m.zip")
    plain.close()
    task = (tmp_path / "diag", tmp_path / "out", cfg["gb_path"], cfg["init_state"])
    _, written = build_memory_dataset.replay_directory(task + (tmp_path / "m.zip", 3))
    assert len(written) == 2
    episode = tmp_path / "out" / written[0]
    assert np.load(episode / "features.npy").shape == (5, 277)
    assert np.load(episode / "labels.npy").shape == (5, len(NAMES))  # RAM labels as stored

    trace = tmp_path / "diag" / "episode-01-actions.jsonl.gz"
    with gzip.open(trace, "rt") as f:
        rows = [json.loads(line) for line in f]
    rows[2]["reward"] += 1.0
    with gzip.open(trace, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with pytest.raises(ValueError, match="diverged"):
        build_memory_dataset.replay_directory(
            (tmp_path / "diag", tmp_path / "out2", cfg["gb_path"], cfg["init_state"])
            + (tmp_path / "m.zip", 3)
        )


def synthetic_dataset(directory, episodes=60, length=120, seed=0):
    """A label set by a one-action cue earlier in the episode: memory, not the screen."""
    rng = np.random.default_rng(seed)
    for e in range(episodes):
        features = np.zeros((length, 277), np.float32)
        features[:, 1:9] = rng.normal(size=(length, 8))  # distractors
        cue = int(rng.integers(20, length - 20))
        features[cue, 0] += 3.0
        labels = np.zeros((length, len(NAMES)), np.float32)
        labels[cue + 1 :, CATCH] = 1.0
        mask = np.zeros((length, len(NAMES)), np.uint8)
        mask[:, CATCH] = 1
        path = directory / f"ep{e:02d}"
        path.mkdir(parents=True)
        np.save(path / "features.npy", features.astype(np.float16))
        np.save(path / "labels.npy", labels)
        np.save(path / "mask.npy", mask)
        # 0xFF (no battle type) masks the battle-type target: only memory is tested here.
        np.save(path / "battle.npy", np.full(length, 0xFF, np.uint8))
        np.save(path / "map.npy", np.zeros(length, np.uint8))
        meta = {"end_reason": "time_limit", "violation_reasons": [], "steps": length}
        (path / "meta.json").write_text(json.dumps(meta))


def test_trained_memory_beats_the_no_memory_control(fake_pyboy, config, tmp_path):
    synthetic_dataset(tmp_path / "ds")
    cfg = dict(config, observation_mode="pixels", session_path=tmp_path / "p")
    plain = DummyVecEnv([lambda: RedGymEnv(cfg)])
    # A stand-in parent is only needed to load; the memory model starts from a seeded
    # random LSTM (loading the parent's LSTM is tested separately).
    parent = RecurrentPPO("MultiInputLstmPolicy", plain, n_steps=8, batch_size=8, seed=0)
    parent.save(tmp_path / "m.zip")
    plain.close()
    torch.set_num_threads(1)
    train_memory.main(
        [
            str(tmp_path / "ds"),
            "--output",
            str(tmp_path / "out"),
            "--feature-model",
            str(tmp_path / "m.zip"),
            "--models",
            "input",
            "memory",
            "--epochs",
            "15",
            "--chunk",
            "64",
            "--batch-episodes",
            "8",
            "--lr",
            "3e-3",
            "--init",
            "random",
        ]
    )
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    control = metrics["models"]["input"]["test"]["encounter_catchable"]
    memory = metrics["models"]["memory"]["test"]["encounter_catchable"]
    assert memory > 0.85 and control < 0.65
    assert (tmp_path / "out" / "memory_model.pt").exists()


def test_menu_peeks_are_spliced_flagged_and_leave_the_replay_verified(
    snapshot_pyboy, config, tmp_path, monkeypatch
):
    cfg = dict(config, observation_mode="pixels", max_steps=5, session_path=tmp_path / "rec")
    env = RedGymEnv(cfg)
    try:
        write_trace_dir(env, tmp_path / "diag", [[4, 5, 0, 1, 2]], cfg)
    finally:
        env.close()
    plain = DummyVecEnv([lambda: RedGymEnv(dict(cfg, session_path=tmp_path / "p"))])
    RecurrentPPO("MultiInputLstmPolicy", plain, n_steps=8, batch_size=8).save(tmp_path / "m.zip")
    plain.close()
    calls = []

    def fake_peek(peek_env, kind, last_wild):
        calls.append(kind)
        if len(calls) == 2:
            return None  # a menu that did not open is discarded
        obs = peek_env.step(6)[0]
        return [(obs, *read_labels(peek_env, last_wild))] * 3

    monkeypatch.setattr(build_memory_dataset, "menu_peek", fake_peek)
    monkeypatch.setattr(build_memory_dataset, "can_peek", lambda env: True)
    task = (tmp_path / "diag", tmp_path / "out", cfg["gb_path"], cfg["init_state"])
    _, written = build_memory_dataset.replay_directory(task + (tmp_path / "m.zip", 4, 1.0, 0))
    episode = tmp_path / "out" / written[0]
    peek = np.load(episode / "peek.npy")
    meta = json.loads((episode / "meta.json").read_text())
    assert len(calls) == 5 and peek.sum() == 3 * 4 and len(peek) == 5 + 12
    assert np.load(episode / "features.npy").shape[0] == len(peek)
    assert meta["peek_actions"] == 12 and sum(meta["peeks"].values()) == 5
    assert not peek[0] and peek[1:4].all() and not peek[4]  # peek follows its source step
