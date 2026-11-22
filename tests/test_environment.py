import json
from pathlib import Path

import numpy as np
import pytest
from experiment import ALGORITHMS, EpisodeCallback, evaluate_main, make_env, resume_main, train_main
from red_gym_env_v2_nuzlocke import RedGymEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


@pytest.mark.parametrize("mode", ["partial", "history", "pixels"])
@pytest.mark.parametrize("explore_map", [True, False])
def test_observation_contract(fake_pyboy, config, mode, explore_map):
    env = RedGymEnv(dict(config, observation_mode=mode, include_explore_map=explore_map))
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=4)
        assert env.observation_space.contains(obs)
        assert ("party_dead" in obs) == (mode == "history")
        assert ("used_regions" in obs) == (mode == "history")
        assert ("map" in obs) == (explore_map and mode != "pixels")
        if mode == "pixels":
            assert set(obs) == {"screens", "recent_actions"}
    finally:
        env.close()
    assert env.pyboy.stopped


def test_individual_deaths_detected_inside_action(fake_pyboy, config):
    env = RedGymEnv(config)
    env.reset()
    env.pyboy.memory[0xD163] = 2
    env.pyboy.set_mon(slot=1, identity=(123, 789))
    env.pyboy.transitions = {2: lambda p: p.set_mon(hp=0), 3: lambda p: p.set_mon(hp=20)}
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["deaths"] == 1
    assert info["violation_reasons"] == ["revived_dead_mon"]
    env.close()


def test_wipe_is_termination_and_time_limit_is_truncation(fake_pyboy, config):
    env = RedGymEnv(dict(config, max_steps=1))
    env.reset()
    _, _, term, trunc, info = env.step(0)
    assert not term and trunc and info["end_reason"] == "time_limit"
    env.reset()
    env.pyboy.transitions = {2: lambda p: p.set_mon(hp=0)}
    _, _, term, trunc, info = env.step(0)
    assert term and not trunc and info["end_reason"] == "wipe"
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.close()


def test_success(fake_pyboy, config):
    env = RedGymEnv(config)
    env.reset()
    env.pyboy.transitions = {1: lambda p: p.memory.__setitem__(0xD356, 1)}
    _, _, term, trunc, info = env.step(0)
    assert term and not trunc and info["is_success"]
    env.close()


def test_milestones_use_lsb_bits_and_persist(fake_pyboy, config):
    env = RedGymEnv(config)
    env.reset()
    env.pyboy.memory[0xD74B] = 1 << 5
    env.pyboy.memory[0xD74E] = 1 << 0
    env.pyboy.set_map(2)
    *_, info = env.step(0)
    assert info["milestones"]["got_pokedex"]
    assert info["milestones"]["delivered_parcel"]
    assert not info["milestones"]["got_parcel"]
    env.pyboy.set_map(0)
    *_, info = env.step(0)
    assert info["milestones"]["reached_pewter"]
    env.close()


def test_map_edge_padding(fake_pyboy, config):
    env = RedGymEnv(config)
    env.reset()
    env.get_global_coords = lambda: (0, 0)
    env.explore_map[0, 0] = 255
    image = env.get_explore_map()
    assert image.shape == (48, 48)
    assert image[24, 24] == 255 and np.count_nonzero(image) == 4
    env.close()


def test_episode_log_survives_vector_auto_reset(fake_pyboy, config, tmp_path):
    vec = DummyVecEnv([lambda: make_env(dict(config, max_steps=1), tmp_path, 0)])
    vec.reset()
    for _ in range(3):
        _, _, dones, infos = vec.step([0])
        assert dones[0] and infos[0]["end_reason"] == "time_limit"
    vec.close()
    records = [
        json.loads(line) for line in (tmp_path / "episodes-0.jsonl").read_text().splitlines()
    ]
    assert [r["episode_index"] for r in records] == [0, 1, 2]
    assert all(r["truncated"] for r in records)


@pytest.mark.parametrize("algorithm", ["ppo", "lstm"])
@pytest.mark.parametrize("mode", ["partial", "pixels"])
def test_actual_policy_train_save_reload_evaluate(fake_pyboy, config, tmp_path, algorithm, mode):
    output = tmp_path / algorithm
    train_main(
        [
            "--rom",
            config["gb_path"],
            "--state",
            config["init_state"],
            "--output",
            str(output),
            "--algorithm",
            algorithm,
            "--observation-mode",
            mode,
            "--timesteps",
            "16",
            "--n-steps",
            "8",
            "--batch-size",
            "8",
            "--n-epochs",
            "1",
            "--max-episode-steps",
            "2",
        ]
    )
    assert (output / "final.zip").is_file()
    assert json.loads((output / "training.json").read_text())["actual_timesteps"] == 16
    assert len((output / "episodes-0.jsonl").read_text().splitlines()) == 8
    evaluate_main([str(output), "--episodes", "2"])
    result = json.loads((output / "evaluation.json").read_text())
    assert result["episodes"] == 2
    assert result["end_reasons"] == {"time_limit": 2}
    evaluate_main(
        [
            str(output),
            "--episodes",
            "2",
            "--deterministic",
            "--output",
            str(output / "deterministic.json"),
        ]
    )
    deterministic = json.loads((output / "deterministic.json").read_text())
    assert deterministic["success_wilson_95"] is None


def test_callback_reads_terminal_info_only():
    callback = EpisodeCallback()
    captured = {}

    class Logger:
        def record_mean(self, key, value):
            captured[key] = value

    callback.model = type("Model", (), {"logger": Logger()})()
    callback.locals = {
        "dones": [True],
        "infos": [
            {
                "is_success": False,
                "deaths": 1,
                "map_coverage": 2,
                "episode_steps": 3,
                "end_reason": "wipe",
                "violation_reasons": [],
            }
        ],
    }
    assert callback._on_step()
    assert captured["nuzlocke/wipe"] == 1


@pytest.mark.parametrize("algorithm", ["ppo", "lstm"])
def test_resume_partial_checkpoint_preserves_optimizer_and_logs(
    fake_pyboy, config, tmp_path, algorithm
):
    source = tmp_path / "source"
    train_main(
        [
            "--rom",
            config["gb_path"],
            "--state",
            config["init_state"],
            "--output",
            str(source),
            "--algorithm",
            algorithm,
            "--timesteps",
            "24",
            "--n-steps",
            "8",
            "--batch-size",
            "8",
            "--n-epochs",
            "1",
            "--max-episode-steps",
            "2",
            "--checkpoint-every",
            "8",
        ]
    )
    original_manifest = (source / "manifest.json").read_bytes()
    original_log = (source / "episodes-0.jsonl").read_bytes()
    checkpoint = source / "checkpoints/policy_16_steps.zip"
    before = ALGORITHMS[algorithm].load(checkpoint)
    previous_optimizer_steps = max(
        state["step"].item() for state in before.policy.optimizer.state.values()
    )
    relocated_rom = tmp_path / "relocated.gb"
    relocated_rom.write_bytes(Path(config["gb_path"]).read_bytes())
    output = tmp_path / "resumed"
    resume_main(
        [
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--timesteps",
            "8",
            "--rom",
            str(relocated_rom),
        ]
    )
    after = ALGORITHMS[algorithm].load(output / "final.zip")
    assert before.num_timesteps == 16 and after.num_timesteps == 24
    assert after._n_updates > before._n_updates
    assert max(state["step"].item() for state in after.policy.optimizer.state.values()) > (
        previous_optimizer_steps
    )
    result = json.loads((output / "training.json").read_text())
    assert result["initial_timesteps"] == 16
    assert result["additional_timesteps"] == 8
    assert result["actual_timesteps"] == 24
    assert (source / "manifest.json").read_bytes() == original_manifest
    assert (source / "episodes-0.jsonl").read_bytes() == original_log
    records = [json.loads(s) for s in (output / "episodes-0.jsonl").read_text().splitlines()]
    assert len(records) == 4 and records[0]["episode_index"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["resume"]["episodes_restarted"]
    assert manifest["config"]["gb_path"] == str(relocated_rom)
    evaluate_main([str(output), "--episodes", "2"])
    assert json.loads((output / "evaluation.json").read_text())["episodes"] == 2
    relocated_rom.write_bytes(b"different ROM")
    with pytest.raises(SystemExit):
        resume_main(
            [
                str(source),
                "--output",
                str(tmp_path / "bad-asset"),
                "--timesteps",
                "8",
                "--rom",
                str(relocated_rom),
            ]
        )
    assert not (tmp_path / "bad-asset").exists()
