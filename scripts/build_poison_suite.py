"""Build natural menu/HP curriculum variants without RAM edits, then score real cures."""

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from poison_curriculum import PoisonEnv, antidote_count
from red_gym_env_v2_nuzlocke import RedGymEnv

BUTTONS = ("down", "left", "right", "up", "A", "B", "start")
CURE_ACTIONS = [(6, 24), (0, 0), (0, 0), (4, 24), (0, 0), (4, 24), (4, 24), (4, 120)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-state", type=Path, required=True)
    p.add_argument("--train-metadata", type=Path, required=True)
    p.add_argument("--heldout-state", type=Path, required=True)
    p.add_argument("--heldout-metadata", type=Path, required=True)
    p.add_argument("--source-actions", type=Path, required=True)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train_meta = json.loads(args.train_metadata.read_text())
    heldout_meta = json.loads(args.heldout_metadata.read_text())
    scenarios = []

    def config(state, meta):
        return {
            "session_path": str(args.output / "emulator"),
            "gb_path": str(args.rom.resolve()),
            "init_state": str(state.resolve()),
            "headless": True,
            "save_final_state": False,
            "print_rewards": False,
            "save_video": False,
            "fast_video": True,
            "action_freq": 24,
            "max_steps": 2000,
            "reward_scale": 0.5,
            "explore_weight": 0,
            "observation_mode": "pixels",
            "include_explore_map": False,
            "position_mode": "coherent-v1",
            "initial_used_regions": meta["initial_used_regions"],
        }

    def idle(env, count):
        for _ in range(count):
            env.pyboy.tick(1, True)
            env._observe_rules()

    def save(env, name, split, weight=1):
        if env.tracker.terminated or env.tracker.dead or env.read_m(0xD057) != 0:
            raise ValueError("Unsafe or in-battle fixture")
        state = args.output / f"{name}.state"
        with state.open("wb") as f:
            env.pyboy.save_state(f)
        Image.fromarray(env.pyboy.screen.ndarray[:, :, :3]).save(state.with_suffix(".png"))
        scenarios.append(
            {
                "id": name,
                "split": split,
                "weight": weight,
                "state_file": state.name,
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "initial_used_regions": sorted(env.tracker.used_regions),
                "dead": [],
                "ram_edited": False,
                "poisoned": any(m.status & 8 for m in env.read_party_mons()),
                "antidotes": antidote_count(env.pyboy.memory),
                "party": [asdict(m) for m in env.read_party_mons()],
                "position": list(env.get_game_coords()),
            }
        )

    for name, prefix, weight in (
        ("train-overworld", 0, 6),
        ("train-bag", 4, 3),
        ("train-party", 7, 1),
    ):
        env = RedGymEnv(config(args.train_state, train_meta))
        try:
            env.reset(seed=0)
            for action, wait in CURE_ACTIONS[:prefix]:
                env.step(action)
                idle(env, wait)
            if not any(m.hp > 0 and m.status & 8 for m in env.read_party_mons()):
                raise ValueError("Preparation accidentally cured the scenario")
            save(env, name, "train", weight)
        finally:
            env.close()

    env = RedGymEnv(config(args.heldout_state, heldout_meta))
    try:
        env.reset(seed=0)
        idle(env, 168)
        save(env, "heldout-other-location", "heldout")
    finally:
        env.close()

    trace = json.loads((args.source_actions.parent / "summary.json").read_text())
    target_episode = int(args.source_actions.name.split("-")[1])
    replay_config = dict(trace["config"])
    replay_config.update(
        gb_path=str(args.rom.resolve()),
        init_state=str((ROOT / "project_work" / Path(replay_config["init_state"]).name).resolve()),
        session_path=str(args.output / "replay-emulator"),
        save_video=False,
    )
    env = RedGymEnv(replay_config)
    try:
        thresholds = [12, 6]
        for episode in range(target_episode + 1):
            env.reset(seed=trace["records"][episode]["evaluation_seed"])
            action_file = args.source_actions.parent / f"episode-{episode:02d}-actions.jsonl.gz"
            with gzip.open(action_file, "rt") as f:
                for line in f:
                    action = json.loads(line)
                    _, _, terminated, truncated, _ = env.step(BUTTONS.index(action["action"]))
                    mon = env.read_party_mons()[0]
                    if (
                        list(env.get_game_coords()) != action["position"]
                        or mon.hp != action["party"][0]["hp"]
                    ):
                        raise AssertionError("Source replay diverged; do not use its HP fixtures")
                    if terminated or truncated:
                        break
                    if (
                        episode == target_episode
                        and thresholds
                        and 0 < mon.hp <= thresholds[0]
                        and mon.status & 8
                        and env.read_m(0xD057) == 0
                        and env.position.valid
                        and antidote_count(env.pyboy.memory) > 0
                    ):
                        threshold = thresholds.pop(0)
                        save(env, f"heldout-low-hp-{threshold}", "heldout")
                    if episode == target_episode and not thresholds:
                        break
        if thresholds:
            raise ValueError("Natural replay did not produce both lower-HP fixtures")
    finally:
        env.close()

    env = RedGymEnv(config(args.train_state, train_meta))
    try:
        env.reset(seed=0)
        for action, wait in CURE_ACTIONS:
            env.step(action)
            idle(env, wait)
        for _ in range(6):
            env.step(5)
            idle(env, 24)
        if any(m.status & 8 for m in env.read_party_mons()):
            raise ValueError("Healthy control was not cured")
        save(env, "healthy-control", "healthy")
    finally:
        env.close()

    suite = {
        "format": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "rom_sha1": hashlib.sha1(args.rom.read_bytes()).hexdigest(),
        "source_state_sha256": hashlib.sha256(args.train_state.read_bytes()).hexdigest(),
        "scenarios": scenarios,
        "contract": "Natural game states, no RAM edits, no prior deaths. Region history preserved; recurrent and exploration history reset. Heldout low-HP fixtures share the source episode; not independent game seeds.",
    }
    suite_path = args.output / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n")
    checks = []
    for scale in (2, 1):
        cfg = config(args.train_state, train_meta)
        cfg.update(task="cure", scenario_suite=str(suite_path.resolve()), screen_scale=scale)
        env = PoisonEnv(cfg)
        try:
            obs, _ = env.reset(seed=0, options={"scenario_index": 0})
            assert env.observation_space.contains(obs) and set(obs) == {"screens", "recent_actions"}
            # Only policy-valid buttons at the standard 24-frame cadence. Repeated
            # A actions allow the menu animations to complete without an extra wait action.
            actions = [6, 0, 0, 4, 0, 4] + [4] * 24
            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if not info["is_success"]:
                raise AssertionError(f"Real-ROM cure did not earn success at scale {scale}: {info}")
            checks.append(
                {
                    "screen_scale": scale,
                    "action_sequence": actions[: info["episode_steps"]],
                    "info": info,
                    "reward_at_success": reward,
                }
            )
            env.reset(seed=0, options={"scenario_index": 2})
            for _ in range(24):
                _, _, done, trunc, party_info = env.step(4)
                if done or trunc:
                    break
            assert party_info["is_success"]
        finally:
            env.close()
    (args.output / "real_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(
        json.dumps(
            {"scenarios": len(scenarios), "real_cures": len(checks), "suite": str(suite_path)}
        )
    )


if __name__ == "__main__":
    main()
