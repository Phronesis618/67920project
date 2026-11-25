"""Replay recorded actions to audit antidote use and capture natural poison saves.

Every action's party, position and battle state must match the original trace.
No RAM writes or policy changes. Captured saves need their used-region metadata
when starting a new rules-from-reset curriculum; they contain no recurrent state.
"""

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
from red_gym_env_v2_nuzlocke import RedGymEnv

BUTTONS = ("down", "left", "right", "up", "A", "B", "start")


def bag(env):
    count = env.read_m(0xD31D)
    if count > 20:
        raise ValueError("Invalid bag length")
    return {str(env.read_m(0xD31E + 2 * i)): env.read_m(0xD31F + 2 * i) for i in range(count)}


def snapshot(env):
    return {
        "position": list(env.get_game_coords()),
        "battle": env.read_m(0xD057),
        "party": json.loads(json.dumps([asdict(m) for m in env.read_party_mons()])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--assets-root", type=Path, default=ROOT / "project_work")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.episodes or min(args.episodes) < 0:
        parser.error("episode indices must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.trace / "summary.json").read_text())
    config = dict(original["config"])
    config.update(session_path=str(args.output / "emulator"), save_video=False)
    for key in ("gb_path", "init_state"):
        config[key] = str(args.assets_root / Path(config[key]).name)
    if hashlib.sha1(Path(config["gb_path"]).read_bytes()).hexdigest() != (
        "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
    ):
        raise ValueError("Unexpected ROM")
    if hashlib.sha1(Path(config["init_state"]).read_bytes()).hexdigest() != (
        "33f3f6dba16714e92d8b5e3bbd9a63e1c00ea16c"
    ):
        raise ValueError("Unexpected starting state")
    result = {
        "source_trace": str(args.trace),
        "source_model_sha256": original["model_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_episodes": args.episodes,
        "reset_order": "Replay every preceding episode in original order; jumping directly to a later reset did not reproduce the original trace.",
        "created_utc": datetime.now(UTC).isoformat(),
        "records": [],
    }
    env = RedGymEnv(config)
    try:
        for episode in range(max(args.episodes) + 1):
            expected = original["records"][episode]
            env.reset(seed=expected["evaluation_seed"])
            initial = snapshot(env)
            previous_bag = bag(env)
            initial_bag = dict(previous_bag)
            previous = initial
            events = []
            poison_save = None
            faint_inventory = []
            trace_file = args.trace / f"episode-{episode:02d}-actions.jsonl.gz"
            with (
                gzip.open(trace_file, "rt") as actions,
                gzip.open(
                    args.output / f"episode-{episode:02d}-inventory.jsonl.gz", "wt"
                ) as inventory,
            ):
                for line in actions:
                    action = json.loads(line)
                    _, _, terminated, truncated, info = env.step(BUTTONS.index(action["action"]))
                    current = snapshot(env)
                    if current != {k: action[k] for k in current}:
                        raise AssertionError(
                            f"Replay diverged at episode {episode}, step {env.step_count}: "
                            f"actual={current!r}; expected="
                            f"{ {k: action[k] for k in current}!r}"
                        )
                    current_bag = bag(env)
                    old_mons = {tuple(m["identity"]): m for m in previous["party"]}
                    current_mons = {tuple(m["identity"]): m for m in current["party"]}
                    cures = [
                        list(k)
                        for k, m in current_mons.items()
                        if k in old_mons
                        and old_mons[k]["status"] & 8
                        and not m["status"] & 8
                        and m["hp"] > 0
                    ]
                    fainted = [
                        list(k)
                        for k, m in current_mons.items()
                        if k in old_mons and old_mons[k]["hp"] > 0 and m["hp"] == 0
                    ]
                    row = {
                        "step": env.step_count,
                        "bag": current_bag,
                        "antidotes": current_bag.get("11", 0),
                        **current,
                    }
                    inventory.write(json.dumps(row, separators=(",", ":")) + "\n")
                    if current_bag != previous_bag or cures or fainted:
                        events.append(
                            {
                                **row,
                                "previous_bag": previous_bag,
                                "poison_cleared_alive": cures,
                                "fainted": fainted,
                            }
                        )
                    if fainted:
                        faint_inventory.append(row)
                    if (
                        poison_save is None
                        and current["battle"] == 0
                        and current_bag.get("11", 0) > 0
                        and env.position.valid
                        and not env.tracker.dead
                        and not env.tracker.terminated
                        and any(m["status"] & 8 and m["hp"] >= 8 for m in current["party"])
                    ):
                        save = args.output / f"episode-{episode:02d}-poisoned.state"
                        with save.open("wb") as f:
                            env.pyboy.save_state(f)
                        poison_save = {
                            "state_file": save.name,
                            "state_sha256": hashlib.sha256(save.read_bytes()).hexdigest(),
                            "source_step": env.step_count,
                            "snapshot": current,
                            "bag": current_bag,
                            "initial_used_regions": sorted(env.tracker.used_regions),
                            "dead": [],
                            "ram_edited": False,
                            "note": "Natural poisoned overworld state; histories/exploration and LSTM reset for a new curriculum. Not yet a trained curriculum.",
                        }
                        Image.fromarray(env.pyboy.screen.ndarray[:, :, :3]).save(
                            save.with_suffix(".png")
                        )
                        save.with_suffix(".json").write_text(
                            json.dumps(poison_save, indent=2) + "\n"
                        )
                    previous_bag, previous = current_bag, current
                    if terminated or truncated:
                        break
            matched_end = all(
                json.dumps(info[k]) == json.dumps(expected["info"][k])
                for k in ("episode_steps", "end_reason", "episode_return", "position", "deaths")
            )
            if not matched_end:
                raise AssertionError("Replay ending differs from original evaluation")
            record = {
                "episode_index": episode,
                "evaluation_seed": expected["evaluation_seed"],
                "every_action_state_matches": True,
                "original_ending_matches": matched_end,
                "initial_bag": initial_bag,
                "final_bag": previous_bag,
                "events": events,
                "faint_inventory": faint_inventory,
                "poison_save": poison_save,
                "info": info,
            }
            result["records"].append(record)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "episode": episode,
                        "events": len(events),
                        "faints_with_antidotes": [r["antidotes"] for r in faint_inventory],
                        "natural_poison_save": poison_save is not None,
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    result["finished_utc"] = datetime.now(UTC).isoformat()
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
