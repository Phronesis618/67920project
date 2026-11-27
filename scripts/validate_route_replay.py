"""Verify automatic trainer-entry replay against a real, unchanged policy trace."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from red_gym_env_v2_nuzlocke import RedGymEnv
from route_replay import RouteReplay

BUTTONS = ("down", "left", "right", "up", "A", "B", "start")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.trace.parent / "summary.json").read_text())
    with gzip.open(args.trace, "rt") as f:
        trace = [json.loads(line) for line in f]
    cfg = dict(
        summary["config"],
        gb_path=str(ROOT / "project_work/PokemonRed.gb"),
        init_state=str(ROOT / "project_work/has_pokedex_nballs.state"),
        save_video=False,
    )
    source = RouteReplay(
        RedGymEnv(dict(cfg, session_path=str(args.output / "source-env"))),
        args.output / "source-bank",
        seed=0,
    )
    reader = RouteReplay(
        RedGymEnv(dict(cfg, session_path=str(args.output / "reader-env"))),
        args.output / "reader-bank",
        sources=[source.directory],
    )
    plain = RedGymEnv(dict(cfg, session_path=str(args.output / "plain-env")))
    entry, expected, start_obs = None, [], None
    checked_actions = 0
    try:
        source.reset(seed=4001)
        plain.reset(seed=4001)
        for row in trace:
            action = BUTTONS.index(row["action"])
            baseline = plain.step(action)
            obs, reward, done, trunc, info = source.step(action)
            checked_actions += 1
            assert (reward, done, trunc) == baseline[1:4], row["step"]
            assert all(info[k] == v for k, v in baseline[4].items()), row["step"]
            for key in obs:
                np.testing.assert_array_equal(obs[key], baseline[0][key])
            assert bytes(source.env.pyboy.memory[0xC000:0xE000]) == bytes(
                plain.pyboy.memory[0xC000:0xE000]
            )
            if entry is not None and len(expected) < 32:
                expected.append(
                    (
                        {k: v.copy() for k, v in obs.items()},
                        reward,
                        done,
                        trunc,
                        bytes(source.env.pyboy.memory[0xC000:0xE000]),
                    )
                )
            if entry is None and source.env.read_m(0xD057) == 2:
                entry = source.env.step_count
                start_obs = {k: v.copy() for k, v in obs.items()}
            if done or trunc:
                break
        anchors = [r for r in source.bank if r["original_step"] == entry]
        assert len(anchors) == 1 and len(expected) == 32
        (source.directory / "bank.json").write_text(json.dumps(anchors))
        obs, restart = reader.reset(seed=999)
        for key in start_obs:
            np.testing.assert_array_equal(obs[key], start_obs[key])
        for row, expected_row in zip(trace[entry : entry + 32], expected):
            obs, reward, done, trunc, _ = reader.step(BUTTONS.index(row["action"]))
            for key in obs:
                np.testing.assert_array_equal(obs[key], expected_row[0][key])
            assert (reward, done, trunc) == expected_row[1:4]
            assert bytes(reader.env.pyboy.memory[0xC000:0xE000]) == expected_row[4]
        report = {
            "passed": True,
            "trace": str(args.trace),
            "original_actions_checked": checked_actions,
            "trainer_entry_step": entry,
            "anchor": anchors[0],
            "matched_branch_actions": 32,
            "matched": [
                "unwrapped_control_rewards_pixels_WRAM_and_all_info",
                "pixels_and_actions",
                "WRAM",
                "branch_rewards",
                "termination",
            ],
            "original_end_reason": info["end_reason"],
            "restart_original_step": restart["restart_original_step"],
            "note": "Actions borrowed from a previous trace; both environments reset freshly together. This is a paired wrapper-integrity check, not a reproduction of the historical evaluation (which reset after preceding episodes). No scripted combat training, RAM edits, or altered reward.",
        }
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        source.close()
        reader.close()
        plain.close()


if __name__ == "__main__":
    main()
