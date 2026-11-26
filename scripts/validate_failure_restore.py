"""Real-ROM snapshot branches must match pixels, WRAM, rewards and rules exactly."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from failure_restart import capture, restore
from red_gym_env_v2_nuzlocke import RedGymEnv

BUTTONS = ("down", "left", "right", "up", "A", "B", "start")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.trace.parent / "summary.json").read_text())
    config = dict(
        summary["config"],
        session_path=str(args.output / "emulator"),
        gb_path=str(ROOT / "project_work/PokemonRed.gb"),
        init_state=str(ROOT / "project_work/has_pokedex_nballs.state"),
        save_video=False,
    )
    with gzip.open(args.trace, "rt") as f:
        actions = [BUTTONS.index(json.loads(line)["action"]) for line in f]
    env = RedGymEnv(config)
    checks = []

    def branch(sequence):
        records = []
        for action in sequence:
            obs, reward, term, trunc, info = env.step(action)
            records.append(
                (
                    {k: v.copy() for k, v in obs.items()},
                    reward,
                    term,
                    trunc,
                    info,
                    bytes(env.pyboy.memory[0xC000:0xE000]),
                )
            )
            if term or trunc:
                break
        return records

    try:
        env.reset(seed=1000)
        step = 0
        targets = {64, 512, 2048, 4096, 8192, 10000}
        while step < min(len(actions), max(targets) + 16):
            obs, _, done, trunc, _ = env.step(actions[step])
            step += 1
            if done or trunc:
                break
            if step not in targets:
                continue
            snapshot = capture(env, obs)
            battle = env.read_m(0xD057)
            position = list(env.get_game_coords())
            first = branch(actions[step : step + 16])
            env.reset(seed=9000)
            restored = restore(env, snapshot)
            for key in obs:
                np.testing.assert_array_equal(restored[key], obs[key])
            second = branch(actions[step : step + 16])
            assert len(first) == len(second)
            for a, b in zip(first, second):
                for key in a[0]:
                    np.testing.assert_array_equal(a[0][key], b[0][key])
                assert a[1:] == b[1:]
            checks.append(
                {
                    "step": step,
                    "matched_actions": len(second),
                    "battle": battle,
                    "position": position,
                    "snapshot_bytes": len(snapshot),
                }
            )
            print(json.dumps(checks[-1]), flush=True)
            step += len(second)
            if second[-1][2] or second[-1][3]:
                break
        if len(checks) < 3:
            raise AssertionError("Insufficient restore coverage")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "roundtrip.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "checks": checks,
                    "matched": [
                        "observation_pixels_and_history",
                        "reward",
                        "termination",
                        "all_info",
                        "all_WRAM",
                    ],
                    "contract": "Snapshots taken after ordinary policy actions. Reset-time snapshots excluded. No RAM edits. Fixed actions validate restoration, not policy quality.",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
