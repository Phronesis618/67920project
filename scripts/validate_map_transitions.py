"""Compare legacy and corrected position reads on a fixed real-ROM action replay.

The 15,150 recorded actions came from pixels-ppo-s2-m01, CPU seed 1000. No
policy training, RAM writes, or checkpoint download is needed for this replay.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from red_gym_env_v2_nuzlocke import RedGymEnv


def digest(obs):
    return hashlib.sha256(obs["screens"].tobytes() + obs["recent_actions"].tobytes()).hexdigest()


def replay(mode, actions, folder):
    env = RedGymEnv(
        {
            "session_path": folder,
            "gb_path": str(ROOT / "project_work/PokemonRed.gb"),
            "init_state": str(ROOT / "project_work/has_pokedex_nballs.state"),
            "headless": True,
            "save_final_state": False,
            "print_rewards": False,
            "save_video": False,
            "fast_video": True,
            "action_freq": 24,
            "max_steps": 16384,
            "reward_scale": 0.5,
            "explore_weight": 0.025,
            "observation_mode": "pixels",
            "navigation_reward": True,
            "navigation_gamma": 0.9995,
            "initial_used_regions": [12],
            "position_mode": mode,
        }
    )
    records = []
    try:
        env.reset(seed=1000)
        for action in actions:
            obs, _, term, trunc, info = env.step(action)
            records.append(
                {
                    "step": info["episode_steps"],
                    "raw": env.raw_game_coords(),
                    "position": env.get_game_coords(),
                    "coherent": info["position_coherent"],
                    "distance": info["navigation_distance"],
                    "path_reward": info["path_reward"],
                    "closest": info["closest_brock_distance"],
                    "deferred_frames": info["deferred_position_frames"],
                    "observation_sha256": digest(obs),
                    "jumping": bool(env.read_m(0xD736) & 0x40),
                }
            )
            if term or trunc:
                break
    finally:
        env.close()
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    actions_path = ROOT / "validation/map_transition_actions.json"
    actions = json.loads(actions_path.read_text())
    folder = ROOT / "runs/map-transition-validation"
    old = replay("legacy-raw", actions, folder / "legacy")
    fixed = replay("coherent-v1", actions, folder / "fixed")
    checks = {
        "complete_replay": len(old) == len(fixed) == len(actions),
        "identical_raw_trajectory": [r["raw"] for r in old] == [r["raw"] for r in fixed],
        "identical_pixel_observations": [r["observation_sha256"] for r in old]
        == [r["observation_sha256"] for r in fixed],
        "legacy_spikes_reproduced": max(abs(r["path_reward"]) for r in old) > 1,
        "fixed_spikes_removed": max(abs(r["path_reward"]) for r in fixed) < 0.1,
        "false_distance_removed": old[-1]["closest"] < 100 < fixed[-1]["closest"],
    }
    changes = [
        {"old": a, "fixed": b}
        for a, b in zip(old, fixed)
        if a["position"] != b["position"] or abs(a["path_reward"] - b["path_reward"]) > 0.05
    ]
    result = {
        "checks": checks,
        "actions": len(actions),
        "old_max_abs_path_reward": max(abs(r["path_reward"]) for r in old),
        "fixed_max_abs_path_reward": max(abs(r["path_reward"]) for r in fixed),
        "old_closest": old[-1]["closest"],
        "fixed_closest": fixed[-1]["closest"],
        "affected_steps": changes,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                actions_path,
                ROOT / "project_work/v2/map_position.py",
                ROOT / "project_work/v2/map_headers.json",
                ROOT / "project_work/v2/red_gym_env_v2_nuzlocke.py",
                ROOT / "project_work/PokemonRed.gb",
                ROOT / "project_work/has_pokedex_nballs.state",
            )
        },
        "limits": [
            "One saved trajectory; not an exhaustive validation of all map transitions.",
            "This establishes a reward/measurement fix, not improved learned performance.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("affected_steps", "source_sha256")},
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
