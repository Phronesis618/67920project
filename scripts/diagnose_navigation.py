"""Replay a checkpoint and capture suspiciously short distances around map changes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from experiment import ALGORITHMS
from red_gym_env_v2_nuzlocke import RedGymEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--threshold", type=int, default=100)
    p.add_argument("--position-mode", choices=("legacy-raw", "coherent-v1"), default="legacy-raw")
    args = p.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = dict(manifest["config"])
    config.update(
        session_path=str(args.output.parent / "emulator"),
        gb_path=str(ROOT / "project_work/PokemonRed.gb"),
        init_state=str(ROOT / "project_work/has_pokedex_nballs.state"),
        position_mode=args.position_mode,
    )
    import hashlib

    for path, expected in (
        (config["gb_path"], manifest["rom_sha1"]),
        (config["init_state"], manifest["state_sha1"]),
    ):
        if hashlib.sha1(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Local asset does not match training manifest")
    torch.set_num_threads(1)
    env = RedGymEnv(config)
    records = []
    found = False
    try:
        model = ALGORITHMS[manifest["algorithm"]].load(
            args.run / "final.zip", env=env, device="cpu"
        )
        set_random_seed(args.seed)
        obs, _ = env.reset(seed=args.seed)
        previous = env.get_game_coords()
        state, episode_start = None, np.array([True])
        remaining = None
        for step in range(config["max_steps"]):
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=False
            )
            episode_start[:] = False
            obs, reward, terminated, truncated, info = env.step(int(action))
            coords = env.get_game_coords()
            distance = info["navigation_distance"]
            suspicious = distance is not None and distance < args.threshold
            if coords[2] != previous[2] or suspicious or remaining is not None:
                records.append(
                    {
                        "step": step + 1,
                        "previous": previous,
                        "coords": coords,
                        "raw_coords": env.raw_game_coords(),
                        "position_coherent": info["position_coherent"],
                        "distance": distance,
                        "reward": reward,
                        "path_reward": info["path_reward"],
                        "milestones": info["milestones"],
                    }
                )
            if suspicious and remaining is None:
                found, remaining = True, 7
            previous = coords
            if remaining is not None:
                remaining -= 1
                if remaining == 0:
                    break
            if terminated or truncated:
                break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"source_run": args.run.name, "seed": args.seed, "found": found, "records": records},
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(records)} records; low-distance observation found: {found}")


if __name__ == "__main__":
    main()
