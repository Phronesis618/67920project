"""Reproduce preceding episodes and capture a natural trainer entry for diagnosis."""

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from failure_restart import capture
from poison_curriculum import antidote_count
from red_gym_env_v2_nuzlocke import RedGymEnv

BUTTONS = ("down", "left", "right", "up", "A", "B", "start")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("diagnostic", type=Path)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.diagnostic / "summary.json").read_text())
    cfg = dict(
        summary["config"],
        session_path=str(args.output / "emulator"),
        gb_path=str(ROOT / "project_work/PokemonRed.gb"),
        init_state=str(ROOT / "project_work/has_pokedex_nballs.state"),
    )
    env = RedGymEnv(cfg)
    matched = 0
    try:
        for episode in range(args.episode + 1):
            seed = summary["records"][episode]["evaluation_seed"]
            env.reset(seed=seed)
            with gzip.open(args.diagnostic / f"episode-{episode:02d}-actions.jsonl.gz", "rt") as f:
                for line in f:
                    row = json.loads(line)
                    obs, reward, term, trunc, _info = env.step(BUTTONS.index(row["action"]))
                    assert abs(reward - row["reward"]) < 1e-8, (
                        episode,
                        row["step"],
                        "reward",
                        reward,
                        row["reward"],
                    )
                    assert list(env.get_game_coords()) == row["position"], (
                        episode,
                        row["step"],
                        "position",
                    )
                    assert env.read_m(0xD057) == row["battle"], (episode, row["step"], "battle")
                    assert (
                        json.loads(json.dumps([asdict(m) for m in env.read_party_mons()]))
                        == row["party"]
                    ), (episode, row["step"], "party")
                    assert antidote_count(env.pyboy.memory) == row["antidotes"], (
                        episode,
                        row["step"],
                        "antidotes",
                    )
                    matched += 1
                    if episode == args.episode and row["battle"] == 2:
                        blob = capture(env, obs)
                        (args.output / "trainer.pkl.gz").write_bytes(blob)
                        Image.fromarray(env.pyboy.screen.ndarray[:, :, :3]).save(
                            args.output / "trainer-entry.png"
                        )
                        report = {
                            "config": cfg,
                            "diagnostic": str(args.diagnostic),
                            "episode": episode,
                            "seed": seed,
                            "step": row["step"],
                            "matched_actions": matched,
                            "snapshot_sha256": hashlib.sha256(blob).hexdigest(),
                            "party": row["party"],
                            "recorded_episode": summary["records"][episode],
                            "trainer_flags": env.read_m(0xD7F3),
                        }
                        (args.output / "capture.json").write_text(
                            json.dumps(report, indent=2) + "\n"
                        )
                        print(
                            json.dumps(
                                {
                                    k: v
                                    for k, v in report.items()
                                    if k not in ["config", "recorded_episode"]
                                }
                            ),
                            flush=True,
                        )
                        return
                    if term or trunc:
                        break
            print(
                json.dumps({"preceding_episode": episode, "matched_actions": matched}), flush=True
            )
        raise ValueError("No trainer reached")
    finally:
        env.close()


if __name__ == "__main__":
    main()
