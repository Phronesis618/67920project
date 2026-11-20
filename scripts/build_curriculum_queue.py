"""Build a matched post-parcel pixels/RAM PPO/LSTM campaign; does not launch it."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Frozen repository snapshot")
    p.add_argument("--python", required=True, help="Python executable with project dependencies")
    p.add_argument("--output", type=Path, required=True, help="New queue.json destination")
    p.add_argument("--baseline-run", type=Path, help="Evaluate the completed bedroom pilot first")
    p.add_argument("--stop-at", default="2026-11-30T04:59:00+00:00")
    args = p.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        p.error("output exists; use a new campaign")
    runner = source / "project_work/v2"
    rom = source / "project_work/PokemonRed.gb"
    state = source / "project_work/has_pokedex_nballs.state"
    jobs = []
    if args.baseline_run:
        jobs.append(
            {
                "id": "bedroom-baseline-eval",
                "timeout_seconds": 21600,
                "command": [
                    args.python,
                    str(runner / "evaluate.py"),
                    str(args.baseline_run),
                    "--episodes",
                    "3",
                    "--output",
                    str(output.parent / "bedroom-evaluation.json"),
                ],
            }
        )
    previous = {}
    for phase, budget, episodes in (("m01", 1048576, 3), ("m05", 4194304, 5), ("m10", 5242880, 10)):
        for seed in range(3):
            for condition, mode in (("pixels", "pixels"), ("ram", "partial")):
                for algorithm in ("ppo", "lstm"):
                    key = f"{condition}-{algorithm}-s{seed}"
                    run = output.parent / "runs" / f"{key}-{phase}"
                    job_id = f"{key}-{phase}-train"
                    if key not in previous:
                        command = [
                            args.python,
                            str(runner / "baseline_fast_v2.py"),
                            "--rom",
                            str(rom),
                            "--state",
                            str(state),
                            "--output",
                            str(run),
                            "--algorithm",
                            algorithm,
                            "--observation-mode",
                            mode,
                            "--seed",
                            str(seed),
                            "--num-envs",
                            "16",
                            "--timesteps",
                            str(budget),
                            "--n-steps",
                            "512",
                            "--batch-size",
                            "512",
                            "--n-epochs",
                            "4",
                            "--max-episode-steps",
                            "16384",
                            "--gamma",
                            "0.9995",
                            "--gae-lambda",
                            "0.99",
                            "--navigation-reward",
                            "--explore-weight",
                            "0.025",
                            "--initial-used-regions",
                            "12",
                            "--device",
                            "cuda",
                            "--checkpoint-every",
                            "262144",
                        ]
                        needs = []
                    else:
                        parent_id, parent_run = previous[key]
                        command = [
                            args.python,
                            str(runner / "resume.py"),
                            str(parent_run),
                            "--output",
                            str(run),
                            "--timesteps",
                            str(budget),
                            "--device",
                            "cuda",
                        ]
                        needs = [parent_id]
                    jobs.append(
                        {
                            "id": job_id,
                            "needs": needs,
                            "command": command,
                            "timeout_seconds": 86400,
                            "condition": condition,
                            "algorithm": algorithm,
                            "seed": seed,
                            "additional_timesteps": budget,
                        }
                    )
                    jobs.append(
                        {
                            "id": f"{key}-{phase}-eval",
                            "needs": [job_id],
                            "timeout_seconds": 21600,
                            "command": [
                                args.python,
                                str(runner / "evaluate.py"),
                                str(run),
                                "--episodes",
                                str(episodes),
                                "--seed",
                                "1000",
                            ],
                        }
                    )
                    previous[key] = (job_id, run)
    config = {
        "description": "Post-parcel PPO/LSTM: pixels+recent actions versus RAM-assisted control.",
        "cwd": str(source),
        "stop_at": args.stop_at,
        "max_consecutive_failures": 2,
        "starting_state_contract": {
            "state": str(state),
            "rules_begin_at_reset": True,
            "prior_history_verified": False,
            "initial_used_regions": [12],
            "note": "Inherited level-six starter, Pokedex, five balls and two antidotes. Route 1 conservatively unavailable for catches. Not a full-game Nuzlocke claim.",
        },
        "reward_contract": "Identical quest-conditioned shortest-path potential shaping and base rewards across observations/algorithms. Gamma 0.9995, GAE lambda 0.99, 512-step worker rollouts; exploration weight 0.025.",
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/67920-matplotlib",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
        },
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(jobs)} jobs to {output}")


if __name__ == "__main__":
    main()
