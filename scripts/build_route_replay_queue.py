"""Finite original/tail/transition comparison with fixed original-route rehearsal."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--python", default="/home/ubuntu/67920-venv/bin/python")
    args = p.parse_args()
    if args.output.exists():
        p.error("Queue exists; use a new campaign")
    campaign = args.output.parent
    jobs, evaluation_ids = [], []

    def evaluate(name, run, needs):
        job_id = f"eval-{name}"
        jobs.append(
            {
                "id": job_id,
                "needs": needs,
                "timeout_seconds": 10800,
                "command": [
                    args.python,
                    str(args.source / "scripts/diagnose_forest.py"),
                    str(run),
                    "--output",
                    str(campaign / "diagnostics" / name),
                    "--episodes",
                    "10",
                    "--seed",
                    "6000",
                    "--videos",
                    "1",
                    "--video-scope",
                    "tail",
                ],
            }
        )
        evaluation_ids.append(job_id)

    for seed in range(3):
        # LSTM first: its trainer-loop/retention regression is strongest.
        for algo in ("lstm", "ppo"):
            parent = args.root / "runs/campaign-failure-20261125/runs" / f"{algo}-ordinary-s{seed}"
            if not (parent / "final.zip").is_file():
                p.error(f"Missing parent {parent}")
            # Compare all three continuations with the same parent's fresh eval.
            # Start useful training immediately; parent evaluation follows it.
            for arm in ("transition", "tail", "ordinary"):
                name = f"{algo}-{arm}-s{seed}"
                output = campaign / "runs" / name
                jobs.append(
                    {
                        "id": f"train-{name}",
                        "timeout_seconds": 21600,
                        "command": [
                            args.python,
                            str(args.source / "project_work/v2/route_replay_experiment.py"),
                            "--parent",
                            str(parent),
                            "--output",
                            str(output),
                            "--arm",
                            arm,
                            "--rom",
                            str(args.source / "project_work/PokemonRed.gb"),
                            "--state",
                            str(args.source / "project_work/has_pokedex_nballs.state"),
                            "--seed",
                            str(seed),
                        ],
                    }
                )
                evaluate(name, output, [f"train-{name}"])
            evaluate(f"{algo}-parent-s{seed}", parent, [])
    jobs.append(
        {
            "id": "report",
            "timeout_seconds": 300,
            "command": [
                args.python,
                str(args.source / "scripts/report_route_replay.py"),
                str(campaign),
            ],
        }
    )
    config = {
        "cwd": str(args.source),
        "required_mount": str(args.root.parent),
        "stop_at": "2026-11-30T04:59:00+00:00",
        "max_consecutive_failures": 2,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/67920-matplotlib",
        },
        "contract": "18 matched continuations: PPO/LSTM x original/tail/transition x three seeds; each adds 1,048,576 actions to the corresponding completed ordinary parent. Mixed arms reserve 8/16 workers for full-route practice and use only their failed-episode banks. No reward, actor-input or action-set changes. RAM context is used for privileged reset sampling. 10 fresh original-start diagnostic evaluations per continuation and per parent, seeds 6000..6009. 240 episodes total. Finite queue; no automatic unbounded extensions.",
        "jobs": jobs,
    }
    campaign.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "queue": str(args.output)}))


if __name__ == "__main__":
    main()
