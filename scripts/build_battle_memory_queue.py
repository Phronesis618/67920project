"""Queue battle training from memory-trained parents: CNN/LSTM trainable vs frozen.

Stage two of two. Stage one (the lstm-memory branch's memory campaign) trains a
recurrent model to predict privileged state; ``export_memory_parent.py`` from that
branch turns each auxiliary model into a plain RecurrentPPO parent. This queue then,
per seed, exports that parent, and trains two battle arms from it with no auxiliary
loss: ``unfrozen`` (everything learns) and ``frozen`` (CNN and actor LSTM fixed by
``--freeze-memory``, so the memory cannot be overwritten). Every model, the exported
parents included, gets the same original-start route evaluation.
"""

import argparse
import json
from pathlib import Path


def build(args):
    campaign = args.output.parent
    if campaign.exists():
        raise SystemExit("Use a new campaign directory")
    py, src = args.python, args.source
    route_runs = sorted((args.root / "runs/campaign-route-replay-20261127/runs").iterdir())
    if not route_runs:
        raise SystemExit("No source runs for the battle bank")
    rom = str(src / "project_work/PokemonRed.gb")
    state = str(src / "project_work/has_pokedex_nballs.state")
    bank = campaign / "bank"
    jobs = [
        {
            "id": "build-bank",
            "timeout_seconds": 7200,
            "command": [
                py,
                str(src / "scripts/build_battle_bank.py"),
                *map(str, route_runs),
                "--output",
                str(bank),
                "--per-group",
                "64",
                "--maps",
                "12",
                "13",
                "51",
                "--area-maps",
                "13",
                "51",
                "--memory-consistent",
                "--screen-rom",
                rom,
                "--screen-state",
                state,
            ],
        }
    ]
    evaluations = []

    def evaluate(name, run, needs):
        job_id = f"route-{name}"
        jobs.append(
            {
                "id": job_id,
                "needs": needs,
                "timeout_seconds": 10800,
                "command": [
                    py,
                    str(src / "scripts/diagnose_forest.py"),
                    str(run),
                    "--output",
                    str(campaign / "route" / name),
                    "--episodes",
                    str(args.episodes),
                    "--seed",
                    "6000",
                    "--videos",
                    "1",
                    "--video-scope",
                    "tail",
                ],
            }
        )
        evaluations.append(job_id)

    for seed in args.seeds:
        auxiliary = args.memory_campaign / f"s{seed}" / "runs" / "auxiliary"
        parent = campaign / "parents" / f"memory-s{seed}"
        jobs.append(
            {
                "id": f"export-s{seed}",
                "timeout_seconds": 1800,
                "command": [
                    py,
                    str(args.memory_source / "scripts/export_memory_parent.py"),
                    str(auxiliary),
                    "--output",
                    str(parent),
                ],
            }
        )
        evaluate(f"parent-s{seed}", parent, [f"export-s{seed}"])
        for arm in ("unfrozen", "frozen"):
            run = campaign / "runs" / f"{arm}-s{seed}"
            command = [
                py,
                str(src / "project_work/v2/battle_experiment.py"),
                "--parent",
                str(parent),
                "--output",
                str(run),
                "--rom",
                rom,
                "--state",
                state,
                "--bank",
                str(bank),
                "--seed",
                str(seed),
                "--timesteps",
                str(args.timesteps),
                "--reward-preset",
                "outcome",
                "--trainer-share",
                "0.5",
                "--area-share",
                "0.5",
                "--learning-rate",
                "1e-4",
                "--target-kl",
                "0.03",
            ]
            if arm == "frozen":
                command.append("--freeze-memory")
            jobs.append(
                {
                    "id": f"train-{arm}-s{seed}",
                    "needs": ["build-bank", f"export-s{seed}"],
                    "timeout_seconds": 21600,
                    "command": command,
                }
            )
            evaluate(f"{arm}-s{seed}", run, [f"train-{arm}-s{seed}"])
    jobs.append(
        {
            "id": "report",
            "needs": evaluations,
            "timeout_seconds": 300,
            "command": [py, str(src / "scripts/report_battle_memory.py"), str(campaign)],
        }
    )
    campaign.mkdir(parents=True)
    args.output.write_text(
        json.dumps(
            {
                "cwd": str(src),
                "required_mount": str(args.root.parent),
                "stop_at": args.stop_at,
                "max_consecutive_failures": 1,
                "environment": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "MPLCONFIGDIR": "/tmp/67920-matplotlib",
                    "PYTHONUNBUFFERED": "1",
                },
                "jobs": jobs,
                "contract": f"Seeds {args.seeds}. Parents: the memory campaign's auxiliary "
                "models, exported as plain RecurrentPPO (head removed). Two battle arms per "
                f"seed, {args.timesteps} actions each, no auxiliary loss: unfrozen and frozen "
                "(CNN + actor LSTM fixed). Outcome preset, trainer share 0.5, area share 0.5, "
                "lr 1e-4, target KL 0.03, memory-consistent screened bank. "
                f"{args.episodes} original-start route evaluations (seeds 6000+) per model, "
                "exported parents included. Finite queue, no automatic expansion.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"queue": str(args.output), "jobs": len(jobs)}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "memory_source", "memory_campaign", "output"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=1048576)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--stop-at", default="2026-12-01T12:00:00+00:00")
    build(p.parse_args(argv))


if __name__ == "__main__":
    main()
