"""Queue the memory-target context comparison: detached control vs auxiliary, per seed.

Same protocol as scripts/context_campaign.py (the context pilot), with the memory target
set (lstm_memory.TARGETS) and one matched pair per parent seed. Each seed's directory has
runs/, cases/ and route/ for the parent, control and auxiliary arms, so
``context_campaign.py report <campaign>/s<seed>`` summarises it. One natural diagnostic
suite (built from the seed-2 parent's recorded failures, as in the pilot) is shared:
the benchmark replays each case's recorded history into every model's own LSTM.
"""

import argparse
import json
from pathlib import Path


def build(args):
    campaign = args.output.parent
    if campaign.exists():
        raise SystemExit("Use a new campaign directory")
    py, src = args.python, args.source
    benchmark = str(src / "scripts/context_benchmark.py")
    diagnostic = args.root / "runs/campaign-route-replay-20261127/diagnostics/lstm-parent-s2"
    suite = campaign / "suite"
    jobs = [
        {
            "id": "build-natural-suite",
            "timeout_seconds": 3600,
            "command": [py, benchmark, "build", str(diagnostic), "--output", str(suite)],
        }
    ]
    for seed in args.seeds:
        parent = args.root / f"runs/campaign-failure-20261125/runs/lstm-ordinary-s{seed}"
        if not (parent / "final.zip").exists():
            raise SystemExit(f"Missing parent {parent}")
        base = campaign / f"s{seed}"
        for arm in ("auxiliary", "control", "parent"):
            run = parent if arm == "parent" else base / "runs" / arm
            needs = ["build-natural-suite"]
            if arm != "parent":
                jobs.append(
                    {
                        "id": f"train-{arm}-s{seed}",
                        "needs": needs,
                        "timeout_seconds": 7200,
                        "command": [
                            py,
                            str(src / "project_work/v2/context_experiment.py"),
                            "--parent",
                            str(parent),
                            "--output",
                            str(run),
                            "--arm",
                            arm,
                            "--seed",
                            str(seed),
                            "--timesteps",
                            str(args.timesteps),
                            "--targets",
                            args.targets,
                            "--rom",
                            str(src / "project_work/PokemonRed.gb"),
                            "--state",
                            str(src / "project_work/has_pokedex_nballs.state"),
                        ],
                    }
                )
                needs = [f"train-{arm}-s{seed}"]
            jobs.append(
                {
                    "id": f"cases-{arm}-s{seed}",
                    "needs": needs,
                    "timeout_seconds": 3600,
                    "command": [
                        py,
                        benchmark,
                        "evaluate",
                        str(run),
                        "--suite",
                        str(suite),
                        "--output",
                        str(base / "cases" / arm),
                    ],
                }
            )
            jobs.append(
                {
                    "id": f"route-{arm}-s{seed}",
                    "needs": needs,
                    "timeout_seconds": 7200,
                    "command": [
                        py,
                        str(src / "scripts/diagnose_forest.py"),
                        str(run),
                        "--output",
                        str(base / "route" / arm),
                        "--episodes",
                        "6",
                        "--seed",
                        "6100",
                        "--videos",
                        "1",
                        "--video-scope",
                        "tail",
                        "--context-diagnostic",
                    ],
                }
            )
        jobs.append(
            {
                "id": f"report-s{seed}",
                "needs": [f"cases-{a}-s{seed}" for a in ("auxiliary", "control", "parent")]
                + [f"route-{a}-s{seed}" for a in ("auxiliary", "control", "parent")],
                "timeout_seconds": 300,
                "command": [py, str(src / "scripts/context_campaign.py"), "report", str(base)],
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
                    "MPLCONFIGDIR": "/tmp/67920-matplotlib",
                    "PYTHONUNBUFFERED": "1",
                },
                "jobs": jobs,
                "contract": f"Parents lstm-ordinary seeds {args.seeds}; per seed two matched "
                f"{args.timesteps}-action original-route continuations with the "
                f"'{args.targets}' target set. Only auxiliary prediction gradients into the "
                "CNN and actor LSTM differ (control: detached head). Rewards, actor inputs and "
                "action set unchanged; on-policy rollouts only. Evaluation as in the context "
                "pilot: natural diagnostic cases and six original-route episodes (6100..6105) "
                "per model, parents included. Finite queue, no automatic expansion.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"queue": str(args.output), "jobs": len(jobs)}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--targets", choices=("context", "memory"), default="memory")
    p.add_argument("--timesteps", type=int, default=262144)
    p.add_argument("--stop-at", default="2026-11-30T12:00:00+00:00")
    build(p.parse_args(argv))


if __name__ == "__main__":
    main()
