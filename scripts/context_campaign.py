"""Build and summarize a bounded context-prediction pilot, not a parameter sweep."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


def build(args):
    campaign = args.output.parent
    if campaign.exists():
        raise ValueError("Use a new campaign directory")
    parent = args.root / "runs/campaign-failure-20261125/runs/lstm-ordinary-s2"
    diagnostic = args.root / "runs/campaign-route-replay-20261127/diagnostics/lstm-parent-s2"
    if not (parent / "final.zip").exists():
        raise ValueError("Missing selected parent")
    suite = campaign / "suite"
    py, src = args.python, args.source
    benchmark = str(src / "scripts/context_benchmark.py")
    jobs = [
        {
            "id": "build-natural-suite",
            "timeout_seconds": 3600,
            "command": [py, benchmark, "build", str(diagnostic), "--output", str(suite)],
        }
    ]
    for arm in ("auxiliary", "control", "parent"):
        run = parent if arm == "parent" else campaign / "runs" / arm
        dependencies = ["build-natural-suite"]
        if arm != "parent":
            jobs.append(
                {
                    "id": f"train-{arm}",
                    "needs": dependencies,
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
                        "2",
                        "--timesteps",
                        "262144",
                        "--rom",
                        str(src / "project_work/PokemonRed.gb"),
                        "--state",
                        str(src / "project_work/has_pokedex_nballs.state"),
                    ],
                }
            )
            dependencies = [f"train-{arm}"]
        jobs.append(
            {
                "id": f"cases-{arm}",
                "needs": dependencies,
                "timeout_seconds": 3600,
                "command": [
                    py,
                    benchmark,
                    "evaluate",
                    str(run),
                    "--suite",
                    str(suite),
                    "--output",
                    str(campaign / "cases" / arm),
                ],
            }
        )
        jobs.append(
            {
                "id": f"route-{arm}",
                "needs": dependencies,
                "timeout_seconds": 7200,
                "command": [
                    py,
                    str(src / "scripts/diagnose_forest.py"),
                    str(run),
                    "--output",
                    str(campaign / "route" / arm),
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
            "id": "report",
            "timeout_seconds": 300,
            "command": [py, str(src / "scripts/context_campaign.py"), "report", str(campaign)],
        }
    )
    campaign.mkdir(parents=True)
    args.output.write_text(
        json.dumps(
            {
                "cwd": str(src),
                "required_mount": str(args.root.parent),
                "stop_at": "2026-11-30T04:59:00+00:00",
                "max_consecutive_failures": 1,
                "environment": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MPLCONFIGDIR": "/tmp/67920-matplotlib",
                    "PYTHONUNBUFFERED": "1",
                },
                "jobs": jobs,
                "contract": "One development-selected LSTM parent (seed 2: best parent forest reach in the inspected comparison), two matched 262144-action original-route continuations. Only auxiliary prediction gradients into the actor representation differ. Six natural diagnostic snapshots, three stochastic branches each, full recorded recurrent history; no snapshot training. Six new original-route evaluation episodes per model (6100..6105), parent included. No automatic expansion, no scripted actions or new rewards. Stop and review the measured behavior and route retention.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"queue": str(args.output), "jobs": len(jobs)}))


def combine_predictions(records):
    names = ("battle_confusion", "poison_confusion", "antidote_confusion")
    result = {k: np.zeros((3, 3) if k == "battle_confusion" else (2, 2), dtype=int) for k in names}
    n, hp = 0, 0.0
    for record in records:
        for name in names:
            result[name] += np.array(record[name])
        n += record["actions"]
        hp += (record["hp_mae"] or 0) * record["actions"]
    out = {k: v.tolist() for k, v in result.items()}
    for name, mat in result.items():
        denom = mat.sum(axis=1)
        out[name.replace("confusion", "recall")] = [
            float(mat[i, i] / denom[i]) if denom[i] else None for i in range(len(mat))
        ]
    return dict(out, actions=n, hp_mae=hp / n if n else None)


def route_metrics(folder):
    summary = json.loads((folder / "summary.json").read_text())
    if not summary.get("finished_utc"):
        raise ValueError("Route evaluation incomplete")
    totals, endings = Counter(), Counter()
    for index, r in enumerate(summary["records"]):
        totals["episodes"] += 1
        totals["forest"] += r["forest_arrival"] is not None
        for milestone in ("reached_pewter", "beat_brock"):
            totals[milestone] += bool(r["info"]["milestones"][milestone])
        totals["trainer_wins"] += r["forest_trainers_beaten"]
        totals["antidote_cures"] += len(r["antidote_cures"])
        totals["trainer_entries"] += sum(b["type"] == 2 for b in r["battles"])
        totals["trainer_frames"] += r["frame_counts"].get("battle2", 0)
        for key in ("trainer_run_rejected_battle2", "run_attempt_battle1", "run_succeeded_battle1"):
            totals[key] += r["run_hooks"].get(key, 0)
        endings[r["info"]["end_reason"]] += 1
        poison_steps = [f["step"] for f in r["faints"] if f["cause"] == "overworld_poison"]
        totals["poison_faints"] += len(poison_steps)
        if poison_steps:
            with gzip.open(folder / f"episode-{index:02d}-actions.jsonl.gz", "rt") as stream:
                for line in stream:
                    action = json.loads(line)
                    if action["step"] in poison_steps:
                        totals["poison_faints_with_antidotes"] += (
                            action["antidotes"] > 0
                        ) * poison_steps.count(action["step"])
    return {
        "counts": dict(totals),
        "endings": dict(endings),
        "prediction": combine_predictions([r["context_prediction"] for r in summary["records"]]),
    }


def report(args):
    result, missing = {}, []
    for arm in ("parent", "control", "auxiliary"):
        try:
            case = json.loads((args.campaign / "cases" / arm / "summary.json").read_text())
            if not case["complete"]:
                raise ValueError("Case evaluation incomplete")
            groups = {}
            for kind in ("wild", "trainer", "poison"):
                rows = [r for r in case["rows"] if r["kind"] == kind]
                groups[kind] = {
                    "trials": len(rows),
                    "successes": sum(r["success"] for r in rows),
                    "trainer_run_rejections": sum(
                        r["run_hooks"].get("trainer_run_rejected_battle2", 0) for r in rows
                    ),
                    "median_success_actions": float(
                        np.median([r["actions"] for r in rows if r["success"]])
                    )
                    if any(r["success"] for r in rows)
                    else None,
                    "prediction": combine_predictions([r["prediction"] for r in rows]),
                }
            result[arm] = {"cases": groups, "route": route_metrics(args.campaign / "route" / arm)}
        except (FileNotFoundError, ValueError) as exc:
            missing.append({"arm": arm, "reason": str(exc)})
    payload = {
        "arms": result,
        "missing": missing,
        "interpretation": "Pilot only. Prediction accuracy is a mechanism check, not task success. Compare verified trainer wins and antidote cures against matched control, retain wild escape and original-route forest/Pewter/Brock progress. Two natural cases per context with repeated stochastic branches are not six independent states; one selected training seed is not a replicated result. No automatic expansion.",
    }
    (args.campaign / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if missing:
        raise ValueError("Incomplete comparison; report lists missing arms")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for name in ("root", "source", "output"):
        b.add_argument(f"--{name}", type=Path, required=True)
    b.add_argument("--python", required=True)
    r = sub.add_parser("report")
    r.add_argument("campaign", type=Path)
    args = p.parse_args()
    (build if args.command == "build" else report)(args)


if __name__ == "__main__":
    main()
