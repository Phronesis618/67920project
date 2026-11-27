"""Bounded optimizer follow-up after the completed route-replay comparison."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from report_route_replay import summarize


def build(args):
    previous = args.root / "runs/campaign-route-replay-20261127"
    if json.loads((previous / "status.json").read_text())["phase"] != "completed":
        raise ValueError("Complete the original comparison first")
    if args.output.exists():
        raise ValueError("Queue already exists")
    campaign = args.output.parent
    config = json.loads((previous / "queue.json").read_text())
    jobs = []
    for seed in range(3):
        for algo in ("ppo", "lstm"):
            name = f"{algo}-gentle-s{seed}"
            parent = args.root / "runs/campaign-failure-20261125/runs" / f"{algo}-ordinary-s{seed}"
            if not (parent / "final.zip").is_file():
                raise ValueError(f"Missing parent: {parent}")
            initial_steps = json.loads((parent / "training.json").read_text())["total_timesteps"]
            output = campaign / "runs" / name
            jobs.append(
                {
                    "id": f"train-{name}",
                    "timeout_seconds": 14400,
                    "command": [
                        args.python,
                        str(args.source / "project_work/v2/route_replay_experiment.py"),
                        "--parent",
                        str(parent),
                        "--output",
                        str(output),
                        "--arm",
                        "ordinary",
                        "--rom",
                        str(args.source / "project_work/PokemonRed.gb"),
                        "--state",
                        str(args.source / "project_work/has_pokedex_nballs.state"),
                        "--seed",
                        str(seed),
                        "--learning-rate",
                        "0.00003",
                        "--target-kl",
                        "0.02",
                    ],
                }
            )
            for stage in ("quarter", "final"):
                command = [
                    args.python,
                    str(args.source / "scripts/diagnose_forest.py"),
                    str(output),
                    "--output",
                    str(campaign / "diagnostics" / f"{name}-{stage}"),
                    "--episodes",
                    "10",
                    "--seed",
                    "6000",
                    "--videos",
                    "1",
                    "--video-scope",
                    "tail",
                ]
                if stage == "quarter":
                    command += [
                        "--checkpoint",
                        str(output / "checkpoints" / f"policy_{initial_steps + 262144}_steps.zip"),
                    ]
                jobs.append(
                    {
                        "id": f"eval-{name}-{stage}",
                        "needs": [f"train-{name}"],
                        "timeout_seconds": 7200,
                        "command": command,
                    }
                )
    jobs.append(
        {
            "id": "report",
            "timeout_seconds": 300,
            "command": [
                args.python,
                str(args.source / "scripts/conservative_followup.py"),
                "report",
                str(campaign),
                "--previous",
                str(previous),
            ],
        }
    )
    config.update(
        cwd=str(args.source),
        jobs=jobs,
        previous_campaign=str(previous),
        contract=(
            "Six ordinary-route continuations: PPO/LSTM x 3 matched seeds, from the same pre-comparison parents. "
            "Change learning rate 0.0003 to 0.00003 and enable target_kl=0.02; retain optimizer moments, "
            "all other training settings, reward and pixel/action inputs. No failure replay. "
            "Train 1,048,576 new actions; evaluate fixed 262,144-action and final checkpoints on original-start "
            "seeds 6000..6009 (120 new episodes). Reuse already-completed matched parent/ordinary evaluations. "
            "Exploratory reuse of inspected seeds, not a new untouched test set. LR and KL are a bundled intervention. "
            "No automatic extensions; retain the existing November 29 Pacific cutoff."
        ),
    )
    campaign.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "queue": str(args.output)}))


def report(args):
    rows, missing = [], []
    for seed in range(3):
        for algo in ("ppo", "lstm"):
            name = f"{algo}-gentle-s{seed}"
            folder = args.campaign / "runs" / name
            manifest_path, training_path = folder / "manifest.json", folder / "training.json"
            if not (manifest_path.exists() and training_path.exists()):
                missing.append(name)
                continue
            manifest = json.loads(manifest_path.read_text())
            training = json.loads(training_path.read_text())
            control = json.loads(
                (args.previous / "runs" / f"{algo}-ordinary-s{seed}" / "manifest.json").read_text()
            )
            for key in (
                "parent_checkpoint_sha256",
                "parent_timesteps",
                "rom_sha1",
                "state_sha1",
                "ruleset",
            ):
                assert manifest[key] == control[key], (name, key)
            ignored = {"gb_path", "init_state", "session_path"}
            assert {k: v for k, v in manifest["config"].items() if k not in ignored} == {
                k: v for k, v in control["config"].items() if k not in ignored
            }
            assert training["additional_timesteps"] == 1048576
            assert manifest["restart"]["arm"] == "ordinary" and not manifest["arguments"]["smoke"]
            assert not manifest["git_dirty"] and training["target_kl"] == 0.02
            assert training["optimizer_learning_rates"] == [0.00003]
            assert sum(s["actions"].get("original", 0) for s in training["worker_stats"]) == 1048576
            for stage in ("parent", "ordinary", "quarter", "final"):
                p = (
                    args.previous / "diagnostics" / f"{algo}-{stage}-s{seed}" / "summary.json"
                    if stage in ("parent", "ordinary")
                    else args.campaign / "diagnostics" / f"{name}-{stage}" / "summary.json"
                )
                if not p.exists():
                    missing.append(f"{name}/{stage}")
                    continue
                summary = json.loads(p.read_text())
                if not summary.get("finished_utc"):
                    missing.append(f"{name}/{stage}")
                    continue
                assert [r["evaluation_seed"] for r in summary["records"]] == list(range(6000, 6010))
                if stage in ("quarter", "final"):
                    assert summary["model_timesteps"] == manifest["parent_timesteps"] + (
                        262144 if stage == "quarter" else 1048576
                    )
                rows.append(
                    {
                        "algorithm": algo,
                        "seed": seed,
                        "stage": stage,
                        "metrics": summarize(summary),
                        "summary": summary,
                        "manifest": manifest,
                        "training": training,
                    }
                )
    totals = defaultdict(Counter)
    for row in rows:
        totals[f"{row['algorithm']}/{row['stage']}"].update(row["metrics"])
    output = args.campaign / "report"
    output.mkdir(exist_ok=True)
    evidence = {
        "complete": not missing,
        "missing": missing,
        "totals": dict(totals),
        "records": rows,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    lines = [
        "# Conservative continuation comparison",
        "",
        "Exploratory comparison on reused evaluation seeds. LR and KL early stopping change together. All evaluation starts from the original save; forest means arrival, not exit. Compare progress and survival jointly, including matched training seeds. Earlier and final checkpoints are fixed in advance, not selected as a winner on these evaluation seeds.",
        "",
        f"Completed model/checkpoint evaluations: {len(rows)}/24. Missing: {', '.join(missing) or 'none'}.",
        "",
        "| Model / stage | Episodes | Forest | Pewter | Brock | Deaths | Timeouts | Trainer encounters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, m in sorted(totals.items()):
        lines.append(
            f"| {group} | "
            + " | ".join(
                str(m[k])
                for k in (
                    "episodes",
                    "forest",
                    "pewter",
                    "brock",
                    "deaths",
                    "timeouts",
                    "trainer_entries",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Matched seeds",
        "",
        "| Model | Seed | Stage | Forest | Pewter | Brock | Timeouts |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['algorithm']} | {row['seed']} | {row['stage']} | {m['forest']} | {m['pewter']} | {m['brock']} | {m['timeouts']} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": not missing, "evaluations": len(rows), "report": str(output)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for key in ("root", "source", "output"):
        b.add_argument(f"--{key}", type=Path, required=True)
    b.add_argument("--python", default="/home/ubuntu/67920-venv/bin/python")
    r = sub.add_parser("report")
    r.add_argument("campaign", type=Path)
    r.add_argument("--previous", type=Path, required=True)
    args = p.parse_args()
    (build if args.command == "build" else report)(args)


if __name__ == "__main__":
    main()
