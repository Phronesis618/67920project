"""Report progress and trainer behavior without treating fewer deaths as success."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from failure_analysis import metrics


def summarize(summary):
    records = summary["records"]
    out = metrics([r["info"] for r in records])
    hooks, faints, frames = Counter(), Counter(), Counter()
    battles = []
    for row in records:
        hooks.update(row["run_hooks"])
        faints.update(event["cause"] for event in row["faints"])
        frames.update(row["frame_counts"])
        battles.extend(row["battles"])
    trainer_frames = frames["battle2"]
    out.update(
        trainer_entries=sum(b["type"] == 2 for b in battles),
        trainer_completed=sum(b["type"] == 2 and "end_step" in b for b in battles),
        trainer_timeouts=sum(
            r["info"]["end_reason"] == "time_limit" and r["final"]["battle"] == 2 for r in records
        ),
        trainer_run_rejected=hooks["trainer_run_rejected_battle2"],
        trainer_frames=trainer_frames,
        wild_run_success=hooks["run_succeeded_battle1"],
        wild_run_attempts=hooks["run_attempt_battle1"],
        trainer_faints=faints["trainer_battle"],
        wild_faints=faints["wild_battle"],
        walking_poison_faints=faints["overworld_poison"],
    )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    args = p.parse_args()
    rows, missing = [], []
    for seed in range(3):
        for algo in ("ppo", "lstm"):
            parents = []
            for arm in ("parent", "ordinary", "tail", "transition"):
                name = f"{algo}-{arm}-s{seed}"
                path = args.campaign / "diagnostics" / name / "summary.json"
                if not path.exists():
                    missing.append(name)
                    continue
                data = json.loads(path.read_text())
                if not data.get("finished_utc") or len(data["records"]) != 10:
                    missing.append(name)
                    continue
                assert [r["evaluation_seed"] for r in data["records"]] == list(range(6000, 6010))
                run = {
                    "name": name,
                    "algorithm": algo,
                    "arm": arm,
                    "seed": seed,
                    "metrics": summarize(data),
                    "evaluation": data,
                }
                if arm != "parent":
                    folder = args.campaign / "runs" / name
                    manifest = json.loads((folder / "manifest.json").read_text())
                    training = json.loads((folder / "training.json").read_text())
                    assert training["additional_timesteps"] == 1048576
                    assert not manifest["git_dirty"] and not manifest["arguments"]["smoke"]
                    parents.append(manifest["parent_checkpoint_sha256"])
                    stats = training["worker_stats"]
                    original = sum(s["actions"].get("original", 0) for s in stats)
                    total = sum(sum(s["actions"].values()) for s in stats)
                    assert total == training["additional_timesteps"]
                    assert original >= total // 2
                    if arm == "ordinary":
                        assert original == total
                    else:
                        assert (
                            sum(sum(s["actions"].values()) for s in stats if s["role"] == "route")
                            == total // 2
                        )
                    run.update(
                        manifest=manifest,
                        training=training,
                        original_action_fraction=original / total,
                    )
                rows.append(run)
            assert len(set(parents)) <= 1, "Unmatched parent checkpoints"
    groups = defaultdict(Counter)
    for row in rows:
        groups[f"{row['algorithm']}/{row['arm']}"].update(row["metrics"])
    report = {
        "complete": not missing,
        "missing": missing,
        "runs": rows,
        "totals": dict(groups),
        "contract": "Fresh original-start evaluation only. Forest means arrival, not exit. Completed trainer battle means observed battle exit, not a separately verified win. RUN rates require battle exposure; a lower rate with fewer trainer entries is not sufficient. Compare matched training seeds, progress and deaths jointly. Parent probes reuse checkpoints, not new training seeds. RAM is privileged reset-selection information only; actor inputs/rewards unchanged.",
    }
    output = args.campaign / "report"
    output.mkdir(exist_ok=True)
    (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Route retention and transition replay",
        "",
        report["contract"],
        "",
        f"Complete evaluated models: {len(rows)}/24. Missing: {', '.join(missing) or 'none'}.",
        "",
        "| Model / starts | Episodes | Forest | Pewter | Brock | Deaths | Trainer timeouts | Rejected RUN | RUN / 1k trainer actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in sorted(groups.items()):
        rate = (
            f"{m['trainer_run_rejected'] * 24000 / m['trainer_frames']:.2f}"
            if m["trainer_frames"]
            else "n/a"
        )
        vals = [
            m[k]
            for k in (
                "episodes",
                "forest",
                "pewter",
                "brock",
                "deaths",
                "trainer_timeouts",
                "trainer_run_rejected",
            )
        ]
        lines.append(f"| {name} | " + " | ".join(map(str, vals)) + f" | {rate} |")
    lines += [
        "",
        "## Matched seeds",
        "",
        "| Run | Forest | Pewter | Brock | Trainer entries | Trainer timeouts | Original-start action share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        fraction = (
            f"{row['original_action_fraction']:.1%}"
            if "original_action_fraction" in row
            else "parent"
        )
        lines.append(
            f"| {row['name']} | {m['forest']} | {m['pewter']} | {m['brock']} | {m['trainer_entries']} | {m['trainer_timeouts']} | {fraction} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"complete": not missing, "evaluated_models": len(rows), "report": str(output)})
    )


if __name__ == "__main__":
    main()
