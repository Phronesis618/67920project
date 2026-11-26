"""Summarize original-start evaluations and choose bounded, exploratory follow-ups."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def metrics(records):
    n = len(records)
    return {
        "episodes": n,
        "forest": sum(r["milestones"]["reached_viridian_forest"] for r in records),
        "pewter": sum(r["milestones"]["reached_pewter"] for r in records),
        "brock": sum(r["is_success"] for r in records),
        "wipes": sum(r["end_reason"] == "wipe" for r in records),
        "timeouts": sum(r["end_reason"] == "time_limit" for r in records),
        "deaths": sum(r["deaths"] for r in records),
        "violations": sum(bool(r["violation_reasons"]) for r in records),
    }


def load_runs(campaign):
    runs, missing = [], []
    for path in sorted((Path(campaign) / "runs").glob("*")):
        required = ("manifest.json", "training.json", "evaluation.json", "final.zip")
        if not all((path / f).is_file() for f in required):
            missing.append(path.name)
            continue
        try:
            manifest, training, evaluation = [read_json(path / f) for f in required[:3]]
        except json.JSONDecodeError:
            missing.append(path.name)
            continue
        if not evaluation["records"]:
            missing.append(path.name)
            continue
        probability = manifest["restart"]["probability"]
        window = manifest["restart"]["window"]
        runs.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "phase": Path(campaign).name,
                "algorithm": manifest["algorithm"],
                "seed": manifest["arguments"]["seed"],
                "arm": "ordinary" if probability == 0 else f"failure-w{window}",
                "manifest": manifest,
                "training": training,
                "evaluation": evaluation,
                "metrics": metrics(evaluation["records"]),
                "manifest_sha256": hashlib.sha256(
                    (path / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
    return runs, missing


def comparable(a, b):
    ma, mb = a["manifest"], b["manifest"]
    keys = ("rom_sha1", "state_sha1", "parent_checkpoint_sha256", "parent_timesteps")
    cfg_keys = (
        "observation_mode",
        "action_freq",
        "max_steps",
        "reward_scale",
        "explore_weight",
        "navigation_reward",
        "navigation_gamma",
        "death_penalty",
        "violation_penalty",
        "level_cap",
        "initial_used_regions",
        "region_map",
        "include_explore_map",
        "navigation_graph_sha256",
        "map_headers_sha256",
    )
    return (
        all(ma[k] == mb[k] for k in keys)
        and ma.get("git_commit") == mb.get("git_commit")
        and not ma.get("git_dirty")
        and not mb.get("git_dirty")
        and all(ma["config"].get(k) == mb["config"].get(k) for k in cfg_keys)
        and ma["config"].get("screen_scale", 2) == mb["config"].get("screen_scale", 2)
        and ma["arguments"]["num_envs"] == mb["arguments"]["num_envs"]
        and not ma["arguments"].get("smoke")
        and not mb["arguments"].get("smoke")
        and a["training"]["additional_timesteps"]
        == b["training"]["additional_timesteps"]
        == 1048576
        and a["evaluation"]["deterministic"] == b["evaluation"]["deterministic"] is False
        and [r["evaluation_seed"] for r in a["evaluation"]["records"]]
        == [r["evaluation_seed"] for r in b["evaluation"]["records"]]
        == list(range(4000, 4010))
    )


def progress(run):
    return tuple(run["metrics"][k] for k in ("brock", "pewter", "forest"))


def retry_stats(runs):
    eligible, repeated = 0, 0
    for run in runs:
        for path in Path(run["path"]).glob("episodes-*.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = record.get("restart_source")
                if not source or source["end_reason"] not in ("wipe", "rule_violation"):
                    continue
                eligible += 1
                repeated += (
                    record["end_reason"] in ("wipe", "rule_violation")
                    and record["fresh_actions"] <= source["lead_actions"]
                )
    return {"eligible_restarts": eligible, "failed_by_previous_failure_time": repeated}


def choose_followup(runs):
    """Cost-allocation rule, not a significance test or a causal diagnosis."""
    complete = []
    for algorithm in ("ppo", "lstm"):
        pairs = []
        for seed in range(3):
            same = [r for r in runs if r["algorithm"] == algorithm and r["seed"] == seed]
            ordinary = [r for r in same if r["arm"] == "ordinary"]
            failure = [r for r in same if r["arm"] == "failure-w1024"]
            if len(ordinary) != 1 or len(failure) != 1 or not comparable(ordinary[0], failure[0]):
                break
            pairs.append((ordinary[0], failure[0]))
        if len(pairs) != 3:
            continue
        deltas = [tuple(y - x for x, y in zip(progress(a), progress(b))) for a, b in pairs]
        aggregate = tuple(sum(d[i] for d in deltas) for i in range(3))
        complete.append(
            {
                "algorithm": algorithm,
                "deltas": deltas,
                "aggregate": aggregate,
                "positive_seeds": sum(d > (0, 0, 0) for d in deltas),
                "retry_stats": retry_stats([b for _, b in pairs]),
            }
        )
    wins = [c for c in complete if c["positive_seeds"] >= 2 and c["aggregate"] > (0, 0, 0)]
    if wins:
        selected = max(wins, key=lambda c: (c["aggregate"], c["positive_seeds"]))
        return {
            "action": "extend_matched",
            "algorithm": selected["algorithm"],
            "evidence": complete,
            "reason": "Positive original-start progress in at least two matched seeds and in aggregate; explore another matched 1M actions with fresh evaluation seeds. This is not statistical significance.",
        }
    trapped = [
        c
        for c in complete
        if c["retry_stats"]["eligible_restarts"] >= 30
        and c["retry_stats"]["failed_by_previous_failure_time"]
        / c["retry_stats"]["eligible_restarts"]
        >= 0.8
    ]
    if trapped:
        selected = max(trapped, key=lambda c: c["retry_stats"]["eligible_restarts"])
        return {
            "action": "earlier_window",
            "algorithm": selected["algorithm"],
            "evidence": complete,
            "reason": "At least 80% of 30+ eligible retries fail by the old failure time, without the paired progress signal. Compare 1024 versus 4096-action windows; unrecoverability is a hypothesis, not established.",
        }
    return {
        "action": "report_only",
        "evidence": complete,
        "reason": "No preregistered follow-up signal, or incomplete/mismatched seed pairs. Preserve the report and traces for interpretation; do not launch an arbitrary treatment.",
    }


def diagnostic_selection(runs):
    selected = []
    for algorithm in ("ppo", "lstm"):
        for arm in ("ordinary", "failure-w1024"):
            candidates = [r for r in runs if r["algorithm"] == algorithm and r["arm"] == arm]
            if not candidates:
                continue
            run = max(candidates, key=lambda r: (progress(r), -r["seed"]))
            records = run["evaluation"]["records"]
            best = max(
                range(len(records)),
                key=lambda i: (
                    records[i]["is_success"],
                    records[i]["milestones"]["reached_pewter"],
                    records[i]["milestones"]["reached_viridian_forest"],
                    records[i]["map_coverage"],
                ),
            )
            failed = [i for i, r in enumerate(records) if not r["is_success"]]
            worst = (
                max(failed, key=lambda i: (records[i]["deaths"], -records[i]["episode_steps"]))
                if failed
                else best
            )
            selected.append(
                {
                    "run": run["path"],
                    "name": run["name"],
                    "episodes": len(records),
                    "seed": records[0]["evaluation_seed"],
                    "video_episodes": sorted({best, worst}),
                }
            )
    return selected


def diagnostic_summary(directory):
    summary = {
        "replayed_episodes": 0,
        "matched_episodes": 0,
        "excluded_mismatches": 0,
        "faint_causes": Counter(),
        "run_hooks": Counter(),
        "videos": [],
    }
    for file in sorted(Path(directory).glob("*/summary.json")):
        for row in read_json(file)["records"]:
            summary["replayed_episodes"] += 1
            if not row["original_evaluation_match"]:
                summary["excluded_mismatches"] += 1
                continue
            summary["matched_episodes"] += 1
            summary["faint_causes"].update(e["cause"] for e in row["faints"])
            summary["run_hooks"].update(row["run_hooks"])
        records = read_json(file)["records"]
        for video in file.parent.glob("*.mp4"):
            episode = int(video.name.split("-")[1])
            summary["videos"].append(
                {
                    "path": str(video),
                    "episode_index": episode,
                    "matches_original_end_state_and_return": episode < len(records)
                    and records[episode]["original_evaluation_match"],
                }
            )
    return summary


def export_report(campaigns, output, diagnostics=None, decision=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_runs, missing = [], {}
    for campaign in campaigns:
        runs, incomplete = load_runs(campaign)
        all_runs.extend(runs)
        missing[str(campaign)] = incomplete
    statuses = {
        str(p): read_json(Path(p) / "status.json")
        for p in campaigns
        if (Path(p) / "status.json").exists()
    }
    (output / "evidence.json").write_text(
        json.dumps({"runs": all_runs, "incomplete": missing, "queue_status": statuses}, indent=2)
        + "\n"
    )
    rows = [
        {k: r[k] for k in ("phase", "name", "algorithm", "seed", "arm")} | r["metrics"]
        for r in all_runs
    ]
    with (output / "per-seed.csv").open("w") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = defaultdict(list)
    for run in all_runs:
        groups[(run["phase"], run["algorithm"], run["arm"])].append(run)
    diag = diagnostic_summary(diagnostics) if diagnostics else None
    if diag:
        (output / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    lines = [
        "# Failure-restart comparison",
        "",
        "Original-start evaluations only; each dot/row is a training seed. Missing runs are not treated as zero success.",
        "",
        "| Phase / algorithm / arm | Seeds | Forest | Pewter | Brock | Wipes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, runs in groups.items():
        m = metrics([x for r in runs for x in r["evaluation"]["records"]])
        lines.append(
            f"| {' / '.join(key)} | {len(runs)} | {m['forest']}/{m['episodes']} | {m['pewter']}/{m['episodes']} | {m['brock']}/{m['episodes']} | {m['wipes']} |"
        )
    lines += [
        "",
        "## Interpretation and provenance",
        "",
        "Two initial antidotes and Squirtle are supplied by the start save. Learned purchasing is unproven. Actor inputs are screens/actions; existing RAM-based route reward and navigation shaping remain. Failure restarts use simulator checkpoints; they add no antidote bonus. LSTM state resets at episode boundaries.",
        "",
        "Training curves describe the training start distribution and are not full-route evaluation curves. Diagnostic replays reuse evaluation trajectories and are not new independent trials. Only matching replays contribute to cause counts. Three training seeds and small evaluations limit precision; exploratory follow-ups are selected using the first comparison and evaluated with fresh seeds.",
        "",
        f"Completed run count: {len(all_runs)}. Incomplete run folders: {json.dumps(missing)}.",
        "",
        "Exact manifests, hashes and episode records: `evidence.json`. Per-seed counts: `per-seed.csv`. Videos and failure counts: `diagnostics.json`.",
    ]
    if decision:
        lines += ["", "## Follow-up decision", "", json.dumps(decision, indent=2)]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "presentation-outline.md").write_text(
        "# Presentation outline\n\n1. Task and supplied start inventory; fixed Squirtle; original-start evaluation.\n"
        "2. Baseline forest bottleneck and audited causes; distinguish observations from hypotheses.\n"
        "3. Automatic failure starts under a fixed reward, with matched ordinary-start control.\n"
        "4. Per-seed original-start outcomes from per-seed.csv and outcomes.png.\n"
        "5. Selected matched replay clips; label selection and replay provenance.\n"
        "6. Conditional follow-up and fresh evaluation seeds, if it ran.\n"
        "7. Limits: supplied items, privileged reward, simulator resets, small seed counts, no claim of general text understanding.\n"
    )
    if groups:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        labels = ["\n".join(k) for k in groups]
        for ax, metric in zip(axes, ("forest", "pewter", "brock")):
            for i, runs in enumerate(groups.values()):
                vals = [r["metrics"][metric] / r["metrics"]["episodes"] for r in runs]
                ax.bar(i, np.mean(vals), color="#6c92ad", alpha=0.5)
                ax.scatter([i] * len(vals), vals, color="#17384d", zorder=3)
            ax.set(
                ylim=(0, 1.05),
                title=metric.capitalize(),
                xticks=range(len(labels)),
                xticklabels=labels,
            )
            ax.tick_params(axis="x", labelsize=7, rotation=15)
        axes[0].set_ylabel("Original-start episode fraction")
        fig.suptitle("Bars: seed means; dots: training seeds. No missing runs imputed.")
        fig.tight_layout()
        fig.savefig(output / "outcomes.png", dpi=160)
        fig.savefig(output / "outcomes.svg")
        plt.close(fig)
        # These traces are explicitly diagnostic: restart and original-start training
        # episode distributions differ. Do not label them evaluation performance.
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        fig, ax = plt.subplots(figsize=(10, 5))
        found = False
        for run in all_runs:
            for event_dir in (Path(run["path"]) / "tensorboard").glob("*"):
                acc = EventAccumulator(str(event_dir)).Reload()
                if "rollout/success_rate" not in acc.Tags()["scalars"]:
                    continue
                data = acc.Scalars("rollout/success_rate")
                ax.plot(
                    [e.step - run["manifest"]["parent_timesteps"] for e in data],
                    [e.value for e in data],
                    label=run["name"],
                    alpha=0.6,
                )
                found = True
        if found:
            ax.set(
                xlabel="Additional policy actions",
                ylabel="Rolling training episode success",
                title="Training distribution diagnostic; not original-start evaluation",
            )
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(output / "training-curves.png", dpi=160)
        plt.close(fig)
    return all_runs
