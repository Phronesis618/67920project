"""Export compact, provenance-preserving campaign metadata from a local run backup."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def summarize(records):
    return {
        "episodes": len(records),
        "forest": sum(r["milestones"]["reached_viridian_forest"] for r in records),
        "pewter": sum(r["milestones"]["reached_pewter"] for r in records),
        "brock": sum(r["is_success"] for r in records),
        "end_reasons": dict(Counter(r["end_reason"] for r in records)),
        "violations": dict(Counter(v for r in records for v in r["violation_reasons"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evaluations, manifests, training, statuses = {}, {}, {}, {}
    all_evaluations, all_training = [], []
    groups = defaultdict(list)
    campaign_summary = {}
    for campaign in args.campaigns:
        statuses[campaign.name] = json.loads((campaign / "status.json").read_text())
        campaign_evaluations, campaign_training = [], []
        for run in sorted((campaign / "runs").iterdir()):
            if not (run / "manifest.json").exists():
                continue
            key = f"{campaign.name}/{run.name}"
            manifest = json.loads((run / "manifest.json").read_text())
            manifests[key] = manifest
            records = []
            incomplete_lines = 0
            for file in sorted(run.glob("episodes-*.jsonl")):
                for line in file.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        incomplete_lines += 1
            all_training.extend(records)
            campaign_training.extend(records)
            training[key] = {
                "completed_episode_summary": summarize(records),
                "incomplete_json_lines": incomplete_lines,
            }
            if (run / "training.json").exists():
                training[key]["training"] = json.loads((run / "training.json").read_text())
            if (run / "evaluation.json").exists():
                evaluation = json.loads((run / "evaluation.json").read_text())
                evaluations[key] = evaluation
                all_evaluations.extend(evaluation["records"])
                campaign_evaluations.extend(evaluation["records"])
                condition, algorithm, _, stage = run.name.split("-")
                groups[(condition, algorithm, stage)].append(evaluation)
        campaign_summary[campaign.name] = {
            "evaluations": summarize(campaign_evaluations),
            "training_episodes": summarize(campaign_training),
            "job_counts": dict(
                Counter(j["status"] for j in statuses[campaign.name]["jobs"].values())
            ),
        }
    progress = {
        "collected_utc": datetime.now(UTC).isoformat(),
        "campaigns": campaign_summary,
        "evaluations": summarize(all_evaluations),
        "training_episodes": summarize(all_training),
        "completed_training_segments": sum("training" in t for t in training.values()),
        "completed_segment_actions": sum(
            t["training"].get("additional_timesteps", t["training"]["actual_timesteps"])
            for t in training.values()
            if "training" in t
        ),
        "interpretation": "Corrected campaign plus its continuation; excludes pre-fix runs. "
        "Checkpoint stages and recovery segments are not independent training seeds. "
        "The archived corrected status is historical, not a live process. Completed-segment "
        "actions exclude partially completed segments, including the interrupted segment. "
        "Episode summaries include completed episodes in partial segments.",
        "groups": [
            {
                "condition": k[0],
                "algorithm": k[1],
                "stage": k[2],
                "training_seeds_evaluated": len(v),
                **summarize([r for e in v for r in e["records"]]),
            }
            for k, v in sorted(groups.items())
        ],
    }
    for name, content in (
        ("progress", progress),
        ("evaluations", evaluations),
        ("manifests", manifests),
        ("training", training),
        ("queue_status", statuses),
    ):
        (args.output / f"{name}.json").write_text(json.dumps(content, indent=2) + "\n")
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
