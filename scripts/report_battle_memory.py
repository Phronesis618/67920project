"""Summarise route evaluations of the battle-from-memory campaign by arm and seed.

Reads ``<campaign>/route/<arm>-s<seed>/summary.json`` (scripts/diagnose_forest.py) for
arms parent, unfrozen and frozen. Forest means arrival, not exit. A completed trainer
battle is a battle exit, not a separately verified win, unless diagnose_forest marks it.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("parent", "unfrozen", "frozen")


def summarise(summary):
    totals, endings = Counter(), Counter()
    for record in summary["records"]:
        info = record["info"]
        totals["episodes"] += 1
        totals["forest"] += record.get("forest_arrival") is not None
        totals["pewter"] += bool(info["milestones"]["reached_pewter"])
        totals["brock"] += bool(info["milestones"]["beat_brock"])
        totals["trainer_wins"] += record.get("forest_trainers_beaten", 0) or 0
        totals["trainer_entries"] += sum(b["type"] == 2 for b in record.get("battles", []))
        totals["antidote_cures"] += len(record.get("antidote_cures", []))
        totals["illegal_catches"] += "illegal_catch" in info.get("violation_reasons", [])
        totals["trainer_run_rejected"] += record.get("run_hooks", {}).get(
            "trainer_run_rejected_battle2", 0
        )
        endings[info["end_reason"]] += 1
    return {"counts": dict(totals), "endings": dict(endings)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    args = p.parse_args(argv)
    per_run, missing = {}, []
    for folder in sorted((args.campaign / "route").iterdir()):
        match = re.fullmatch(r"(\w+)-s(\d+)", folder.name)
        summary_path = folder / "summary.json"
        if not match or match.group(1) not in ARMS:
            continue
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if not summary or not summary.get("finished_utc"):
            missing.append(folder.name)
            continue
        per_run[folder.name] = summarise(summary)
    by_arm = defaultdict(Counter)
    for name, result in per_run.items():
        by_arm[name.rsplit("-s", 1)[0]].update(result["counts"])
    report = {"by_arm": {k: dict(v) for k, v in by_arm.items()}, "runs": per_run}
    report["missing"] = missing
    (args.campaign / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ("episodes", "forest", "pewter", "brock", "trainer_wins", "antidote_cures")
    columns += ("illegal_catches", "trainer_run_rejected")
    print("arm".ljust(10) + "".join(c[:14].rjust(15) for c in columns))
    for arm in ARMS:
        if arm in by_arm:
            print(arm.ljust(10) + "".join(str(by_arm[arm].get(c, 0)).rjust(15) for c in columns))
    if missing:
        raise SystemExit(f"Incomplete evaluations: {missing}")


if __name__ == "__main__":
    main()
