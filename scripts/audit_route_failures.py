"""Offline failure audit of existing action/frame traces; no new training."""

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def distribution(values):
    return (
        {
            "n": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
        if values
        else {"n": 0}
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    graph = json.loads(
        (Path(__file__).resolve().parents[1] / "project_work/v2/navigation_graph.json").read_text()
    )
    episodes = []
    for folder in sorted((args.campaign / "diagnostics").iterdir()):
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        assert summary.get("finished_utc") and len(summary["records"]) == 10
        for index, record in enumerate(summary["records"]):
            path = folder / f"episode-{index:02d}-actions.jsonl.gz"
            with gzip.open(path, "rt") as stream:
                actions = [json.loads(line) for line in stream]
            assert len(actions) == record["info"]["episode_steps"]
            seen = set()
            last_new = 0
            longest_static = 0
            static = 0
            previous = None
            visits = Counter()
            buttons = Counter()
            inventory_decreases = []
            cures = []
            unique_overworld = set()
            distances = []
            for a in actions:
                pos = tuple(a["position"])
                buttons[a["action"]] += 1
                if a["battle"] == 0:
                    visits[pos] += 1
                    unique_overworld.add(pos)
                    if pos not in seen:
                        seen.add(pos)
                        last_new = a["step"]
                    if (
                        previous
                        and previous["battle"] == 0
                        and previous["position"] == a["position"]
                    ):
                        static += 1
                    else:
                        static = 1
                    longest_static = max(longest_static, static)
                    dist = graph["distances"]["brock"].get(f"{pos[2]},{pos[0]},{pos[1]}")
                    if dist is not None:
                        distances.append(dist)
                else:
                    static = 0
                if previous:
                    if a["antidotes"] < previous["antidotes"]:
                        inventory_decreases.append(
                            {
                                "step": a["step"],
                                "before": previous["antidotes"],
                                "after": a["antidotes"],
                                "battle": a["battle"],
                                "position": a["position"],
                            }
                        )
                    old = {tuple(m["identity"]): m for m in previous["party"]}
                    for m in a["party"]:
                        before = old.get(tuple(m["identity"]))
                        if before and before["status"] & 8 and not m["status"] & 8 and m["hp"] > 0:
                            cures.append(
                                {
                                    "step": a["step"],
                                    "battle": a["battle"],
                                    "position": a["position"],
                                    "antidotes": a["antidotes"],
                                }
                            )
                previous = a
            trainers = []
            for b in [b for b in record["battles"] if b["type"] == 2]:
                end = b.get("end_step", len(actions))
                segment = actions[b["start_step"] - 1 : end]
                first = b["party_at_entry"][0]
                hp = [
                    m["hp"]
                    for a in segment
                    for m in a["party"]
                    if m["identity"] == first["identity"]
                ]
                trainers.append(
                    dict(
                        b,
                        action_duration=end - b["start_step"] + 1,
                        first_mon_hp_min=min(hp),
                        first_mon_hp_max=max(hp),
                        action_histogram=dict(Counter(a["action"] for a in segment)),
                        entry_remaining_actions=summary["config"]["max_steps"] - b["start_step"],
                    )
                )
            poison_faints = []
            for f in record["faints"]:
                if f["cause"] == "overworld_poison":
                    a = actions[min(f["step"] - 1, len(actions) - 1)]
                    poison_faints.append(
                        {"step": f["step"], "antidotes": a["antidotes"], "position": a["position"]}
                    )
            forest_step = record["forest_arrival"]["step"] if record["forest_arrival"] else None
            wild = [b for b in record["battles"] if b["type"] == 1]
            row = {
                "run": folder.name,
                "seed": record["evaluation_seed"],
                "episode": index,
                "info": record["info"],
                "forest_arrival_step": forest_step,
                "forest_remaining_actions": summary["config"]["max_steps"] - forest_step
                if forest_step
                else None,
                "frame_counts": record["frame_counts"],
                "run_hooks": record["run_hooks"],
                "faint_causes": dict(Counter(f["cause"] for f in record["faints"])),
                "poison_faints": poison_faints,
                "trainer_battles": trainers,
                "wild_battles": len(wild),
                "forest_wild_battles": record["forest_wild_battles"],
                "final": record["final"],
                "unique_overworld_tiles": len(unique_overworld),
                "best_distance": min(distances) if distances else None,
                "actions_since_new_overworld_tile": len(actions) - last_new,
                "longest_same_tile_overworld_actions": longest_static,
                "top_overworld_tiles": [
                    {"position": pos, "actions": n} for pos, n in visits.most_common(3)
                ],
                "last_1000_unique_positions": len({tuple(a["position"]) for a in actions[-1000:]}),
                "antidote_decreases": inventory_decreases,
                "poison_clear_events": cures,
                "all_action_histogram": dict(buttons),
            }
            episodes.append(row)
        print(json.dumps({"audited": folder.name, "episodes": 10}), flush=True)
    totals = Counter()
    causes = Counter()
    groups = defaultdict(Counter)
    all_trainers = []
    wild_attempts = wild_success = poison_stock = 0
    for e in episodes:
        info = e["info"]
        group = "-".join(e["run"].split("-")[:2])
        totals[info["end_reason"]] += 1
        groups[group][info["end_reason"]] += 1
        totals["forest_arrivals"] += e["forest_arrival_step"] is not None
        totals[f"final_battle_{e['final']['battle']}"] += 1
        if info["end_reason"] == "time_limit":
            totals[f"timeout_battle_{e['final']['battle']}"] += 1
        causes.update(e["faint_causes"])
        all_trainers.extend(e["trainer_battles"])
        wild_attempts += e["run_hooks"].get("run_attempt_battle1", 0)
        wild_success += e["run_hooks"].get("run_succeeded_battle1", 0)
        poison_stock += sum(f["antidotes"] > 0 for f in e["poison_faints"])
    completed = [b for b in all_trainers if "end_step" in b]
    healthy_static = [
        dict(run=e["run"], seed=e["seed"], **b)
        for e in episodes
        for b in e["trainer_battles"]
        if b["first_mon_hp_min"] == b["first_mon_hp_max"] == b["party_at_entry"][0]["max_hp"]
        and b["action_duration"] >= 1000
    ]
    report = {
        "episodes": episodes,
        "totals": dict(totals),
        "by_group": dict(groups),
        "faint_causes": dict(causes),
        "wild_run": {"attempts": wild_attempts, "successes": wild_success},
        "poison_faints_with_antidotes": poison_stock,
        "antidote_inventory_decrease_count": sum(len(e["antidote_decreases"]) for e in episodes),
        "poison_clear_event_count": sum(len(e["poison_clear_events"]) for e in episodes),
        "forest_arrival_step": distribution(
            [e["forest_arrival_step"] for e in episodes if e["forest_arrival_step"] is not None]
        ),
        "forest_actions_remaining": distribution(
            [
                e["forest_remaining_actions"]
                for e in episodes
                if e["forest_remaining_actions"] is not None
            ]
        ),
        "trainer_entry_remaining_actions": distribution(
            [b["entry_remaining_actions"] for b in all_trainers]
        ),
        "completed_trainer_action_duration": distribution(
            [b["action_duration"] for b in completed]
        ),
        "trainer_entries": len(all_trainers),
        "trainer_exits": len(completed),
        "trainer_entries_single_mon": sum(len(b["party_at_entry"]) == 1 for b in all_trainers),
        "trainer_entries_full_health": sum(
            all(m["hp"] == m["max_hp"] and m["status"] == 0 for m in b["party_at_entry"])
            for b in all_trainers
        ),
        "healthy_static_trainer_examples": healthy_static,
        "notes": "Existing stochastic evaluations, including six parent checkpoints. No new policy trials. Battle exits are not independently labeled wins. Inventory decreases are not automatically item uses. Static overworld position includes dialogue and menus; it is not always an obstacle.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["episodes", "healthy_static_trainer_examples"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
