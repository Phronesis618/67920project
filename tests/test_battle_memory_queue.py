import json

import pytest

from scripts import build_battle_memory_queue, report_battle_memory


def test_queue_exports_then_trains_both_arms_in_dependency_order(tmp_path):
    root = tmp_path / "root"
    (root / "runs/campaign-route-replay-20261127/runs/ppo-tail-s0").mkdir(parents=True)
    queue = tmp_path / "campaign" / "queue.json"
    build_battle_memory_queue.main(
        [
            "--root", str(root),
            "--source", str(tmp_path / "src"),
            "--memory-source", str(tmp_path / "memsrc"),
            "--memory-campaign", str(tmp_path / "mem"),
            "--output", str(queue),
            "--python", "py",
            "--seeds", "0", "1",
        ]
    )  # fmt: skip
    jobs = json.loads(queue.read_text())["jobs"]
    seen = set()
    for job in jobs:  # the queue runner requires dependencies before dependants
        assert set(job.get("needs", [])) <= seen, job["id"]
        seen.add(job["id"])
    ids = [j["id"] for j in jobs]
    assert ids[0] == "build-bank" and ids[-1] == "report"
    for seed in (0, 1):
        export = next(j for j in jobs if j["id"] == f"export-s{seed}")
        assert export["command"][1] == str(tmp_path / "memsrc/scripts/export_memory_parent.py")
        frozen = next(j for j in jobs if j["id"] == f"train-frozen-s{seed}")["command"]
        unfrozen = next(j for j in jobs if j["id"] == f"train-unfrozen-s{seed}")["command"]
        assert "--freeze-memory" in frozen and "--freeze-memory" not in unfrozen
        assert "--aux-coef" not in frozen and "--aux-coef" not in unfrozen
        assert frozen[frozen.index("--parent") + 1].endswith(f"parents/memory-s{seed}")
    assert sum(i.startswith("route-") for i in ids) == 6
    bank = jobs[0]["command"]
    assert "--memory-consistent" in bank and "--screen-rom" in bank
    with pytest.raises(SystemExit):
        build_battle_memory_queue.main(
            ["--root", str(root), "--source", "s", "--memory-source", "m",
             "--memory-campaign", "c", "--output", str(queue), "--python", "py"]
        )  # fmt: skip


def episode(end, forest, wins, violations=()):
    return {
        "info": {
            "end_reason": end,
            "violation_reasons": list(violations),
            "milestones": {"reached_pewter": False, "beat_brock": False},
        },
        "forest_arrival": {"step": 1} if forest else None,
        "forest_trainers_beaten": wins,
        "battles": [{"type": 2}] * wins,
        "antidote_cures": [],
        "run_hooks": {"trainer_run_rejected_battle2": 2},
    }


def test_report_sums_arms_and_flags_missing_runs(tmp_path):
    for name, records in (
        ("frozen-s0", [episode("wipe", True, 1), episode("time_limit", False, 0)]),
        ("frozen-s1", [episode("rule_violation", True, 2, ["illegal_catch"])]),
    ):
        folder = tmp_path / "route" / name
        folder.mkdir(parents=True)
        summary = {"records": records, "finished_utc": "now"}
        (folder / "summary.json").write_text(json.dumps(summary))
    report_battle_memory.main([str(tmp_path)])
    frozen = json.loads((tmp_path / "report.json").read_text())["by_arm"]["frozen"]
    assert frozen["episodes"] == 3 and frozen["forest"] == 2 and frozen["trainer_wins"] == 3
    assert frozen["illegal_catches"] == 1 and frozen["trainer_run_rejected"] == 6
    (tmp_path / "route" / "unfrozen-s0").mkdir()
    with pytest.raises(SystemExit, match="unfrozen-s0"):
        report_battle_memory.main([str(tmp_path)])
