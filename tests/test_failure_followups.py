import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from failure_analysis import choose_followup, diagnostic_selection, export_report
from run_failure_followups import refinement_queue


def make_run(tmp_path, seed, arm, forest=0):
    path = tmp_path / "primary" / "runs" / f"ppo-{arm}-s{seed}"
    path.mkdir(parents=True)
    manifest = {
        "algorithm": "ppo",
        "rom_sha1": "same-rom",
        "state_sha1": "same-save",
        "parent_checkpoint_sha256": f"parent-{seed}",
        "parent_timesteps": 10485760,
        "config": {"observation_mode": "pixels"},
        "arguments": {"num_envs": 16, "seed": seed, "smoke": False},
        "restart": {"probability": 0 if arm == "ordinary" else 0.5, "window": 1024},
    }
    records = [
        {
            "evaluation_seed": 4000 + i,
            "is_success": False,
            "end_reason": "time_limit",
            "deaths": 0,
            "violation_reasons": [],
            "episode_steps": 16384,
            "map_coverage": 100,
            "milestones": {"reached_viridian_forest": i < forest, "reached_pewter": False},
        }
        for i in range(10)
    ]
    evaluation = {"records": records, "deterministic": False}
    training = {"additional_timesteps": 1048576}
    for name, obj in (("manifest", manifest), ("training", training), ("evaluation", evaluation)):
        (path / f"{name}.json").write_text(json.dumps(obj))
    (path / "final.zip").write_bytes(b"not a policy; report fixture")
    return {
        "path": str(path),
        "name": path.name,
        "algorithm": "ppo",
        "seed": seed,
        "arm": "ordinary" if arm == "ordinary" else "failure-w1024",
        "manifest": manifest,
        "training": training,
        "evaluation": evaluation,
        "metrics": {"forest": forest, "pewter": 0, "brock": 0},
    }


def pairs(tmp_path, gain=True):
    return [
        make_run(tmp_path, seed, arm, forest=3 if gain and arm == "failure" else 1)
        for seed in range(3)
        for arm in ("ordinary", "failure")
    ]


def test_followup_requires_complete_comparable_seed_pairs(tmp_path):
    runs = pairs(tmp_path)
    assert choose_followup(runs)["action"] == "extend_matched"
    assert choose_followup(runs[:-1])["action"] == "report_only"
    for key in ("budget", "seed", "parent", "smoke"):
        altered = copy.deepcopy(runs)
        if key == "budget":
            altered[1]["training"]["additional_timesteps"] = 524288
        if key == "seed":
            altered[1]["evaluation"]["records"][0]["evaluation_seed"] = 999
        if key == "parent":
            altered[1]["manifest"]["parent_checkpoint_sha256"] = "different"
        if key == "smoke":
            altered[1]["manifest"]["arguments"]["smoke"] = True
        assert choose_followup(altered)["action"] == "report_only"


def test_no_automatic_treatment_for_mixed_results(tmp_path):
    runs = pairs(tmp_path, gain=False)
    assert choose_followup(runs)["action"] == "report_only"
    runs[1]["metrics"]["forest"] = 10
    assert choose_followup(runs)["action"] == "report_only"  # only one positive seed


def test_earlier_window_requires_observed_repeat_failures(tmp_path):
    runs = pairs(tmp_path, gain=False)
    failure = next(r for r in runs if r["arm"] == "failure-w1024")
    record = {
        "restart_source": {"end_reason": "wipe", "lead_actions": 128},
        "end_reason": "wipe",
        "fresh_actions": 100,
    }
    path = Path(failure["path"]) / "episodes-0.jsonl"
    path.write_text((json.dumps(record) + "\n") * 29)
    assert choose_followup(runs)["action"] == "report_only"
    path.write_text((json.dumps(record) + "\n") * 30)
    decision = choose_followup(runs)
    assert decision["action"] == "earlier_window"
    queue = refinement_queue(
        decision,
        runs,
        tmp_path / "source",
        tmp_path / "followup",
        "python",
        "2026-11-30T04:59:00+00:00",
    )
    train = [j for j in queue["jobs"] if j["id"].startswith("train")]
    assert len(train) == 6 and len(queue["jobs"]) == 12
    assert {j["command"][j["command"].index("--restart-window") + 1] for j in train} == {
        "1024",
        "4096",
    }
    for seed in range(3):
        matched = [j for j in train if j["id"].endswith(f"s{seed}")]
        parents = [j["command"][j["command"].index("--parent") + 1] for j in matched]
        assert parents[0] == parents[1]
    assert all(j["command"][-1] == "5000" for j in queue["jobs"] if j["id"].startswith("eval"))


def test_diagnostic_selection_and_report_handle_missing_data(tmp_path):
    runs = pairs(tmp_path)
    selected = diagnostic_selection(runs)
    assert len(selected) == 2
    assert all(s["episodes"] == 10 and s["seed"] == 4000 for s in selected)
    assert all(1 <= len(s["video_episodes"]) <= 2 for s in selected)
    incomplete = tmp_path / "primary/runs/not-finished"
    incomplete.mkdir()
    report = tmp_path / "report"
    result = export_report([tmp_path / "primary"], report)
    assert len(result) == 6
    assert (report / "outcomes.png").stat().st_size > 1000
    evidence = json.loads((report / "evidence.json").read_text())
    assert evidence["incomplete"][str(tmp_path / "primary")] == ["not-finished"]
    assert len(evidence["runs"]) == 6
    assert (report / "presentation-outline.md").exists()


def test_empty_campaign_still_has_report(tmp_path):
    assert export_report([tmp_path / "absent"], tmp_path / "report") == []
    assert (tmp_path / "report/report.md").exists()
    assert choose_followup([])["action"] == "report_only"
