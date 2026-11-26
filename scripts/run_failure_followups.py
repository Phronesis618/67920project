"""After the active campaign: reports, representative replays, bounded refinement.

Training/emulator jobs obey the existing cutoff. CPU-only report generation still
runs after cutoff or partial campaign failures, and never treats missing data as zero.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from failure_analysis import choose_followup, diagnostic_selection, export_report


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def refinement_queue(decision, runs, source, output, executable, stop_at):
    algorithm = decision["algorithm"]
    jobs = []
    for seed in range(3):
        if decision["action"] == "extend_matched":
            arms = (("ordinary", 0, 1024, "ordinary"), ("failure", 0.5, 1024, "failure-w1024"))
        else:
            arms = (
                ("window1024", 0.5, 1024, "failure-w1024"),
                ("window4096", 0.5, 4096, "failure-w1024"),
            )
        for label, probability, window, parent_arm in arms:
            parent = next(
                r
                for r in runs
                if r["algorithm"] == algorithm and r["seed"] == seed and r["arm"] == parent_arm
            )
            name = f"{algorithm}-{label}-s{seed}"
            run = output / "runs" / name
            jobs.append(
                {
                    "id": f"train-{name}",
                    "timeout_seconds": 21600,
                    "command": [
                        executable,
                        str(source / "project_work/v2/failure_experiment.py"),
                        "--parent",
                        parent["path"],
                        "--output",
                        str(run),
                        "--rom",
                        str(source / "project_work/PokemonRed.gb"),
                        "--state",
                        str(source / "project_work/has_pokedex_nballs.state"),
                        "--restart-probability",
                        str(probability),
                        "--restart-window",
                        str(window),
                        "--seed",
                        str(seed),
                    ],
                }
            )
            jobs.append(
                {
                    "id": f"eval-{name}",
                    "needs": [f"train-{name}"],
                    "timeout_seconds": 7200,
                    "command": [
                        executable,
                        str(source / "project_work/v2/evaluate.py"),
                        str(run),
                        "--episodes",
                        "10",
                        "--seed",
                        "5000",
                    ],
                }
            )
    return {
        "cwd": str(source),
        "required_mount": str(output.parents[2]),
        "stop_at": stop_at,
        "max_consecutive_failures": 2,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/67920-matplotlib",
        },
        "decision": decision,
        "jobs": jobs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "output", "source", "mount"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--stop-at", default="2026-11-30T04:59:00+00:00")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.mount.is_mount() or not args.output.resolve().is_relative_to(args.mount.resolve()):
        p.error("Follow-ups must live on the attached persistent filesystem")
    cutoff = datetime.fromisoformat(args.stop_at)
    if cutoff.tzinfo is None:
        p.error("stop-at needs a timezone")
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "status.json").exists():
            p.error("Controller already has a record; inspect before recovery")
        state = {"phase": "running", "controller_pid": os.getpid(), "active_job": None, "jobs": {}}

        def save():
            state["updated_utc"] = datetime.now(UTC).isoformat()
            write_json(args.output / "status.json", state)

        def remaining():
            return (cutoff - datetime.now(UTC)).total_seconds()

        def run(name, command, timeout, limited=True):
            if limited and remaining() <= 0:
                state["jobs"][name] = {"status": "skipped", "reason": "training/emulator cutoff"}
                save()
                return False
            timeout = min(timeout, remaining()) if limited else timeout
            state["active_job"] = name
            record = {
                "status": "running",
                "command": command,
                "started_utc": datetime.now(UTC).isoformat(),
            }
            state["jobs"][name] = record
            save()
            with (args.output / f"{name}.log").open("ab") as log:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                record["pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=timeout)
                    record.update(status="succeeded" if code == 0 else "failed", exit_code=code)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    record.update(status="timed_out", exit_code=child.returncode)
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                    raise
            record["finished_utc"] = datetime.now(UTC).isoformat()
            state["active_job"] = None
            save()
            return record["status"] == "succeeded"

        state["active_job"] = "wait-for-primary"
        save()
        compute_safe = True
        # This wait consumes no GPU and leaves the active immutable queue untouched.
        while True:
            primary = json.loads((args.primary / "status.json").read_text())
            if primary["phase"] != "running":
                break
            if remaining() < -120:
                compute_safe = False
                state["primary_warning"] = (
                    "Primary did not confirm shutdown after cutoff; report only."
                )
                break
            heartbeat = datetime.fromisoformat(primary["updated_utc"])
            if (datetime.now(UTC) - heartbeat).total_seconds() > 300:
                pids = [
                    primary.get("controller_pid"),
                    primary["jobs"].get(primary.get("active_job"), {}).get("pid"),
                ]
                if not any(pid and Path(f"/proc/{pid}").exists() for pid in pids):
                    state["primary_warning"] = (
                        "Primary heartbeat stopped and its recorded processes are absent."
                    )
                    break
            save()
            time.sleep(15)
        if (args.primary / "STOP_AFTER_CURRENT").exists():
            compute_safe = False
            state["primary_warning"] = "Explicit primary stop marker retained; analysis only."
        state["primary_final_phase"] = primary["phase"]
        write_json(args.output / "primary-status.json", primary)
        state["active_job"] = "primary-report"
        save()
        runs = export_report([args.primary], args.output / "primary-report")
        decision = choose_followup(runs)
        selected = diagnostic_selection(runs)
        write_json(args.output / "diagnostic-selection.json", selected)
        for selection in selected:
            output = args.output / "diagnostics" / selection["name"]
            command = [
                sys.executable,
                str(args.source / "scripts/diagnose_forest.py"),
                selection["run"],
                "--output",
                str(output),
                "--assets-root",
                str(args.source / "project_work"),
                "--episodes",
                str(selection["episodes"]),
                "--seed",
                str(selection["seed"]),
                "--device",
                "cuda",
                "--video-scope",
                "tail",
                "--video-episodes",
                *map(str, selection["video_episodes"]),
            ]
            if compute_safe:
                run("diagnose-" + selection["name"], command, 3600)
        # Keep at least eight hours for the finite six-run matched extension;
        # individual jobs retain hard deadlines even if throughput falls later.
        if decision["action"] != "report_only" and (remaining() < 8 * 3600 or not compute_safe):
            decision = {
                "action": "report_only",
                "proposed": decision,
                "reason": "Less than eight hours remain before the fixed training cutoff.",
            }
        write_json(args.output / "decision.json", decision)
        campaigns = [args.primary]
        if decision["action"] != "report_only":
            followup = args.output / "refinement"
            followup.mkdir()
            queue = refinement_queue(
                decision, runs, args.source, followup, sys.executable, args.stop_at
            )
            # The declared mount is explicit, avoiding dependence on path depth.
            queue["required_mount"] = str(args.mount)
            write_json(followup / "queue.json", queue)
            # This inner controller owns its GPU child and enforces the cutoff.
            # Do not time out just its parent process and orphan a training job.
            run(
                "refinement",
                [
                    sys.executable,
                    str(args.source / "scripts/run_experiment_queue.py"),
                    str(followup / "queue.json"),
                ],
                None,
                limited=False,
            )
            campaigns.append(followup)
        state["active_job"] = "final-report"
        save()
        export_report(campaigns, args.output / "report", args.output / "diagnostics", decision)
        state["active_job"] = None
        state["phase"] = (
            "completed_with_failures"
            if any(j["status"] in ("failed", "timed_out") for j in state["jobs"].values())
            else "completed"
        )
        state["decision"] = decision["action"]
        save()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            status = Path(sys.argv[sys.argv.index("--output") + 1]) / "status.json"
            if status.exists():
                record = json.loads(status.read_text())
                record.update(
                    phase="halted", error=str(exc), updated_utc=datetime.now(UTC).isoformat()
                )
                write_json(status, record)
        raise
