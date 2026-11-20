"""Run an explicit, finite experiment queue with durable status and per-job logs.

Run inside tmux. STOP_AFTER_CURRENT pauses between jobs. Failed jobs are not retried
automatically; their dependents are skipped, and repeated failures halt the queue.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def utc_now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_child(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run_queue(config_path):
    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes()
    config = json.loads(raw)
    jobs = config["jobs"]
    seen = set()
    for job in jobs:
        job_id = job["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) or job_id in seen:
            raise ValueError(f"Invalid or duplicate job ID: {job_id}")
        if not job["command"] or not all(isinstance(x, str) for x in job["command"]):
            raise ValueError(f"Expected command argument list: {job_id}")
        if not set(job.get("needs", [])).issubset(seen):
            raise ValueError(f"Dependencies must precede their job: {job_id}")
        if job.get("timeout_seconds", 21600) <= 0:
            raise ValueError("Job timeouts must be positive")
        seen.add(job_id)
    folder = config_path.parent
    cwd = Path(config["cwd"]).resolve()
    if not cwd.is_dir():
        raise ValueError(f"Working directory does not exist: {cwd}")
    stop_at = datetime.fromisoformat(config["stop_at"]) if config.get("stop_at") else None
    if stop_at is not None and stop_at.tzinfo is None:
        raise ValueError("stop_at must include a timezone")
    fingerprint = hashlib.sha256(raw).hexdigest()
    status_path = folder / "status.json"
    with (folder / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            state = json.loads(status_path.read_text())
            if state["config_sha256"] != fingerprint:
                raise ValueError("Queue definition changed; use a new campaign directory")
            if any(j["status"] == "running" for j in state["jobs"].values()):
                raise ValueError(
                    "An earlier job was interrupted or is still running. Inspect its PID and "
                    "checkpoint before creating a continuation queue; no duplicate job started."
                )
        else:
            state = {
                "config_sha256": fingerprint,
                "created_utc": utc_now(),
                "jobs": {j["id"]: {"status": "pending"} for j in jobs},
            }
        state.update(phase="running", controller_pid=os.getpid(), active_job=None)
        failures = 0
        environment = os.environ.copy()
        environment.update(config.get("environment", {}))

        def save():
            state["updated_utc"] = utc_now()
            write_json(status_path, state)

        def event(job_id, status):
            with (folder / "events.jsonl").open("a") as f:
                f.write(json.dumps({"time": utc_now(), "job": job_id, "status": status}) + "\n")
            print(f"{utc_now()} {job_id}: {status}", flush=True)

        save()
        for job in jobs:
            job_id = job["id"]
            record = state["jobs"][job_id]
            if record["status"] != "pending":
                continue
            if (folder / "STOP_AFTER_CURRENT").exists() or (
                stop_at is not None and datetime.now(UTC) >= stop_at
            ):
                state.update(phase="paused", active_job=None)
                save()
                return 0
            if any(state["jobs"][d]["status"] != "succeeded" for d in job.get("needs", [])):
                record.update(status="skipped", reason="dependency did not succeed")
                event(job_id, "skipped")
                save()
                continue
            start = time.monotonic()
            timeout = job.get("timeout_seconds", 21600)
            if stop_at is not None:
                timeout = min(timeout, (stop_at - datetime.now(UTC)).total_seconds())
            child = None
            with (folder / f"{job_id}.log").open("ab") as log:
                try:
                    child = subprocess.Popen(
                        job["command"],
                        cwd=cwd,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    record.update(status="running", pid=child.pid, started_utc=utc_now())
                    state["active_job"] = job_id
                    event(job_id, "running")
                    save()
                    while True:
                        remaining = timeout - (time.monotonic() - start)
                        if remaining <= 0:
                            stop_child(child)
                            record.update(status="timed_out", exit_code=child.returncode)
                            break
                        try:
                            code = child.wait(timeout=min(15, remaining))
                            record.update(
                                status="succeeded" if code == 0 else "failed", exit_code=code
                            )
                            break
                        except subprocess.TimeoutExpired:
                            record["elapsed_seconds"] = time.monotonic() - start
                            save()
                except OSError as exc:
                    record.update(status="failed", error=str(exc))
                except BaseException:
                    if child is not None:
                        stop_child(child)
                    record.update(status="interrupted", finished_utc=utc_now())
                    state.update(phase="halted", active_job=None)
                    save()
                    raise
            record.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - start)
            state["active_job"] = None
            event(job_id, record["status"])
            failures = 0 if record["status"] == "succeeded" else failures + 1
            if failures >= config.get("max_consecutive_failures", 2):
                state["phase"] = "halted"
                save()
                return 1
            save()
        failed = any(r["status"] != "succeeded" for r in state["jobs"].values())
        state["phase"] = "completed_with_failures" if failed else "completed"
        save()
        return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    raise SystemExit(run_queue(parser.parse_args().config))
