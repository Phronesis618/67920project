"""Pull an experiment directory to this Mac while its queue is active."""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--queue-status", required=True, help="Path relative to the copied directory")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--stop-at", required=True)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    stop_at = datetime.fromisoformat(args.stop_at)
    if stop_at.tzinfo is None or args.interval < 1:
        p.error("stop-at needs a timezone and interval must be positive")
    args.local.mkdir(parents=True, exist_ok=True)
    ssh = shlex.join(
        [
            "ssh",
            "-i",
            str(args.identity),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=15",
        ]
    )
    command = [
        "rsync",
        "-az",
        "--partial",
        "--timeout=120",
        "-e",
        ssh,
        f"{args.host}:{args.remote.rstrip('/')}/",
        str(args.local) + "/",
    ]
    with (args.local / "backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while datetime.now(UTC) < stop_at and not (args.local / "STOP_BACKUP").exists():
            started = datetime.now(UTC).isoformat()
            print(f"{started} pulling results", flush=True)
            try:
                result = subprocess.run(command, timeout=1800, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
            state = {
                "pid": os.getpid(),
                "started_utc": started,
                "finished_utc": datetime.now(UTC).isoformat(),
                "exit_code": code,
            }
            (args.local / "backup_status.json").write_text(json.dumps(state, indent=2) + "\n")
            if args.once:
                return code
            queue_status = args.local / args.queue_status
            if code == 0 and queue_status.is_file():
                phase = json.loads(queue_status.read_text())["phase"]
                if phase in {"completed", "completed_with_failures", "halted", "paused"}:
                    print(f"Queue {phase}; final pull complete", flush=True)
                    return 0
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
