"""Recover an interrupted curriculum from a backup without repeating completed jobs."""

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path


def checkpoint_steps(path):
    with zipfile.ZipFile(path) as archive:
        if not {"policy.pth", "policy.optimizer.pth", "data"}.issubset(archive.namelist()):
            raise ValueError(f"Incomplete checkpoint: {path}")
        return int(json.loads(archive.read("data"))["num_timesteps"])


def build(backup, destination, source, python, campaign, mount):
    raw = (backup / "queue.json").read_bytes()
    original = json.loads(raw)
    status = json.loads((backup / "status.json").read_text())
    if hashlib.sha256(raw).hexdigest() != status["config_sha256"]:
        raise ValueError("Backup queue and status do not match")
    for path in (destination, source, campaign):
        if not path.is_absolute() or not path.is_relative_to(mount):
            raise ValueError(f"Recovery paths must be under the persistent mount: {path}")
    jobs = original["jobs"]
    completed = {key for key, value in status["jobs"].items() if value["status"] == "succeeded"}
    if any(v["status"] not in {"succeeded", "running", "pending"} for v in status["jobs"].values()):
        raise ValueError("This recovery requires a queue without prior failed or skipped jobs")
    old_source = original["cwd"]
    old_campaign = str(Path(jobs[0]["command"][jobs[0]["command"].index("--output") + 1]).parents[1])
    remaining = []
    resumed = []
    for job in jobs:
        job_id = job["id"]
        if job_id in completed:
            continue
        entry = copy.deepcopy(job)
        command = entry["command"]
        command[0] = python
        for i, value in enumerate(command[1:], 1):
            if value.startswith(old_source + "/"):
                command[i] = str(source / Path(value).relative_to(old_source))
            elif value.startswith(old_campaign + "/runs/"):
                relative = Path(value).relative_to(old_campaign)
                run_name = relative.parts[1]
                root = destination if f"{run_name}-train" in completed else campaign
                command[i] = str(root / relative)
        entry["needs"] = [key for key in entry.get("needs", []) if key not in completed]
        if job_id.endswith("-train"):
            run_name = job_id.removesuffix("-train")
            local_run = backup / "runs" / run_name
            if status["jobs"][job_id]["status"] == "running":
                parent = json.loads((local_run / "manifest.json").read_text())
                target = parent.get("resume", {}).get("initial_timesteps", 0) + int(
                    parent["arguments"]["timesteps"]
                )
                candidates = sorted(
                    local_run.glob("checkpoints/policy_*_steps.zip"),
                    key=lambda p: int(p.stem.split("_")[1]), reverse=True,
                )
                if not candidates:
                    raise ValueError(f"No saved checkpoint for interrupted job: {job_id}")
                checkpoint = candidates[0]
                steps = checkpoint_steps(checkpoint)
                if not 0 < steps < target:
                    raise ValueError(f"Checkpoint is outside the unfinished stage budget: {steps}")
                remaining_steps = target - steps
                command = [
                    python, str(source / "project_work/v2/resume.py"),
                    str(destination / "runs" / run_name),
                    "--checkpoint", str(destination / checkpoint.relative_to(backup)),
                    "--output", str(campaign / "runs" / run_name),
                    "--timesteps", str(remaining_steps), "--device", "cuda",
                ]
                entry["additional_timesteps"] = remaining_steps
                resumed.append({"job": job_id, "checkpoint_steps": steps,
                                "target_steps": target, "remaining_steps": remaining_steps,
                                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            if Path(command[1]).name == "resume.py":
                command += ["--rom", str(source / "project_work/PokemonRed.gb"),
                            "--state", str(source / "project_work/has_pokedex_nballs.state")]
        entry["command"] = command
        remaining.append(entry)
    config = {key: copy.deepcopy(value) for key, value in original.items() if key != "jobs"}
    config.update(
        description="Persistent-storage recovery of the corrected November 22 campaign",
        cwd=str(source), required_mount=str(mount), jobs=remaining,
        recovery={"backup_status_utc": status["updated_utc"],
                  "original_config_sha256": status["config_sha256"],
                  "completed_jobs_preserved": len(completed), "resumed_jobs": resumed,
                  "episodes_restarted": True, "rng_reseeded": True},
    )
    config["starting_state_contract"]["state"] = str(source / "project_work/has_pokedex_nballs.state")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("backup", "destination", "source", "campaign", "mount", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh recovery queue")
    config = build(args.backup, args.destination, args.source, args.python, args.campaign, args.mount)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"jobs": len(config["jobs"]), **config["recovery"]}, indent=2))


if __name__ == "__main__":
    main()
