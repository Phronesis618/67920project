"""Build the paired antidote curriculum queue on the attached Lambda filesystem."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", default="/home/ubuntu/67920-venv/bin/python")
    p.add_argument("--stop-at", default="2026-11-30T04:59:00+00:00")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists")
    script = str(args.source / "project_work/v2/poison_experiment.py")
    jobs = []
    for seed in range(3):
        for algorithm in ("ppo", "lstm"):
            parent = (
                args.root
                / "runs/campaign-recovery-20261123/runs"
                / f"pixels-{algorithm}-s{seed}-m10"
            )
            if not (parent / "final.zip").is_file():
                p.error(f"Missing parent: {parent}")
            for scale, resolution in ((2, "half"), (1, "native")):
                name = f"{algorithm}-{resolution}-s{seed}"
                output = args.output.parent / "runs" / name
                jobs.append(
                    {
                        "id": f"train-{name}",
                        "timeout_seconds": 21600,
                        "command": [
                            args.python,
                            script,
                            "train",
                            "--parent",
                            str(parent),
                            "--suite",
                            str(args.suite),
                            "--rom",
                            str(args.source / "project_work/PokemonRed.gb"),
                            "--output",
                            str(output),
                            "--screen-scale",
                            str(scale),
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
                            args.python,
                            script,
                            "evaluate",
                            "--run",
                            str(output),
                        ],
                    }
                )
    config = {
        "cwd": str(args.source),
        "required_mount": str(args.root.parent),
        "stop_at": args.stop_at,
        "max_consecutive_failures": 2,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/67920-matplotlib",
        },
        "contract": "12 paired fine-tunes: PPO/LSTM x half/native x 3 seeds; 1,048,576 actions each. Each gets initial/final cure, healthy-control and route-transfer evaluation. Originals untouched.",
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
