"""Queue a paired route-only failure-restart comparison after the active diagnostic."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--python", default="/home/ubuntu/67920-venv/bin/python")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists")
    prior = args.root / "runs/campaign-poison-20261125"
    jobs = [
        {
            "id": "wait-for-diagnostic",
            "timeout_seconds": 7200,
            "command": [
                args.python,
                "-c",
                (
                    "import json,time,sys\nfrom pathlib import Path\np=Path(sys.argv[1])\n"
                    "while json.loads(p.read_text())['phase'] == 'running':\n    time.sleep(15)\n"
                ),
                str(prior / "status.json"),
            ],
        },
        {
            "id": "evaluate-cure-diagnostic",
            "needs": ["wait-for-diagnostic"],
            "timeout_seconds": 7200,
            "command": [
                args.python,
                str(args.source / "project_work/v2/poison_experiment.py"),
                "evaluate",
                "--run",
                str(prior / "runs/ppo-half-s0"),
            ],
        },
    ]
    for seed in range(3):
        for algo in ("ppo", "lstm"):
            parent = (
                args.root / "runs/campaign-recovery-20261123/runs" / f"pixels-{algo}-s{seed}-m10"
            )
            if not (parent / "final.zip").is_file():
                p.error(f"Missing parent: {parent}")
            for label, probability in (("failure", "0.5"), ("ordinary", "0")):
                name = f"{algo}-{label}-s{seed}"
                output = args.output.parent / "runs" / name
                jobs.append(
                    {
                        "id": f"train-{name}",
                        "needs": ["wait-for-diagnostic"],
                        "timeout_seconds": 21600,
                        "command": [
                            args.python,
                            str(args.source / "project_work/v2/failure_experiment.py"),
                            "--parent",
                            str(parent),
                            "--output",
                            str(output),
                            "--rom",
                            str(args.source / "project_work/PokemonRed.gb"),
                            "--state",
                            str(args.source / "project_work/has_pokedex_nballs.state"),
                            "--restart-probability",
                            probability,
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
                            str(args.source / "project_work/v2/evaluate.py"),
                            str(output),
                            "--episodes",
                            "10",
                            "--seed",
                            "4000",
                        ],
                    }
                )
    config = {
        "cwd": str(args.source),
        "required_mount": str(args.root.parent),
        "stop_at": "2026-11-30T04:59:00+00:00",
        "max_consecutive_failures": 2,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/67920-matplotlib",
        },
        "contract": "PPO/LSTM x failure/ordinary resets x 3 seeds. Same original rewards and 1,048,576 added actions per arm. Failure arm samples 50% original starts, 50% prior failure anchors once available. Final evaluation always original start. First finish/evaluate the one active cure diagnostic; no other poison-grid jobs.",
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "queue": str(args.output)}))


if __name__ == "__main__":
    main()
