"""Continue a route policy with matched ordinary/failure-restart training."""

import argparse
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from experiment import ALGORITHMS, EpisodeCallback, EpisodeLog, git_metadata, sha1
from failure_restart import FailureRestart
from poison_experiment import sha256, write_json
from red_gym_env_v2_nuzlocke import RedGymEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(config, output, rank, probability, seed):
    env = RedGymEnv(dict(config, session_path=str(output / f"worker-{rank}")))
    env = FailureRestart(env, output / "failure-banks" / f"worker-{rank}", probability, seed + rank)
    env = EpisodeLog(env, output / f"episodes-{rank}.jsonl", rank)
    return Monitor(env, info_keywords=("is_success", "end_reason", "deaths", "training_reset"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "rom", "state"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--restart-probability", type=float, choices=(0, 0.5), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=1048576)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="128-action episodes and 32-step rollouts; not an experiment arm",
    )
    args = p.parse_args(argv)
    if min(args.timesteps, args.num_envs) < 1:
        p.error("timesteps and num-envs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Choose an empty output directory")
    parent = json.loads((args.parent / "manifest.json").read_text())
    if parent["config"]["observation_mode"] != "pixels":
        p.error("This paired experiment requires pixel parents")
    if sha1(args.rom) != parent["rom_sha1"] or sha1(args.state) != parent["state_sha1"]:
        p.error("Parent assets do not match")
    config = dict(
        parent["config"],
        session_path=str(args.output.resolve()),
        gb_path=str(args.rom.resolve()),
        init_state=str(args.state.resolve()),
        save_video=False,
        save_final_state=False,
    )
    if args.smoke:
        config["max_steps"] = 128
    args.output.mkdir(parents=True, exist_ok=True)
    factories = [
        partial(make_env, config, args.output, i, args.restart_probability, args.seed)
        for i in range(args.num_envs)
    ]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
    try:
        algorithm = parent["algorithm"]
        overrides = {"n_steps": 32, "batch_size": 32} if args.smoke else {}
        model = ALGORITHMS[algorithm].load(
            args.parent / "final.zip",
            env=env,
            device=args.device,
            seed=args.seed,
            custom_objects={"_last_lstm_states": None},
            **overrides,
        )
        model.set_random_seed(args.seed)
        model.tensorboard_log = str(args.output / "tensorboard")
        model.verbose = 1
        # Parent rolling metrics describe a different training segment; retaining
        # them would mislabel the new arm's early success/length statistics.
        model.ep_info_buffer = None
        model.ep_success_buffer = None
        initial_steps = model.num_timesteps
        manifest = {
            "algorithm": algorithm,
            "config": config,
            "ruleset": parent["ruleset"],
            "rom_sha1": parent["rom_sha1"],
            "state_sha1": parent["state_sha1"],
            "parent_checkpoint": str((args.parent / "final.zip").resolve()),
            "parent_checkpoint_sha256": sha256(args.parent / "final.zip"),
            "parent_timesteps": initial_steps,
            "created_utc": datetime.now(UTC).isoformat(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "restart": {
                "probability": args.restart_probability,
                "stride": 64,
                "min_lead": 64,
                "window": 1024,
                "anchors_per_failure": 4,
                "max_failures_per_worker": 8,
            },
            "contract": "Original route reward, observation, optimizer and hyperparameters retained. Both arms capture failure banks; only reset sampling differs. Failure includes wipe, rule violation and unsuccessful time limit. No item-specific predicates/rewards. Original remaining episode budget and all rule/reward history retained; recurrent hidden state resets. Fresh policy actions only. Evaluation always starts from the original post-parcel save.",
            **git_metadata(),
        }
        write_json(args.output / "manifest.json", manifest)
        model.save(args.output / "initial.zip")
        print(
            json.dumps(
                {"initialized": str(args.output), "restart_probability": args.restart_probability}
            ),
            flush=True,
        )
        start = time.monotonic()
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=[
                EpisodeCallback(),
                CheckpointCallback(
                    save_freq=max(262144 // args.num_envs, 1),
                    save_path=str(args.output / "checkpoints"),
                    name_prefix="policy",
                ),
            ],
        )
        model.save(args.output / "final.zip")
        write_json(
            args.output / "training.json",
            {
                "additional_timesteps": model.num_timesteps - initial_steps,
                "total_timesteps": model.num_timesteps,
                "seconds": time.monotonic() - start,
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
