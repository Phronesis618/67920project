"""Matched route-only, fixed-share tail replay, and transition replay training."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from experiment import ALGORITHMS, EpisodeCallback, EpisodeLog, git_metadata, sha1
from poison_experiment import sha256, write_json
from red_gym_env_v2_nuzlocke import RedGymEnv
from route_replay import RouteReplay
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(config, output, rank, arm, seed, route_workers):
    env = RedGymEnv(dict(config, session_path=str(output / f"worker-{rank}")))
    banks = output / "failure-banks"
    sources = [banks / f"worker-{i}" for i in range(route_workers)] if rank >= route_workers else []
    env = RouteReplay(
        env,
        banks / f"worker-{rank}",
        sources=sources,
        mode="transition" if arm == "transition" else "tail",
        seed=seed + rank,
    )
    env = EpisodeLog(env, output / f"episodes-{rank}.jsonl", rank)
    return Monitor(env, info_keywords=("is_success", "end_reason", "deaths", "training_reset"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "rom", "state"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--arm", choices=("ordinary", "tail", "transition"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=1048576)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--learning-rate", type=float, help="Optional explicit continuation override")
    p.add_argument("--target-kl", type=float, help="Optional KL early-stopping target")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="128-action episodes and 32-step rollouts; not an experiment arm",
    )
    args = p.parse_args(argv)
    if min(args.timesteps, args.num_envs) < 1:
        p.error("timesteps and num-envs must be positive")
    if args.num_envs % 2 or args.num_envs < 2:
        p.error("Use an even number of workers, at least two")
    for name in ("learning_rate", "target_kl"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"{name} must be finite and positive")
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
    route_workers = args.num_envs if args.arm == "ordinary" else args.num_envs // 2
    factories = [
        partial(
            make_env,
            config,
            args.output,
            i,
            args.arm,
            args.seed,
            route_workers,
        )
        for i in range(args.num_envs)
    ]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
    try:
        algorithm = parent["algorithm"]
        overrides = {"n_steps": 32, "batch_size": 32} if args.smoke else {}
        for name in ("learning_rate", "target_kl"):
            if getattr(args, name) is not None:
                overrides[name] = getattr(args, name)
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
            "optimizer_settings": {
                "learning_rate_at_start": model.lr_schedule(1.0),
                "target_kl": model.target_kl,
                "n_epochs": model.n_epochs,
                "explicit_overrides": {
                    k: v for k, v in overrides.items() if k in ("learning_rate", "target_kl")
                },
            },
            "created_utc": datetime.now(UTC).isoformat(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "restart": {
                "arm": args.arm,
                "route_workers": route_workers,
                "replay_workers": args.num_envs - route_workers,
                "stride": 64,
                "min_lead": 64,
                "tail_window": 1024,
                "transition_reservoir_per_context": 8,
                "anchors_per_context_per_failure": 2,
                "max_failures_per_route_worker": 8,
                "context_selection": "Uniform available overworld/wild/trainer contexts, then uniform anchor",
            },
            "contract": "Original route reward, pixel/action observations and optimizer moments retained. Hyperparameters retained except explicit learning-rate/target-KL overrides recorded in optimizer_settings (and smoke-only rollout/batch overrides). Ordinary arm uses all original-start workers. Mixed arms dedicate half the workers/actions to original-route practice; replay workers sample only failures from those route workers (fallback original when bank empty). Tail anchors are 64..1024 actions before failure; transition anchors follow battle-context or coherent map changes anywhere in the failed episode, plus the first action. Both samplers stratify by RAM-derived overworld/wild/trainer context; this is privileged reset selection, not an actor input. No item-specific rewards, forced fighting or action masks. Original remaining horizon and rule/reward history retained; LSTM hidden state resets. Evaluation always original post-parcel save.",
            **git_metadata(),
        }
        write_json(args.output / "manifest.json", manifest)
        model.save(args.output / "initial.zip")
        print(
            json.dumps(
                {"initialized": str(args.output), "arm": args.arm, "route_workers": route_workers}
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
                "worker_stats": env.env_method("replay_stats"),
                "optimizer_learning_rates": [g["lr"] for g in model.policy.optimizer.param_groups],
                "target_kl": model.target_kl,
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
