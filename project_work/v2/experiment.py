"""Small, reproducible training and evaluation entry points; no public streaming."""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
from navigation import graph_sha256
from red_gym_env_v2_nuzlocke import RedGymEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = {"ppo": PPO, "lstm": RecurrentPPO}
RULESET = "core-strict-revival-battle-entry-cap-v1"


class EpisodeLog(gym.Wrapper):
    """Write terminal info BEFORE a vector environment auto-resets the worker."""

    def __init__(self, env, path, worker):
        super().__init__(env)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.worker = worker
        self.episode_index = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            record = dict(
                info,
                worker=self.worker,
                episode_index=self.episode_index,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            self.episode_index += 1
        return obs, reward, terminated, truncated, info


class EpisodeCallback(BaseCallback):
    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                for key in ("is_success", "deaths", "map_coverage", "episode_steps"):
                    self.logger.record_mean(f"nuzlocke/{key}", float(info[key]))
                self.logger.record_mean("nuzlocke/wipe", float(info["end_reason"] == "wipe"))
                self.logger.record_mean(
                    "nuzlocke/violation", float(bool(info["violation_reasons"]))
                )
        return True


def make_env(config, output, rank):
    env = RedGymEnv(config)
    env = EpisodeLog(env, Path(output) / f"episodes-{rank}.jsonl", rank)
    return Monitor(
        env,
        str(Path(output) / f"worker-{rank}.monitor.csv"),
        info_keywords=("is_success", "end_reason", "deaths"),
    )


def sha1(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha1").hexdigest()


def git_metadata():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {"git_commit": commit.stdout.strip(), "git_dirty": bool(dirty.stdout.strip())}


def train_parser():
    p = argparse.ArgumentParser(description="Train the limited Nuzlocke ruleset; see README.")
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--state", type=Path, default=ROOT / "project_work/init.state")
    p.add_argument(
        "--output", type=Path, default=ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    p.add_argument("--algorithm", choices=ALGORITHMS, default="ppo")
    p.add_argument(
        "--observation-mode", choices=("partial", "history", "pixels"), default="partial"
    )
    p.add_argument("--no-explore-map", action="store_true")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=16_384)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--navigation-reward", action="store_true")
    p.add_argument("--explore-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level-cap", type=int, default=14)
    p.add_argument("--start-jitter-frames", type=int, default=0)
    p.add_argument("--initial-used-regions", type=int, nargs="*", default=[])
    p.add_argument("--region-map", type=Path)
    p.add_argument("--death-penalty", type=float, default=0)
    p.add_argument("--violation-penalty", type=float, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    return p


def train_main(argv=None):
    parser = train_parser()
    args = parser.parse_args(argv)
    for name in ("rom", "state"):
        if not getattr(args, name).is_file():
            parser.error(f"{name} does not exist: {getattr(args, name)}")
    for name in (
        "timesteps",
        "num_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "max_episode_steps",
        "checkpoint_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    rollout = args.n_steps * args.num_envs
    if rollout < 2 or args.batch_size < 2 or rollout % args.batch_size:
        parser.error("rollout size (n-steps * num-envs) must be divisible by batch-size >= 2")
    if not 0 < args.gamma <= 1:
        parser.error("gamma must be in (0, 1]")
    if not 0 <= args.gae_lambda <= 1:
        parser.error("gae-lambda must be in [0, 1]")
    if not np.isfinite(args.explore_weight) or args.explore_weight < 0:
        parser.error("explore-weight must be finite and nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a new run directory")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "session_path": str(args.output),
        "gb_path": str(args.rom.resolve()),
        "init_state": str(args.state.resolve()),
        "headless": True,
        "save_final_state": False,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "action_freq": 24,
        "max_steps": args.max_episode_steps,
        "reward_scale": 0.5,
        "explore_weight": args.explore_weight,
        "navigation_reward": args.navigation_reward,
        "navigation_gamma": args.gamma,
        "navigation_graph_sha256": graph_sha256() if args.navigation_reward else None,
        "observation_mode": args.observation_mode,
        "include_explore_map": not args.no_explore_map and args.observation_mode != "pixels",
        "level_cap": args.level_cap,
        "start_jitter_frames": args.start_jitter_frames,
        "initial_used_regions": args.initial_used_regions,
        "region_map": json.loads(args.region_map.read_text()) if args.region_map else {},
        "death_penalty": args.death_penalty,
        "violation_penalty": args.violation_penalty,
    }
    manifest = {
        "ruleset": RULESET,
        "algorithm": args.algorithm,
        "config": config,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **git_metadata(),
        "rom_sha1": sha1(args.rom),
        "state_sha1": sha1(args.state),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    factories = [partial(make_env, config, args.output, rank) for rank in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
    start = time.monotonic()
    try:
        model = ALGORITHMS[args.algorithm](
            "MultiInputPolicy" if args.algorithm == "ppo" else "MultiInputLstmPolicy",
            env,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=0.01,
            tensorboard_log=str(args.output / "tensorboard"),
            device=args.device,
            verbose=1,
        )
        callbacks = [
            EpisodeCallback(),
            CheckpointCallback(
                save_freq=max(args.checkpoint_every // args.num_envs, 1),
                save_path=str(args.output / "checkpoints"),
                name_prefix="policy",
            ),
        ]
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(args.output / "final.zip")
        elapsed = time.monotonic() - start
        result = {
            "actual_timesteps": model.num_timesteps,
            "seconds": elapsed,
            "steps_per_second": model.num_timesteps / elapsed,
        }
        (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


def resume_main(argv=None):
    p = argparse.ArgumentParser(
        description="Continue PPO/LSTM weights and optimizer in a new run; episodes restart."
    )
    p.add_argument("run", type=Path, help="Source run containing manifest.json")
    p.add_argument(
        "--checkpoint", type=Path, help="Checkpoint inside source run; default final.zip"
    )
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--timesteps", type=int, required=True, help="Additional environment steps")
    p.add_argument("--rom", type=Path, help="Relocated ROM; must match the original hash")
    p.add_argument(
        "--state", type=Path, help="Relocated initial save; must match the original hash"
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    source = args.run.resolve()
    checkpoint = (args.checkpoint or source / "final.zip").resolve()
    if not (source / "manifest.json").is_file():
        p.error("source run must contain manifest.json")
    if not checkpoint.is_file() or not checkpoint.is_relative_to(source):
        p.error("checkpoint must be an existing file inside the source run")
    if args.timesteps < 1:
        p.error("timesteps must be positive")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error("output directory is not empty; choose a new run directory")
    parent = json.loads((source / "manifest.json").read_text())
    if parent["ruleset"] != RULESET or parent["algorithm"] not in ALGORITHMS:
        p.error("source run uses an unsupported algorithm or ruleset")
    config = dict(parent["config"])
    for name, key, hash_key in (
        ("rom", "gb_path", "rom_sha1"),
        ("state", "init_state", "state_sha1"),
    ):
        path = getattr(args, name) or Path(config[key])
        if not path.is_file():
            p.error(f"{name} not found; supply its new location with --{name}")
        if sha1(path) != parent[hash_key]:
            p.error(f"{name} differs from the original training asset")
        config[key] = str(path.resolve())
    config["session_path"] = str(output)
    arguments = dict(parent["arguments"])
    arguments.update(
        output=str(output),
        timesteps=args.timesteps,
        device=args.device,
        rom=config["gb_path"],
        state=config["init_state"],
    )
    num_envs = arguments["num_envs"]
    output.mkdir(parents=True, exist_ok=True)
    factories = [partial(make_env, config, output, rank) for rank in range(num_envs)]
    env = DummyVecEnv(factories) if num_envs == 1 else SubprocVecEnv(factories)
    start = time.monotonic()
    try:
        model = ALGORITHMS[parent["algorithm"]].load(
            checkpoint,
            env=env,
            device=args.device,
            force_reset=True,
            tensorboard_log=str(output / "tensorboard"),
            seed=arguments["seed"],
        )
        initial_timesteps = model.num_timesteps
        manifest = {
            "ruleset": RULESET,
            "algorithm": parent["algorithm"],
            "config": config,
            "arguments": arguments,
            **git_metadata(),
            "rom_sha1": parent["rom_sha1"],
            "state_sha1": parent["state_sha1"],
            "created_utc": datetime.now(UTC).isoformat(),
            "resume": {
                "source_run": str(source),
                "checkpoint": str(checkpoint),
                "checkpoint_sha1": sha1(checkpoint),
                "initial_timesteps": initial_timesteps,
                "episodes_restarted": True,
                "rng_reseeded": True,
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        callbacks = [
            EpisodeCallback(),
            CheckpointCallback(
                save_freq=max(arguments["checkpoint_every"] // num_envs, 1),
                save_path=str(output / "checkpoints"),
                name_prefix="policy",
            ),
        ]
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=False)
        model.save(output / "final.zip")
        elapsed = time.monotonic() - start
        additional = model.num_timesteps - initial_timesteps
        result = {
            "actual_timesteps": model.num_timesteps,
            "initial_timesteps": initial_timesteps,
            "additional_timesteps": additional,
            "seconds": elapsed,
            "steps_per_second": additional / elapsed,
        }
        (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


def wilson_interval(successes, n):
    if n < 1:
        raise ValueError("At least one episode is required")
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate_main(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a saved run with its original observation/rule config"
    )
    p.add_argument("run", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.episodes < 1:
        p.error("episodes must be positive")
    output = args.output or args.run / "evaluation.json"
    if output.exists():
        p.error("evaluation output exists; choose a new --output path")
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = dict(manifest["config"])
    for key, hash_key in (("gb_path", "rom_sha1"), ("init_state", "state_sha1")):
        if sha1(config[key]) != manifest[hash_key]:
            p.error(f"{key} differs from the training asset")
    env = RedGymEnv(config)
    records = []
    try:
        model = ALGORITHMS[manifest["algorithm"]].load(
            args.model or args.run / "final.zip", env=env
        )
        set_random_seed(args.seed)
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            state, episode_start = None, np.array([True])
            while True:
                action, state = model.predict(
                    obs, state=state, episode_start=episode_start, deterministic=args.deterministic
                )
                obs, _, terminated, truncated, info = env.step(int(action))
                episode_start[:] = False
                if terminated or truncated:
                    records.append(dict(info, evaluation_seed=args.seed + episode))
                    break
    finally:
        env.close()
    successes = sum(r["is_success"] for r in records)
    stochastic_trials = not args.deterministic or config.get("start_jitter_frames", 0) > 0
    result = {
        "episodes": args.episodes,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "success_rate": successes / args.episodes,
        "success_wilson_95": wilson_interval(successes, args.episodes)
        if stochastic_trials
        else None,
        "stochastic_trials": stochastic_trials,
        "end_reasons": dict(Counter(r["end_reason"] for r in records)),
        "records": records,
        "interpretation": "Episode sampling uncertainty for this checkpoint and start setup; not variation across training seeds.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
