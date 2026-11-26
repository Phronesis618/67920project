"""Fine-tune paired pixel policies on natural poison saves, then measure transfer."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
from experiment import ALGORITHMS, EpisodeCallback, EpisodeLog, git_metadata, sha1
from poison_curriculum import MatchedCombinedExtractor, PoisonEnv, load_suite
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_env(config, output, rank):
    config = dict(config, session_path=str(Path(output) / f"worker-{rank}"))
    env = EpisodeLog(PoisonEnv(config), Path(output) / f"episodes-{rank}.jsonl", rank)
    return Monitor(env, info_keywords=("is_success", "end_reason", "deaths"))


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    for key in ("timesteps", "num_envs", "n_steps", "batch_size", "max_steps"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.batch_size < 2 or args.num_envs * args.n_steps % args.batch_size:
        raise ValueError("Rollout must be divisible by batch_size >= 2")
    parent = json.loads((args.parent / "manifest.json").read_text())
    if parent["config"]["observation_mode"] != "pixels":
        raise ValueError("Parent must be a pixel policy")
    suite = json.loads(args.suite.read_text())
    if sha1(args.rom) != parent["rom_sha1"] or suite["rom_sha1"] != parent["rom_sha1"]:
        raise ValueError("ROM differs from parent or scenario suite")
    scenarios = load_suite(args.suite, "train")
    config = dict(parent["config"])
    config.update(
        task="cure",
        scenario_suite=str(args.suite.resolve()),
        split="train",
        screen_scale=args.screen_scale,
        gb_path=str(args.rom.resolve()),
        init_state=scenarios[0]["state_path"],
        session_path=str(args.output.resolve()),
        max_steps=args.max_steps,
        save_video=False,
        save_final_state=False,
        navigation_reward=False,
        explore_weight=0,
        start_jitter_frames=0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    factories = [partial(make_env, config, args.output, i) for i in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
    start = time.monotonic()
    try:
        algorithm = parent["algorithm"]
        pretrained = ALGORITHMS[algorithm].load(args.parent / "final.zip", device="cpu")
        # Architecture and every learned weight are shared within each resolution pair.
        # Optimizers are fresh in both arms; native input changes visual activations.
        policy_kwargs = dict(pretrained.policy_kwargs)
        policy_kwargs["features_extractor_class"] = MatchedCombinedExtractor
        policy_kwargs.pop("features_extractor_kwargs", None)
        model = ALGORITHMS[algorithm](
            "MultiInputPolicy" if algorithm == "ppo" else "MultiInputLstmPolicy",
            env,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=0.0003,
            policy_kwargs=policy_kwargs,
            device=args.device,
            verbose=1,
            tensorboard_log=str(args.output / "tensorboard"),
        )
        model.policy.load_state_dict(pretrained.policy.state_dict(), strict=True)
        manifest = {
            "algorithm": algorithm,
            "config": config,
            "parent_config": parent["config"],
            "parent_state_sha1": parent["state_sha1"],
            "rom_sha1": parent["rom_sha1"],
            "parent_checkpoint": str((args.parent / "final.zip").resolve()),
            "parent_checkpoint_sha256": sha256(args.parent / "final.zip"),
            "parent_timesteps": pretrained.num_timesteps,
            "suite_sha256": sha256(args.suite),
            "created_utc": datetime.now(UTC).isoformat(),
            "parameters": sum(p.numel() for p in model.policy.parameters()),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "contract": "Actor sees screens and recent actions only. RAM scores cures. Fresh optimizer; all parent policy weights loaded. Initial model is evaluated at its own resolution before comparison with fine-tuned model. No auxiliary loss or route reward changes.",
            **git_metadata(),
        }
        del pretrained
        write_json(args.output / "manifest.json", manifest)
        model.save(args.output / "initial.zip")
        print(
            json.dumps({"initialized": str(args.output), "parameters": manifest["parameters"]}),
            flush=True,
        )
        model.learn(
            total_timesteps=args.timesteps,
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
                "actual_timesteps": model.num_timesteps,
                "seconds": time.monotonic() - start,
            },
        )
    finally:
        env.close()


def evaluate(args):
    output = args.run / "evaluation.json"
    if output.exists():
        raise ValueError("Evaluation already exists")
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = manifest["config"]
    if sha256(config["scenario_suite"]) != manifest["suite_sha256"]:
        raise ValueError("Scenario suite changed")
    if sha1(config["gb_path"]) != manifest["rom_sha1"]:
        raise ValueError("ROM changed")
    if sha1(manifest["parent_config"]["init_state"]) != manifest["parent_state_sha1"]:
        raise ValueError("Route start changed")
    results = {
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha256(args.run / "manifest.json"),
        "interpretation": "Stochastic policy rollouts from fixed natural saves. Heldout low-HP saves share the training source episode. Route evaluation is a small transfer pilot, not evidence of architecture superiority. Healthy controls check false cure scoring; the game itself rejects antidotes on healthy Pokemon.",
        "conditions": [],
    }
    for checkpoint in ("initial", "final"):
        model_path = args.run / f"{checkpoint}.zip"
        model = ALGORITHMS[manifest["algorithm"]].load(model_path, device=args.device)
        for split in ("train", "heldout", "healthy", "route"):
            cfg = dict(
                config, split=split, session_path=str(args.run / "eval" / checkpoint / split)
            )
            count = args.episodes
            if split == "route":
                cfg = dict(
                    manifest["parent_config"],
                    task="route",
                    screen_scale=config["screen_scale"],
                    session_path=cfg["session_path"],
                    max_steps=args.route_max_steps,
                    save_video=False,
                    save_final_state=False,
                )
                count = args.route_episodes
            env = PoisonEnv(cfg)
            records = []
            set_random_seed(args.seed)
            try:
                for episode in range(count):
                    options = (
                        {} if split == "route" else {"scenario_index": episode % len(env.scenarios)}
                    )
                    obs, _ = env.reset(seed=args.seed + episode, options=options)
                    state, episode_start = None, np.array([True])
                    while True:
                        action, state = model.predict(
                            obs, state=state, episode_start=episode_start, deterministic=False
                        )
                        obs, _, terminated, truncated, info = env.step(int(action))
                        episode_start[:] = False
                        if terminated or truncated:
                            records.append(dict(info, evaluation_seed=args.seed + episode))
                            break
            finally:
                env.close()
            condition = {
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256(model_path),
                "split": split,
                "episodes": count,
                "successes": sum(r["is_success"] for r in records),
                "records": records,
            }
            results["conditions"].append(condition)
            write_json(args.run / "evaluation.partial.json", results)
            print(json.dumps({k: v for k, v in condition.items() if k != "records"}), flush=True)
        del model
    write_json(output, results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    for name in ("parent", "suite", "rom", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--screen-scale", type=int, choices=(1, 2), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=1048576)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--episodes", type=int, default=9, help="Per save split, balanced across scenarios"
    )
    p.add_argument("--route-episodes", type=int, default=3)
    p.add_argument("--route-max-steps", type=int, default=16384)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (
        args.command == "evaluate"
        and min(args.episodes, args.route_episodes, args.route_max_steps) < 1
    ):
        parser.error("Evaluation counts must be positive")
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
