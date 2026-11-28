"""One matched recurrent continuation: detached readout versus auxiliary learning."""

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from context_aux import ContextCallback, ContextLabels, ContextPPO
from experiment import EpisodeCallback, EpisodeLog, git_metadata, sha1
from poison_experiment import sha256, write_json
from red_gym_env_v2_nuzlocke import RedGymEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(config, output, rank):
    env = RedGymEnv(dict(config, session_path=str(output / f"worker-{rank}")))
    env = EpisodeLog(env, output / f"episodes-{rank}.jsonl", rank)
    return Monitor(ContextLabels(env), info_keywords=("is_success", "end_reason", "deaths"))


def convert_parent(parent, env, *, seed, backprop, device, n_steps=None, batch_size=None):
    """Retain all actor/critic weights and Adam moments; only the readout is new."""
    kwargs = {
        key: getattr(parent, key)
        for key in (
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "clip_range_vf",
            "normalize_advantage",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "policy_kwargs",
        )
    }
    if n_steps is not None:
        kwargs.update(n_steps=n_steps, batch_size=batch_size)
    model = ContextPPO(
        "MultiInputLstmPolicy",
        env,
        seed=seed,
        device=device,
        context_backprop=backprop,
        context_coef=0.1,
        **kwargs,
    )
    missing, unexpected = model.policy.load_state_dict(parent.policy.state_dict(), strict=False)
    if set(missing) != {"context_head.weight", "context_head.bias"} or unexpected:
        raise ValueError((missing, unexpected))
    old = copy.deepcopy(parent.policy.optimizer.state_dict())
    if len(old["param_groups"]) != 1:
        raise ValueError("Expected an ordinary upstream parent optimizer")
    new = model.policy.optimizer.state_dict()
    assert old["param_groups"][0]["params"] == new["param_groups"][0]["params"]
    old["param_groups"].append(new["param_groups"][1])
    model.policy.optimizer.load_state_dict(old)
    model.num_timesteps = parent.num_timesteps
    model._n_updates = parent._n_updates
    model.set_random_seed(seed)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "rom", "state"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--arm", choices=("control", "auxiliary"), required=True)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=262144)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.num_envs < 1 or args.timesteps < 1:
        p.error("positive workers and timesteps required")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Choose a new output directory")
    parent = json.loads((args.parent / "manifest.json").read_text())
    if parent["algorithm"] != "lstm" or parent["config"]["observation_mode"] != "pixels":
        p.error("Requires a pixel LSTM parent")
    for path, key in ((args.rom, "rom_sha1"), (args.state, "state_sha1")):
        if sha1(path) != parent[key]:
            p.error("Parent assets differ")
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(
        parent["config"],
        gb_path=str(args.rom.resolve()),
        init_state=str(args.state.resolve()),
        session_path=str(args.output.resolve()),
        save_video=False,
        save_final_state=False,
    )
    if args.smoke:
        config["max_steps"] = 64
    factories = [partial(make_env, config, args.output, i) for i in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
    try:
        source = RecurrentPPO.load(args.parent / "final.zip", device=args.device)
        model = convert_parent(
            source,
            env,
            seed=args.seed,
            backprop=args.arm == "auxiliary",
            device=args.device,
            n_steps=16 if args.smoke else None,
            batch_size=16 if args.smoke else None,
        )
        initial = model.num_timesteps
        manifest = {
            "algorithm": "lstm",
            "training_class": "context_aux.ContextPPO",
            "config": config,
            "ruleset": parent["ruleset"],
            "rom_sha1": parent["rom_sha1"],
            "state_sha1": parent["state_sha1"],
            "created_utc": datetime.now(UTC).isoformat(),
            "parent_checkpoint": str((args.parent / "final.zip").resolve()),
            "parent_checkpoint_sha256": sha256(args.parent / "final.zip"),
            "parent_timesteps": initial,
            "arm": args.arm,
            "context_coef": model.context_coef,
            "optimizer": {
                k: getattr(model, k)
                for k in (
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "target_kl",
                    "ent_coef",
                )
            },
            "learning_rate": model.lr_schedule(1.0),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "contract": "Same ordinary post-parcel starts, pixel/action actor observations, route rewards, pretrained actor/critic weights and Adam state. Both arms train an identical RAM-supervised context head; control detaches its LSTM features, auxiliary backpropagates a 0.1-weight prediction loss into CNN/actor LSTM. Head and base gradients clipped separately. Labels describe the observation BEFORE its action, never its outcome. No action targets, masks, scripted fighting, item rewards or replay starts. One development-selected parent seed, bounded pilot; no broad claim without replication.",
            **git_metadata(),
        }
        write_json(args.output / "manifest.json", manifest)
        model.tensorboard_log = str(args.output / "tensorboard")
        model.verbose = 1
        model.save(args.output / "initial.zip")
        start = time.monotonic()
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=[
                ContextCallback(),
                EpisodeCallback(),
                CheckpointCallback(
                    save_freq=max(65536 // args.num_envs, 1),
                    save_path=str(args.output / "checkpoints"),
                    name_prefix="policy",
                ),
            ],
        )
        model.save(args.output / "final.zip")
        write_json(
            args.output / "training.json",
            {
                "additional_timesteps": model.num_timesteps - initial,
                "total_timesteps": model.num_timesteps,
                "seconds": time.monotonic() - start,
                "optimizer_learning_rates": [g["lr"] for g in model.policy.optimizer.param_groups],
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
