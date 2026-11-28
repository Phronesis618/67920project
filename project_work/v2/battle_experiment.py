"""Continue a parent with battle rewards and battle-only replay workers.

Route workers play the original post-parcel save so route-following keeps being
trained; replay workers start at fixed battle snapshots (scripts/build_battle_bank.py)
and stop when that battle ends. Every worker uses the same reward: the parent's route
reward plus BattleReward terms, death and rule-violation penalties.
"""

import argparse
import json
import time
from dataclasses import asdict, fields
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from battle_training import (
    PRESET_PENALTIES,
    PRESETS,
    BattleReplay,
    BattleReward,
    BattleRewardWeights,
    load_bank,
    snapshot_config_sha,
)
from experiment import ALGORITHMS, EpisodeCallback, EpisodeLog, git_metadata, sha1
from poison_experiment import sha256, write_json
from red_gym_env_v2_nuzlocke import RedGymEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(config, output, rank, route_workers, args, weights, config_sha):
    env = RedGymEnv(dict(config, session_path=str(output / f"worker-{rank}")))
    if rank >= route_workers:
        env = BattleReplay(
            env,
            args.bank,
            seed=args.seed * 1000 + rank,
            trainer_share=args.trainer_share,
            max_battle_actions=args.max_battle_actions,
            timeout_penalty=args.battle_timeout_penalty,
            config_sha=config_sha,
            area_share=args.area_share,
            max_area_actions=args.max_area_actions,
        )
    env = BattleReward(env, weights, gamma=args.gamma, rom=args.rom)
    env = EpisodeLog(env, output / f"episodes-{rank}.jsonl", rank)
    return Monitor(env, info_keywords=("is_success", "end_reason", "deaths", "training_reset"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "rom", "state", "bank"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=1048576)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--replay-workers", type=int, help="Default: half of --num-envs")
    p.add_argument("--trainer-share", type=float, default=0.5)
    p.add_argument("--max-battle-actions", type=int, default=1024)
    p.add_argument("--area-share", type=float, default=0.0, help="Replay starts in fresh areas")
    p.add_argument("--max-area-actions", type=int, default=2048)
    p.add_argument("--reward-preset", choices=sorted(PRESETS), default="dense")
    p.add_argument("--battle-timeout-penalty", type=float, help="Default: preset's")
    p.add_argument("--death-penalty", type=float, help="Default: preset's")
    p.add_argument("--violation-penalty", type=float, help="Default: preset's")
    p.add_argument("--learning-rate", type=float, help="Default: parent's")
    p.add_argument("--target-kl", type=float, help="Default: parent's (none)")
    p.add_argument("--clip-range", type=float, default=0.2, help="Parents used constant 0.2")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="128-action route episodes, 64-action battles, 32-step rollouts; not an arm",
    )
    for f in fields(BattleRewardWeights):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), help="Preset value")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.timesteps, args.num_envs) < 1:
        p.error("timesteps and num-envs must be positive")
    replay_workers = args.num_envs // 2 if args.replay_workers is None else args.replay_workers
    if not 0 <= replay_workers < args.num_envs:
        p.error("Keep at least one original-route worker")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Choose an empty output directory")
    parent = json.loads((args.parent / "manifest.json").read_text())
    if parent["config"]["observation_mode"] != "pixels":
        p.error("This experiment requires pixel parents")
    if sha1(args.rom) != parent["rom_sha1"] or sha1(args.state) != parent["state_sha1"]:
        p.error("Parent assets do not match")
    config_sha = snapshot_config_sha(parent["config"], parent["rom_sha1"], parent["state_sha1"])
    bank_rows, bank_manifest = load_bank(args.bank)
    if bank_manifest["config_sha256"] != config_sha:
        p.error("Battle bank was captured under a different environment config")
    preset = asdict(PRESETS[args.reward_preset])
    preset.update(
        {
            f.name: getattr(args, f.name)
            for f in fields(BattleRewardWeights)
            if getattr(args, f.name) is not None
        }
    )
    weights = BattleRewardWeights(**preset)
    for name, value in PRESET_PENALTIES[args.reward_preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    config = dict(
        parent["config"],
        session_path=str(args.output.resolve()),
        gb_path=str(args.rom.resolve()),
        init_state=str(args.state.resolve()),
        save_video=False,
        save_final_state=False,
        death_penalty=args.death_penalty,
        violation_penalty=args.violation_penalty,
    )
    if args.smoke:
        config["max_steps"] = 128
        args.max_battle_actions = min(args.max_battle_actions, 64)
        args.max_area_actions = min(args.max_area_actions, 128)
    args.gamma = float(config.get("navigation_gamma", 0.9995))
    args.output.mkdir(parents=True, exist_ok=True)
    route_workers = args.num_envs - replay_workers
    factories = [
        partial(make_env, config, args.output, i, route_workers, args, weights, config_sha)
        for i in range(args.num_envs)
    ]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories)
    try:
        algorithm = parent["algorithm"]
        overrides = {"n_steps": 32, "batch_size": 32} if args.smoke else {}
        if args.learning_rate is not None:
            overrides["learning_rate"] = args.learning_rate
        if args.target_kl is not None:
            overrides["target_kl"] = args.target_kl
        model = ALGORITHMS[algorithm].load(
            args.parent / "final.zip",
            env=env,
            device=args.device,
            seed=args.seed,
            # Schedules are pickled functions whose bytecode only runs on the Python
            # that saved them (Lambda 3.11 crashes 3.12 with SIGTRAP). Rebuild them
            # from plain values; lr_schedule is rebuilt from learning_rate on setup.
            custom_objects={
                "_last_lstm_states": None,
                "clip_range": args.clip_range,
                "lr_schedule": None,
            },
            **overrides,
        )
        if abs(model.gamma - args.gamma) > 1e-12:
            p.error("Parent PPO gamma differs from its navigation gamma")
        model.set_random_seed(args.seed)
        model.tensorboard_log = str(args.output / "tensorboard")
        model.verbose = 1
        # Parent rolling metrics describe a different training segment.
        model.ep_info_buffer = None
        model.ep_success_buffer = None
        initial_steps = model.num_timesteps
        manifest = {
            "algorithm": algorithm,
            "config": config,
            "ruleset": parent["ruleset"],
            "rom_sha1": parent["rom_sha1"],
            "state_sha1": parent["state_sha1"],
            "snapshot_config_sha256": config_sha,
            "parent_checkpoint": str((args.parent / "final.zip").resolve()),
            "parent_checkpoint_sha256": sha256(args.parent / "final.zip"),
            "parent_timesteps": initial_steps,
            "created_utc": datetime.now(UTC).isoformat(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "optimizer": {
                "learning_rate": model.learning_rate,
                "target_kl": model.target_kl,
                "clip_range": args.clip_range,
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "n_epochs": model.n_epochs,
                "gamma": model.gamma,
            },
            "battle": {
                "route_workers": route_workers,
                "replay_workers": replay_workers,
                "reward_preset": args.reward_preset,
                "trainer_share": args.trainer_share,
                "area_share": args.area_share,
                "max_area_actions": args.max_area_actions,
                "max_battle_actions": args.max_battle_actions,
                "battle_timeout_penalty": args.battle_timeout_penalty,
                "weights": asdict(weights),
                "bank": str(args.bank.resolve()),
                "bank_manifest": bank_manifest,
                "bank_rows": len(bank_rows),
            },
            "contract": "Parent route reward retained and extended for every worker by the "
            f"'{args.reward_preset}' battle preset (weights above; wild terms off at the level "
            "cap). dense: damage/KO/win, levels up to the cap, potential-based HP shaping, "
            "rejected trainer RUN and overtime, death and violation penalties. outcome: only "
            "win = 10 x party HP fraction at exit, flee 0, legal catch +5, illegal catch or other "
            "violation -10, wipe in battle -10, battle stalled to the cap -10. Area replay "
            "starts (fresh area, no encounter yet) end on leaving the map. Replay workers start at fixed battle snapshots and truncate "
            "at battle exit or max_battle_actions (with penalty). RAM is privileged reward/"
            "reset information only; actor inputs and action set unchanged. Evaluation must "
            "still use original starts.",
            **git_metadata(),
        }
        write_json(args.output / "manifest.json", manifest)
        model.save(args.output / "initial.zip")
        print(
            json.dumps(
                {
                    "initialized": str(args.output),
                    "route_workers": route_workers,
                    "replay_workers": replay_workers,
                }
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
                "worker_stats": env.env_method("battle_stats"),
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
