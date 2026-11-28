"""Play a frozen policy from battle-bank snapshots and report BattleReward terms.

Real-game check of RAM addresses, RUN hooks and battle-exit truncation. It does not
train, and its outcomes describe the frozen parent, not the battle curriculum.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from battle_training import (
    PRESET_PENALTIES,
    PRESETS,
    BattleReplay,
    BattleReward,
    snapshot_config_sha,
)
from experiment import ALGORITHMS
from red_gym_env_v2_nuzlocke import RedGymEnv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bank", "rom", "state", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--trainer-share", type=float, default=0.5)
    p.add_argument("--max-battle-actions", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-preset", choices=sorted(PRESETS), default="dense")
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    parent = json.loads((args.parent / "manifest.json").read_text())
    config = dict(
        parent["config"],
        session_path=str(args.output / "env"),
        gb_path=str(args.rom.resolve()),
        init_state=str(args.state.resolve()),
        save_video=False,
        save_final_state=False,
    )
    sha = snapshot_config_sha(parent["config"], parent["rom_sha1"], parent["state_sha1"])
    env = BattleReward(
        BattleReplay(
            RedGymEnv(config),
            args.bank,
            seed=args.seed,
            trainer_share=args.trainer_share,
            max_battle_actions=args.max_battle_actions,
            config_sha=sha,
            timeout_penalty=PRESET_PENALTIES[args.reward_preset]["battle_timeout_penalty"],
        ),
        PRESETS[args.reward_preset],
        gamma=float(config["navigation_gamma"]),
        rom=args.rom,
    )
    model = ALGORITHMS[parent["algorithm"]].load(
        args.parent / "final.zip", device="cpu", custom_objects={"_last_lstm_states": None}
    )
    records = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            source = info["battle_source"]
            state, start = None, np.ones((1,), dtype=bool)
            done, reward_sum = False, 0.0
            while not done:
                action, state = model.predict(
                    obs, state=state, episode_start=start, deterministic=False
                )
                obs, reward, terminated, truncated, info = env.step(int(action))
                start[:] = False
                reward_sum += reward
                done = terminated or truncated
            record = {
                "episode": episode,
                "battle": source["battle"],
                "map": source["map"],
                "anchor": source["anchor"],
                "end_reason": info["end_reason"],
                "actions": info["battle_replay_actions"],
                "return": reward_sum,
                "totals": info["battle_reward_totals"],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()
    summary = {
        "episodes": len(records),
        "end_reasons": Counter(r["end_reason"] for r in records),
        "hooks": dict(env.lifetime),
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
