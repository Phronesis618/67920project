"""Replay recorded evaluation episodes into a supervised memory dataset.

Each input is a diagnostics directory (scripts/diagnose_forest.py) holding summary.json
and episode-NN-actions.jsonl.gz. Its episodes are replayed in their original order in a
single environment: PyBoy keeps some state across save loads, so replaying an episode
alone can diverge (an encounter thousands of actions in), while replaying the directory
in order reproduces every episode. Each step is checked against the trace (position,
battle, party, reward); a mismatch aborts that directory.

Per episode it writes ``features.npy`` (float16, T x 277: the frozen LSTM parent's CNN +
recent-action features of the observation before each action), ``labels.npy`` and
``mask.npy`` (lstm_memory.LABELS, read from RAM for that same observation), ``battle.npy``,
``map.npy`` and ``peek.npy``, plus ``meta.json``.

``--peek-rate`` inserts menu checks: at that chance per overworld action, the current
state is copied into a second emulator, which opens the start menu and looks at the bag
or the party (lstm_memory.menu_peek). Those actions are spliced into the sequence and
flagged in ``peek.npy``; the main replay never pauses or reloads, so it still matches its
trace. The bag shows ball and antidote counts and the party screen HP, which ordinary
play rarely shows; looking changes none of them, so later labels stay true.
"""

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from failure_restart import capture, restore
from lstm_memory import (
    BUTTONS,
    NAMES,
    PEEK_TARGETS,
    WILD,
    can_peek,
    menu_peek,
    policy_features,
    read_labels,
)
from memory_addresses import BATTLE_TYPE_ADDRESS
from red_gym_env_v2_nuzlocke import RedGymEnv
from sb3_contrib import RecurrentPPO

PORTABLE = {"_last_lstm_states": None, "clip_range": 0.2, "lr_schedule": None}


def replay_directory(task):
    directory, output, rom, state, feature_model, batch = task[:6]
    peek_rate, seed = (task[6], task[7]) if len(task) > 6 else (0.0, 0)
    torch.set_num_threads(1)
    rng = np.random.default_rng([seed, *directory.name.encode()])
    policy = RecurrentPPO.load(feature_model, device="cpu", custom_objects=PORTABLE).policy
    summary = json.loads((directory / "summary.json").read_text())
    config = dict(
        summary["config"],
        session_path=str(output / "_env" / directory.name),
        gb_path=str(rom),
        init_state=str(state),
        save_video=False,
        save_final_state=False,
    )
    env = RedGymEnv(config)
    peek_env = RedGymEnv(dict(config, session_path=config["session_path"] + "-peek"))
    peek_env.reset(seed=0)
    written = []
    try:
        for index, record in enumerate(summary["records"]):
            trace = directory / f"episode-{index:02d}-actions.jsonl.gz"
            if not trace.exists():
                raise ValueError(f"{trace} missing: episodes must be replayed in order")
            with gzip.open(trace, "rt") as f:
                rows = [json.loads(line) for line in f]
            obs, _ = env.reset(seed=record["evaluation_seed"])
            observations, labels, masks, battles, maps, peeks = [], [], [], [], [], []
            features, last_wild, info, peek_counts = [], env.step_count, None, {}
            for row in rows:
                label, mask = read_labels(env, last_wild)
                observations.append(obs)
                labels.append(label)
                masks.append(mask)
                battles.append(env.read_m(BATTLE_TYPE_ADDRESS))
                maps.append(env.get_game_coords()[2])
                peeks.append(False)
                if peek_rate and can_peek(env) and rng.random() < peek_rate:
                    kind = str(rng.choice(sorted(PEEK_TARGETS)))
                    restore(peek_env, capture(env, obs))
                    steps = menu_peek(peek_env, kind, last_wild)
                    key = kind if steps else f"{kind}_discarded"
                    peek_counts[key] = peek_counts.get(key, 0) + 1
                    for peek_obs, label, mask in steps or ():
                        observations.append(peek_obs)
                        labels.append(label)
                        masks.append(mask)
                        battles.append(0)
                        maps.append(maps[-1])
                        peeks.append(True)
                obs, reward, terminated, truncated, info = env.step(BUTTONS.index(row["action"]))
                if env.read_m(BATTLE_TYPE_ADDRESS) == WILD:
                    last_wild = env.step_count
                # Trace rows describe the state after their action.
                party = [asdict(m) for m in env.read_party_mons()]
                after = [list(env.get_game_coords()), env.read_m(BATTLE_TYPE_ADDRESS), party]
                got = json.loads(json.dumps(after))
                expected = [row["position"], row["battle"], row["party"]]
                if got != expected or abs(reward - row["reward"]) > 1e-9:
                    raise ValueError(f"{directory} episode {index} diverged at {row['step']}")
                if len(observations) >= batch:
                    features.append(policy_features(policy, observations))
                    observations = []
                if terminated or truncated:
                    break
            if observations:
                features.append(policy_features(policy, observations))
            played = len(labels) - sum(peeks)
            if played != len(rows) or info["end_reason"] != record["info"]["end_reason"]:
                raise ValueError(f"{directory} episode {index} ended differently")
            target = output / f"{directory.parent.parent.name}__{directory.name}__{index:02d}"
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "features.npy", np.concatenate(features).astype(np.float16))
            np.save(target / "labels.npy", np.stack(labels))
            np.save(target / "mask.npy", np.stack(masks).astype(np.uint8))
            np.save(target / "battle.npy", np.asarray(battles, dtype=np.uint8))
            np.save(target / "map.npy", np.asarray(maps, dtype=np.uint8))
            np.save(target / "peek.npy", np.asarray(peeks, dtype=bool))
            meta = {
                "source": str(directory),
                "episode": index,
                "evaluation_seed": record["evaluation_seed"],
                "end_reason": record["info"]["end_reason"],
                "violation_reasons": record["info"].get("violation_reasons", []),
                "steps": len(rows),
                "peek_actions": int(sum(peeks)),
                "peeks": peek_counts,
                "labels": list(NAMES),
            }
            (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            written.append(target.name)
    finally:
        env.close()
        peek_env.close()
    return str(directory), written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("diagnostics", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--feature-model", type=Path, required=True, help="LSTM parent final.zip")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--peek-rate", type=float, default=0.0, help="Menu checks per overworld action")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            d,
            args.output,
            args.rom.resolve(),
            args.state.resolve(),
            args.feature_model,
            args.batch,
            args.peek_rate,
            args.seed,
        )
        for d in args.diagnostics
    ]
    failed = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(replay_directory, t): t[0] for t in tasks}
        for future, directory in futures.items():
            try:
                source, written = future.result()
                print(json.dumps({"source": source, "episodes": len(written)}), flush=True)
            except ValueError as error:
                failed[str(directory)] = str(error)
                print(json.dumps({"source": str(directory), "error": str(error)}), flush=True)
    (args.output / "build.json").write_text(
        json.dumps({"sources": [str(d) for d in args.diagnostics], "failed": failed}, indent=2)
    )
    if failed:
        raise SystemExit(f"{len(failed)} directories failed; see build.json")


if __name__ == "__main__":
    main()
