"""Natural failure-state diagnostics with full recorded visual-history LSTM warmup.

This suite is diagnostic, not the training reset distribution. It never edits
RAM or supplies RAM to the actor. Scores come from independent game observations.
"""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from diagnose_forest import BUTTONS, ForestObserver, write_json
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from context_aux import context_target
from experiment import ALGORITHMS
from failure_restart import capture, restore
from poison_curriculum import CureSignals, antidote_count
from red_gym_env_v2_nuzlocke import RedGymEnv


def relocated(config, assets, session):
    return dict(
        config,
        gb_path=str(assets / Path(config["gb_path"]).name),
        init_state=str(assets / Path(config["init_state"]).name),
        session_path=str(session),
        save_video=False,
    )


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.diagnostic / "summary.json").read_text())
    config = relocated(summary["config"], args.assets_root, args.output / "emulator")
    env = RedGymEnv(config)
    counts, cases, matched = Counter(), [], 0
    try:
        for episode, record in enumerate(summary["records"]):
            obs, _ = env.reset(seed=record["evaluation_seed"])
            history = {key: [] for key in obs}
            captured = set()
            with gzip.open(args.diagnostic / f"episode-{episode:02d}-actions.jsonl.gz", "rt") as f:
                for line in f:
                    row = json.loads(line)
                    for key, values in history.items():
                        values.append(obs[key].copy())
                    obs, reward, term, trunc, _ = env.step(BUTTONS.index(row["action"]))
                    assert abs(reward - row["reward"]) < 1e-8, (episode, row["step"], "reward")
                    assert list(env.get_game_coords()) == row["position"]
                    assert env.read_m(0xD057) == row["battle"]
                    assert (
                        json.loads(json.dumps([asdict(m) for m in env.read_party_mons()]))
                        == row["party"]
                    )
                    assert antidote_count(env.pyboy.memory) == row["antidotes"]
                    matched += 1
                    target = context_target(env)
                    kind = (
                        "trainer"
                        if target[0] == 2 and row["position"][2] == 51
                        else "wild"
                        if target[0] == 1
                        else "poison"
                        if target[0] == 0 and target[1] and target[3]
                        else None
                    )
                    if (
                        not term
                        and not trunc
                        and kind
                        and kind not in captured
                        and counts[kind] < 2
                        and config["max_steps"] - env.step_count >= 2048
                    ):
                        name = f"{kind}-episode{episode}-step{env.step_count}"
                        folder = args.output / name
                        folder.mkdir()
                        blob = capture(env, obs)
                        (folder / "state.pkl.gz").write_bytes(blob)
                        np.savez_compressed(
                            folder / "history.npz",
                            **{key: np.stack(value) for key, value in history.items()},
                        )
                        cases.append(
                            {
                                "id": name,
                                "kind": kind,
                                "episode": episode,
                                "seed": record["evaluation_seed"],
                                "step": env.step_count,
                                "history_actions": len(history["screens"]),
                                "snapshot_sha256": hashlib.sha256(blob).hexdigest(),
                                "history_sha256": hashlib.sha256(
                                    (folder / "history.npz").read_bytes()
                                ).hexdigest(),
                                "initial_context": target.tolist(),
                            }
                        )
                        counts[kind] += 1
                        captured.add(kind)
                        print(json.dumps(cases[-1]), flush=True)
                    if sum(counts.values()) == 6:
                        break
                    if term or trunc:
                        break
            if sum(counts.values()) == 6:
                break
        if dict(counts) != {"wild": 2, "trainer": 2, "poison": 2}:
            raise ValueError(f"Incomplete natural diagnostic bank: {counts}")
        write_json(
            args.output / "suite.json",
            {
                "config": config,
                "source": str(args.diagnostic),
                "cases": cases,
                "matched_reconstruction_actions": matched,
                "contract": "First two eligible distinct-episode states of each kind in the selected parent diagnostic. At least 2048 original actions remain. Full within-episode pixel/action history saved BEFORE the captured observation. Development diagnostic selection, not untouched test data; never used as training resets or action targets.",
            },
        )
    finally:
        env.close()


class CureObserver(ForestObserver):
    def begin(self, stream):
        super().begin(stream)
        self.signals = CureSignals(
            self.read_party_mons(),
            antidote_count(self.pyboy.memory),
            self.read_m(0xD057),
            allowed_battles=(0, 1, 2),
        )

    def _observe_rules(self):
        super()._observe_rules()
        if self.recording:
            self.signals.observe(
                self.read_party_mons(),
                antidote_count(self.pyboy.memory),
                self.read_m(0xD057),
                self.tracker.dead,
            )


def warm_history(model, path, chunk=64):
    """Replay observations through memory only; never execute or imitate old actions."""
    if not hasattr(model.policy, "lstm_actor"):
        return None
    shape = model.policy.lstm_hidden_state_shape
    state = (torch.zeros(shape, device=model.device), torch.zeros(shape, device=model.device))
    with np.load(path) as history, torch.no_grad():
        n = len(history["screens"])
        arrays = {key: history[key] for key in history.files}
        for start in range(0, n, chunk):
            obs = {key: array[start : start + chunk] for key, array in arrays.items()}
            tensors, _ = model.policy.obs_to_tensor(obs)
            episode_start = torch.zeros(len(obs["screens"]), device=model.device)
            if start == 0:
                episode_start[0] = 1
            _, state = model.policy.get_distribution(tensors, state, episode_start)
    return tuple(s.cpu().numpy() for s in state)


class PredictionMetrics:
    def __init__(self):
        self.battle = np.zeros((3, 3), dtype=int)
        self.poison = np.zeros((2, 2), dtype=int)
        self.item = np.zeros((2, 2), dtype=int)
        self.n, self.hp_error = 0, 0.0

    def add(self, logits, target):
        logits = logits.reshape(-1, 6)[0].detach().cpu()
        if target[0] >= 0:
            self.battle[int(target[0]), int(logits[:3].argmax())] += 1
        self.poison[int(target[1]), int(logits[3] >= 0)] += 1
        self.item[int(target[3]), int(logits[5] >= 0)] += 1
        self.hp_error += abs(float(logits[4].sigmoid()) - float(target[2]))
        self.n += 1

    def result(self):
        return {
            "actions": self.n,
            "battle_confusion": self.battle.tolist(),
            "poison_confusion": self.poison.tolist(),
            "antidote_confusion": self.item.tolist(),
            "hp_mae": self.hp_error / self.n if self.n else None,
            "confusion_convention": "rows=true label, columns=predicted label; correlated action samples, not independent trials",
        }


def evaluate(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    suite = json.loads((args.suite / "suite.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    checkpoint = args.run / "final.zip"
    model = ALGORITHMS[manifest["algorithm"]].load(checkpoint, device=args.device)
    rows = []
    for case in suite["cases"]:
        folder = args.suite / case["id"]
        blob = (folder / "state.pkl.gz").read_bytes()
        assert hashlib.sha256(blob).hexdigest() == case["snapshot_sha256"]
        assert (
            hashlib.sha256((folder / "history.npz").read_bytes()).hexdigest()
            == case["history_sha256"]
        )
        history_state = warm_history(model, folder / "history.npz")
        for rep in range(args.repeats):
            seed = 6200 + 10 * suite["cases"].index(case) + rep
            config = relocated(
                suite["config"], args.assets_root, args.output / "emulator" / f"{case['id']}-{rep}"
            )
            env = CureObserver(config)
            try:
                env.reset(seed=case["seed"])
                obs = restore(env, blob)
                initial_flags = env.read_m(0xD7F3) & 28
                state = (
                    tuple(s.copy() for s in history_state) if history_state is not None else None
                )
                set_random_seed(seed)
                predictions = PredictionMetrics()
                success = False
                with gzip.open(args.output / f"{case['id']}-{rep}-events.jsonl.gz", "wt") as stream:
                    env.begin(stream)
                    for step in range(args.actions):
                        target = context_target(env)
                        action, state = model.predict(
                            obs, state=state, episode_start=np.array([False]), deterministic=False
                        )
                        if hasattr(model.policy, "last_context_logits"):
                            predictions.add(model.policy.last_context_logits, target)
                        obs, _reward, term, trunc, info = env.step(int(action))
                        if case["kind"] == "trainer":
                            success = bool((env.read_m(0xD7F3) & 28) & ~initial_flags)
                        elif case["kind"] == "poison":
                            success = bool(env.signals.events)
                        else:
                            success = env.hook_counts.get("run_succeeded_battle1", 0) > 0
                        if success or term or trunc:
                            break
                row = {
                    "case": case["id"],
                    "kind": case["kind"],
                    "rep": rep,
                    "seed": seed,
                    "success": success,
                    "actions": step + 1,
                    "end_reason": info["end_reason"],
                    "final": env.snapshot(),
                    "run_hooks": dict(env.hook_counts),
                    "faints": env.faints,
                    "cures": env.signals.events,
                    "prediction": predictions.result(),
                }
                rows.append(row)
                write_json(
                    args.output / "summary.json",
                    {
                        "run": str(args.run),
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "suite": str(args.suite),
                        "rows": rows,
                        "complete": False,
                    },
                )
                print(
                    json.dumps(
                        {k: row[k] for k in ("case", "rep", "success", "actions", "run_hooks")}
                    ),
                    flush=True,
                )
            finally:
                env.close()
    report = json.loads((args.output / "summary.json").read_text())
    report["complete"] = True
    report["contract"] = (
        "Identical naturally reached snapshots, original remaining horizon/rewards/rules, stochastic matched seeds, full recorded history for recurrent warmup. At most 2048 fresh actions. Trainer success needs a new forest trainer event flag; poison success needs a frame-matched antidote cure; wild success needs the game's successful RUN hook. Small development diagnostic, not full-route performance."
    )
    write_json(args.output / "summary.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("diagnostic", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("run", type=Path)
    e.add_argument("--suite", type=Path, required=True)
    e.add_argument("--repeats", type=int, default=3)
    e.add_argument("--actions", type=int, default=2048)
    e.add_argument("--device", default="cuda")
    for parser in (b, e):
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--assets-root", type=Path, default=ROOT / "project_work")
    args = p.parse_args()
    (build if args.command == "build" else evaluate)(args)


if __name__ == "__main__":
    main()
