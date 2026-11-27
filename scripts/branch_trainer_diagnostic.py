"""Controlled button-only branch of one natural failed trainer encounter.

This is a diagnostic feasibility intervention, never policy training/evaluation.
No RAM, rewards, rules, party, opponent, inventory or remaining budget is edited.
"""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

from diagnose_forest import BUTTONS, ForestObserver
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from failure_restart import restore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.capture / "capture.json").read_text())
    blob = (args.capture / "trainer.pkl.gz").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == metadata["snapshot_sha256"]
    with gzip.open(
        Path(metadata["diagnostic"]) / f"episode-{metadata['episode']:02d}-actions.jsonl.gz", "rt"
    ) as f:
        trace = [json.loads(line) for line in f]
    results = []
    for mode in ["recorded_policy", "press_A"]:
        cfg = dict(metadata["config"], session_path=str(args.output / mode))
        env = ForestObserver(cfg)
        env.reset(seed=metadata["seed"])
        restore(env, blob)
        start_return = env.episode_return
        start_potential = env.previous_potential
        start_scores = dict(env.progress_reward)
        scores = dict(start_scores)
        reward_events = []
        battle_exit = None
        discounted = 0.0
        gamma = cfg["navigation_gamma"]
        with gzip.open(args.output / f"{mode}-events.jsonl.gz", "wt") as stream:
            env.begin(stream)
            try:
                for i in range(cfg["max_steps"] - env.step_count):
                    action = (
                        BUTTONS.index(trace[metadata["step"] + i]["action"])
                        if mode == "recorded_policy"
                        else 4
                    )
                    _, reward, term, trunc, info = env.step(action)
                    discounted += gamma**i * reward
                    if mode == "recorded_policy":
                        row = trace[metadata["step"] + i]
                        assert abs(reward - row["reward"]) < 1e-8, (i, reward, row["reward"])
                        assert list(env.get_game_coords()) == row["position"]
                    changed = {
                        k: env.progress_reward[k] - scores[k]
                        for k in scores
                        if env.progress_reward[k] != scores[k]
                    }
                    if changed:
                        reward_events.append(
                            {
                                "fresh_action": i + 1,
                                "step": env.step_count,
                                "score_delta": changed,
                                "reward": reward,
                                "party": env.snapshot()["party"],
                            }
                        )
                    scores = dict(env.progress_reward)
                    if env.read_m(0xD057) == 0 and battle_exit is None:
                        battle_exit = i + 1
                    # The first trainer's event flag is 0xD7F3 bit 2, verified in events.json.
                    beat_first = bool(env.read_m(0xD7F3) & 4)
                    if mode == "press_A" and (
                        beat_first or battle_exit and i + 1 >= battle_exit + 64
                    ):
                        break
                    if term or trunc:
                        break
                Image.fromarray(env.pyboy.screen.ndarray[:, :, :3]).save(
                    args.output / f"{mode}-final.png"
                )
                result = {
                    "mode": mode,
                    "fresh_actions": i + 1,
                    "battle_exit_after_actions": battle_exit,
                    "first_trainer_flag": beat_first,
                    "end_reason": info["end_reason"],
                    "final": env.snapshot(),
                    "run_hooks": dict(env.hook_counts),
                    "reward_score_changes": {k: scores[k] - start_scores[k] for k in scores},
                    "undiscounted_reward": env.episode_return - start_return,
                    "discounted_reward": discounted,
                    "start_potential": start_potential,
                    "end_potential": env.previous_potential,
                    "reward_events": reward_events,
                    "faints": env.faints,
                }
                results.append(result)
                print(
                    json.dumps(
                        {k: v for k, v in result.items() if k not in ["reward_events", "faints"]}
                    ),
                    flush=True,
                )
            finally:
                env.close()
    report = {
        "snapshot_sha256": metadata["snapshot_sha256"],
        "source_episode": metadata["seed"],
        "source_step": metadata["step"],
        "initial_party": metadata["party"],
        "branches": results,
        "contract": "Same naturally reached trainer-entry snapshot. First branch replays original policy buttons and validates rewards/positions. Second branch repeatedly presses A (ordinary 24-frame actions) to test whether selecting Fight/first move can progress. No policy weights or environment rewards/rules/state changed. One controlled state, not a success-rate estimate.",
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
