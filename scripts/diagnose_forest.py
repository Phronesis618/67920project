"""Replay frozen checkpoints with per-action traces and frame-level battle/faint events.

This observer does not change actions, observations, rewards, or the episode limit.
RUN hooks are pinned to the checked Red ROM and executable bytes, not menu guesses.
See pret/pokered ba9c0568 engine/battle/core.asm: TryRunningFromBattle.
"""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from experiment import ALGORITHMS
from poison_curriculum import antidote_count
from red_gym_env_v2_nuzlocke import RedGymEnv

ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
# Bank 0x0f; verify instruction bytes before installing PyBoy execution hooks.
HOOKS = {
    "run_attempt": (0x4ACF, "fa57d03d"),
    "wild_run_failed": (0x4B47, "21974b"),
    "trainer_run_rejected": (0x4B4C, "219c4b"),
    "run_succeeded": (0x4B89, "21a14b"),
}
BUTTONS = ("down", "left", "right", "up", "A", "B", "start")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ForestObserver(RedGymEnv):
    def __init__(self, config):
        self.recording = False
        super().__init__(config)
        rom = Path(config["gb_path"]).read_bytes()
        if hashlib.sha1(rom).hexdigest() != ROM_SHA1:
            raise ValueError("Battle hooks only support the verified Red ROM")
        for kind, (address, expected) in HOOKS.items():
            offset = 0xF * 0x4000 + address - 0x4000
            if rom[offset : offset + len(bytes.fromhex(expected))] != bytes.fromhex(expected):
                raise ValueError(f"Unexpected opcode at {kind}")
            self.pyboy.hook_register(0xF, address, self._hook, kind)

    def snapshot(self):
        return {
            "position": list(self.get_game_coords()),
            "battle": self.read_m(0xD057),
            "party": [asdict(mon) for mon in self.read_party_mons()],
            "antidotes": antidote_count(self.pyboy.memory),
        }

    def begin(self, event_stream):
        self.events = event_stream
        self.frame = 0
        self.previous = self.snapshot()
        self.frame_counts = Counter()
        self.hook_counts = Counter()
        self.battles = []
        self.current_battle = None
        self.faints = []
        self.first_forest = None
        self.recording = True

    def emit(self, kind, **details):
        event = {
            "kind": kind,
            "frame": self.frame,
            "step": self.step_count + 1,
            "position": list(self.get_game_coords()),
            **details,
        }
        self.events.write(json.dumps(event, separators=(",", ":")) + "\n")
        return event

    def _hook(self, kind):
        if not self.recording:
            return
        battle_type = self.read_m(0xD057)
        self.hook_counts[f"{kind}_battle{battle_type}"] += 1
        if self.current_battle is not None:
            counts = self.current_battle["run_hooks"]
            counts[kind] = counts.get(kind, 0) + 1
        # Hook executes during the next tick; the observer runs after that tick.
        self.emit(kind, battle=battle_type, frame=self.frame + 1)

    def _observe_rules(self):
        super()._observe_rules()
        if not self.recording:
            return
        self.frame += 1
        current = self.snapshot()
        battle = current["battle"]
        in_forest = current["position"][2] == 51
        self.frame_counts[f"battle{battle}"] += 1
        if in_forest:
            self.frame_counts[f"forest_battle{battle}"] += 1
            if self.first_forest is None:
                self.first_forest = self.emit("forest_arrival", party=current["party"])
        if battle != self.previous["battle"]:
            self.emit("battle_transition", previous=self.previous["battle"], battle=battle)
            if self.current_battle is not None:
                self.current_battle.update(end_frame=self.frame, end_step=self.step_count + 1)
                self.current_battle = None
            if battle in (1, 2):
                self.current_battle = {
                    "type": battle,
                    "start_frame": self.frame,
                    "start_step": self.step_count + 1,
                    "position": current["position"],
                    "party_at_entry": current["party"],
                    "run_hooks": {},
                }
                self.battles.append(self.current_battle)
        previous_party = {tuple(m["identity"]): m for m in self.previous["party"]}
        for mon in current["party"]:
            old = previous_party.get(tuple(mon["identity"]))
            if old is None:
                self.emit("party_added", mon=mon, battle=battle)
            elif mon["hp"] != old["hp"] or mon["status"] != old["status"]:
                self.emit("hp_or_status", before=old, after=mon, battle=battle)
                if old["hp"] > 0 and mon["hp"] == 0:
                    cause = {1: "wild_battle", 2: "trainer_battle"}.get(battle, "unknown")
                    if battle == 0 and old["status"] & 8:
                        cause = "overworld_poison"
                    self.faints.append(
                        self.emit("faint", cause=cause, battle=battle, before=old, after=mon)
                    )
        self.previous = current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=ROOT / "project_work")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument(
        "--video-episodes",
        type=int,
        nargs="*",
        help="Record selected zero-based episodes; preceding episodes still replay",
    )
    parser.add_argument("--video-scope", choices=("forest", "tail"), default="forest")
    args = parser.parse_args()
    if args.episodes < 1 or args.videos < 0:
        parser.error("episodes must be positive and videos nonnegative")
    if args.video_episodes is not None and any(
        i < 0 or i >= args.episodes for i in args.video_episodes
    ):
        parser.error("video episodes must be within the replayed episode range")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = dict(manifest["config"])
    config.setdefault("position_mode", "legacy-raw")
    config.update(session_path=str(args.output / "emulator"), save_video=False)
    for key, hash_key in (("gb_path", "rom_sha1"), ("init_state", "state_sha1")):
        config[key] = str(args.assets_root / Path(config[key]).name)
        if hashlib.sha1(Path(config[key]).read_bytes()).hexdigest() != manifest[hash_key]:
            raise ValueError(f"{key} does not match the training manifest")
    torch.set_num_threads(1)
    env = ForestObserver(config)
    original = args.run / "evaluation.json"
    original_records = json.loads(original.read_text())["records"] if original.exists() else []
    result = {
        "source_run": str(args.run),
        "training_commit": manifest["git_commit"],
        "model_sha256": hashlib.sha256((args.run / "final.zip").read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "device": args.device,
        "seed": args.seed,
        "video_scope": args.video_scope,
        "video_episodes": args.video_episodes,
        "config": config,
        "hooks": HOOKS,
        "records": [],
        "notes": "Stochastic policy; seed once before the episode loop, as in evaluate_main. "
        "Frame-level observer only. Forest videos start on arrival, one frame/action at 10fps "
        "(about 4x game speed). Battle rate denominator includes battle and menu actions. "
        "End-state match is checked against original evaluation; CPU/GPU replays can differ.",
    }
    try:
        model = ALGORITHMS[manifest["algorithm"]].load(
            args.run / "final.zip", env=env, device=args.device
        )
        set_random_seed(args.seed)
        for episode in range(args.episodes):
            env.recording = False
            obs, _ = env.reset(seed=args.seed + episode)
            state, episode_start = None, np.array([True])
            forest_actions = 0
            writer = None
            record_video = (
                episode in args.video_episodes
                if args.video_episodes is not None
                else episode < args.videos
            )
            tail_frames = deque(maxlen=600)
            prefix = args.output / f"episode-{episode:02d}"
            with (
                gzip.open(f"{prefix}-events.jsonl.gz", "wt") as events,
                gzip.open(f"{prefix}-actions.jsonl.gz", "wt") as actions,
            ):
                env.begin(events)
                try:
                    while True:
                        action, state = model.predict(
                            obs, state=state, episode_start=episode_start, deterministic=False
                        )
                        obs, reward, terminated, truncated, info = env.step(int(action))
                        episode_start[:] = False
                        snapshot = env.snapshot()
                        actions.write(
                            json.dumps(
                                {
                                    "step": info["episode_steps"],
                                    "action": BUTTONS[int(action)],
                                    "reward": reward,
                                    **snapshot,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        if snapshot["position"][2] == 51:
                            forest_actions += 1
                        if record_video and args.video_scope == "tail":
                            tail_frames.append(env.pyboy.screen.ndarray[:, :, :3].copy())
                        if (
                            env.first_forest is not None
                            and record_video
                            and args.video_scope == "forest"
                        ):
                            if writer is None:
                                writer = media.VideoWriter(
                                    str(prefix) + "-forest.mp4", shape=(144, 160), fps=10
                                )
                                writer.__enter__()
                            writer.add_image(env.pyboy.screen.ndarray[:, :, :3])
                        if terminated or truncated:
                            break
                    if record_video and args.video_scope == "tail" and tail_frames:
                        with media.VideoWriter(
                            str(prefix) + "-tail.mp4", shape=(144, 160), fps=10
                        ) as tail_writer:
                            for frame in tail_frames:
                                tail_writer.add_image(frame)
                finally:
                    if writer is not None:
                        writer.__exit__(None, None, None)
                    env.recording = False
            forest_battles = [b for b in env.battles if b["position"][2] == 51]
            wild_count = sum(b["type"] == 1 for b in forest_battles)
            record = {
                "evaluation_seed": args.seed + episode,
                "info": info,
                "forest_arrival": env.first_forest,
                "forest_actions": forest_actions,
                "forest_wild_battles": wild_count,
                "forest_trainers": sum(b["type"] == 2 for b in forest_battles),
                "wild_battles_per_1000_forest_actions": 1000 * wild_count / forest_actions
                if forest_actions
                else None,
                "frame_counts": dict(env.frame_counts),
                "run_hooks": dict(env.hook_counts),
                "faints": env.faints,
                "battles": env.battles,
                "final": env.snapshot(),
            }
            expected = next(
                (r for r in original_records if r["evaluation_seed"] == args.seed + episode), None
            )
            record["original_evaluation_match"] = expected is not None and all(
                json.dumps(info[k]) == json.dumps(expected[k])
                for k in ("episode_steps", "end_reason", "position", "deaths", "episode_return")
            )
            result["records"].append(record)
            write_json(args.output / "summary.json", result)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "evaluation_seed",
                            "forest_actions",
                            "forest_wild_battles",
                            "forest_trainers",
                            "run_hooks",
                            "original_evaluation_match",
                        )
                    }
                    | {
                        "end_reason": info["end_reason"],
                        "faint_causes": [e["cause"] for e in env.faints],
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    result["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
