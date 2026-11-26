"""Failure-driven start-state sampling; transitions and rewards stay on-policy.

Snapshots include emulator and Python-side rule/reward history. Pickle files are
private run artifacts produced by this module, never accepted from external input.
"""

import copy
import gzip
import hashlib
import io
import json
import pickle
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from pyboy.utils import WindowEvent

# Mutable episode state used by RedGymEnv's transition, observation and reward.
# Configuration, output handles, and lifetime logging counters are not restored.
EPISODE_FIELDS = (
    "tracker",
    "position",
    "_episode_done",
    "episode_return",
    "start_delay",
    "explore_map",
    "recent_screens",
    "recent_actions",
    "seen_coords",
    "levels_satisfied",
    "base_explore",
    "max_opponent_level",
    "max_event_rew",
    "max_level_rew",
    "last_health",
    "total_healing_rew",
    "died_count",
    "party_size",
    "step_count",
    "base_event_flags",
    "current_event_flags_set",
    "max_map_progress",
    "progress_reward",
    "total_reward",
    "milestones",
    "path_reward_total",
    "navigation_goal",
    "navigation_distance",
    "previous_potential",
    "closest_brock_distance",
    "_np_random",
)


def capture(env, observation):
    if env._episode_done or env.tracker.terminated:
        raise ValueError("Cannot restart from a terminal state")
    stream = io.BytesIO()
    env.pyboy.save_state(stream)
    payload = {
        "format": 1,
        "emulator": stream.getvalue(),
        # PyBoy 2.4 save_state omits API inputs waiting for the next tick.
        # This environment uses immediate inputs only, never delayed inputs.
        "pending_inputs": [int(e) for e in getattr(env.pyboy, "events", ())],
        "attributes": {name: getattr(env, name) for name in EPISODE_FIELDS},
        "observation": observation,
    }
    return gzip.compress(pickle.dumps(payload, protocol=5), compresslevel=1, mtime=0)


def restore(env, blob):
    payload = pickle.loads(gzip.decompress(blob))
    if payload["format"] != 1 or set(payload["attributes"]) != set(EPISODE_FIELDS):
        raise ValueError("Unsupported snapshot format")
    attrs = payload["attributes"]
    if attrs["_episode_done"] or attrs["tracker"].terminated:
        raise ValueError("Terminal snapshot rejected")
    env.pyboy.load_state(io.BytesIO(payload["emulator"]))
    if hasattr(env.pyboy, "events"):
        env.pyboy.events[:] = [WindowEvent(e) for e in payload["pending_inputs"]]
    for key, value in attrs.items():
        setattr(env, key, value)
    # Return the saved observation rather than advancing the frame stack a second time.
    return copy.deepcopy(payload["observation"])


class FailureRestart(gym.Wrapper):
    """Mix original starts with snapshots 64..1024 actions before a failure.

    Each worker keeps at most 8 failed episodes, 4 anchors each. Bank entries are
    selected without looking at poison, inventory, map, or failure subtype.
    The original episode's remaining action allowance is preserved. LSTM state
    resets as at any new SB3 episode; no old actions/advantages are replayed.
    """

    def __init__(
        self,
        env,
        directory,
        probability=0.5,
        seed=0,
        stride=64,
        window=1024,
        min_lead=64,
        max_failures=8,
        anchors=4,
    ):
        super().__init__(env)
        if not 0 <= probability <= 1 or min(stride, window, min_lead, max_failures, anchors) < 1:
            raise ValueError("Invalid restart parameters")
        if min_lead > window:
            raise ValueError("min_lead exceeds window")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "bank.json").exists():
            raise ValueError("Choose a new bank; worker restart requires explicit recovery")
        self.probability = probability
        self.rng = np.random.default_rng(seed)
        self.stride, self.window, self.min_lead = stride, window, min_lead
        self.max_failures, self.anchors = max_failures, anchors
        self.tail = deque()
        self.bank = []
        self.episode_index = -1
        self.failures = 0
        self.active_source = None
        self.original_step = 0
        self.fresh_actions = 0

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        force = options.pop("failure_restart", None)
        obs, info = self.env.reset(options=options, **kwargs)
        self.episode_index += 1
        use_bank = bool(self.bank) and (
            bool(force) if force is not None else self.rng.random() < self.probability
        )
        self.active_source = None
        if use_bank:
            row = self.bank[int(self.rng.integers(len(self.bank)))]
            blob = (self.directory / row["file"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != row["sha256"]:
                raise ValueError("Failure snapshot hash mismatch")
            obs = restore(self.env, blob)
            self.active_source = dict(row)
        self.original_step = self.env.step_count
        self.fresh_actions = 0
        # These two totals only report the new branch; reward accounting is retained.
        self.env.episode_return = 0.0
        self.env.path_reward_total = 0.0
        self.tail.clear()
        # Capture only after an ordinary action, not immediately after loading an
        # external save. PyBoy's initial renderer state can otherwise differ on
        # the first rendered action even when WRAM/rewards match exactly.
        return obs, dict(info, **self._info())

    def _capture(self, obs):
        self.tail.append((self.fresh_actions, self.env.step_count, capture(self.env, obs)))
        while self.tail and self.fresh_actions - self.tail[0][0] > self.window:
            self.tail.popleft()

    def _info(self):
        return {
            "training_reset": "failure" if self.active_source else "original",
            "restart_source": self.active_source,
            "restart_original_step": self.original_step,
            "fresh_actions": self.fresh_actions,
            "failure_bank_size": len(self.bank),
        }

    def _promote_failure(self, info):
        eligible = [
            s for s in self.tail if self.min_lead <= self.fresh_actions - s[0] <= self.window
        ]
        if not eligible:
            return
        chosen = self.rng.choice(len(eligible), min(self.anchors, len(eligible)), replace=False)
        self.failures += 1
        for index in sorted(chosen):
            fresh_step, original_step, blob = eligible[int(index)]
            name = f"failure-{self.failures:05d}-step-{original_step:05d}.pkl.gz"
            target = self.directory / name
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(blob)
            temporary.replace(target)
            self.bank.append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "failure": self.failures,
                    "episode_index": self.episode_index,
                    "original_step": original_step,
                    "lead_actions": self.fresh_actions - fresh_step,
                    "end_reason": info["end_reason"],
                    "source_reset": "failure" if self.active_source else "original",
                }
            )
        old = [r for r in self.bank if r["failure"] <= self.failures - self.max_failures]
        self.bank = [r for r in self.bank if r not in old]
        temporary = self.directory / "bank.tmp"
        temporary.write_text(json.dumps(self.bank, indent=2) + "\n")
        temporary.replace(self.directory / "bank.json")
        for row in old:
            (self.directory / row["file"]).unlink()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.fresh_actions += 1
        if terminated or truncated:
            if not info["is_success"]:
                self._promote_failure(info)
        elif self.fresh_actions == 1 or self.fresh_actions % self.stride == 0:
            self._capture(obs)
        return obs, reward, terminated, truncated, dict(info, **self._info())
