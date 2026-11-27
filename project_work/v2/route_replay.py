"""Keep route workers separate from workers replaying failed route states.

Only original-start workers publish snapshots. Readers use an atomic bank index
and immutable blobs, so retiring an index entry never races a pending restore.
RAM battle/map context controls training-state sampling, never actor inputs.
"""

import hashlib
import json
from collections import Counter, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from failure_restart import capture, restore


class RouteReplay(gym.Wrapper):
    def __init__(
        self,
        env,
        directory,
        sources=(),
        mode="transition",
        seed=0,
        stride=64,
        window=1024,
        min_lead=64,
        max_failures=8,
        anchors_per_context=2,
        reservoir_size=8,
    ):
        super().__init__(env)
        if mode not in ("tail", "transition"):
            raise ValueError("Unknown replay mode")
        if min(stride, window, min_lead, max_failures, anchors_per_context, reservoir_size) < 1:
            raise ValueError("Replay parameters must be positive")
        if min_lead > window or anchors_per_context > reservoir_size:
            raise ValueError("Inconsistent replay parameters")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "bank.json").exists():
            raise ValueError("Choose a fresh worker bank")
        self.sources = tuple(Path(p) for p in sources)
        self.role = "replay" if self.sources else "route"
        self.mode, self.rng = mode, np.random.default_rng(seed)
        self.stride, self.window, self.min_lead = stride, window, min_lead
        self.max_failures, self.anchors = max_failures, anchors_per_context
        self.reservoir_size = reservoir_size
        self.bank, self.tail = [], deque()
        self.transitions = {k: [] for k in (0, 1, 2)}
        self.transition_counts = Counter()
        self.action_counts, self.reset_counts = Counter(), Counter()
        self.episode_index, self.failures = -1, 0
        self.active_source = None
        self.fresh_actions, self.original_step = 0, 0

    def context(self):
        battle = self.env.read_m(0xD057)
        # get_game_coords uses the same coherent map tracker as navigation.
        return (battle, self.env.get_game_coords()[2])

    def _available(self):
        rows = []
        for directory in self.sources:
            index = directory / "bank.json"
            if index.exists():
                rows.extend(
                    dict(row, directory=str(directory)) for row in json.loads(index.read_text())
                )
        return rows

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_index += 1
        self.active_source = None
        rows = self._available() if self.sources else []
        if rows:
            # Equal probability per available context, then per anchor. The tail
            # comparator uses exactly the same stratification as transition replay.
            contexts = sorted({row["battle"] for row in rows})
            context = contexts[int(self.rng.integers(len(contexts)))]
            candidates = [row for row in rows if row["battle"] == context]
            row = candidates[int(self.rng.integers(len(candidates)))]
            blob = (Path(row["directory"]) / row["file"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != row["sha256"]:
                raise ValueError("Route replay snapshot hash mismatch")
            obs = restore(self.env, blob)
            self.active_source = row
        self.original_step = self.env.step_count
        self.fresh_actions = 0
        self.env.episode_return = 0.0
        self.env.path_reward_total = 0.0
        self.tail.clear()
        self.transitions = {k: [] for k in (0, 1, 2)}
        self.transition_counts.clear()
        self.previous_context = self.context()
        self.reset_counts["replay" if self.active_source else "original"] += 1
        return obs, dict(info, **self._info())

    def _info(self):
        return {
            "worker_role": self.role,
            "training_reset": "failure" if self.active_source else "original",
            "restart_source": self.active_source,
            "restart_original_step": self.original_step,
            "fresh_actions": self.fresh_actions,
            "failure_bank_size": len(self.bank),
        }

    def replay_stats(self):
        return {
            "role": self.role,
            "actions": dict(self.action_counts),
            "resets": dict(self.reset_counts),
            "failures_published": self.failures,
            "active_bank_entries": len(self.bank),
        }

    def _record(self, obs, context):
        battle, map_id = context
        if battle not in self.transitions:
            return
        row = {"original_step": self.env.step_count, "battle": battle, "map": map_id}
        if self.mode == "tail":
            if self.fresh_actions == 1 or self.fresh_actions % self.stride == 0:
                self.tail.append((row, capture(self.env, obs)))
            while (
                self.tail and self.env.step_count - self.tail[0][0]["original_step"] > self.window
            ):
                self.tail.popleft()
        elif self.fresh_actions == 1 or context != self.previous_context:
            # Uniform bounded reservoir within each context, not just the last
            # transitions (which can miss the first healthy battle entry).
            self.transition_counts[battle] += 1
            entries = self.transitions[battle]
            index = int(self.rng.integers(self.transition_counts[battle]))
            if len(entries) < self.reservoir_size:
                entries.append((row, capture(self.env, obs)))
            elif index < self.reservoir_size:
                entries[index] = (row, capture(self.env, obs))

    def _publish(self, info):
        candidates = (
            list(self.tail)
            if self.mode == "tail"
            else [pair for rows in self.transitions.values() for pair in rows]
        )
        candidates = [
            pair
            for pair in candidates
            if self.env.step_count - pair[0]["original_step"] >= self.min_lead
            and (
                self.mode != "tail" or self.env.step_count - pair[0]["original_step"] <= self.window
            )
        ]
        if not candidates:
            return
        self.failures += 1
        for context in (0, 1, 2):
            eligible = [pair for pair in candidates if pair[0]["battle"] == context]
            chosen = self.rng.choice(len(eligible), min(self.anchors, len(eligible)), replace=False)
            for i in chosen:
                row, blob = eligible[int(i)]
                name = f"failure-{self.failures:05d}-step-{row['original_step']:05d}.pkl.gz"
                (self.directory / name).write_bytes(blob)
                self.bank.append(
                    dict(
                        row,
                        file=name,
                        sha256=hashlib.sha256(blob).hexdigest(),
                        failure=self.failures,
                        episode_index=self.episode_index,
                        lead_actions=self.env.step_count - row["original_step"],
                        end_reason=info["end_reason"],
                        source_reset="original",
                        mode=self.mode,
                    )
                )
        self.bank = [row for row in self.bank if row["failure"] > self.failures - self.max_failures]
        temporary = self.directory / "bank.tmp"
        temporary.write_text(json.dumps(self.bank, indent=2) + "\n")
        temporary.replace(self.directory / "bank.json")
        # Keep retired blobs for audit and readers that already loaded the old
        # index. This campaign is finite; only active sampling indices are bounded.

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.fresh_actions += 1
        self.action_counts["replay" if self.active_source else "original"] += 1
        if self.role == "route":
            context = self.context()
            if terminated or truncated:
                if not info["is_success"]:
                    self._publish(info)
            else:
                self._record(obs, context)
            self.previous_context = context
        return obs, reward, terminated, truncated, dict(info, **self._info())
