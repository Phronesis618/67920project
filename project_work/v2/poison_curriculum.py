"""Screen-only antidote curriculum and a capacity-matched resolution comparison.

RAM is used for scoring and diagnostics, never supplied as additional actor input.
The ordinary route reward remains available via task='route'.
"""

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from red_gym_env_v2_nuzlocke import RedGymEnv
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MatchedNatureCNN(BaseFeaturesExtractor):
    """NatureCNN parameter names/shapes; pool native feature maps to the 72x80 size."""

    def __init__(self, space, features_dim=256):
        super().__init__(space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 6)),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64 * 5 * 6, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class MatchedCombinedExtractor(BaseFeaturesExtractor):
    """Compatible with the existing screen-only CombinedExtractor's state_dict."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
        if set(observation_space.spaces) != {"screens", "recent_actions"}:
            raise ValueError("This experiment requires screens/actions only")
        extractors, size = {}, 0
        for key, space in observation_space.spaces.items():
            if is_image_space(space):
                extractors[key] = MatchedNatureCNN(space)
                size += 256
            else:
                extractors[key] = nn.Flatten()
                size += get_flattened_obs_dim(space)
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = size

    def forward(self, observations):
        return torch.cat(
            [module(observations[key]) for key, module in self.extractors.items()], dim=1
        )


def antidote_count(memory):
    count = memory[0xD31D]
    if count > 20:
        raise ValueError("Invalid bag length")
    return sum(memory[0xD31F + 2 * i] for i in range(count) if memory[0xD31E + 2 * i] == 0x0B)


class CureSignals:
    """Match living poison clears to antidote consumption within two emulator frames.

    No HP increase is allowed (a Center heal is not an antidote cure). Dead identities
    are excluded even if a later game animation restores HP. A small frame window
    allows the status write and inventory write to straddle an emulator tick.
    """

    def __init__(self, party, antidotes, battle):
        self.previous = {m.identity: m for m in party}
        self.antidotes = antidotes
        self.battle = battle
        self.frame = 0
        self.spends = deque()
        self.clears = deque()
        self.events = []
        self.antidotes_spent = 0

    def observe(self, party, antidotes, battle, dead):
        self.frame += 1
        current = {m.identity: m for m in party}
        spent = max(0, self.antidotes - antidotes)
        self.antidotes_spent += spent
        if battle == self.battle == 0:
            self.spends.extend([self.frame] * spent)
            for key, mon in current.items():
                old = self.previous.get(key)
                if (
                    old is not None
                    and old.status & 8
                    and not mon.status & 8
                    and old.hp == mon.hp > 0
                    and key not in dead
                ):
                    self.clears.append((self.frame, key))
        else:
            self.spends.clear()
            self.clears.clear()
        while self.spends and self.frame - self.spends[0] > 2:
            self.spends.popleft()
        while self.clears and self.frame - self.clears[0][0] > 2:
            self.clears.popleft()
        while self.spends and self.clears:
            _, key = self.clears.popleft()
            self.spends.popleft()
            mon = current.get(key)
            if mon is not None and mon.hp > 0 and not mon.status & 8 and key not in dead:
                self.events.append({"frame": self.frame, "identity": key})
        self.previous, self.antidotes, self.battle = current, antidotes, battle


def load_suite(path, split):
    path = Path(path).resolve()
    suite = json.loads(path.read_text())
    selected = []
    for entry in suite["scenarios"]:
        if entry["split"] != split:
            continue
        state = path.parent / entry["state_file"]
        if hashlib.sha256(state.read_bytes()).hexdigest() != entry["state_sha256"]:
            raise ValueError(f"State hash mismatch: {entry['id']}")
        if entry.get("dead") or entry.get("ram_edited", True):
            raise ValueError("Curriculum requires natural, no-death fixtures")
        selected.append(dict(entry, state_path=str(state)))
    if not selected:
        raise ValueError(f"No scenarios in split {split}")
    return selected


class PoisonEnv(RedGymEnv):
    def __init__(self, config):
        config = dict(config)
        if config.get("observation_mode") != "pixels":
            raise ValueError("Poison experiment actor inputs must be screens/actions only")
        self.task = config.get("task", "cure")
        if self.task not in ("cure", "route"):
            raise ValueError("Unknown task")
        self.scenarios = (
            load_suite(config["scenario_suite"], config.get("split", "train"))
            if self.task == "cure"
            else []
        )
        self.scenario_rng = None
        self.signals = None
        self.task_return = 0.0
        self.scenario = None
        super().__init__(config)

    def reset(self, seed=None, options=None):
        self.signals = None
        options = options or {}
        if self.scenarios:
            if seed is not None or self.scenario_rng is None:
                self.scenario_rng = np.random.default_rng(seed)
            index = options.get("scenario_index")
            if index is None:
                weights = np.array([s.get("weight", 1.0) for s in self.scenarios], dtype=float)
                index = int(
                    self.scenario_rng.choice(len(self.scenarios), p=weights / weights.sum())
                )
            self.scenario = self.scenarios[index]
            self.init_state = self.scenario["state_path"]
            self.initial_used_regions = tuple(self.scenario["initial_used_regions"])
        obs, info = super().reset(seed=seed, options=options)
        party = self.read_party_mons()
        count = antidote_count(self.pyboy.memory)
        self.targets = {m.identity for m in party if m.hp > 0 and m.status & 8}
        if self.scenario:
            if (
                bool(self.targets) != self.scenario["poisoned"]
                or count != self.scenario["antidotes"]
            ):
                raise ValueError("Loaded scenario does not match its poison/inventory contract")
            if self.read_m(0xD057) != 0 or (self.targets and count < 1):
                raise ValueError("Cure fixtures must start outside battle with an antidote")
        self.signals = CureSignals(party, count, self.read_m(0xD057))
        self.initial_antidotes = count
        self.task_return = 0.0
        return obs, dict(info, scenario=self.scenario["id"] if self.scenario else "route")

    def _observe_rules(self):
        super()._observe_rules()
        if self.signals is not None:
            self.signals.observe(
                self.read_party_mons(),
                antidote_count(self.pyboy.memory),
                self.read_m(0xD057),
                self.tracker.dead,
            )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        party = self.read_party_mons()
        info.update(
            scenario=self.scenario["id"] if self.scenario else "route",
            antidotes_remaining=antidote_count(self.pyboy.memory),
            antidotes_spent=self.signals.antidotes_spent,
            overworld_antidote_cures=len(self.signals.events),
            poisoned_alive=sum(m.hp > 0 and bool(m.status & 8) for m in party),
        )
        if self.task == "cure":
            cured = {e["identity"] for e in self.signals.events}
            alive_unpoisoned = {m.identity for m in party if m.hp > 0 and not m.status & 8}
            success = (
                bool(self.targets)
                and self.targets <= cured & alive_unpoisoned
                and not self.tracker.terminated
                and not self.tracker.dead
            )
            failed = terminated and not success
            reward = -0.001 + float(success) - float(failed)
            self.task_return += reward
            if success:
                terminated, truncated = True, False
                self._episode_done = True
                self._close_video()
            info.update(
                is_success=success,
                end_reason="antidote_cure" if success else info["end_reason"],
                episode_return=self.task_return,
            )
        return obs, reward, terminated, truncated, info
