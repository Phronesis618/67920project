"""Nuzlocke adapter. partial/history differ only in supplied monitor history.

Neither mode is a full-state oracle. RAM, frame stacking and the optional exploration
map remain common inputs. See the root README for the limited rule contract.
"""

from pathlib import Path

import numpy as np
from gymnasium import Env, spaces
from memory_addresses import (
    BATTLE_TYPE_ADDRESS,
    BOX_COUNT_ADDRESS,
    MAP_N_ADDRESS,
    NUM_MAPS,
    PARTY_BASE,
    PARTY_SIZE_ADDRESS,
    PARTY_STRIDE,
)
from navigation import graph_sha256, navigation_state, shaping_reward
from nuzlocke_rules import NuzlockeTracker, PartyMon
from red_gym_env_v2 import RedGymEnv as BaseRedGymEnv


class RedGymEnv(BaseRedGymEnv):
    def __init__(self, config):
        config = dict(config)
        config["session_path"] = Path(config["session_path"])
        config["session_path"].mkdir(parents=True, exist_ok=True)
        self.observation_mode = config.get("observation_mode", "partial")
        if self.observation_mode not in ("partial", "history", "pixels"):
            raise ValueError("observation_mode must be partial, history or pixels")
        self.include_explore_map = config.get("include_explore_map", True)
        self.level_cap = config.get("level_cap", 14)
        self.initial_used_regions = tuple(config.get("initial_used_regions", ()))
        self.region_map = {int(k): int(v) for k, v in config.get("region_map", {}).items()}
        self.start_jitter_frames = config.get("start_jitter_frames", 0)
        self.death_penalty = float(config.get("death_penalty", 0))
        self.violation_penalty = float(config.get("violation_penalty", 0))
        self.navigation_reward = config.get("navigation_reward", False)
        self.navigation_gamma = float(config.get("navigation_gamma", 0.997))
        if self.navigation_reward:
            if not 0 < self.navigation_gamma <= 1:
                raise ValueError("navigation_gamma must be in (0, 1]")
            expected = config.get("navigation_graph_sha256")
            if expected is not None and expected != graph_sha256():
                raise ValueError("Navigation graph differs from the training manifest")
        if config["action_freq"] < 8 or config["max_steps"] < 1:
            raise ValueError("action_freq must be >= 8 and max_steps must be positive")
        if self.start_jitter_frames < 0:
            raise ValueError("start_jitter_frames must be nonnegative")
        super().__init__(config)
        self.metadata = {"render_modes": ["rgb_array"]}
        self.render_mode = "rgb_array"
        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces.update(
            {
                "party_hp": spaces.Box(0, 1, shape=(6,), dtype=np.float32),
                "party_level": spaces.Box(0, 1, shape=(6,), dtype=np.float32),
                "party_present": spaces.MultiBinary(6),
                "party_status": spaces.MultiBinary(6),
            }
        )
        if self.observation_mode == "history":
            obs_spaces["party_dead"] = spaces.MultiBinary(6)
            obs_spaces["used_regions"] = spaces.MultiBinary(NUM_MAPS)
        if not self.include_explore_map:
            del obs_spaces["map"]
        if self.observation_mode == "pixels":
            obs_spaces = {k: obs_spaces[k] for k in ("screens", "recent_actions")}
        self.observation_space = spaces.Dict(obs_spaces)
        self._episode_done = True

    def reset(self, seed=None, options=None):
        Env.reset(self, seed=seed)
        self._close_video()
        self.tracker = NuzlockeTracker(self.level_cap, self.initial_used_regions, self.region_map)
        self.episode_return = 0.0
        self._episode_done = False
        # Base reset calls our observation reader; tracker must already exist.
        super().reset(seed=seed, options=options or {})
        self._observe_rules()
        self.start_delay = int(self.np_random.integers(self.start_jitter_frames + 1))
        for _ in range(self.start_delay):
            self.pyboy.tick(1, True)
            self._observe_rules()
        if self.tracker.terminated:
            raise ValueError("Initial state/jitter already violates the configured rules")
        if self.read_m(0xD356) & 1:
            raise ValueError("Initial state already has the Boulder Badge")
        self.last_health = self.read_hp_fraction()
        self.party_size = self.read_m(PARTY_SIZE_ADDRESS)
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum(self.progress_reward.values())
        self.recent_screens.fill(0)
        self.milestones = self.read_milestones()
        self.path_reward_total = 0.0
        self.navigation_goal, self.navigation_distance, self.previous_potential = (
            self.read_navigation()
        )
        self.closest_brock_distance = (
            self.navigation_distance if self.navigation_goal == "brock" else None
        )
        return self._get_obs(), {"start_delay_frames": self.start_delay}

    def read_navigation(self):
        if not self.navigation_reward:
            return None, None, 0.0
        x, y, map_id = self.get_game_coords()
        return navigation_state(
            map_id,
            x,
            y,
            got_parcel=self.read_bit(0xD74E, 1),
            got_pokedex=self.read_bit(0xD74B, 5),
        )

    def read_milestones(self):
        # LSB-numbered flags, checked against pret/pokered constants/event_constants.asm.
        return {
            "got_parcel": self.read_bit(0xD74E, 1),
            "delivered_parcel": self.read_bit(0xD74E, 0),
            "got_pokedex": self.read_bit(0xD74B, 5),
            "beat_brock": self.read_bit(0xD755, 7),
            "reached_viridian_forest": self.read_m(MAP_N_ADDRESS) == 51,
            "reached_pewter": self.read_m(MAP_N_ADDRESS) == 2,
        }

    def read_party_mons(self):
        party = []
        for i in range(min(self.read_m(PARTY_SIZE_ADDRESS), 6)):
            b = PARTY_BASE + i * PARTY_STRIDE
            party.append(
                PartyMon(
                    identity=(self.read_hp(b + 0x0C), self.read_hp(b + 0x1B)),
                    species=self.read_m(b),
                    hp=self.read_hp(b + 1),
                    max_hp=self.read_hp(b + 0x22),
                    level=self.read_m(b + 0x21),
                    status=self.read_m(b + 4),
                )
            )
        return party

    def read_hp_fraction(self):
        mons = self.read_party_mons()
        return sum(m.hp for m in mons) / max(sum(m.max_hp for m in mons), 1)

    def _observe_rules(self):
        self.tracker.observe(
            self.read_party_mons(),
            map_id=self.read_m(MAP_N_ADDRESS),
            battle_type=self.read_m(BATTLE_TYPE_ADDRESS),
            owned_count=self.read_m(PARTY_SIZE_ADDRESS) + self.read_m(BOX_COUNT_ADDRESS),
        )
        self.died_count = len(self.tracker.dead)

    def _get_obs(self):
        obs = super()._get_obs()
        if self.observation_mode == "pixels":
            return {k: obs[k] for k in ("screens", "recent_actions")}
        obs["health"] = obs["health"].astype(np.float32)
        mons = self.read_party_mons()
        obs["level"] = self.fourier_encode(0.02 * sum(m.level for m in mons)).astype(np.float32)
        for key in ("party_hp", "party_level"):
            obs[key] = np.zeros(6, dtype=np.float32)
        for key in ("party_present", "party_status"):
            obs[key] = np.zeros(6, dtype=np.int8)
        for i, mon in enumerate(mons):
            obs["party_hp"][i] = mon.hp / max(mon.max_hp, 1)
            obs["party_level"][i] = mon.level / 100
            obs["party_present"][i] = 1
            obs["party_status"][i] = mon.status != 0
        if self.observation_mode == "history":
            obs["party_dead"] = np.zeros(6, dtype=np.int8)
            for i, mon in enumerate(mons):
                obs["party_dead"][i] = mon.identity in self.tracker.dead
            obs["used_regions"] = np.zeros(NUM_MAPS, dtype=np.int8)
            obs["used_regions"][list(self.tracker.used_regions)] = 1
        if not self.include_explore_map:
            obs.pop("map")
        return obs

    def get_explore_map(self):
        # Fixed-size, zero-padded crop at global-map edges.
        y, x = self.get_global_coords()
        p = self.coords_pad
        out = np.zeros((2 * p, 2 * p), dtype=np.uint8)
        h, w = self.explore_map.shape
        y0, y1 = max(0, y - p), min(h, y + p)
        x0, x1 = max(0, x - p), min(w, x + p)
        if y1 > y0 and x1 > x0:
            out[y0 - y + p : y1 - y + p, x0 - x + p : x1 - x + p] = self.explore_map[y0:y1, x0:x1]
        return out.repeat(2, axis=0).repeat(2, axis=1)

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        # Observe each frame so a 24-frame action cannot hide a transient faint
        # or battle transition. Measure this overhead before scaling workers.
        for frame in range(self.act_freq):
            if frame == 8:
                self.pyboy.send_input(self.release_actions[action])
            render = self.save_video or not self.headless or frame == self.act_freq - 1
            self.pyboy.tick(1, render)
            self._observe_rules()
        self.pyboy.send_input(self.release_actions[action])
        if self.save_video:
            self.add_video_frame()

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if (
            cur_health > self.last_health > 0
            and self.read_m(PARTY_SIZE_ADDRESS) == self.party_size
            and not self.tracker.terminated
        ):
            self.total_healing_rew += (cur_health - self.last_health) ** 2

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.save_video and self.step_count == 0:
            self.start_video()
        previous_deaths = len(self.tracker.dead)
        self.run_action_on_emulator(int(action))
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_heal_reward()
        self.party_size = self.read_m(PARTY_SIZE_ADDRESS)
        reward = self.update_reward()
        reward -= self.death_penalty * (len(self.tracker.dead) - previous_deaths)
        if self.tracker.violations:
            reward -= self.violation_penalty
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        for name, reached in self.read_milestones().items():
            self.milestones[name] = self.milestones[name] or reached
        self.step_count += 1
        success = bool(self.read_m(0xD356) & 1) and not self.tracker.terminated
        terminated = bool(success or self.tracker.terminated)
        truncated = self.step_count >= self.max_steps and not terminated
        self._episode_done = terminated or truncated
        self.navigation_goal, self.navigation_distance, potential = self.read_navigation()
        path_reward = 0.0
        if self.navigation_reward:
            path_reward = shaping_reward(
                self.previous_potential, potential, self.navigation_gamma, terminated
            )
            reward += path_reward
            self.path_reward_total += path_reward
            self.previous_potential = potential
            if self.navigation_goal == "brock" and self.navigation_distance is not None:
                self.closest_brock_distance = min(
                    self.navigation_distance,
                    self.closest_brock_distance
                    if self.closest_brock_distance is not None
                    else self.navigation_distance,
                )
        self.episode_return += float(reward)
        reason = (
            "wipe"
            if self.tracker.wiped
            else "rule_violation"
            if self.tracker.violations
            else "brock_defeated"
            if success
            else "time_limit"
            if truncated
            else "running"
        )
        info = {
            "is_success": success,
            "end_reason": reason,
            "violation_reasons": sorted(self.tracker.violations),
            "deaths": len(self.tracker.dead),
            "regions_used": len(self.tracker.used_regions),
            "badges": self.get_badges(),
            "map_coverage": len(self.seen_coords),
            "max_map_progress": self.max_map_progress,
            "episode_steps": self.step_count,
            "episode_return": self.episode_return,
            "milestones": dict(self.milestones),
            "navigation_goal": self.navigation_goal,
            "navigation_distance": self.navigation_distance,
            "closest_brock_distance": self.closest_brock_distance,
            "path_reward": path_reward,
            "path_reward_total": self.path_reward_total,
        }
        obs = self._get_obs()
        if self.print_rewards:
            print(f"step={self.step_count} return={self.episode_return:.3f} {reason}")
        if self._episode_done:
            self._close_video()
        return obs, float(reward), terminated, truncated, info

    def _close_video(self):
        for name in ("full_frame_writer", "model_frame_writer", "map_frame_writer"):
            writer = getattr(self, name, None)
            if writer is not None:
                writer.close()
                setattr(self, name, None)

    def close(self):
        self._close_video()
        self.pyboy.stop(save=False)
