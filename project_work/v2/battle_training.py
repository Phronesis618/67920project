"""Battle-focused training: dense battle rewards and battle-only replay starts.

The base route reward gives almost nothing for fighting well: no level reward, a
squared heal term, about +2 for a trainer event flag and nothing for wild battles,
while losing a battle ends the episode. Stalling or RUN spam is therefore the safe
choice. ``BattleReward`` adds battle terms on top of the unchanged base reward;
``BattleReplay`` starts training episodes at battle snapshots taken from earlier
failed episodes and ends them when that battle ends.

All terms read RAM after each action. RAM is privileged reward/reset information
only; actor inputs are unchanged. Snapshot blobs are private run artifacts written
by ``failure_restart.capture``, never accepted from external input.
"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from failure_restart import restore
from memory_addresses import BATTLE_TYPE_ADDRESS, BOX_COUNT_ADDRESS, PARTY_SIZE_ADDRESS

# wEnemyMon fields (pret/pokered ram/wram.asm). Checked on restored campaign
# snapshots: species/HP/max HP/level match the visible wild and trainer battles.
ENEMY_SPECIES, ENEMY_HP, ENEMY_LEVEL, ENEMY_MAX_HP = 0xCFE5, 0xCFE6, 0xCFF3, 0xCFF4
WILD, TRAINER = 1, 2

ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
# Bank 0x0f; verify instruction bytes before installing PyBoy execution hooks.
# See pret/pokered ba9c0568 engine/battle/core.asm: TryRunningFromBattle.
RUN_HOOKS = {
    "run_attempt": (0x4ACF, "fa57d03d"),
    "wild_run_failed": (0x4B47, "21974b"),
    "trainer_run_rejected": (0x4B4C, "219c4b"),
    "run_succeeded": (0x4B89, "21a14b"),
}

# Config keys that do not change emulator dynamics or restored episode state.
# Snapshots stay valid when only these differ between producer and consumer.
_SNAPSHOT_NEUTRAL_KEYS = {
    "session_path",
    "gb_path",
    "init_state",
    "headless",
    "save_final_state",
    "print_rewards",
    "save_video",
    "fast_video",
    "death_penalty",
    "violation_penalty",
}


def snapshot_config_sha(config, rom_sha1, state_sha1):
    relevant = {k: v for k, v in config.items() if k not in _SNAPSHOT_NEUTRAL_KEYS}
    payload = json.dumps(
        {"config": relevant, "rom_sha1": rom_sha1, "state_sha1": state_sha1},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_rom(path):
    rom = Path(path).read_bytes()
    if hashlib.sha1(rom).hexdigest() != ROM_SHA1:
        raise ValueError("Battle RUN hooks only support the verified Red ROM")
    for kind, (address, expected) in RUN_HOOKS.items():
        offset = 0xF * 0x4000 + address - 0x4000
        if rom[offset : offset + len(bytes.fromhex(expected))] != bytes.fromhex(expected):
            raise ValueError(f"Unexpected opcode at {kind}")
    return rom


@dataclass(frozen=True)
class BattleRewardWeights:
    """Initial guesses, scaled against the route reward (event flag +2, wipe about -2..-4).

    Wild terms are small and switch off at the level cap so farming Route 1 cannot
    outweigh progress; trainer battles are the ones that block the route.
    """

    trainer_damage: float = 1.0  # per full enemy HP bar removed
    trainer_ko: float = 1.0
    trainer_win: float = 2.0
    wild_damage: float = 0.2
    wild_ko: float = 0.2
    wild_win: float = 0.2
    level: float = 1.0  # per party level gained, counted only up to the level cap
    hp_potential: float = 1.0  # potential-based: losing HP costs what healing returns
    trainer_run_rejected: float = 0.05  # per rejected RUN in a trainer battle
    overtime_after: int = 512  # actions within one battle before overtime applies
    overtime: float = 0.001  # per battle action beyond overtime_after
    win_hp_scaled: float = 0.0  # per win, times party HP fraction at battle exit
    loss: float = 0.0  # when the party wipes during a battle

    def __post_init__(self):
        if any(v < 0 for v in asdict(self).values()):
            raise ValueError("Battle reward weights must be nonnegative")


# "outcome" rewards only how a battle ends: a win is worth up to 10, scaled by the
# HP left; fleeing or catching is 0; a wipe in battle is -10. No shaping terms.
_OFF = dict.fromkeys(
    (
        "trainer_damage",
        "trainer_ko",
        "trainer_win",
        "wild_damage",
        "wild_ko",
        "wild_win",
        "level",
        "hp_potential",
        "trainer_run_rejected",
        "overtime",
    ),
    0.0,
)
PRESETS = {
    "dense": BattleRewardWeights(),
    "outcome": BattleRewardWeights(**_OFF, win_hp_scaled=10.0, loss=10.0),
}

# Dense keeps the earlier death/violation penalties. Outcome already charges a wipe in
# battle (so no death penalty), and treats both a rule violation (an illegal catch
# ends the episode inside the battle) and a replay battle stalled to the cap as a
# loss: trainer battles cannot be fled, so a zero-cost stall would beat a risky fight.
PRESET_PENALTIES = {
    "dense": {"battle_timeout_penalty": 1.0, "death_penalty": 2.0, "violation_penalty": 2.0},
    "outcome": {"battle_timeout_penalty": 10.0, "death_penalty": 0.0, "violation_penalty": 10.0},
}


class BattleReward(gym.Wrapper):
    """Add battle terms to the unchanged base reward.

    Damage counts only decreases while enemy species and max HP are unchanged, so a
    switch-in or the stale enemy slot at battle entry is never rewarded. A win is a
    battle exit whose last in-battle enemy HP was zero; running and catching are not
    wins. HP is potential shaping (gamma * phi' - phi, phi' = 0 on termination), so a
    damage/heal cycle cannot be farmed.
    """

    def __init__(self, env, weights=None, gamma=0.9995, rom=None):
        super().__init__(env)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.weights = weights or BattleRewardWeights()
        self.gamma = gamma
        self.totals = Counter()
        self.lifetime = Counter()
        self.pending_rejected = 0
        self.battle = 0
        if rom is not None:
            verify_rom(rom)
            for kind, (address, _) in RUN_HOOKS.items():
                self.env.unwrapped.pyboy.hook_register(0xF, address, self._hook, kind)

    def _hook(self, kind):
        self.lifetime[f"hook_{kind}"] += 1
        # The hook runs mid-action, so read the battle type now, not last step's.
        if kind == "trainer_run_rejected" and self._battle_type() == TRAINER:
            self.pending_rejected += 1

    def _read_enemy(self):
        base = self.env.unwrapped
        return (
            base.read_m(ENEMY_SPECIES),
            base.read_hp(ENEMY_HP),
            base.read_hp(ENEMY_MAX_HP),
        )

    def _battle_type(self):
        value = self.env.unwrapped.read_m(BATTLE_TYPE_ADDRESS)
        return value if value in (WILD, TRAINER) else 0

    def _start(self):
        base = self.env.unwrapped
        self.battle = self._battle_type()
        self.enemy = self._read_enemy() if self.battle else None
        self.battle_actions = 0
        self.levels = {m.identity: m.level for m in base.read_party_mons()}
        self.phi = self.weights.hp_potential * base.read_hp_fraction()
        self.pending_rejected = 0
        self.lost = False
        self.totals = Counter()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._start()
        return obs, self._info(info, {})

    def _info(self, info, terms):
        return dict(
            info,
            training_reset=info.get("training_reset", "original"),
            battle_reward=dict(terms),
            battle_reward_totals=dict(self.totals),
        )

    def _wild_open(self):
        cap = self.env.unwrapped.level_cap
        levels = [m.level for m in self.env.unwrapped.read_party_mons()]
        return cap is None or not levels or max(levels) < cap

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base, w = self.env.unwrapped, self.weights
        terms = Counter()
        previous, now = self.battle, self._battle_type()
        kind = "trainer" if TRAINER in (previous, now) else "wild"
        scale = 1.0 if kind == "trainer" or self._wild_open() else 0.0

        if now:
            if now != previous:
                self.battle_actions, self.enemy = 0, self._read_enemy()
            else:
                self.battle_actions += 1
                enemy = self._read_enemy()
                old_species, old_hp, old_max = self.enemy
                species, hp, max_hp = enemy
                if species == old_species and max_hp == old_max and max_hp > 0:
                    if hp < old_hp:
                        terms[f"{kind}_damage"] += (
                            scale * getattr(w, f"{kind}_damage") * (old_hp - hp) / max_hp
                        )
                    if old_hp > 0 and hp == 0:
                        terms[f"{kind}_ko"] += scale * getattr(w, f"{kind}_ko")
                        self.totals[f"{kind}_kos"] += 1
                self.enemy = enemy
            if self.battle_actions > w.overtime_after:
                terms["overtime"] -= w.overtime
        elif previous:
            won = self.enemy is not None and self.enemy[1] == 0 and not base.tracker.wiped
            self.totals[f"{kind}_battles"] += 1
            if won:
                terms[f"{kind}_win"] += scale * getattr(w, f"{kind}_win")
                terms["win_hp"] += scale * w.win_hp_scaled * base.read_hp_fraction()
                self.totals[f"{kind}_wins"] += 1
            self.enemy = None
        if (previous or now) and base.tracker.wiped and not self.lost:
            # A wipe ends the episode, often while the battle flag is still set.
            terms["loss"] -= w.loss
            self.totals[f"{kind}_losses"] += 1
            self.lost = True

        cap = base.level_cap if base.level_cap is not None else 100
        for mon in base.read_party_mons():
            old = self.levels.get(mon.identity)
            if old is not None and mon.level > old:
                terms["level"] += w.level * max(0, min(mon.level, cap) - min(old, cap))
            self.levels[mon.identity] = mon.level

        phi = 0.0 if terminated else w.hp_potential * base.read_hp_fraction()
        terms["hp"] += self.gamma * phi - self.phi
        self.phi = phi

        if self.pending_rejected:
            terms["trainer_run_rejected"] -= w.trainer_run_rejected * self.pending_rejected
            self.totals["trainer_run_rejected"] += self.pending_rejected
            self.pending_rejected = 0

        self.battle = now
        terms = {k: float(v) for k, v in terms.items() if v}
        for key, value in terms.items():
            self.totals[f"reward_{key}"] += value
        added = sum(terms.values())
        self.lifetime["reward_added"] += added
        return obs, reward + added, terminated, truncated, self._info(info, terms)

    def battle_stats(self):
        stats = {"lifetime": dict(self.lifetime), "weights": asdict(self.weights)}
        if isinstance(self.env, BattleReplay):
            stats["replay"] = self.env.replay_stats()
        return stats


def catch_in_progress(env, blob, actions=60, b_button=5):
    """True when a catch completes while only B is pressed: a ball is already thrown.

    Such snapshots end in a catch whatever the policy does, so they teach nothing.
    """
    restore(env, blob)
    owned = env.read_m(PARTY_SIZE_ADDRESS) + env.read_m(BOX_COUNT_ADDRESS)
    for _ in range(actions):
        _, _, terminated, truncated, _ = env.step(b_button)
        if env.read_m(PARTY_SIZE_ADDRESS) + env.read_m(BOX_COUNT_ADDRESS) > owned:
            return True
        if terminated or truncated or env.read_m(BATTLE_TYPE_ADDRESS) != WILD:
            return False
    return False


def load_bank(directory):
    directory = Path(directory)
    rows = json.loads((directory / "bank.json").read_text())
    manifest = json.loads((directory / "bank_manifest.json").read_text())
    for row in rows:
        if row["battle"] not in (WILD, TRAINER):
            raise ValueError("Battle bank contains a non-battle snapshot")
        if not (directory / row["file"]).is_file():
            raise ValueError(f"Missing battle snapshot {row['file']}")
    if not rows:
        raise ValueError("Battle bank is empty")
    return rows, manifest


class BattleReplay(gym.Wrapper):
    """Start every episode at a stored battle snapshot; end it when the battle ends.

    The bank is fixed for the whole run, so every replay worker sees the same states.
    A battle exit is a time-limit-style truncation (the game continues), so SB3
    bootstraps the overworld value. A battle that lasts ``max_battle_actions`` is
    truncated with ``timeout_penalty``. LSTM hidden state resets as at any episode.
    """

    def __init__(
        self,
        env,
        bank,
        seed=0,
        trainer_share=0.5,
        max_battle_actions=1024,
        timeout_penalty=1.0,
        config_sha=None,
    ):
        super().__init__(env)
        if not 0 <= trainer_share <= 1 or max_battle_actions < 1 or timeout_penalty < 0:
            raise ValueError("Invalid battle replay parameters")
        self.directory = Path(bank)
        self.rows, manifest = load_bank(self.directory)
        if config_sha is not None and manifest["config_sha256"] != config_sha:
            raise ValueError("Battle bank was captured under a different environment config")
        self.by_type = {t: [r for r in self.rows if r["battle"] == t] for t in (WILD, TRAINER)}
        self.trainer_share = trainer_share
        if (trainer_share > 0 and not self.by_type[TRAINER]) or (
            trainer_share < 1 and not self.by_type[WILD]
        ):
            raise ValueError("Battle bank lacks the battle type this share requires")
        self.rng = np.random.default_rng(seed)
        self.max_battle_actions = max_battle_actions
        self.timeout_penalty = timeout_penalty
        self.active = None
        self.actions = 0
        self.original_step = 0
        self.stats = Counter()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        kind = TRAINER if self.rng.random() < self.trainer_share else WILD
        pool = self.by_type[kind]
        row = pool[int(self.rng.integers(len(pool)))]
        blob = (self.directory / row["file"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise ValueError("Battle snapshot hash mismatch")
        obs = restore(self.env.unwrapped, blob)
        if self.env.unwrapped.read_m(BATTLE_TYPE_ADDRESS) != row["battle"]:
            raise ValueError("Restored snapshot is not in the recorded battle")
        # Report only this branch; restored reward accounting is retained. The
        # step counter only drives the time limit (no reward reads it); restart it so
        # a late-episode snapshot gets its whole battle, bounded by max_battle_actions.
        base = self.env.unwrapped
        base.episode_return = 0.0
        base.path_reward_total = 0.0
        self.original_step, base.step_count = base.step_count, 0
        self.active, self.actions = row, 0
        self.stats[f"starts_battle{kind}"] += 1
        return obs, dict(info, **self._info())

    def _info(self):
        return {
            "training_reset": "battle",
            "battle_source": self.active,
            "battle_replay_actions": self.actions,
            "restart_original_step": self.original_step,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions += 1
        self.stats["actions"] += 1
        if not (terminated or truncated):
            if self.env.unwrapped.read_m(BATTLE_TYPE_ADDRESS) not in (WILD, TRAINER):
                truncated = True
                info = dict(info, end_reason="battle_over")
            elif self.actions >= self.max_battle_actions:
                truncated = True
                reward -= self.timeout_penalty
                info = dict(info, end_reason="battle_timeout")
        if terminated or truncated:
            self.stats[f"end_{info['end_reason']}"] += 1
        return obs, reward, terminated, truncated, dict(info, **self._info())

    def replay_stats(self):
        return dict(self.stats)
