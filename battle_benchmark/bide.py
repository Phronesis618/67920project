"""Finite-horizon Bide timing pilot, NOT a Gen 1 damage/Brock simulator.

The player attacks (uniform damage 2 or 3) or waits, then the living opponent acts.
Idle opponents start Bide with probability 1/4; otherwise attack for 2 or 3.
Bide stores damage over the next 2 or 3 player turns (equiprobable) and then
returns twice the accumulated damage. The player acts first even on release turns.
Only the start/release announcements and both HP values are observable. The timer
and accumulated damage are hidden. Ordinary turns erase the announcement.

Every kernel probability and finite-horizon value is a rational Fraction. This is
an exact answer key for this model only; it deliberately omits cartridge mechanics.
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import cache
from pathlib import Path

ATTACK, WAIT = 0, 1
QUIET, START, RELEASE = 0, 1, 2
ACTIONS = (ATTACK, WAIT)


@dataclass(frozen=True)
class State:
    hp: int = 12
    enemy_hp: int = 12
    remaining: int = 0
    stored: int = 0
    signal: int = QUIET

    @property
    def terminal(self):
        return self.hp == 0 or self.enemy_hp == 0


def observation(state):
    return state.hp, state.enemy_hp, state.signal


@cache
def outcomes(state, action):
    """Return (probability, successor) pairs, with duplicate outcomes combined."""
    if action not in ACTIONS:
        raise ValueError("action must be ATTACK or WAIT")
    if state.terminal:
        return ((Fraction(1), state),)
    result = defaultdict(Fraction)
    damages = (
        ((Fraction(1, 2), 2), (Fraction(1, 2), 3)) if action == ATTACK else ((Fraction(1), 0),)
    )
    for chance, damage in damages:
        damage = min(damage, state.enemy_hp)
        new = replace(state, enemy_hp=state.enemy_hp - damage, signal=QUIET)
        if new.enemy_hp == 0:
            result[new] += chance
        elif state.remaining:
            stored = state.stored + damage
            if state.remaining == 1:
                new = replace(
                    new, hp=max(0, state.hp - 2 * stored), remaining=0, stored=0, signal=RELEASE
                )
            else:
                new = replace(new, remaining=state.remaining - 1, stored=stored)
            result[new] += chance
        else:
            for duration in (2, 3):
                result[replace(new, remaining=duration, stored=0, signal=START)] += chance / 8
            for enemy_damage in (2, 3):
                result[replace(new, hp=max(0, state.hp - enemy_damage))] += chance * Fraction(3, 8)
    return tuple((chance, new) for new, chance in result.items())


def policy(name, obs, memory=False):
    """Heuristics receive only public observations, never the hidden physical state."""
    _, _, signal = obs
    if name == "always_attack":
        return ((Fraction(1), ATTACK, False),)
    if name == "random":
        return ((Fraction(1, 2), ATTACK, False), (Fraction(1, 2), WAIT, False))
    if name == "reactive":
        return ((Fraction(1), WAIT if signal == START else ATTACK, False),)
    if name == "remember_bide":
        if signal == START:
            memory = True
        elif signal == RELEASE:
            memory = False
        return ((Fraction(1), WAIT if memory else ATTACK, memory),)
    raise ValueError(f"Unknown policy {name}")


class AnswerKey:
    """Backward induction; timeout counts as failure to win within the horizon."""

    def __init__(self):
        # Per-solve caches do not retain every previous AnswerKey instance globally.
        self.optimal = cache(self._optimal)
        self.evaluate = cache(self._evaluate)

    def _optimal(self, state, turns):
        if state.terminal:
            return Fraction(state.enemy_hp == 0 and state.hp > 0)
        if turns == 0:
            return Fraction(0)
        return max(self.action_value(state, turns, action) for action in ACTIONS)

    def action_value(self, state, turns, action):
        return sum(p * self.optimal(s, turns - 1) for p, s in outcomes(state, action))

    def choose(self, state, turns):
        return max(ACTIONS, key=lambda action: self.action_value(state, turns, action))

    def _evaluate(self, name, state, turns, memory=False):
        if state.terminal:
            return Fraction(state.enemy_hp == 0 and state.hp > 0)
        if turns == 0:
            return Fraction(0)
        return sum(
            p_action * p_next * self.evaluate(name, new, turns - 1, new_memory)
            for p_action, action, new_memory in policy(name, observation(state), memory)
            for p_next, new in outcomes(state, action)
        )


def sample_outcome(rng, choices):
    draw = rng.random()
    cumulative = 0.0
    for chance, result in choices:
        cumulative += float(chance)
        if draw < cumulative:
            return result
    return choices[-1][1]  # protect against floating point sum roundoff


def monte_carlo(key, name, initial, horizon, episodes, seed):
    rng = random.Random(seed)
    wins = 0
    for _ in range(episodes):
        state, memory = initial, False
        for turns in range(horizon, 0, -1):
            if state.terminal:
                break
            if name == "oracle":
                action = key.choose(state, turns)
            else:
                action, memory = sample_outcome(
                    rng, tuple((p, (a, m)) for p, a, m in policy(name, observation(state), memory))
                )
            state = sample_outcome(rng, outcomes(state, action))
        wins += state.enemy_hp == 0 and state.hp > 0
    return wins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hp", type=int, default=12)
    parser.add_argument("--enemy-hp", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/bide-pilot.json"))
    args = parser.parse_args(argv)
    if min(args.horizon, args.hp, args.enemy_hp, args.episodes) < 1:
        parser.error("horizon, HP values and episodes must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    initial, key = State(args.hp, args.enemy_hp), AnswerKey()
    optimum = key.optimal(initial, args.horizon)
    results = {}
    for name in ("always_attack", "random", "reactive", "remember_bide", "oracle"):
        exact = optimum if name == "oracle" else key.evaluate(name, initial, args.horizon)
        wins = monte_carlo(key, name, initial, args.horizon, args.episodes, args.seed)
        results[name] = {
            "exact_win_probability": float(exact),
            "exact_fraction": str(exact),
            "gap_to_full_state_oracle": float(optimum - exact),
            "sampled_wins": wins,
            "sampled_win_rate": wins / args.episodes,
        }
    report = {
        "model": "abstract-bide-v1",
        "cartridge_validated": False,
        "scope": "Synthetic timing pilot. Not a Brock/Nuzlocke result; oracle sees hidden timer and stored damage.",
        "initial_state": asdict(initial),
        "horizon": args.horizon,
        "episodes_per_policy": args.episodes,
        "seed": args.seed,
        "results": results,
        "oracle_cache": key.optimal.cache_info()._asdict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
