from fractions import Fraction

import pytest

from battle_benchmark.bide import (
    ATTACK,
    QUIET,
    RELEASE,
    START,
    WAIT,
    AnswerKey,
    State,
    monte_carlo,
    observation,
    outcomes,
    policy,
)


def test_probability_mass_on_all_reachable_states():
    frontier = {State()}
    visited = set()
    for _ in range(5):
        successors = set()
        for state in frontier - visited:
            for action in (ATTACK, WAIT):
                distribution = outcomes(state, action)
                assert sum(p for p, _ in distribution) == 1
                assert all(p > 0 for p, _ in distribution)
                successors.update(s for _, s in distribution)
        visited |= frontier
        frontier = successors
    assert len(visited) > 100


def test_bide_counts_damage_and_releases_twice():
    state = State(hp=12, enemy_hp=12, remaining=2, stored=0, signal=START)
    branches = outcomes(state, ATTACK)
    assert {(s.remaining, s.stored) for _, s in branches} == {(1, 2), (1, 3)}
    for _, state in branches:
        [(p, released)] = outcomes(state, WAIT)
        assert p == 1
        assert released.hp == 12 - 2 * state.stored
        assert released.signal == RELEASE
        assert released.remaining == released.stored == 0


def test_player_knockout_prevents_release():
    [(p, state)] = outcomes(State(1, 2, 1, 10), ATTACK)
    assert p == 1 and state.hp == 1 and state.enemy_hp == 0


def test_public_observation_hides_timer_and_damage():
    assert observation(State(12, 10, 1, 2)) == observation(State(12, 10, 3, 8))


def test_memory_policy_uses_old_announcement():
    [(_, action, memory)] = policy("remember_bide", (12, 12, START))
    assert action == WAIT and memory
    [(_, action, memory)] = policy("remember_bide", (12, 12, QUIET), memory)
    assert action == WAIT and memory
    [(_, action, memory)] = policy("remember_bide", (12, 12, RELEASE), memory)
    assert action == ATTACK and not memory
    assert policy("reactive", (12, 12, QUIET))[0][1] == ATTACK


def test_hand_computed_one_turn_answer():
    # Against 3 HP, only the 3-damage roll wins in one turn.
    key = AnswerKey()
    assert key.optimal(State(12, 3), 1) == Fraction(1, 2)
    assert key.optimal(State(12, 2), 1) == 1
    assert key.optimal(State(12, 2), 0) == 0
    assert key.optimal(State(0, 2), 1) == 0


def test_oracle_dominates_and_mc_agrees():
    key, initial, horizon = AnswerKey(), State(8, 8), 8
    optimum = key.optimal(initial, horizon)
    for name in ("always_attack", "random", "reactive", "remember_bide"):
        exact = key.evaluate(name, initial, horizon)
        assert 0 <= exact <= optimum <= 1
        # A reproducible stochastic cross-check of DP, not a cartridge validation.
        sampled = monte_carlo(key, name, initial, horizon, 3000, 32) / 3000
        assert sampled == pytest.approx(float(exact), abs=0.035)
