from collections import defaultdict
from itertools import pairwise

import pytest
from navigation import graph, navigation_state, shaping_reward


def test_shortest_routes_use_forest_gates_and_directed_ledges():
    data = graph()
    edges = defaultdict(list)
    for a, b, cost, kind, _ in data["edges"]:
        edges[tuple(a)].append((tuple(b), cost, kind))

    def route(start, goal):
        distance = data["distances"][goal]
        node = start
        maps = [node[0]]
        kinds = []
        for _ in range(1024):
            d = distance[",".join(map(str, node))]
            if d == 0:
                return maps, kinds
            candidates = [
                (b, k)
                for b, c, k in edges[node]
                if distance.get(",".join(map(str, b)), 1e9) + c == d
            ]
            assert candidates
            node, kind = candidates[0]
            if node[0] != maps[-1]:
                maps.append(node[0])
            kinds.append(kind)
        pytest.fail("route did not terminate")

    maps, _ = route((40, 5, 3), "brock")
    assert maps == [40, 0, 12, 1, 13, 50, 51, 47, 13, 2, 54]
    maps, kinds = route((42, 2, 5), "oak")
    assert maps == [42, 1, 12, 0, 40] and "ledge" in kinds
    for a, b, _, kind, _ in data["edges"]:
        if kind == "ledge":
            assert not any(n == tuple(a) for n, _, _ in edges[tuple(b)])


def test_objective_changes_and_unknown_map():
    assert navigation_state(42, 2, 5, got_parcel=False, got_pokedex=False)[:2] == ("parcel", 0)
    assert navigation_state(42, 2, 5, got_parcel=True, got_pokedex=False)[0] == "oak"
    assert navigation_state(40, 5, 3, got_parcel=True, got_pokedex=True)[:2] == ("brock", 386)
    assert navigation_state(255, 0, 0, got_parcel=True, got_pokedex=True)[1] is None


def test_shaping_terminal_truncation_and_cycles():
    gamma = 0.9995
    assert shaping_reward(3, 3.1, gamma, True) == -3
    assert shaping_reward(3, 3.1, gamma, False) == pytest.approx(gamma * 3.1 - 3)
    potentials = [2.5, 2.6, 2.5, 2.7]
    total = sum(
        gamma**i * shaping_reward(a, b, gamma, i == 2)
        for i, (a, b) in enumerate(pairwise(potentials))
    )
    assert total == pytest.approx(-potentials[0])
    truncated = sum(
        gamma**i * shaping_reward(a, b, gamma, False)
        for i, (a, b) in enumerate(pairwise(potentials))
    )
    assert truncated == pytest.approx(-potentials[0] + gamma**3 * potentials[-1])
