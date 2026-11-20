"""Privileged, quest-conditioned potential shaping from static walking distances."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

GRAPH_PATH = Path(__file__).with_name("navigation_graph.json")


@lru_cache(maxsize=1)
def graph():
    return json.loads(GRAPH_PATH.read_text())


def graph_sha256():
    return hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest()


def navigation_state(map_id, x, y, *, got_parcel, got_pokedex):
    # The campaign starts after the parcel quest. Earlier phases are supplied
    # for distance inspection, not a validated curriculum from a pre-starter save.
    phase, goal = (2, "brock") if got_pokedex else (1, "oak") if got_parcel else (0, "parcel")
    distance = graph()["distances"][goal].get(f"{map_id},{x},{y}")
    # Bounded potential on ALL states, including unsupported maps/transient tiles.
    # Stage offsets keep advancing the quest from erasing navigation progress.
    potential = 2.0 * (phase - min(distance if distance is not None else 512, 512) / 512)
    return goal, distance, potential


def shaping_reward(previous, current, gamma, terminated):
    # True terminal states have zero potential. Time-limit truncation retains
    # its potential because SB3 bootstraps the terminal observation's value.
    return gamma * (0.0 if terminated else current) - previous
