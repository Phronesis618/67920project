"""Build directed walking distances from a pinned pret/pokered checkout.

Only the Pallet-to-Brock corridor and its Centers/Marts are included. Static
collision tiles, land tile-pair restrictions, directional ledges, map connections
and warp destinations are used. Moving NPCs, battle danger and travel time are
not modeled. This is a navigation heuristic, not a game solver.
"""

import argparse
import hashlib
import heapq
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

MAPS = [
    "PalletTown",
    "Route1",
    "ViridianCity",
    "Route2",
    "ViridianForest",
    "ViridianForestSouthGate",
    "ViridianForestNorthGate",
    "PewterCity",
    "OaksLab",
    "PewterGym",
    "ViridianMart",
    "ViridianPokecenter",
    "PewterMart",
    "PewterPokecenter",
    "RedsHouse1F",
    "RedsHouse2F",
]
TILESETS = {
    "OVERWORLD": ("Overworld", "overworld"),
    "DOJO": ("Dojo", "gym"),
    "GYM": ("Gym", "gym"),
    "FOREST": ("Forest", "forest"),
    "FOREST_GATE": ("ForestGate", "gate"),
    "MART": ("Mart", "pokecenter"),
    "POKECENTER": ("Pokecenter", "pokecenter"),
    "REDS_HOUSE_1": ("RedsHouse1", "reds_house"),
    "REDS_HOUSE_2": ("RedsHouse2", "reds_house"),
}
LAST = {
    "OaksLab": "PALLET_TOWN",
    "RedsHouse1F": "PALLET_TOWN",
    "ViridianMart": "VIRIDIAN_CITY",
    "ViridianPokecenter": "VIRIDIAN_CITY",
    "PewterGym": "PEWTER_CITY",
    "PewterMart": "PEWTER_CITY",
    "PewterPokecenter": "PEWTER_CITY",
    "ViridianForestSouthGate": "ROUTE_2",
    "ViridianForestNorthGate": "ROUTE_2",
}
DIRECTIONS = {"north": (0, -1, 3), "south": (0, 1, 0), "west": (-1, 0, 1), "east": (1, 0, 2)}


def key(node):
    return ",".join(map(str, node))


def build(root):
    used = {}

    def read(name, binary=False):
        raw = (root / name).read_bytes()
        used[name] = hashlib.sha256(raw).hexdigest()
        return raw if binary else raw.decode()

    constants = {
        m[0]: (i, int(m[1]), int(m[2]))
        for i, m in enumerate(
            re.findall(
                r"^\s*map_const\s+(\w+),\s*(\d+),\s*(\d+)",
                read("constants/map_constants.asm"),
                re.MULTILINE,
            )
        )
    }
    collisions = {}
    labels = []
    for line in read("data/tilesets/collision_tile_ids.asm").splitlines():
        match = re.match(r"(\w+)_Coll::", line)
        if match:
            labels.append(match[1])
        if line.strip().startswith("coll_tiles"):
            values = {int(v, 16) for v in re.findall(r"\$([0-9a-fA-F]+)", line)}
            for label in labels:
                collisions[label] = values
            labels = []
    forbidden = defaultdict(set)
    pairs = read("data/tilesets/pair_collision_tile_ids.asm").split("TilePairCollisionsWater")[0]
    for ts, a, b in re.findall(r"db\s+(\w+),\s*\$([\dA-Fa-f]+),\s*\$([\dA-Fa-f]+)", pairs):
        forbidden[ts].add(frozenset((int(a, 16), int(b, 16))))
    ledges = []
    for direction, a, b in re.findall(
        r"db SPRITE_FACING_(\w+),\s*\$([\dA-Fa-f]+),\s*\$([\dA-Fa-f]+)",
        read("data/tilesets/ledge_tiles.asm"),
    ):
        d = {"DOWN": "south", "LEFT": "west", "RIGHT": "east"}[direction]
        ledges.append((DIRECTIONS[d], int(a, 16), int(b, 16)))
    maps = {}
    by_constant = {}
    tiles = {}
    walkable = set()
    warps = {}
    connections = []
    for name in MAPS:
        header = read(f"data/maps/headers/{name}.asm")
        constant, ts = re.search(r"map_header\s+\w+,\s*(\w+),\s*(\w+)", header).groups()
        mid, bw, bh = constants[constant]
        w, h = 2 * bw, 2 * bh
        label, blockname = TILESETS[ts]
        blocks = read(f"maps/{name}.blk", True)
        blockset = read(f"gfx/blocksets/{blockname}.bst", True)
        if len(blocks) != bw * bh:
            raise ValueError(f"Block dimensions mismatch: {name}")
        maps[mid] = {"name": name, "width": w, "height": h, "tileset": ts, "constant": constant}
        by_constant[constant] = mid
        for y in range(h):
            for x in range(w):
                block = blocks[(y // 2) * bw + x // 2]
                tile = blockset[block * 16 + (2 * (y % 2) + 1) * 4 + 2 * (x % 2)]
                node = (mid, x, y)
                tiles[node] = tile
                if tile in collisions[label]:
                    walkable.add(node)
        obj = read(f"data/maps/objects/{name}.asm")
        warps[mid] = [
            (int(x), int(y), LAST[name] if dest == "LAST_MAP" else dest, int(warp) - 1)
            for x, y, dest, warp in re.findall(
                r"warp_event\s+(\d+),\s*(\d+),\s*(\w+),\s*(\d+)", obj
            )
        ]
        # Fixed visible sprites after the parcel quest. Transient/toggleable Oak
        # sprites are handled explicitly; moving NPCs remain an approximation.
        for x, y, sprite, movement, rest in re.findall(
            r"object_event\s+(\d+),\s*(\d+),\s*(\w+),\s*(\w+),(.*)", obj
        ):
            if movement != "STAY":
                continue
            if name == "PalletTown" and sprite == "SPRITE_OAK":
                continue
            if name == "OaksLab" and (int(x), int(y)) != (5, 2):
                continue
            if name == "ViridianCity" and "OLD_MAN_SLEEPY" in rest:
                continue
            walkable.discard((mid, int(x), int(y)))
        for direction, target, _, offset in re.findall(
            r"connection (\w+),\s*(\w+),\s*(\w+),\s*(-?\d+)", header
        ):
            connections.append((mid, direction, target, int(offset)))
    edges = defaultdict(dict)
    name_to_id = {m["name"]: i for i, m in maps.items()}

    def edge(a, b, cost, kind, action):
        if a in walkable and b in walkable:
            edges[a][b] = (cost, kind, action)

    for n in sorted(walkable):
        mid, x, y = n
        ts = maps[mid]["tileset"]
        for dx, dy, action in DIRECTIONS.values():
            target = (mid, x + dx, y + dy)
            if target in walkable and frozenset((tiles[n], tiles[target])) not in forbidden[ts]:
                edge(n, target, 1, "walk", action)
        if ts == "OVERWORLD":
            for (dx, dy, action), floor, ledge in ledges:
                if tiles[n] == floor and tiles.get((mid, x + dx, y + dy)) == ledge:
                    edge(n, (mid, x + 2 * dx, y + 2 * dy), 2, "ledge", action)
    for mid, direction, name, offset in connections:
        if name not in name_to_id:
            continue
        dest = name_to_id[name]
        dx, dy, action = DIRECTIONS[direction]
        for n in sorted(walkable):
            if n[0] != mid:
                continue
            _, x, y = n
            m = maps[mid]
            d = maps[dest]
            if dy == -1 and y == 0:
                edge(n, (dest, x - 2 * offset, d["height"] - 1), 1, "connection", action)
            if dy == 1 and y == m["height"] - 1:
                edge(n, (dest, x - 2 * offset, 0), 1, "connection", action)
            if dx == -1 and x == 0:
                edge(n, (dest, d["width"] - 1, y - 2 * offset), 1, "connection", action)
            if dx == 1 and x == m["width"] - 1:
                edge(n, (dest, 0, y - 2 * offset), 1, "connection", action)
    for mid, entries in warps.items():
        for x, y, dest_constant, index in entries:
            if dest_constant not in by_constant:
                continue
            dest = by_constant[dest_constant]
            tx, ty, *_ = warps[dest][index]
            # Warp nodes include the arrival tile. Actual execution may also take
            # an automatic step; distances are approximate tile counts.
            edge((mid, x, y), (dest, tx, ty), 1, "warp", None)
    reverse = defaultdict(list)
    for a, targets in edges.items():
        for b, (cost, _, _) in targets.items():
            reverse[b].append((a, cost))
    goals = {"parcel": (42, 2, 5), "oak": (40, 5, 3), "brock": (54, 4, 2)}
    distances = {}
    for name, goal in goals.items():
        if goal not in walkable:
            raise ValueError(f"Goal not walkable: {goal}")
        best = {goal: 0}
        heap = [(0, goal)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance != best[node]:
                continue
            for parent, cost in reverse[node]:
                candidate = distance + cost
                if candidate < best.get(parent, float("inf")):
                    best[parent] = candidate
                    heapq.heappush(heap, (candidate, parent))
        distances[name] = {key(n): v for n, v in sorted(best.items())}
    serial_edges = [
        [list(a), list(b), cost, kind, action]
        for a in sorted(edges)
        for b, (cost, kind, action) in sorted(edges[a].items())
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "format": 1,
        "source_repository": "https://github.com/pret/pokered",
        "source_commit": commit,
        "source_sha256": used,
        "scope": "Static walking corridor after parcel delivery; no Cut/Surf, moving NPCs, battle danger or timing.",
        "maps": maps,
        "goals": goals,
        "distances": distances,
        "edges": serial_edges,
        "nodes": len(walkable),
        "edge_count": len(serial_edges),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = build(a.checkout)
    a.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "nodes": data["nodes"],
                "edges": data["edge_count"],
                "start_distances": {g: d.get("40,5,3") for g, d in data["distances"].items()},
            },
            indent=2,
        )
    )
