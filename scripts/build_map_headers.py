"""Extract map identity/size checks from a pinned pret/pokered checkout."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def build(root):
    files = [root / "constants/map_constants.asm", root / "constants/tileset_constants.asm"]
    ids = {
        name: (i, int(w), int(h))
        for i, (name, w, h) in enumerate(
            re.findall(
                r"^\s*map_const\s+(\w+),\s*(\d+),\s*(\d+)", files[0].read_text(), re.MULTILINE
            )
        )
    }
    tilesets = {
        name: i
        for i, name in enumerate(
            re.findall(r"^\s*const\s+(\w+)", files[1].read_text(), re.MULTILINE)
        )
    }
    maps = {}
    for path in sorted((root / "data/maps/headers").glob("*.asm")):
        match = re.search(r"map_header\s+(\w+),\s*(\w+),\s*(\w+)", path.read_text())
        if match is None:
            continue
        name, constant, tileset = match.groups()
        mid, w, h = ids[constant]
        maps[mid] = {"name": name, "header": [tilesets[tileset], h, w]}
        files.append(path)
    return {
        "source_repository": "https://github.com/pret/pokered",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "maps": maps,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(build(args.checkout), indent=2) + "\n")
