"""Sample a fixed battle-start bank from existing runs' failure banks.

Each source run directory needs ``manifest.json`` and ``failure-banks/worker-*/bank.json``
(written by route_replay.RouteReplay). Only wild/trainer snapshots are kept. Rows are
grouped by (battle type, map, anchor mode) and up to ``--per-group`` are drawn per group
with a seeded RNG, so the commonest failure location does not crowd out the rest.
With ``--screen-rom/--screen-state``, wild snapshots whose catch is already underway
(a ball was thrown before the snapshot) are dropped. Blobs are hash-checked,
deduplicated by content and copied; sources are never modified.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from battle_training import TRAINER, WILD, catch_in_progress, snapshot_config_sha


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def collect(runs):
    rows, shas, configs = [], {}, []
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        configs.append(manifest["config"])
        sha = snapshot_config_sha(manifest["config"], manifest["rom_sha1"], manifest["state_sha1"])
        shas[str(run)] = sha
        for index in sorted(run.glob("failure-banks/worker-*/bank.json")):
            for row in json.loads(index.read_text()):
                if row["battle"] in (WILD, TRAINER):
                    rows.append(
                        dict(row, source_run=run.name, source_file=str(index.parent / row["file"]))
                    )
    return rows, shas, configs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+", help="Run directories holding failure-banks/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maps", type=int, nargs="*", help="Keep only these map ids")
    p.add_argument("--screen-rom", type=Path, help="Drop wild snapshots with a catch underway")
    p.add_argument("--screen-state", type=Path)
    args = p.parse_args(argv)
    if args.per_group < 1:
        p.error("per-group must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Choose an empty output directory")
    if (args.screen_rom is None) != (args.screen_state is None):
        p.error("Pass both --screen-rom and --screen-state, or neither")
    rows, shas, configs = collect(args.runs)
    if len(set(shas.values())) != 1:
        p.error("Source runs use different environment configs; build one bank per config")
    if args.maps:
        rows = [r for r in rows if r["map"] in args.maps]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["battle"], row["map"], row["mode"])].append(row)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    bank, seen, screened_out = [], set(), Counter()
    screen, scratch = None, None
    if args.screen_rom:
        from red_gym_env_v2_nuzlocke import RedGymEnv

        scratch = tempfile.TemporaryDirectory()
        screen = RedGymEnv(
            dict(
                configs[0],
                session_path=scratch.name,
                gb_path=str(args.screen_rom),
                init_state=str(args.screen_state),
                save_video=False,
                save_final_state=False,
            )
        )
        screen.reset(seed=0)
    for key in sorted(groups):
        group = groups[key]
        order = rng.permutation(len(group))
        taken = 0
        for i in order:
            if taken >= args.per_group:
                break
            row = group[int(i)]
            blob = Path(row["source_file"]).read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Hash mismatch in {row['source_file']}")
            if digest in seen:
                continue
            seen.add(digest)
            if screen is not None and row["battle"] == WILD and catch_in_progress(screen, blob):
                screened_out[f"map{row['map']}/{row['mode']}"] += 1
                continue
            name = f"battle{row['battle']}-map{row['map']:03d}-{digest[:16]}.pkl.gz"
            (args.output / name).write_bytes(blob)
            bank.append(
                {
                    "file": name,
                    "sha256": digest,
                    "battle": row["battle"],
                    "map": row["map"],
                    "mode": row["mode"],
                    "anchor": "battle_entry" if row["mode"] == "transition" else "mid_tail",
                    "original_step": row["original_step"],
                    "lead_actions": row["lead_actions"],
                    "end_reason": row["end_reason"],
                    "source_run": row["source_run"],
                    "source_file": row["source_file"],
                }
            )
            taken += 1
    if screen is not None:
        screen.close()
        scratch.cleanup()
    if not bank:
        p.error("No battle snapshots matched")
    write_json(args.output / "bank.json", bank)
    counts = Counter(f"battle{r['battle']}/map{r['map']}/{r['anchor']}" for r in bank)
    write_json(
        args.output / "bank_manifest.json",
        {
            "config_sha256": next(iter(shas.values())),
            "created_utc": datetime.now(UTC).isoformat(),
            "sources": sorted(shas),
            "per_group": args.per_group,
            "seed": args.seed,
            "maps": args.maps,
            "available": len(rows),
            "screened": args.screen_rom is not None,
            "screened_out_catch_in_progress": dict(screened_out),
            "selected": len(bank),
            "counts": dict(sorted(counts.items())),
        },
    )
    print(json.dumps({"bank": str(args.output), "selected": len(bank), "counts": counts}))


if __name__ == "__main__":
    main()
