"""Replay real emulator inputs for catches, a faint and battle-entry cap enforcement.

No RAM edits in the gameplay cases. The separate reward wiring audit deliberately
sets event bits; it is not evidence of completing the parcel quest through play.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from red_gym_env_v2_nuzlocke import RedGymEnv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_for(folder, state="has_pokedex_nballs.state", **extra):
    return {
        "session_path": folder,
        "gb_path": str(ROOT / "project_work/PokemonRed.gb"),
        "init_state": str(ROOT / "project_work" / state),
        "headless": True,
        "save_final_state": False,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "action_freq": 24,
        "max_steps": 2000,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
        **extra,
    }


def replay(folder, actions, **extra):
    env = RedGymEnv(config_for(folder, **extra))
    health = []
    valid = True
    missing_path_states = 0
    try:
        obs, _ = env.reset(seed=0)
        valid &= env.observation_space.contains(obs)
        for action in actions:
            obs, _, terminated, truncated, info = env.step(action)
            valid &= env.observation_space.contains(obs)
            if env.navigation_reward and info["navigation_distance"] is None:
                missing_path_states += 1
            hp = env.read_party_mons()[0].hp
            if not health or hp != health[-1]["hp"]:
                health.append({"step": env.step_count, "hp": hp})
            if terminated or truncated:
                break
        return {
            "info": info,
            "all_observations_valid": bool(valid),
            "party_size": len(env.read_party_mons()),
            "health_changes": health,
            "missing_path_states": missing_path_states,
            "observation_keys": sorted(obs),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actions_path = ROOT / "validation/first_catch_actions.json"
    actions = json.loads(actions_path.read_text())
    folder = ROOT / "runs/real-rule-validation"
    cases = {
        "legal_catch": replay(folder / "legal", actions),
        "illegal_catch": replay(folder / "illegal", actions, initial_used_regions=[12]),
        "battle_entry_cap": replay(folder / "cap", actions, level_cap=5),
        "faint_and_wipe": replay(folder / "faint", actions[:64] + [4, 0, 4, 4] + [4] * 1200),
        "navigation_pixels": replay(
            folder / "navigation",
            actions,
            observation_mode="pixels",
            navigation_reward=True,
            navigation_gamma=0.9995,
            explore_weight=0.025,
        ),
    }
    checks = {
        "navigation_pixels": cases["navigation_pixels"]["missing_path_states"] == 0
        and cases["navigation_pixels"]["observation_keys"] == ["recent_actions", "screens"],
        "legal_catch": cases["legal_catch"]["party_size"] == 2
        and cases["legal_catch"]["info"]["end_reason"] == "running"
        and cases["legal_catch"]["info"]["deaths"] == 0,
        "illegal_catch": cases["illegal_catch"]["info"]["violation_reasons"] == ["illegal_catch"],
        "battle_entry_cap": cases["battle_entry_cap"]["info"]["violation_reasons"] == ["level_cap"],
        "faint_and_wipe": cases["faint_and_wipe"]["info"]["end_reason"] == "wipe"
        and cases["faint_and_wipe"]["info"]["deaths"] == 1,
        "observations": all(c["all_observations_valid"] for c in cases.values()),
    }
    env = RedGymEnv(config_for(folder / "reward", state="init.state"))
    wiring = {}
    try:
        env.reset(seed=0)
        for name, address, bit in (
            ("parcel", 0xD74E, 1),
            ("delivery", 0xD74E, 0),
            ("pokedex", 0xD74B, 5),
        ):
            env.pyboy.memory[address] |= 1 << bit
            wiring[name] = {
                "first_reward": env.update_reward(),
                "repeat_reward": env.update_reward(),
            }
        key = "x:3 y:6 m:38"
        env.seen_coords[key] = 599
        wiring["stuck_at_599"] = env.get_game_state_reward()["stuck"]
        env.seen_coords[key] = 600
        wiring["stuck_at_600"] = env.get_game_state_reward()["stuck"]
    finally:
        env.close()
    checks["reward_wiring"] = all(
        wiring[k] == {"first_reward": 2.0, "repeat_reward": 0.0}
        for k in ("parcel", "delivery", "pokedex")
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "cases": cases,
        "reward_wiring_injected_ram": wiring,
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__),
                actions_path,
                ROOT / "project_work/PokemonRed.gb",
                ROOT / "project_work/has_pokedex_nballs.state",
                ROOT / "project_work/v2/red_gym_env_v2_nuzlocke.py",
                ROOT / "project_work/v2/nuzlocke_rules.py",
                ROOT / "project_work/v2/navigation.py",
                ROOT / "project_work/v2/navigation_graph.json",
            )
        },
        "limitations": [
            "Inherited save history is unknown; these are rules-from-reset fixtures.",
            "Illegal catch uses the same real catch with Route 1 declared used at reset.",
            "Cap test lowers the cap to five for a real battle entered by a level-six starter.",
            "Reward audit injects flags and does not validate the complete parcel route.",
            "Full-party box catches, PC identity changes, revival and Brock remain unvalidated in real play.",
            "Battle item bans and set-mode enforcement remain unimplemented.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
