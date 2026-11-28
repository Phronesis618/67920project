# Battle rewards and battle-only replay (implementation, not yet trained)

**November 27.** This adds a battle curriculum and validates it on CPU. No campaign has
been launched and nothing here claims better play.

## Why

The route-replay comparison ([2026-11-27](../2026-11-27/README.md), final report on
Lambda) did not stop trainer stalls: continuations stalled in 18 of 60 trainer battles
versus 6 of 42 for their parents, with rejected RUNs up to 27–30 per 1,000 trainer
actions. Of the 108 evaluation episodes that never reached the forest, 56 timed out
wandering, 27 ended in illegal catches and 25 were wipes of a lone level 6–7 Squirtle.

The reward explains much of this. The `level` and `op_lvl` terms are commented out,
heal reward is squared (tiny for small heals), `death_penalty` and `violation_penalty`
are 0, a wild win earns nothing and a trainer win about +2 (one event flag). Losing a
battle ends the episode at once (about −2 to −4 terminal shaping); stalling costs
only about 0.001–0.002 per action through `γΦ′ − Φ`. Under `γ = 0.9995` that makes
stalling or RUN spam the safe choice. Replaying failed battles without a battle reward
mostly rehearses that choice; route replay also split starts evenly across overworld,
wild and trainer contexts and kept playing after the battle ended.

## What changed

`project_work/v2/battle_training.py`

- **`BattleReward`** adds terms to the unchanged base reward for every worker:
  enemy damage (per full HP bar), KOs and wins; levels gained up to the level cap;
  potential-based HP shaping; −0.05 per rejected trainer RUN (ROM-verified PyBoy hook);
  overtime cost after 512 actions in one battle. Wild terms are 0.2 and switch off at
  the level cap so Route 1 farming cannot outweigh the route; trainer terms are 1/1/2.
  Damage counts only while enemy species and max HP are unchanged: at trainer-battle
  entry the enemy slot still holds the previous wild mon (seen on real snapshots).
  A win is a battle exit after the enemy reached 0 HP; running and catching are not.
- **`BattleReplay`** starts each episode at a fixed battle snapshot (trainer share 0.5)
  and truncates when the battle ends, so SB3 bootstraps the overworld value. Battles
  longer than 1,024 actions are truncated with −1. The restored step counter restarts
  so late-episode snapshots are not cut off by the old remaining horizon.
- `battle_experiment.py` continues a parent with 8 route + 8 replay workers, adds
  `death_penalty`/`violation_penalty` 2.0, and exposes every weight plus
  `--learning-rate`, `--target-kl` and `--clip-range` (parents: 3e-4, none, 0.2).
- `scripts/build_battle_bank.py` samples battle snapshots from existing runs' failure
  banks, at most `--per-group` per (battle type, map, anchor mode), hash-checked and
  deduplicated; all sources must share one snapshot config hash.

RAM is privileged reward/reset information only. Actor inputs and the action set are
unchanged. Evaluation must still use original starts (`scripts/diagnose_forest.py`).

## Bank

From the 18 route-replay runs' failure banks (2,256 battle snapshots), maps 12/13/51,
64 per group, seed 0: **512 snapshots**, 64 each of wild Route 1 / Route 2 / forest and
trainer forest, half battle-entry (transition anchors) and half mid-battle (tail
anchors). [bank_manifest.json](bank_manifest.json). The bank was built on the Mac from
a copy; rebuild it on Lambda with the same command before a campaign.

## Validation

- 94 tests pass (12 new). Lint clean on the new files.
- Real game, frozen parents, 16 bank battles each, 512-action cap
  ([PPO](validation-ppo-parent.json), [LSTM](validation-lstm-parent.json)): hooks fired
  (16 and 7 rejected trainer RUNs), terms matched visible play (a trainer win ≈ +4.3
  including damage, KO, level and win). Parent baselines from these starts: PPO won 3
  of 7 trainer battles (2 timeouts, 2 wipes); LSTM won 1 of 7 (4 timeouts). Both won 1
  of 9 wild battles and fled the rest. Sixteen battles is a sanity check, not a baseline.
- CPU smoke training ([smoke.json](smoke.json)), PPO and LSTM, 1,024 actions, 4 workers,
  lr 1e-4, target KL 0.03: both completed and saved; replay workers ended on battle exit,
  smoke battle cap and wipes; hooks fired in subprocess workers.

Two defects were found and fixed during the smoke run: snapshots carried their original
step count, so with the smoke horizon every replay episode ended after one action
(covered by a new test); and loading Lambda (Python 3.11) checkpoints on Python 3.12
crashed with SIGTRAP inside the pickled `clip_range` schedule, so the experiment now
rebuilds schedules from plain values.

## Open before launching

- The weights are initial guesses. Wild terms may be too small to stop fleeing.
- This changes the reward, which the route-replay write-up advised against
  mid-comparison; agree on it before a campaign.
- `rl-conservative-20261127` was using the Lambda GPU when this was written.
