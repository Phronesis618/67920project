# Offline LSTM memory from replayed episodes

**November 27.** Supervised training of an LSTM memory, separate from PPO. No reward,
policy update or environment rollout by a learner is involved.

## Why

A catch is legal only at the first encounter in an area, which no screen shows. The
pixel policies see three stacked 72×80 screens and their last three buttons, so only a
recurrent memory can track it. A probe of the LSTM parent found it does not: catch
legality was near chance from its memory. Training the memory offline keeps policy
training (the `battle-training` branch) and memory training independent.

## Data

Evaluation runs (`scripts/diagnose_forest.py`) record every action. Replaying a
diagnostics directory's episodes **in their original order in one environment**
reproduces each episode exactly. Replaying a single episode on its own can diverge:
one diverged 6,666 actions in, at a wild encounter, because PyBoy keeps some state
across save loads. `scripts/build_memory_dataset.py` checks every step against the
trace (position, battle, party, reward) and the final outcome, and refuses a mismatch.

All 280 recorded episodes (28 directories from `campaign-route-replay-20261127` and
`campaign-followups-20261125`, PPO and LSTM, several training arms) replayed with no
mismatch: 106 wipes, 136 time limits, 38 illegal catches. Each step pairs the frozen
LSTM parent's 277-d input features (CNN + recent buttons) for the observation with RAM
labels for that same state (`project_work/v2/lstm_memory.py`):
`encounter_catchable` (overworld on an encounter map: area unused; wild battle: this
encounter catchable), `used_route2/forest/route22`, `poisoned`, `owned_gt1`, and scaled
`party_hp`, `enemy_hp`, `balls`, `antidotes`, `event_flags`, `since_wild_battle`.

## Training

`scripts/train_memory.py`: episodes split by episode and stratified by ending into
179 train, 45 validation and 56 test (713,553 test actions). Truncated BPTT over
256-action chunks, Adam 1e-3, 6 epochs, the epoch with the lowest validation loss kept,
binary labels class-balanced. Three models on identical features:

- `input`: linear head on the current features (standardised) — no memory;
- `parent_frozen`: the parent's actor LSTM, frozen, plus a trained head (a probe);
- `memory`: LSTM initialised from the parent's actor LSTM plus head, both trained.

## Results (held-out test episodes; balanced accuracy, chance 0.5)

| Fact | input | parent_frozen | memory |
|---|---:|---:|---:|
| Encounter here is catchable | 0.638 | 0.584 | **0.934** |
| … at the first action of each wild battle (998) | 0.643 | 0.577 | **0.966** |
| … at every wild-battle action (72,481) | 0.500 | 0.518 | **0.848** |
| Route 2 used | 0.661 | 0.668 | **0.976** |
| Forest used | 0.708 | 0.738 | **0.954** |
| Poisoned | 0.789 | 0.827 | **0.915** |
| More than one Pokémon owned | 0.502 | 0.506 | 0.501 |

The trained memory holds catch legality and area history; neither the current screen
nor the parent's memory does. `owned_gt1` has almost no positive examples (few legal
catches happen), so it stays at chance.

**The scaled (regression) labels are not learned yet.** Test R² for the memory model:
`since_wild_battle` 0.56, but `party_hp` −0.93, `enemy_hp` −0.52, `balls` −2.99,
`antidotes` −0.57, `event_flags` −2.18 (the controls are near or below 0 too). The
validation loss also fluctuated between epochs (0.85–1.47). Likely causes are one
summed loss across very differently scaled terms and per-episode shifts in these
quantities; they need per-label scaling or separate early stopping before use.

## Limits

- Test episodes come from the same policies and campaigns as training (held out by
  episode). A policy that behaves differently may show different situations.
- One seed and one split. Features are the LSTM parent's frozen CNN; a different
  policy's CNN would need its own dataset.
- The trained memory (`memory_model.pt`, not committed) is not yet connected to a
  policy. It could initialise a recurrent policy's actor LSTM, or feed a policy its
  predictions; either is a separate step.
