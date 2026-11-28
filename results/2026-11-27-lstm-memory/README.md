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

## Menu checks (`--peek-rate 0.002`)

Ball and antidote counts appear only in the bag, and overworld HP only on the party
screen; recorded play rarely opens either. `build_memory_dataset.py --peek-rate` copies
the state at random overworld actions into a second emulator that opens START, looks at
the bag or the party, and closes it (navigation checked against the start-menu border
tile and cursor in RAM; a check where the menu did not open is discarded). Those actions
are spliced into the sequence and flagged; the main replay never reloads and still
matched every trace. At the policy's 72×80 resolution the bag shows "×5" and "×2" and the
party screen the HP bar and "22/22". Peek actions are trained on but never scored.

All 280 episodes again, 29,609 peek actions, the same 56 test episodes
([metrics-peek.json](metrics-peek.json)). Memory model, no peeks → with peeks:

| Label | no peeks | with peeks |
|---|---:|---:|
| Party HP (R²) | −0.93 | **0.21** |
| Balls (R²) | −2.99 | **0.17** |
| Enemy HP / antidotes (R²) | −0.52 / −0.57 | −0.04 / −0.13 |
| Event flags (R²) | −2.18 | −2.91 |
| Encounter catchable | **0.93** | 0.81 |
| … at wild-battle entry / during wild battles | **0.97 / 0.85** | 0.83 / 0.61 |

Peeks made party HP and ball counts learnable from memory for the first time (modestly).
Event flags never appear in a menu and were not helped. Catch legality fell, but the
selected epoch also changed (epoch 2 instead of 4): one epoch is chosen per model from a
single summed validation loss across all twelve labels, which fluctuated by ±0.3 between
epochs in both runs. This run does not separate the effect of peeks from that selection
noise; per-label epoch selection and repeated seeds are needed before concluding that
peeks hurt legality.

## Three seeds, no epoch selection

Epoch selection by a summed validation loss was noisy (it moved by about ±0.3 between
epochs), so `train_memory.py` now decays the learning rate on a cosine (1e-3 → 5e-5 over
6 epochs) and evaluates the average of the last two epochs' weights. Both datasets were
retrained with seeds 0–2 on the same split ([metrics-seeds.json](metrics-seeds.json)).
Memory model, mean ± sd over seeds:

| Label | input | no peeks | with peeks |
|---|---:|---:|---:|
| Encounter catchable | 0.63 | 0.96 ± 0.00 | 0.96 ± 0.01 |
| … at wild-battle entry / during wild battles | 0.62 / 0.51 | 0.97 / 0.90 | 0.97 / 0.89 |
| Route 2 used / forest used | 0.66 / 0.69 | 0.98 / 0.95 | 0.97 / 0.96 |
| Poisoned | 0.71 | 0.90 | 0.90 |
| More than one Pokémon owned | 0.50 | 0.70 ± 0.07 | 0.69 ± 0.01 |
| Party HP (R²) | 0.01 | 0.42 ± 0.03 | 0.37 ± 0.08 |
| Enemy HP / balls (R²) | 0.01 / 0.01 | 0.22 / 0.20 | 0.24 / 0.19 |
| Time since last wild battle (R²) | 0.02 | 0.64 | 0.63 |
| Antidotes / event flags (R²) | < 0 | < 0 | < 0 |

The earlier drop in catch legality with peeks was selection noise: with a fixed
procedure both datasets reach 0.96. Menu checks made no measurable difference to any
label over three seeds; the earlier count and HP gains came from the training procedure,
not the peeks. Party HP, enemy HP, balls and time since a wild battle are now partly
learned; antidote counts and event flags are not.
