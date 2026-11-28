# Focused pilot: can learned state recognition improve combat and recovery?

The [failure audit](../failure-analysis/README.md) found trainer RUN loops and
poison deaths with antidotes still available. It did not establish why the policy
fails to select the required actions. This pilot tests one candidate mechanism:
the pixel LSTM may not represent battle type and recovery needs well enough.

## Fixed comparison

Both arms start from the same `campaign-failure-20261125/runs/lstm-ordinary-s2`
checkpoint and retain its actor/critic weights and Adam moments. Seed 2 was
selected because it reached the forest in 8/10 inspected parent episodes (versus
4/10 and 7/10 for seeds 0 and 1), providing more exposure to the failures of
interest. This is a selected development pilot, not a three-seed replicated result.

Each arm adds **262,144 ordinary-route actions**, with the parent's PPO settings,
episode limit, original post-parcel starts, rewards and rules unchanged. Both
have the same new linear prediction head on the actor LSTM's output:

| Arm | Prediction head training | Effect on actor representation |
|---|---|---|
| Control | RAM-supervised readout of detached LSTM features | None from prediction loss; ordinary PPO updates continue |
| Auxiliary | Identical readout and targets | Prediction loss also trains the visual encoder and actor LSTM |

Targets are current overworld/wild/trainer context, whether a living party member
is poisoned, total party HP fraction, and whether an antidote is available.
The combined auxiliary loss has weight 0.1. Categorical and binary losses average
over classes present in each minibatch; HP uses sigmoid mean-squared error.
Padding is excluded. Original network and readout gradients are clipped separately
so the detached control's readout cannot change its PPO gradient scaling.

The actor still receives **only screens and recent actions**. Labels are attached
to training `info`, sampled before the action, and stored in a separate rollout
buffer field. The head's predictions are diagnostics, not additional actor inputs.
There are no action targets, scripted battle actions, action masks, new rewards,
or failure-state training resets.

## What will be measured

1. **Six naturally reached diagnostic states:** the first two eligible entries
   from different episodes for each of wild battle, forest trainer battle and
   walking poison with an antidote. At least 2,048 original actions remain.
   The suite replays the selected parent's existing traces exactly and preserves
   emulator RNG, inventory, rule/reward history and remaining horizon. Each
   LSTM receives the entire recorded within-episode pixel/action history before
   choosing new actions. The old buttons are not executed during evaluation or
   used as imitation targets.
2. **Three stochastic branches per state/model:** up to 2,048 new actions.
   Trainer success requires a newly set forest trainer event flag. Poison success
   requires a living poison clear matched to antidote consumption within two
   emulator frames, without healing or revival; in-battle cures also count.
   Wild success requires the game's actual successful RUN hook. Rejected trainer
   RUN selections, faints and state-prediction confusion matrices are recorded.
3. **Original-route transfer:** six additional episodes per parent/control/auxiliary
   checkpoint, using evaluation stream 6100–6105. Report forest arrivals, verified
   trainer wins, poison deaths with inventory, cures, Pewter/Brock progress, and
   wild escape. These evaluations are separate from the selected diagnostic states.

The queue has two training jobs, three diagnostic evaluations (54 branches total),
three route evaluations (18 episodes total), suite construction, and a report.
It stops after that comparison; no additional seeds or optimizer sweep are queued.
The experiment cutoff remains **November 29 at 8:59 p.m. Pacific**.

Better state-prediction accuracy alone is not a successful intervention. We need
better combat/recovery behavior than the control while retaining route navigation
and wild escape. Two states per task with repeated stochastic branches are not six
independent states. Any promising result needs replication; any failure should be
interpreted against the prediction and behavior measurements before further work.

## Validation and execution

The focused local checks passed: 28 tests across context supervision, cure
measurement and snapshot restoration. They cover pre-action target alignment,
auto-reset alignment, recurrent shuffle/padding alignment, prediction-gradient
isolation, equivalence of the detached control to an upstream PPO update with
existing Adam state, inference/training checkpoint reload, and batched recurrent
history warmup versus stepwise memory.

Entry points:

- `project_work/v2/context_experiment.py`: matched training and provenance.
- `scripts/context_benchmark.py`: natural suite reconstruction and branch scoring.
- `scripts/context_campaign.py`: finite queue and comparison report.
- `scripts/diagnose_forest.py --context-diagnostic`: original-route transfer and
  state-prediction diagnostics.

Run status and launch evidence will be recorded beside this document. All remote
source snapshots, cases, checkpoints and logs live on the attached persistent
filesystem. The earlier optimizer sweep remains paused.
