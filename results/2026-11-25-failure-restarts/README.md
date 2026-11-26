# Failure-driven restarts with the existing route reward

## Antidote acquisition: what we actually know

The original post-parcel save supplies two antidotes and a level-6 Squirtle. This
is setup inventory, not demonstrated purchasing behavior. One recorded pixel-PPO
trajectory collected a third antidote in Viridian Forest through its own actions
(action 6,797, next to the hidden pickup). No purchasing skill is established by
this audit. See the [inventory evidence](../2026-11-25-antidote/README.md).

Scripts prepared the cure task's menu-start variants and verified a legal cure;
they did not provide actions during policy training. The additional +1 antidote
reward and curated poison starts make that task a diagnostic intervention. It
should not be presented as general end-to-end learning.

## General experiment

Save emulator and environment snapshots during ordinary play. If an episode ends
in a wipe, rule violation, or unsuccessful time limit, retain up to four snapshots
from 64–1,024 actions before that ending. Selection does not inspect poison, item
inventory, map ID, or failure subtype. Each worker keeps eight failed episodes.

After its bank has entries, the treatment samples 50% original starts and 50%
failure snapshots. The current policy then takes **new actions**, using the
unchanged original reward. This is start-state sampling, not repeatedly optimizing
old action/reward traces. PPO alternates fresh policy interaction and optimization;
see [Schulman et al.](https://arxiv.org/abs/1707.06347). Changing the training start
distribution has precedent in [curriculum work](https://proceedings.mlr.press/v78/florensa17a.html),
but this failure-driven sampler is not an implementation of that paper's
goal-outward reverse curriculum, and no novelty claim is made.

The control also records failure snapshots but always starts from the original
save. Thus the reward, actor inputs, architecture, optimizer, training budget, and
logging overhead match; only reset sampling differs. Neither arm receives an
antidote bonus or scripted item-use actions. Both start from the original pixel
baseline checkpoint, never from the cure-fine-tuned model.

The existing route reward still uses events, healing, badges, exploration,
revisits, and privileged navigation shaping. We have **not** removed that shaping
or demonstrated learning from only sparse game success. The new claim being tested
is whether automatic failure starts improve learning under a fixed reward.

## State and evaluation contract

Snapshots preserve the emulator, pending button inputs, frame/action history,
rule monitor (deaths, encounter eligibility and used regions), coherent position,
visited tiles, reward accumulators, navigation potential, milestones and remaining
episode allowance. Restarts do not erase a recorded death, renew catch eligibility,
make visited tiles new, or refill inventory. Only logging totals start anew.
Snapshots are captured after ordinary actions; reset-time captures were excluded
after a PyBoy initialization rendering mismatch was found. Recurrent hidden state
resets at every new episode; this is a reset curriculum, not exact continuation
of a policy's past internal state. Simulator save/restore access is a training
assumption and does not add RAM fields to actor observations.

Twelve paired runs: PPO/LSTM × ordinary/failure starts × seeds 0/1/2. Each continues
the corresponding 10,485,760-action baseline for 1,048,576 fresh actions, using its
original optimizer and training hyperparameters, 16 workers and 16,384-action
original horizon. A restart inherits the original remaining allowance. The bank
starts empty, so early episodes are ordinary starts in both arms.

Each final policy is evaluated for ten stochastic episodes from the original
post-parcel save (seeds 4000–4009), with no failure resets. Training returns and
success rates under different reset distributions must not be compared as route
performance. Evaluate forest/Pewter/Brock progress, deaths, rule violations and
timeouts from these original-start episodes. Near-failure states may already be
unrecoverable, and sampling them does not guarantee the policy will learn to cure
poison or purchase supplies. The current sampler uses a fixed backward window;
expanding it based on recovery outcomes is a later possible ablation.

## Completed validation

- All 72 repository tests pass, including restoration of reward/rule history,
  pending inputs, unchanged ordinary-arm rewards, bounded banks, hash checks,
  terminal-state rejection, and wipe/violation/time-limit selection.
- Real-emulator branches match exactly at six route positions, including a trainer
  battle: 96 repeated actions with identical observations, WRAM, rewards,
  termination flags and info. See [roundtrip.json](roundtrip.json).
- PPO and LSTM each completed a 1,024-action real-emulator smoke training run with
  two workers and actual failure resets, then checkpoint reload and two
  original-start evaluation episodes. The smoke uses a shortened 128-action
  horizon and is not performance evidence. See [training-smoke.json](training-smoke.json).
- Ruff and `git diff --check` pass.

## Queue and storage

The previous poison queue has `STOP_AFTER_CURRENT`. Its one active PPO half-size
run finishes and receives its planned initial/final evaluation. The remaining
poison-resolution grid is not launched. Those diagnostic results remain separate.

`scripts/build_failure_queue.py` builds the new finite queue: a wait for the old
controller, one diagnostic evaluation, then twelve train/evaluate pairs. It retains
the November 29, 8:59 p.m. Pacific cutoff. The controller requires the persistent
mount, checkpoints every 262,144 actions, and retains failure banks there too.

Persistent root: `/lambda/nfs/jeff-67920-rl-checkpoints/67920project`.
Source: `source-failure-20261125`; campaign: `runs/campaign-failure-20261125`;
tmux: `67920-failure`. The second Mac backup is `runs/failure-backup-20261125`,
updated every three minutes while its process/network remain available.
The controller launched at **4:27 p.m. Pacific, November 25**, using clean frozen
source commit `0655df9`. It is waiting for the active cure diagnostic to finish;
the paired training jobs have not started at this snapshot. The first new-campaign
Mac backup succeeded. See [launch.json](launch.json) and the exact [queue.json](queue.json).
All implementation tests are complete; full-budget training/evaluation outcomes
are still pending. No failure-versus-control performance result is claimed here.
