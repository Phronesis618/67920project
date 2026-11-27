# Route retention and earlier failure replay

**Latest: [failure-mechanism audit](failure-analysis/README.md).** The follow-up
sweep is paused after one in-flight run. A controlled branch wins a naturally
stalled trainer encounter in 290 actions; the original policy times out after
12,995 more actions and 441 rejected RUN selections. See the audit for limits,
poison/navigation findings and verified reward behavior.

**Update: the campaign completed November 27 at 9:35 a.m. Pacific.** All 18 training
runs and 240 evaluation episodes finished; none reached Pewter/Brock. See the
[completed results and next comparison](completed.md). The launch details below
describe the original setup.

**Launched November 26, 10:27 p.m. Pacific (November 27, 06:27 UTC).** The first
job is `train-lstm-transition-s0`. The finite queue has 18 training jobs, 24
ten-episode evaluations, and one report job. The source snapshot is commit
`87a258d`; [launch.json](launch.json) records the observed running state and CUDA
checks. This is launch evidence, not a claim of improved game performance.

The previous comparison did not clear the forest. This follow-up tests whether
practice starts too late and displaces ordinary route training. It changes the
training-start distribution, not the reward or policy action set.

One concrete stalled episode is LSTM failure seed 2, evaluation seed 4001:
trainer battle from action 6,492 through 16,384 (9,893 actions), Squirtle HP
22/22 throughout, 71 rejected RUN selections, ending at the time limit.
This is not evidence that poison or low HP causes every forest failure. Other
recorded episodes beat the first trainer, lose in combat, or faint from walking
poison. The [previous report](../2026-11-26/README.md) separates those outcomes.

## Comparison

PPO and LSTM, three matched seeds, three continuation arms:

| Arm | Full-route workers | Replay workers | Replay anchors |
|---|---:|---:|---|
| Ordinary | 16 | 0 | None |
| Tail + route practice | 8 | 8 | 64–1,024 actions before a failure |
| Transition + route practice | 8 | 8 | After map/battle-context transitions anywhere in a failed episode |

All arms start from the corresponding **ordinary-training** checkpoint from the
completed comparison, at 11,534,336 actions, and add 1,048,576 actions. No curated
antidote checkpoint is used. Only full-route workers publish replay states; replay
branches never populate the shared bank. Empty-bank replay workers fall back to
ordinary starts. Half the collected actions in either mixed arm therefore come
from dedicated original-start workers, with additional original actions during
warmup. This differs from a 50% episode-reset probability.

Both replay arms sample uniformly among available overworld, wild-battle, and
trainer-battle contexts, then uniformly among their anchors. This is **privileged
RAM-based reset selection**, not an unprivileged curriculum. Pixel/action actor
inputs and the existing RAM-based reward remain unchanged. There is no new RUN
penalty, antidote reward, combat script, or action mask.

Transition snapshots include the first action and changes of battle context or
coherent map identity. A bounded reservoir retains up to eight candidates per
context; a failed original episode publishes at most two per context, at least
64 actions before its end. Sampling retains the latest eight failed episodes per
route worker. Full emulator state, rules, reward history, frame stack, pending
buttons, and original remaining episode allowance are restored. LSTM hidden state
resets. Blob files remain immutable for concurrent readers and auditing.

## Evaluation and limits

Each of the 18 continuations and six parent checkpoints receives ten fresh
original-start evaluations, seeds 6000–6009: **240 episodes total**. Per-action
traces, battle RUN hooks, faint causes, and one tail video per model accompany
the route outcomes. No training-reset evaluations are substituted for full-route
results. Report trainer RUN rates alongside encounter exposure, trainer timeouts,
forest arrival, Pewter/Brock, and deaths; avoiding encounters is not success.
The diagnostic script's legacy `original_evaluation_match` field is not applicable
to this fresh seed range; no prior seed-6000 evaluation exists to match. The
campaign report uses these new episode outcomes directly.

This is a hypothesis test, not a demonstrated fix. Tail versus transition compares
anchor timing under the same fixed worker allocation and context stratification.
Ordinary versus either mixed arm changes several aspects of reset sampling.
Three training seeds are still a small sample. A battle exit is recorded as a
completed encounter, not a separately verified trainer victory.

The finite queue is built by `scripts/build_route_replay_queue.py`, reports through
`scripts/report_route_replay.py`, and stops training/evaluation by
**November 29, 8:59 p.m. Pacific**. Checkpoints, snapshots, logs and videos live on
the attached persistent filesystem; a separate Mac backup mirrors the campaign.

## Validation and storage

All 82 tests and lint checks passed. PPO and LSTM each passed both CPU and CUDA
integration runs, including shared-bank replay. A real-game paired check matched
10,832 actions against an unwrapped environment, then restored a trainer-entry
anchor 1,115 actions before the failure and matched the next 32 actions' pixels,
WRAM, rewards and termination. [validation.json](validation.json) retains these
checks and their scope; they test implementation, not learned performance.

Remote campaign:
`/lambda/nfs/jeff-67920-rl-checkpoints/67920project/runs/campaign-route-replay-20261127`.
The first Mac backup succeeded at **06:27:40 UTC**, and the backup process is set
to repeat every three minutes while this Mac is connected and awake. Persistent
filesystem saves are independent of the Mac backup. The local mirror is
`runs/route-replay-backup-20261127`.
