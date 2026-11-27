# Route retention and earlier failure replay

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

This is a hypothesis test, not a demonstrated fix. Tail versus transition compares
anchor timing under the same fixed worker allocation and context stratification.
Ordinary versus either mixed arm changes several aspects of reset sampling.
Three training seeds are still a small sample. A battle exit is recorded as a
completed encounter, not a separately verified trainer victory.

The finite queue is built by `scripts/build_route_replay_queue.py`, reports through
`scripts/report_route_replay.py`, and stops training/evaluation by
**November 29, 8:59 p.m. Pacific**. Checkpoints, snapshots, logs and videos live on
the attached persistent filesystem; a separate Mac backup mirrors the campaign.
