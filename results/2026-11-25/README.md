# Completed baseline campaign — November 25, 2026

Checked November 25 at 11:07 a.m. Pacific (19:07 UTC). The corrected campaign and
its recovery completed the original 72-job plan: 42 historical completed jobs plus
30 successful recovery jobs. There are no failed recovery jobs. All 12 configurations
completed 10,485,760 training actions: pixels/RAM × PPO/LSTM × training seeds 0–2.
The recovery queue finished November 24 at 2:41:58 a.m. Pacific (10:41:58 UTC).
Its final automatic Mac backup succeeded at 2:45:20 a.m. Pacific. Persistent storage
still holds all checkpoints. The archived corrected status still says `running`;
it describes the old terminated instance, not a live job.

## Final-checkpoint evaluation

Each row includes three training seeds and ten stochastic episodes per seed.

| Inputs | Policy | Forest arrivals | Pewter arrivals | Brock wins | Timeouts | Wipes | Illegal catches |
|---|---|---:|---:|---:|---:|---:|---:|
| pixels | PPO | 20/30 | 0/30 | 0/30 | 18 | 7 | 5 |
| pixels | LSTM | 10/30 | 0/30 | 0/30 | 12 | 13 | 5 |
| ram | PPO | 17/30 | 0/30 | 0/30 | 19 | 6 | 5 |
| ram | LSTM | 11/30 | 0/30 | 0/30 | 11 | 5 | 14 |

Pixel PPO has the highest observed forest-arrival rate in this final snapshot.
Three training seeds and no evaluated forest exits do not establish a general
architecture winner. These results show a navigation/survival bottleneck, not
successful end-to-end Nuzlocke play.

Across all three checkpoint stages, there are **216 evaluation episodes: 90 forest
arrivals, zero Pewter arrivals and zero Brock wins**. Endings: 114 timeouts, 49 wipes,
53 illegal catches. Earlier stages are repeated measurements of the same training
seeds, not additional independent runs. The earlier 17 diagnostic replays are not
added to these totals.

## Two training episodes reached Pewter

There are 10,408 logged completed training episodes. Two episodes from
`ram-ppo-s1-m10` finish in Pewter (map 2), with zero deaths and no rule violations;
both time out. Their raw and coherent end positions agree: `(32,10,2)` and
`(34,8,2)`. No training episode records a Brock win. These are rare training
trajectories, not held-out performance, and no video of these two trajectories
was recorded.

The [two original records](pewter_training_episodes.json) identify workers 1 and 11
and episode indices 17 and 15. Summing per-worker episode lengths and multiplying
by 16 synchronous environments places their endings at global steps 8,094,976 and
8,413,488. Episodes span policy updates, so neither trajectory proves a single
saved checkpoint would reproduce it, or that later training forgot the behavior.

## Follow-up now running

The completed queue left the A10 idle. On November 25 at 11:09 a.m. Pacific,
`campaign-checkpoint-probe-20261125` was started in tmux session
`67920-checkpoint-probe`. It evaluates the RAM PPO seed-1 policies saved at
**7,864,320; 8,126,464; 8,388,608; and 10,485,760** training actions, sequentially.
Each gets 20 stochastic episodes with evaluation seed initialization 2000,
for **80 additional episodes**. The final model provides a comparison against the
intermediate snapshots. This is an exploratory checkpoint search; any selected
model needs a separate held-out evaluation before reporting a selection-adjusted
performance claim.

The [exact queue](checkpoint_probe_queue.json) records checkpoint SHA-256 hashes,
commands and budgets. It uses the frozen source and original environment/reward
configuration. Outputs and logs are under the persistent project directory at
`runs/campaign-checkpoint-probe-20261125/`, including `evaluations/`. The queue
requires the persistent mount and keeps the original November 29, 8:59 p.m. Pacific
training cutoff (November 30, 04:59 UTC). Automatic Mac backups were restarted for
this queue. No new survival, antidote or auxiliary-prediction training treatment
has been launched.

The [November 23 diagnosis](../2026-11-23/README.md) still supplies the direct
behavioral evidence: rejected trainer RUN attempts, trainer/wild-battle faints,
walking-poison deaths and delayed forest arrival. The starting save contains five
Poké Balls and two antidotes. Inventory consumption before poison deaths remains
unmeasured by those traces. Prioritize a controlled forest/trainer or survival
intervention over simply extending unchanged training, within the remaining
November 29 compute window. The report deadline is December 10 at 2 p.m. Pacific
(5 p.m. Eastern).

## Files and counting

[Progress](progress.json), [individual evaluations](evaluations.json),
[manifests](manifests.json), [training summaries](training.json), and
[archived queue status](queue_status.json) preserve campaign provenance.
`completed_segment_actions` excludes the interrupted segment's unfinished work;
it is not the sum of final model timestep counters. Episode summaries include
completed episodes within that interrupted segment.
