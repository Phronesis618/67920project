# Completed route replay comparison

All 43 jobs succeeded: 18 training continuations, 24 ten-episode evaluations
(including six parent models), and the report. The queue finished **November 27
at 9:35 a.m. Pacific**; its final Mac backup completed at **9:42 a.m.** Models,
replay snapshots, traces and videos remain on the attached persistent filesystem.

**No evaluation episode reached Pewter or beat Brock (0/240).** Results below
combine three training seeds with ten original-start episodes each. Forest means
arrival, not exit.

| Training | PPO forest arrivals | LSTM forest arrivals |
|---|---:|---:|
| Parent before continuation | 26/30 | 19/30 |
| Ordinary continued training | 24/30 | 17/30 |
| Tail replay + route practice | 4/30 | 17/30 |
| Transition replay + route practice | 18/30 | 7/30 |

PPO tail replay is worse than ordinary continuation in all three pairs
(-5, -8, -7 forest arrivals). LSTM transition replay is also worse in all three
pairs (-3, -4, -3). PPO transition replay changes arrivals by +1, +1, -8; LSTM
tail replay by -3, -1, +4. The latter two comparisons are inconsistent across
seeds. This does not support expanding either replay recipe as a route solution.

Ordinary continuation itself reduces deaths from 37 to 10 across the two
architectures, while timeouts rise from 16 to 43 (60 episodes per condition).
Trainer encounters decline from 42 to 24. Those changes do not demonstrate better
combat or poison treatment. Trainer RUN loops persist: PPO transition seed 0
has seven trainer encounters, all ending in trainer-battle timeouts.

The fixed worker allocation was honored. Replayed training actions are additional
fresh policy actions from saved states; exactly half of mixed-arm actions come
from dedicated original-route workers. Including bank warmup, 56.5–58.9% of mixed
arm actions belong to original-start episodes. Preserving that share did not
prevent the observed navigation regression.

## Next bounded comparison

**Update: paused November 27 at 10:53 a.m. Pacific** after the first in-flight PPO
training job completed. Its evaluations and the other five training runs have
not started. The user requested diagnosis before more sweeping experiments;
see the [failure-mechanism audit](failure-analysis/README.md).

**Launched November 27 at 10:37 a.m. Pacific**, starting with PPO seed 0.
The immutable training source is `9ae1c5d`; the queue contains six training jobs,
twelve checkpoint evaluations, and one report. [Launch evidence](conservative-launch.json)
records the live optimizer settings and first successful Mac backup at 10:37:39.
All 82 tests, CPU and CUDA smoke training for both algorithms, and explicit
earlier-checkpoint evaluation checks passed; these validate implementation, not
learned performance. [Local validation details](conservative-validation.json).

Test a **conservative optimizer setting on ordinary route training only**:
PPO/LSTM × three seeds, starting from the same earlier parent checkpoints, each
adding 1,048,576 actions. Reduce learning rate from 0.0003 to 0.00003 and enable
KL early stopping with target 0.02. Retain optimizer moments, other training
settings, original rewards, screen/action inputs and original reset distribution.
No replay, combat macros, action masking or item-specific rewards are introduced.

Evaluate predetermined checkpoints after 262,144 and 1,048,576 added actions,
ten original-start episodes each: 120 new episodes. Compare against the existing
matched parent and ordinary-control evaluations. This tests retention and progress
at both durations; it does not assume the final checkpoint is best.

This is an exploratory comparison using the already-inspected evaluation seeds
6000–6009. Learning rate and KL early stopping change together, so an improvement
would support the combined setting rather than isolate either cause. The previous
large KL updates motivate this test but do not prove why navigation regressed.
The queue remains finite and retains the **November 29, 8:59 p.m. Pacific** cutoff.

New campaign: `/lambda/nfs/jeff-67920-rl-checkpoints/67920project/runs/campaign-conservative-20261127`.
Local mirror: `runs/conservative-backup-20261127`, copied every three minutes while
the Mac remains connected and awake. Persistent saves do not depend on the Mac.

[Per-seed report](completed-detail.md) · [Exact counts, episode outcomes,
manifests, training allocation and completion/backup records](completed.json).
