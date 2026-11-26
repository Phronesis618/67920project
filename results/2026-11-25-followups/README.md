# Queued analysis, replay and conditional curriculum refinement

The active failure/ordinary-start comparison stays immutable. A separate controller
waits for it to finish, fail, or reach its deadline, then performs the following work.
This is a finite pipeline, not an unlimited experiment search.

1. Export original-start evaluation counts for every completed training seed:
   forest, Pewter, Brock, wipes, timeouts, deaths and violations. Include exact
   manifests, source hashes, queue status and missing-run labels.
2. Produce PNG/SVG outcome plots and diagnostic training curves. Training curves
   are explicitly labeled as measuring different training-start distributions.
3. Select at most four completed policies (one per architecture/arm). Replay their
   ten evaluation episodes in the original order. Save per-action position,
   battle, party and antidote inventory; frame-level faint causes and actual RUN
   rejection hooks. Export up to two selected tail videos per policy, at most the
   final 600 actions each. Selection favors progress and severe failures and is
   illustrative, not a new unbiased sample. Replays are not independent trials.
   Only replays matching original end state and return contribute to cause counts;
   this check does not independently prove every intermediate state matched.
4. Apply the declared follow-up rules below. No reward or actor input changes.
5. Assemble the final evidence package, including a report, per-seed CSV, plots,
   trace/video index and presentation outline. Partial results still get a report.

## Bounded follow-up rules

An architecture must have all three complete, comparable seed pairs: matching
parents, assets, configuration, clean source revision, action budgets, and the
same ten stochastic evaluation seeds (4000–4009). Missing runs are not zeros.

- **Extend a promising comparison:** require better original-start progress in
  at least two of three seed pairs and in aggregate. Progress is compared in
  priority order: Brock, Pewter, forest. Select at most one architecture, then
  continue its ordinary/failure arms for another 1,048,576 actions per seed.
- **Test earlier starts:** if the progress criterion is absent, require at least
  30 restarts originating from wipes/rule violations, with at least 80% failing
  again by the source failure time. Compare lookback windows of 1,024 and 4,096
  actions for one architecture, with three seeds and 1,048,576 actions per arm.
  Each pair starts from the same failure-policy checkpoint. This is a test of
  the "too late to recover" hypothesis, not proof of irrecoverability.
- **Otherwise:** generate the report and traces without launching an arbitrary
  new treatment. These gates allocate compute; they are not significance tests.

Either training branch is capped at six runs and six final original-start
evaluations, using fresh evaluation seeds 5000–5009. Checkpoint selection and
follow-up choice remain exploratory. The bank starts empty in each continuation;
only new policy rollouts contribute training data.

At least eight hours must remain before the existing **November 29, 8:59 p.m.
Pacific** cutoff to start a training branch. All training and emulator jobs obey
that cutoff. CPU-only reporting can finish afterward. An explicit stop marker on
the primary campaign also prevents downstream GPU/emulator work. If the primary
does not confirm shutdown after the deadline, only reporting proceeds.

## Implementation and verification

- `scripts/run_failure_followups.py`: persistent controller and bounded child queue.
- `scripts/failure_analysis.py`: matching rules, decisions, reports and figures.
- `scripts/diagnose_forest.py`: ordered diagnostic replay and selected tail videos.
- `project_work/v2/failure_experiment.py`: optional 4,096-action window; default unchanged.
- All 77 repository tests pass, including incomplete/mismatched data, selection
  gates, equal-parent window pairs, fresh evaluation seeds and report generation.

The separate source snapshot is `source-followups-20261125` under
`/lambda/nfs/jeff-67920-rl-checkpoints/67920project`. Outputs live in
`runs/campaign-followups-20261125`; tmux session is `67920-followups`.
The Mac copy is `runs/followups-backup-20261125`, pulled every three minutes while
the backup process/network remain available. `launch.json` records actual launch
and video validation status. This plan does not claim the future experiments have
already completed or that a conditional branch is guaranteed to run.

The controller launched **November 25 at 8:17 p.m. Pacific**, from clean frozen
source `f58dd41`, and is waiting for the primary comparison. The first Mac backup
succeeded. Lambda validation produced the selected 160×144, 128-frame tail video,
with both short replays matching a same-host CPU reference. A separate controller
smoke test verified report generation after cutoff with incomplete primary data.
The copied Mac reference diverged on Lambda and was excluded; production replays
use the same host/device as their evaluation and still verify the match explicitly.
See [launch.json](launch.json) for this deployment snapshot.
