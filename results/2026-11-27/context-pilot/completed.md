# Completed context-prediction pilot

The finite queue completed **November 27 at 7:46 p.m. Pacific**. All ten jobs
succeeded. The final Mac backup succeeded at **7:49 p.m.**; checkpoints and raw
traces also remain on the attached Lambda filesystem. No extension was launched.
Both continuations used the same parent, optimizer settings and 262,144-action
budget. [Verification and fatal encounters](completed-verification.json).

**The auxiliary arm retained route navigation in this pilot, but did not solve
combat or recovery.** The ordinary continuation lost its ability to reach the
forest. This is one selected training seed, not a replicated general result.

| Original-route evaluations | Parent | Ordinary continuation/control | Auxiliary prediction |
|---|---:|---:|---:|
| Forest arrivals | 6/6 | 0/6 | 6/6 |
| Verified forest trainer wins, total | 2 | 0 | 2 |
| Verified antidote cures, total | 1 | 0 | 1 |
| Reached Pewter / beat Brock | 0/6 | 0/6 | 0/6 |
| Wipes / timeouts / rule violations | 5 / 1 / 0 | 1 / 3 / 2 | 5 / 1 / 0 |

The starting model and auxiliary model have the same aggregate route outcomes,
but their checkpoints and trajectories differ. The auxiliary treatment has not
demonstrated better final task performance than the starting model.

| Natural-state diagnostic branches | Parent | Control | Auxiliary |
|---|---:|---:|---:|
| Trainer wins | 1/6 | 3/6 | 3/6 |
| Successful wild escapes | 6/6 | 6/6 | 6/6 |
| Antidote cures from poisoned starts | 0/6 | 0/6 | 0/6 |

Each diagnostic task uses **two states with three stochastic branches each**.
These are not six independent starting states. The auxiliary arm's trainer win
count does not exceed the control's. Its 49 rejected trainer RUN selections
versus 74 for control also occurred over fewer actions (7,571 versus 9,435), so
raw counts alone overstate the behavioral difference.

## What still causes failures

All **five auxiliary route wipes occurred during trainer battles**. The entry
states separate preparation from combat execution:

- Three fatal encounters began at level 6 with 14/22, 19/22 and 22/22 HP.
  The fully healthy case shows that low entry HP alone does not explain the losses.
- The other two began after a trainer win, at level 8 with **3/26 HP while poisoned**
  and **4/26 HP without poison**. The policy proceeds into another fight before
  recovering. Both still had antidotes in the bag, although an antidote cannot
  replace missing HP.
- Four of the five faints had poison status immediately before fainting. The
  current observer identifies that they happened in battle; it does not identify
  whether the final HP loss was an enemy attack or an in-battle poison tick.

The auxiliary head classified trainer context correctly on 87.6% of true-trainer
action observations in the diagnostic branches (control: 59.4%), but these are
correlated observations on different trajectories. This is a representation
measurement, not proof that recognition caused improved behavior.

Poison recognition remains poor: **0/508** poisoned action observations were
identified at the 0.5 threshold in the auxiliary poison-start diagnostics; HP
fraction mean absolute error there was **0.662**. On original-route trajectories,
poison recall was **229/2,550 = 9.0%**. The control encountered no poisoned states
in its failed route evaluation, so that route cannot support a poison-recognition
comparison. Every scored inventory label was "antidote available"; perfect
availability accuracy does not establish that the model tracks item depletion.

## Decision

Keep the parent and auxiliary checkpoints, and do not extend the ordinary control
that lost navigation. The intervention is worth retaining as a candidate, but
more training is not yet justified as a demonstrated cure for the failure mode.

The next diagnosis should distinguish poor attack/move/menu choices from battle
damage and poison, while checking why the policy re-enters combat at 3–4 HP.
Poison/status recognition across the transition from battle to walking also needs
attention. Any next change should preserve original-route navigation and be
assessed on these behaviors, not just prediction accuracy or shaped return.

Evidence: [complete comparison](completed.json), [parent route](completed-evidence/route/parent.json),
[control route](completed-evidence/route/control.json), [auxiliary route](completed-evidence/route/auxiliary.json),
and the corresponding [parent](completed-evidence/cases/parent.json),
[control](completed-evidence/cases/control.json), and
[auxiliary](completed-evidence/cases/auxiliary.json) natural-state diagnostics.
