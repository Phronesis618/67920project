# Interim results — November 27, 12:56 a.m. Pacific

The queue is healthy: three of 18 training continuations and four of 24 evaluation
sets have completed, with no job failures. PPO transition seed 0 is training.
This is **one LSTM training seed and 40 evaluation episodes**, not the final
comparison. All four models were evaluated from the original save on seeds
6000–6009. Forest means arrival, not exit.

| LSTM seed 0 | Forest arrivals | Pewter / Brock | Deaths | Timeouts | Trainer encounters |
|---|---:|---:|---:|---:|---:|
| Parent before continuation | 4/10 | 0 / 0 | 4 | 1/10 | 4 |
| Ordinary continued training | 5/10 | 0 / 0 | 0 | 10/10 | 0 |
| Tail replay + route practice | 2/10 | 0 / 0 | 1 | 8/10 | 1 |
| Transition replay + route practice | 2/10 | 0 / 0 | 0 | 7/10 | 1 |

Both replay variants have lower route progress on this seed. The ordinary model
survives but never enters a trainer battle. Fewer deaths therefore do not establish
better combat or poison handling. Each replay model encounters a trainer once and
times out in battle; rejected RUN counts are 90 for tail replay and 24 for
transition replay. That exposure is too small to establish a trainer improvement.

The sampler is working as configured: exactly half of mixed-arm training actions
come from dedicated full-route workers. Including empty-bank warmup, approximately
59% of actions start from original-route episodes. The remaining actions use the
shared failure banks; this is not a failure to activate replay.

## What needs attention

The current recipe has not demonstrated better play. Retention and stalling remain
the priorities. All three continuations use learning rate 0.0003 with no KL early
stop; their logged median approximate KL values are 0.121–0.128. A separate
comparison with smaller optimizer steps and a KL limit is a reasonable next test
of whether fine-tuning changes the policy too quickly. These logs do not prove
that optimizer size caused the regression, nor establish which settings will win.

**Decision: leave the running comparison unchanged.** Finish PPO and the other two
training seeds before interpreting this as a general effect. Do not add further
task-specific rewards or change settings mid-comparison based on one seed.

## Rule-check audit and preservation

Ten illegal-catch endings show an uninitialized new party slot during the catch
animation. The rule monitor decides catches from the ownership count and encounter
history, not that slot's identity. All ten are supported by the recorded history:
the region was either already used in the starting save or had an earlier wild
encounter. No false-catch termination was established by this audit, and no rule
change was made. This audit does not claim every possible rule-monitor edge case
has been ruled out.

Persistent saves and the Mac backup are functioning; the latest backup observed
for this report completed at 12:57:10 a.m. Pacific. The queue retains the
November 29, 8:59 p.m. Pacific cutoff.

[Exact evaluation records, optimizer summaries, sampler counts, catch audit, and
queue/backup snapshot](interim-seed0.json).
