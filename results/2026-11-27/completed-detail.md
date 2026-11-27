# Route retention and transition replay

Fresh original-start evaluation only. Forest means arrival, not exit. Completed trainer battle means observed battle exit, not a separately verified win. RUN rates require battle exposure; a lower rate with fewer trainer entries is not sufficient. Compare matched training seeds, progress and deaths jointly. Parent probes reuse checkpoints, not new training seeds. RAM is privileged reset-selection information only; actor inputs/rewards unchanged.

Complete evaluated models: 24/24. Missing: none.

| Model / starts | Episodes | Forest | Pewter | Brock | Deaths | Trainer timeouts | Rejected RUN | RUN / 1k trainer actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lstm/ordinary | 30 | 17 | 0 | 0 | 2 | 2 | 309 | 20.04 |
| lstm/parent | 30 | 19 | 0 | 0 | 17 | 4 | 390 | 7.48 |
| lstm/tail | 30 | 17 | 0 | 0 | 10 | 4 | 196 | 6.79 |
| lstm/transition | 30 | 7 | 0 | 0 | 14 | 1 | 37 | 6.92 |
| ppo/ordinary | 30 | 24 | 0 | 0 | 8 | 4 | 1707 | 27.01 |
| ppo/parent | 30 | 26 | 0 | 0 | 20 | 2 | 945 | 15.81 |
| ppo/tail | 30 | 4 | 0 | 0 | 6 | 0 | 115 | 30.15 |
| ppo/transition | 30 | 18 | 0 | 0 | 11 | 7 | 709 | 13.32 |

## Matched seeds

| Run | Forest | Pewter | Brock | Trainer entries | Trainer timeouts | Original-start action share |
|---|---:|---:|---:|---:|---:|---:|
| ppo-parent-s0 | 7 | 0 | 0 | 6 | 0 | parent |
| ppo-ordinary-s0 | 6 | 0 | 0 | 6 | 4 | 100.0% |
| ppo-tail-s0 | 1 | 0 | 0 | 0 | 0 | 58.3% |
| ppo-transition-s0 | 7 | 0 | 0 | 7 | 7 | 58.4% |
| lstm-parent-s0 | 4 | 0 | 0 | 4 | 0 | parent |
| lstm-ordinary-s0 | 5 | 0 | 0 | 0 | 0 | 100.0% |
| lstm-tail-s0 | 2 | 0 | 0 | 1 | 1 | 58.5% |
| lstm-transition-s0 | 2 | 0 | 0 | 1 | 1 | 58.9% |
| ppo-parent-s1 | 10 | 0 | 0 | 9 | 2 | parent |
| ppo-ordinary-s1 | 8 | 0 | 0 | 0 | 0 | 100.0% |
| ppo-tail-s1 | 0 | 0 | 0 | 0 | 0 | 56.5% |
| ppo-transition-s1 | 9 | 0 | 0 | 5 | 0 | 56.9% |
| lstm-parent-s1 | 7 | 0 | 0 | 7 | 2 | parent |
| lstm-ordinary-s1 | 9 | 0 | 0 | 3 | 2 | 100.0% |
| lstm-tail-s1 | 8 | 0 | 0 | 3 | 3 | 58.2% |
| lstm-transition-s1 | 5 | 0 | 0 | 3 | 0 | 57.7% |
| ppo-parent-s2 | 9 | 0 | 0 | 5 | 0 | parent |
| ppo-ordinary-s2 | 10 | 0 | 0 | 13 | 0 | 100.0% |
| ppo-tail-s2 | 3 | 0 | 0 | 3 | 0 | 58.4% |
| ppo-transition-s2 | 2 | 0 | 0 | 2 | 0 | 58.8% |
| lstm-parent-s2 | 8 | 0 | 0 | 11 | 2 | parent |
| lstm-ordinary-s2 | 3 | 0 | 0 | 2 | 0 | 100.0% |
| lstm-tail-s2 | 7 | 0 | 0 | 11 | 0 | 57.6% |
| lstm-transition-s2 | 0 | 0 | 0 | 0 | 0 | 57.3% |
