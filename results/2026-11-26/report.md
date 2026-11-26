# Failure-restart comparison

Original-start evaluations only; each dot/row is a training seed. Missing runs are not treated as zero success.

| Phase / algorithm / arm | Seeds | Forest | Pewter | Brock | Wipes |
|---|---:|---:|---:|---:|---:|
| campaign-failure-20261125 / lstm / failure-w1024 | 3 | 17/30 | 0/30 | 0/30 | 6 |
| campaign-failure-20261125 / lstm / ordinary | 3 | 23/30 | 0/30 | 0/30 | 17 |
| campaign-failure-20261125 / ppo / failure-w1024 | 3 | 26/30 | 0/30 | 0/30 | 11 |
| campaign-failure-20261125 / ppo / ordinary | 3 | 23/30 | 0/30 | 0/30 | 18 |

## Interpretation and provenance

Two initial antidotes and Squirtle are supplied by the start save. Learned purchasing is unproven. Actor inputs are screens/actions; existing RAM-based route reward and navigation shaping remain. Failure restarts use simulator checkpoints; they add no antidote bonus. LSTM state resets at episode boundaries.

Training curves describe the training start distribution and are not full-route evaluation curves. Diagnostic replays reuse evaluation trajectories and are not new independent trials. Only matching replays contribute to cause counts. Three training seeds and small evaluations limit precision; exploratory follow-ups are selected using the first comparison and evaluated with fresh seeds.

Completed run count: 12. Incomplete run folders: {"/lambda/nfs/jeff-67920-rl-checkpoints/67920project/runs/campaign-failure-20261125": []}.

Exact manifests, hashes and episode records: `evidence.json`. Per-seed counts: `per-seed.csv`. Videos and failure counts: `diagnostics.json`.

## Follow-up decision

{
  "action": "report_only",
  "evidence": [
    {
      "algorithm": "ppo",
      "deltas": [
        [
          0,
          0,
          0
        ],
        [
          0,
          0,
          0
        ],
        [
          0,
          0,
          3
        ]
      ],
      "aggregate": [
        0,
        0,
        3
      ],
      "positive_seeds": 1,
      "retry_stats": {
        "eligible_restarts": 80,
        "failed_by_previous_failure_time": 22
      }
    },
    {
      "algorithm": "lstm",
      "deltas": [
        [
          0,
          0,
          -3
        ],
        [
          0,
          0,
          -1
        ],
        [
          0,
          0,
          -2
        ]
      ],
      "aggregate": [
        0,
        0,
        -6
      ],
      "positive_seeds": 0,
      "retry_stats": {
        "eligible_restarts": 70,
        "failed_by_previous_failure_time": 12
      }
    }
  ],
  "reason": "No preregistered follow-up signal, or incomplete/mismatched seed pairs. Preserve the report and traces for interpretation; do not launch an arbitrary treatment."
}
