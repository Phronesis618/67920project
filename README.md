# Pokémon Red Nuzlocke — 6.7920

Jeffrey Li and Kevin Li. Target: Pallet Town through Brock, including navigation.
`previous_work/` preserves the imported Whidden baseline. New experiments live in
`project_work/v2/`; the ROM-free mechanism pilot lives in `battle_benchmark/`.
The old `TIMELINE.md` is historical planning, not a record of completed work.

## Antidote curriculum and resolution follow-up — November 25, 2026

The [curriculum experiment](results/2026-11-25-curriculum/README.md) adds an
outside-battle antidote task using natural poisoned saves, plus a capacity-matched
half/native-resolution comparison. The 12 fine-tunes pair PPO/LSTM and three seeds;
initial/final evaluations include held-out HP/location saves and full-route transfer.
The queue launched at 4:09 p.m. Pacific on November 25; PPO half-resolution seed 0
is running, with the remaining train/evaluate jobs queued on persistent storage.
See its launch record for the snapshot. All 66 tests and real-emulator cure
checks passed; those scripted checks are not learned-policy results.

## Antidote audit and curriculum fixture — November 25, 2026

The [inventory audit](results/2026-11-25-antidote/README.md) resolves the supply question
for all three previously diagnosed walking-poison deaths: the agents still had
**3, 2, and 2 antidotes**. Five successful cures were observed, all during battles.
A natural poisoned-overworld save with level-6 Squirtle at 18/22 HP and three
antidotes is now available, with a verified scripted cure and preserved region history.
The report also compares native/current menu resolution and confirms the fixed
Squirtle starter. The follow-up above implements the curriculum/resolution treatment.

The 80-episode intermediate-checkpoint search completed with **49 forest arrivals,
zero Pewter arrivals and zero Brock wins**. Its earlier launch snapshot follows.

## Completed baseline and checkpoint follow-up — November 25, 2026

The [completed comparison](results/2026-11-25/README.md) contains all 12 configurations
at 10.49M training actions. Final-checkpoint forest arrivals are 20/30 for pixel PPO,
10/30 for pixel LSTM, 17/30 for RAM PPO and 11/30 for RAM LSTM. No evaluated episode
reaches Pewter or beats Brock. Two RAM-PPO training episodes finish alive in Pewter;
these are rare training outcomes, not reliable evaluation performance.

All 72 planned jobs finished on November 24, and their final backup succeeded.
An 80-episode evaluation of four saved RAM-PPO checkpoints around the two Pewter
training episodes subsequently completed: 49 forest arrivals, zero Pewter/Brock.
See the dated report
for exact status, counts, checkpoint hashes and the remaining November 29 cutoff.

## Forest diagnosis and current results — November 23, 2026

The [corrected/recovery snapshot and forest report](results/2026-11-23/README.md)
now include the results that were previously only on Lambda: **146 evaluation
episodes, 53 forest arrivals, zero Pewter arrivals and zero Brock wins**. All twelve
configurations completed the 5.24M-action stage; the 10.49M stage remains incomplete.
The pixel-only PPO and LSTM models already receive screens and recent actions only.

Frame-level replay diagnostics distinguish trainer/wild battles, walking poison,
illegal catches and timeouts, and count actual rejected RUN selections. They leave
the training campaign unchanged. See the report for reproducible traces, videos,
input/reward provenance and the proposed controlled follow-up.

## Persistent-storage recovery — November 23, 2026

The original A10 was terminated. The last successful Mac backup captured the
corrected campaign at **November 22, 5:18 p.m. Eastern**: 42 completed jobs,
21 completed training segments, 81 evaluation episodes, and the interrupted
`pixels-lstm-s2-m05` run. The backup contains 21 final models and 197 periodic
checkpoints. Those evaluations reached Viridian Forest 20 times, but never Pewter
or Brock; the remaining comparisons are incomplete.

The replacement uses a dedicated Virginia filesystem, `jeff-67920-rl-checkpoints`,
mounted at `/lambda/nfs/jeff-67920-rl-checkpoints`. Source, restored results, new
checkpoints, queue status and logs belong under its `67920project/` directory.
The resumed queue runs in tmux session `67920-recovery`, with new outputs in
`67920project/runs/campaign-recovery-20261123` on that filesystem. All 21 completed
models and 197 historical periodic checkpoints have been restored there.
The Python environment can remain on the instance's local disk and be rebuilt
from the saved `pyproject.toml` and `uv.lock`. Keep this filesystem when terminating
an instance, then attach it when launching another instance in the same region.

`scripts/build_recovery_queue.py` generates a fresh 30-job continuation without
rewriting the historical campaign. The interrupted run resumes weights and
optimizer state at **2,359,296 actions**, with **2,883,584 additional actions** to
reach its original 5,242,880-action target. Episodes restart and RNGs are reseeded;
this is not an exact continuation of emulator state. The other completed jobs are
preserved and the original November 29 training cutoff is retained.

Recovery queues declare `required_mount`; the controller refuses to start unless
the queue resides on that mounted filesystem, and checks again before each job.
The Mac backup remains a second copy. Training's cutoff stops the experiment queue;
it does not terminate the instance or delete the filesystem.

## Map-transition correction — November 22, 2026

The current implementation uses `coherent-v1` position reads. It holds the last
valid position while the map header is stale, the LCD is disabled for loading,
coordinates lie outside the map, or a ledge jump is in progress. It commits the
destination after the map/header/coordinates are consistent. Header identity also
guards transitions between maps with equal dimensions and tilesets. Expected
headers are generated from pinned `pret/pokered` sources.

Path rewards, exploration credit/map inputs, region checks and map-progress metrics
use the same coherent position. Health and battle rules still run every frame.
Logs retain raw and accepted positions, coherence and deferred-frame counts. The
potential reward formula and terminal/truncation handling are unchanged.

A fixed **15,150-action real-ROM replay** reproduces both door and ledge artifacts.
Maximum absolute path reward changes from **1.15725 to 0.04840**, and the false
minimum distance changes from **68 to 216**. Raw emulator trajectories and pixel
policy observations match exactly. This validates the correction on that trajectory;
it does not establish improved learning. See the [replay evidence](validation/map_transition_fix_2026-11-22.json).
Reproduce it with `python scripts/validate_map_transitions.py --output runs/map-check.json`.
Unit/integration checks and the real-ROM catch/faint/cap replays also pass.

The corrected campaign is `runs/campaign-coherent-20261122`, with frozen source at
`/home/ubuntu/experiments/67920-coherent-20261122/source` and tmux session
`67920-coherent`. It restarts the same 12 configurations from fresh policy weights,
with matched settings and the same 72-job budget/cutoff. Older campaigns and
checkpoints remain archived. Fresh runs separate corrected learning from weights
trained with transition artifacts. Manifests record position version and map-header
hashes. Evaluation/resume of historical manifests defaults to their legacy position
semantics, preserving the original experimental conditions.

## Pre-fix results — November 22, 2026

Historical snapshot: **1:14 a.m. Eastern** (06:14 UTC). At that time the Lambda queue was active:
**46/72 jobs succeeded, one running, 25 pending, zero failed**. All 12 seeds/configurations
finished their 1.05M-action stage; 11 finished 5.24M. The remaining 5.24M run was
`ram-lstm-s2-m05-train`. That campaign is now superseded by the corrected runs.

Across **91 completed evaluation episodes**, 24 reached Viridian Forest, **zero
reached Pewter City and zero defeated Brock**. These are repeated checkpoint
measurements, not 91 independent training seeds. The 5.24M comparison is still
incomplete:

| Inputs | Policy | Seeds evaluated at 5.24M | Forest episodes | Pewter / Brock |
|---|---|---:|---:|---:|
| pixels | PPO | 3/3 | 4/15 | 0 / 0 |
| pixels | LSTM | 3/3 | 5/15 | 0 / 0 |
| ram | PPO | 3/3 | 9/15 | 0 / 0 |
| ram | LSTM | 2/3 | 4/10 | 0 / 0 |

No architecture winner is established. In completed training episodes, the two
terminal failure types are wipes and forbidden catches; all recorded rule violations
are `illegal_catch`. Time limits are also common. Route 1 was deliberately marked
used in this staged benchmark, which makes catches there illegal.

**Known navigation-reward issue:** map ID and coordinates update at different times
during warps. A local replay observed `(map 50, x4, y7)` briefly become
`(map 13, x4, y7)`, giving distance 68, before settling at `(map 13, x3, y43)`,
whose distance is 226. The path reward spiked by about +0.611, then -0.619.
Therefore `closest_brock_distance` can overstate progress and the shaping signal
contains transition artifacts. This has not been established as the cause of
stalling. The correction above addresses this issue. These historical results
retain the original frozen code/configuration and are not relabeled as corrected.

- [Full progress snapshot and checkpoint hashes](results/2026-11-22/progress.json)
- [Per-episode evaluations](results/2026-11-22/evaluations.json),
  [training totals](results/2026-11-22/training.json),
  [run manifests](results/2026-11-22/manifests.json),
  [queue definition](results/2026-11-22/queue.json)
- [Map-transition diagnostic](results/2026-11-22/map_transition_diagnostic.json)
- [Download experiment checkpoints and logs](https://github.com/Phronesis618/67920project/releases/tag/experiments-2026-11-22)

The release contains the final checkpoint from every completed training segment,
plus logs and metadata for those segments. Periodic intermediate checkpoints remain
on Lambda and in local backups. Training data and checkpoints were not committed
as large Git blobs. The report deadline remains December 10, 5 p.m. Eastern;
training is capped at November 29, 11:59 p.m. Eastern.

## Current status

- Runnable PPO and recurrent PPO entry points with matching observation options.
- Separate, tested Nuzlocke monitor; terminal episode logs survive vector auto-reset.
- Checkpoint evaluation reports success, wipes, violations, time limits and coverage.
- An exact finite-horizon answer key for a **synthetic Bide timing model**.
- Tests exercise real SB3 PPO/LSTM updates and save/load using a fake emulator.
- **Real-ROM startup, observations, PPO/LSTM updates, checkpoint reload and evaluation
  passed a local smoke test on November 20, 2026.** Each policy trained for only 256
  steps, then completed three 128-step evaluation episodes. This establishes that
  the pipeline runs, not that either policy has learned the task.
- Real-ROM button replays now pass for a legal catch, an illegal catch with a region
  declared used, a faint/wipe, and battle-entry level-cap enforcement. All replay
  observations remained valid. See the [rule replay record](validation/real_rule_replay_2026-11-20.json).
  Full-party box catches, PC identity handling, revival and Brock still need real-play checks.

## Deadline and compute planning — November 20

The course instructions supplied in our project chat give a hard report deadline of
**December 10, 5 p.m. Eastern**. Presentation sessions are listed as December 8/10;
the slide deadline is inconsistent in those instructions. Use December 4 as our
internal slide-ready target until the current course announcement is confirmed.

Working priorities for the remaining 20 days:

- November 20–23: validate real Nuzlocke events, fix the experimental rule contract,
  finish training-resume validation, and prepare/test a remote job.
- November 24–29: run matched PPO/LSTM experiments and simple baselines. Decide by
  November 24 whether to use the proposal's staged-gym fallback for the main study.
  Aim for three independent seeds per learned policy; pilot first to choose a budget.
- November 30–December 3: held-out evaluation, figures, analysis and report draft.
- December 4–10: slides, rehearsal and final report. Avoid relying on new experiments
  during this period. Full-route exact solving, dialogue embeddings and broad risk
  sweeps are stretch work until the main comparison is complete.

An 8,192-step CPU timing run for each algorithm measured about 284 actions/second
for PPO and 170 for LSTM with one emulator, 128-step rollouts and four update epochs.
That extrapolates to about 1.0 and 1.6 hours per million actions respectively, but
does not establish the sample budget needed to learn the task. A six-run experiment
(two algorithms, three seeds, five million actions per run) would take approximately
39 serial hours at those rates, excluding evaluation, queueing and reruns. See the
[timing record](validation/compute_benchmark_2026-11-20.json).

Jeffrey prioritizes iteration speed over price. The instance actually launched on
November 20 is **one NVIDIA A10, 30 vCPUs, about 222 GiB RAM**, running Lambda Stack
22.04. SSH access works using the Mac's existing key. There is no attached persistent
filesystem; results are copied back to the Mac while it is awake and connected.
PPO timing with 8,192 actions measured 127 actions/s on CPU with 8 workers, 274 on
GPU with 8 workers, and 393 on GPU with 16 workers. The large LSTM CPU timing trial
was stopped to prioritize the requested training campaign. Both algorithms passed
multi-worker GPU startup, training, checkpoint reload and evaluation checks.

## Active Lambda campaign — November 20

The queue runs in remote tmux session `67920-queue`, using 16 emulator workers and
one GPU per job, sequentially. It uses a frozen source snapshot so later local edits
do not silently change a running experiment. This is an **exploratory campaign under
the current core rules**, not complete hardcore-Nuzlocke evidence.

The original queue contained 18 independent runs: PPO/LSTM × seeds 0, 1, 2 × three observation conditions:
standard `partial`, `partial` without the exploration map, and `history` with explicit
death/encounter history. All start from the pre-starter bedroom save. Each run is
queued for cumulative budgets of 1,048,576, 5,242,880, and 10,485,760 actions, with
3, 5, and 10 stochastic evaluation episodes respectively. Continuation stages depend
on successful completion of their preceding training stage. These are 54 training
segments plus 54 evaluation jobs. The bedroom PPO pilot is retained at its saved
checkpoints; the original queue is superseded by the revised campaign below.

Shared settings: 128-step rollouts per worker, batch size 256, four update epochs,
gamma 0.997, episode cap 16,384 actions, and checkpointing every 262,144 actions.
GPU/CPU/thread settings and complete commands are recorded in `queue.json`.

Remote project: `/home/ubuntu/projects/67920project`.
Remote campaign: `/home/ubuntu/projects/67920project/runs/campaign-20261120`.
Frozen source: `/home/ubuntu/experiments/67920-20261120/source`.
Local backup: `runs/lambda-backup/`, pulled every three minutes while the Mac is on.
The implementation is published on the `codex/nuzlocke-foundation` branch. The active
campaign uses the frozen snapshot taken before that commit; its manifests retain
the original Git metadata and the snapshot is preserved separately.

`status.json` identifies the active job; `controller.log` records transitions and
each job has its own log. To pause safely between jobs, create `STOP_AFTER_CURRENT`
in the remote campaign directory. Two consecutive command failures halt the queue;
failed dependencies are skipped instead of producing invalid continuation runs.
The queue's hard cutoff is **November 29, 11:59 p.m. Eastern**, leaving the planned
analysis/writing period before the December 10 report. It exits when the queued jobs
finish; it does not generate arbitrary new experiments or terminate the instance.

### Revised post-parcel campaign

Feedback identified two risks: the opening parcel quest requires long backtracking,
and RAM/event/map inputs already supply substantial state and memory. The revised
queue is in `runs/campaign-postparcel-20261120`, generated by
`scripts/build_curriculum_queue.py`. It trains 12 independent runs:
PPO/LSTM × seeds 0, 1, 2 × `pixels`/RAM-assisted.
Each run receives the same 1,048,576 → 5,242,880 → 10,485,760 cumulative budgets,
with 3/5/10 evaluation episodes. That is 72 training/evaluation jobs,
capped by the same November 29 cutoff. Pixel pilots run first within
each seed. The old campaign's unfinished jobs are retained but not automatically run.
The revised source is frozen at `/home/ubuntu/experiments/67920-postparcel-20261120/source`;
its tmux session is `67920-postparcel`. Backups follow the revised queue until it exits.

`pixels` supplies only three grayscale frames and the last three actions to both
actor and critic. The RAM-assisted control retains the original `partial` inputs.
Rules, rewards, starting state and hyperparameters match between the two conditions.
RAM still computes the training reward and rule monitor in both. This is an
architecture comparison; model capacity is not matched, and success alone does not
prove that a particular memory mechanism caused the difference.

Both conditions start at `has_pokedex_nballs.state`: Oak's lab after parcel delivery,
one healthy level-six starter, five Poké Balls and two antidotes. The inherited save's
prior encounter/death history is not established. Rules begin at reset; Route 1
(map 12) is conservatively marked used in both conditions. Treat this as a staged
benchmark, not evidence of a legal full-game run from the bedroom.

The original reward pays +2 per increase in the maximum event-bit count, +0.0125
per newly visited tile, and a small stuck adjustment after 600 visits to a tile.
Ordinary backtracking has no direct revisit penalty, but the delayed quest reward
is difficult to discover and propagate. A controlled RAM-bit audit confirms +2
for each newly set parcel/delivery/Pokédex flag and zero for repeating it. This
does not establish successful end-to-end quest learning or prevent unrelated event
bit resets from masking future progress under the count-based reward.

The revised runs add **quest-conditioned shortest-path potential shaping** and
reduce the exploration weight from 0.25 to 0.025. Both architectures use gamma
0.9995, GAE lambda 0.99, 512-step rollouts per worker, batch size 512, and four
update epochs. These are separate from the original pilot's settings; the old
pilot is not a controlled shaping ablation.

`navigation_graph.json` is generated from pinned `pret/pokered` map blocks,
collision tables, directional ledges, warps and outdoor connections. Its source
commit and input hashes are embedded. Reverse Dijkstra computes all distances
once offline; runtime uses a dictionary lookup. The graph has 4,487 walkable
positions and 14,787 directed edges. Oak-to-Brock distance is 386 approximate
tile/warp moves, through both forest gates; the shop-to-Oak return path uses
one-way ledges. It is a static navigation heuristic, not a verified planner for
moving NPCs, battle risk, healing or all game states. The supported campaign starts
after the parcel quest; earlier goal phases are not a validated pre-starter curriculum.

The goal is the shop before receiving the parcel, Oak while carrying it, and
Brock after receiving the Pokédex. The potential is
`2 * (quest_phase - min(distance, 512) / 512)`, with unknown positions using 512.
The added reward is `gamma * Phi(next) - Phi(current)`. True terminal states use
zero potential; time-limit truncations retain their bootstrap potential. The
graph's hash and shaping gamma are recorded in every new training manifest.
Distances/objectives are logged but are not observation inputs in `pixels` mode.
This is privileged reward guidance shared across both observation conditions.
Reaching Brock's position is not success; success still requires the Boulder Badge.

## Setup

Run these commands from the repository root. Python 3.11 and the dependencies are
pinned in `pyproject.toml` / `uv.lock`. Use this environment for the new experiments;
the inherited requirements files describe the old baseline.

```sh
uv sync --python 3.11 --locked
uv run pytest -q
```

Kevin's November 20 upload supplies `project_work/PokemonRed.gb`; its SHA-1 matches
the version expected by Whidden: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`.
The training command requires an explicit ROM path and records its SHA-1 and the
starting save state's SHA-1. It works from any current working directory.
No streaming/broadcast wrapper is enabled.

## First real-game smoke run

```sh
uv run python project_work/v2/baseline_fast_v2.py \
  --rom project_work/PokemonRed.gb \
  --output runs/ppo-smoke \
  --timesteps 256 --n-steps 32 --batch-size 32 --n-epochs 1 \
  --max-episode-steps 128

uv run python project_work/v2/evaluate.py runs/ppo-smoke --episodes 3
```

This checks startup, stepping, updates, checkpoints and evaluation. It is not enough
training to learn the task. Output directories must be new to prevent mixing runs.
There is no automatic restart loop and no automatic loading of the inherited
checkpoint: its observation space differs from the new one.

## Continuing training on Lambda or locally

Use `resume.py` with a run produced by the new runner. `--timesteps` means additional
steps, and `--output` must be a fresh directory. Algorithm, observations, rules,
worker count and training hyperparameters are inherited. A changed device is allowed.

```sh
uv run python project_work/v2/resume.py runs/ppo-smoke \
  --output runs/ppo-continued --timesteps 100000 \
  --rom project_work/PokemonRed.gb --state project_work/init.state --device cpu
```

On a CUDA-capable Lambda instance use `--device cuda` after verifying the installed
PyTorch build can see the GPU. For an interrupted run with no `final.zip`, explicitly
choose an intact periodic checkpoint, for example
`--checkpoint runs/ppo-smoke/checkpoints/policy_100000_steps.zip`.
Checkpoint paths are relative to the current working directory and must be inside
the source run. Copy the source run's manifest along with its checkpoint when moving
machines. `--rom` and `--state` relocate assets; their hashes must still match.

Resume restores policy/value weights, optimizer state and the cumulative timestep
counter. **Emulator episodes restart, RNGs are reseeded, and any partial rollout is
discarded.** This is training continuation, not an exact restoration of an interrupted
trajectory or an independent training seed. A new manifest records the parent run,
checkpoint hash and starting timestep; prior logs are preserved. The resumed run
can be evaluated with the same `evaluate.py` entry point.

Each run writes:

- `manifest.json`: configuration, seed, asset hashes, Git commit and dirty status.
- `episodes-N.jsonl`: every completed episode from worker N, with ending reason,
  violation reasons, individual deaths, success, badges, coverage, steps and return.
- `worker-N.monitor.csv`, TensorBoard data and periodic checkpoints.
- `final.zip` and `training.json` (actual timesteps and measured steps/second).
- Evaluation writes `evaluation.json` separately, including individual records.

## Observation comparison

| Mode | Shared inputs | Additional history |
|---|---|---|
| `partial` (default) | Screen stack, health, levels, badges, events, recent actions, party HP/levels/presence/status, exploration map | None |
| `history` | Same | Persistent `party_dead` and `used_regions` |
| `pixels` | Screen stack and recent actions only | None |

`partial` and `history` include privileged RAM measurements. The default exploration map is also
supplied memory; `--no-explore-map` removes it from **both** conditions when desired.
`history` is not a full-state oracle: it does not expose enemy stats or Bide timers.
Party status is one bit per member (any condition), not the identity of the condition.
The map shape is 48×48×1, screen shape 72×80×3, and events length 2,488.

For the first experiment, keep the same `partial` observations, rules, rewards,
starting state and training budget; change only the policy class:

```sh
uv run python project_work/v2/baseline_fast_v2.py \
  --rom project_work/PokemonRed.gb --algorithm ppo --observation-mode partial \
  --output runs/ppo-seed0 --seed 0 --num-envs 4 --timesteps 1000000

uv run python project_work/v2/baseline_fast_v2.py \
  --rom project_work/PokemonRed.gb --algorithm lstm --observation-mode partial \
  --output runs/lstm-seed0 --seed 0 --num-envs 4 --timesteps 1000000
```

These are starting configurations, not a claim that a million steps is sufficient.
Benchmark CPU throughput before increasing worker count. LSTM has more parameters,
so describe this as a policy-architecture comparison; add a capacity-matched
feedforward control before attributing all differences specifically to memory.
Repeat with independent training seeds and a fixed held-out evaluation setup.

The evaluator carries recurrent states between actions and resets them at every
episode. Evaluation defaults to stochastic actions and uses seeds starting at 1000.
With `--deterministic` and zero start jitter, repeatedly loading one state produces
the same deterministic trial; do not treat repetitions as independent evidence.
`--start-jitter-frames N` samples 0..N idle frames on reset using the episode seed.
This changes emulator timing but is not a substitute for varied starting teams/maps.
Confidence intervals quantify episode sampling for one checkpoint, not training-seed
variability or generalization across unseen game states.

## Exact rule contract of the current monitor

Preset identifier: `core-strict-revival-battle-entry-cap-v1`.

- A party Pokémon observed at zero HP is permanently recorded as dead. A party with
  no living members ends the run. An empty party before receiving a starter does not.
- **Strict revival convention inherited from Kevin's implementation:** any positive
  HP on a recorded dead Pokémon fails the run, including automatic Center healing.
  This is stricter than allowing a revived Pokémon to remain unusable. It is a
  deliberate experimental convention, not a claim about every Nuzlocke variant.
- First observed wild encounter consumes the region, whether caught, defeated or
  fled. Only that encounter may be caught. Tracking begins at the supplied save state.
  Starter/gifts are exempt in this preset; there is no duplicate-species exception.
- A region defaults to its raw map ID. `--region-map path.json` can map several map
  IDs to one region (JSON object mapping IDs to IDs, all in 0..255).
- Catch detection uses a party+current-box count increase during a wild battle or
  its exit snapshot. PC changes outside battle are ignored. Real catches into a
  nonfull party have passed replay validation; full parties and box changes remain unvalidated.
- `--level-cap 14` checks the entire party **at battle entry**. Level-ups during a
  battle are allowed. Boxed Pokémon do not count toward this check.
- Death identity uses original-trainer ID + DVs, surviving evolution and party
  reordering. These fields are not globally unique. Simultaneous party collisions
  raise an instrumentation error instead of silently assigning both Pokémon a death.
  Full PC-roster identity reconciliation is still future work.
- **Battle-item bans and set-mode/no-free-switch enforcement are not implemented.**
  Do not describe these runs as a complete hardcore Nuzlocke experiment yet.

Starting states must be before Brock's victory. A save state cannot reconstruct
earlier encounter/death history. Use `--initial-used-regions` to supply known encounter
history; only use states whose prior death history is empty for now. The bundled
`project_work/init.state` was checked in the emulator: it starts in the player's
bedroom (map 38), with an empty party, no badges, and none of the tracked milestones.

The monitor samples every emulator frame, including within each 24-frame action.
That avoids skipping transitions visible briefly in RAM, at a possible throughput
cost. Catch, faint and battle-entry transitions have passed limited real-ROM replays;
this does not cover all RAM synchronization during animations.
Rule failures and wipes are terminations; step limits are truncations. Getting the
Boulder Badge is success only if there is no rule failure. Individual faint counts
are independent of healing rewards. The inherited exploration/event/healing reward
remains the default; `--death-penalty` and `--violation-penalty` are configurable and
must be held fixed in architecture comparisons. Shaped return is not the success rate.

## ROM-free Bide timing pilot

```sh
uv run python -m battle_benchmark.bide --output runs/bide-pilot.json
```

This is a small synthetic model to check transition enumeration, backward induction,
history-dependent policies and Monte Carlo evaluation **before** building the Gen 1
engine. It is not Charmander versus Onix and its numbers are not Pokémon results.

Both sides start with 12 HP. The player attacks for 2 or 3 damage with equal
probability, or waits. The player acts first. If the opponent survives and is idle,
it attacks for 2 or 3 damage with total probability 3/4, or begins Bide with probability
1/4. Bide stores damage over the next 2 or 3 player turns (equal probabilities), then
returns twice the stored damage. Knocking out the enemy prevents its action.
The horizon is 12 turns; not winning by the horizon is failure.

The policy sees both HP values and the most recent start/release announcement;
ordinary turns erase the announcement. Remaining Bide duration and stored damage are
hidden. Baselines are random, always attacking, reacting only to the current start
message, and remembering the start until release. The exact optimum is a full-state
oracle with access to both hidden fields. It is **not** the perfect-memory POMDP
optimum. All transition probabilities and solved values use rational arithmetic.

The pilot omits real damage rolls, stats, speed order, moves, PP, critical hits,
stat stages, status effects, trainer AI, healing items and cartridge Bide quirks.
Next on the solver track: implement and validate a Gen 1 turn kernel for one staged
Charmander–Onix matchup, then replace the synthetic kernel behind the same kind of
`outcomes(state, action)` interface. Keep model-exact and cartridge-validated claims
separate throughout the report.

## Next validation checkpoint

1. The real-ROM smoke test above passed for both PPO and LSTM; repeat after changes
   that affect emulator integration. Retain its manifests and episode logs in `runs/`.
2. Stage first/second catches, a faint plus Center heal, PC changes and an over-cap
   battle; check episode reasons against a recorded emulator trace.
3. Confirm the desired revival/gift/cap conventions and implement missing hardcore
   rules before collecting any results labeled hardcore Nuzlocke.
4. Run one comparable PPO/LSTM pair and inspect actual failure modes.
5. Build the cartridge-backed single-fight transition tests before widening the engine.

Upstream references: [Whidden environment](https://github.com/PWhiddy/PokemonRedExperiments),
[pokered RAM structures](https://github.com/pret/pokered/blob/master/macros/ram.asm),
[SB3 mixed-input policy](https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/custom_policy.html),
[recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
