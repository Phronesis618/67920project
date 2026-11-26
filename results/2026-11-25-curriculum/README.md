# Antidote curriculum and resolution comparison

This follows the [inventory audit](../2026-11-25-antidote/README.md). All three
diagnosed walking-poison deaths still had antidotes (3, 2, 2); all five observed
successful cures occurred in battle. Supply alone does not explain those deaths.
The starter is fixed by our initial save: level-6 Squirtle, not an agent choice.

## Task and saves

The actor receives only three grayscale screens and its last three actions.
RAM is used to score and audit outcomes, not as an additional model input.
An outside-battle poison clear earns success only when a living, previously
poisoned Pokémon consumes an antidote, retains its HP, and has no recorded death.
Inventory/status writes may be separated by at most two emulator frames.
Reward is +1 for success, -1 for failure, and -0.001 per action. Episodes end on
success, ruleset termination, or 256 actions. In-battle cures do not solve this task.
No new reward is added to the original full-route evaluation.

Seven natural saves are listed with hashes in [suite.json](suite.json). No RAM
injection was used. Training starts 60% in the overworld (18/22 HP, 3 antidotes),
30% with the bag open, and 10% at the party target menu. Held-out saves cover
12 HP, 6 HP, and a different forest location/inventory from a different policy.
The low-HP saves share the training source episode: they are not independent game
seeds. Region encounter history is preserved; exploration and recurrent history
restart. A healthy control was obtained through a real antidote cure.

Healthy controls detect false-positive scoring. They do not prove the policy
avoids unnecessary menus: the game itself refuses to consume an antidote on a
healthy Pokémon. Trainer RUN behavior and auxiliary RAM prediction are separate
future experiments, not changes in this comparison.

## Matched experiment

Twelve fine-tunes: PPO/LSTM × 80×72/160×144 pixels × seeds 0/1/2. Each loads all
weights from its corresponding completed 10.49M-action pixel baseline and starts
with a fresh optimizer. Adaptive pooling keeps parameter counts equal across
resolutions; the half-resolution extractor is exactly equivalent to the original
extractor after loading weights. Native inputs change visual activations even
before training, so both `initial.zip` and `final.zip` are evaluated.

Each run gets 1,048,576 curriculum actions, 8 workers, 128-step rollouts, batch 256,
4 epochs, learning rate 0.0003, gamma 0.99, GAE lambda 0.95 and entropy 0.01.
Checkpoints are saved every 262,144 actions. Each initial/final model gets nine
episodes per train/held-out/healthy split, then three original-route episodes of
up to 16,384 actions. The route test is a transfer pilot, not a powered architecture
comparison. Menu-task success alone is not forest progress or proof of text reading.

`scripts/build_poison_suite.py` constructs and validates the natural fixtures.
`project_work/v2/poison_experiment.py` trains/evaluates them.
`scripts/build_poison_queue.py` creates the 24-job serial train/evaluate queue.
The queue retains the November 29, 8:59 p.m. Pacific training cutoff.

## Validation and storage

All 66 repository tests passed. They cover genuine cure matching, false cures,
death precedence, state hashes, parameter parity, exact half-resolution feature
equivalence, and PPO/LSTM training plus checkpoint round trips at both resolutions.
The real emulator completed an eight-action outside-battle cure at both resolutions
(18/22 HP retained, antidotes 3→2, poison cleared, no deaths), plus party-menu-start
cures. See [real_validation.json](real_validation.json). These scripted checks
validate the task; they are not learned-policy results.

Persistent root: `/lambda/nfs/jeff-67920-rl-checkpoints/67920project`.
New source: `source-poison-20261125`; saves: `curricula/poison-20261125`;
campaign: `runs/campaign-poison-20261125`. Baseline source and results stay intact.
The queue requires the persistent mount. A second copy is pulled to the Mac every
three minutes while its backup process and network remain available.

Launch status and the remote smoke result are recorded in `launch.json` after
deployment. Training outcomes will be reported only after logs/evaluations exist.
