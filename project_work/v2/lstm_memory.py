"""Train an LSTM memory offline, by supervised learning on replayed episodes.

The policy sees only screens and recent buttons. Facts such as "an encounter here would
be catchable" are not on screen, so only memory can track them. Evaluation runs record
every action; replaying those actions from the start save reproduces each episode
exactly, so every step can be paired with the true RAM state the rewards and rules use.
An LSTM over the frozen policy features is then trained to predict those facts. No PPO
and no reward are involved; the trained LSTM can later initialise or feed a policy.
"""

import numpy as np
import torch as th
from memory_addresses import BATTLE_TYPE_ADDRESS, BOX_COUNT_ADDRESS, PARTY_SIZE_ADDRESS

BUTTONS = ("down", "left", "right", "up", "A", "B", "start")  # the env's action order
ENEMY_HP, ENEMY_MAX_HP = 0xCFE6, 0xCFF4  # wEnemyMon HP and max HP (pret/pokered)
WILD, TRAINER = 1, 2
POKE_BALL, ANTIDOTE = 0x04, 0x0B
# Maps with wild encounters on the route to Pewter: Route 1, Route 22, Route 2, forest.
ENCOUNTER_MAPS = (12, 33, 13, 51)

# (name, kind). Binary labels use BCE, regression labels (scaled to about 0..1) MSE.
LABELS = (
    # The catch-legality fact, seen at two moments. Overworld on an encounter map: the
    # area is unused, so the next encounter here is catchable. Wild battle: this
    # encounter is catchable (the tracker marks the area used as the battle starts).
    ("encounter_catchable", "binary"),
    ("used_route2", "binary"),
    ("used_forest", "binary"),
    ("used_route22", "binary"),
    ("poisoned", "binary"),
    ("owned_gt1", "binary"),
    ("party_hp", "regression"),
    ("enemy_hp", "regression"),
    ("balls", "regression"),  # / 5, the starting count
    ("antidotes", "regression"),  # / 5
    ("event_flags", "regression"),  # / 32: events and beaten trainers
    ("since_wild_battle", "regression"),  # actions / 2048, capped
)
NAMES = tuple(name for name, _ in LABELS)

# Training targets: the RAM labels above plus the targets of Jeffrey's context pilot
# (project_work/v2/context_aux.py) that they do not already cover. Party HP and poison
# are shared. Battle type is a 3-way class (overworld / wild / trainer) and antidote
# availability a yes/no; both are derived from stored data, so old datasets still work.
TARGETS = LABELS + (("battle_type", "categorical"), ("has_antidote", "binary"))
TARGET_NAMES = tuple(name for name, _ in TARGETS)
KINDS = tuple(kind for _, kind in TARGETS)
WIDTHS = {"binary": 1, "regression": 1, "categorical": 3}
HEAD_SLICES, _start = {}, 0
for _name, _kind in TARGETS:
    HEAD_SLICES[_name] = slice(_start, _start + WIDTHS[_kind])
    _start += WIDTHS[_kind]
HEAD_DIM = _start


def targets_from(labels, mask, battle):
    """Extend stored RAM labels with the derived targets; class indices for categories."""
    labels, mask, battle = np.asarray(labels), np.asarray(mask), np.asarray(battle)
    battle_type = np.where(np.isin(battle, (0, 1, 2)), battle, -1).astype(np.float32)
    antidotes = NAMES.index("antidotes")
    extra = np.stack([battle_type, (labels[:, antidotes] > 0).astype(np.float32)], 1)
    extra_mask = np.stack([battle_type >= 0, mask[:, antidotes] > 0], 1).astype(np.float32)
    return (
        np.concatenate([labels, extra], 1).astype(np.float32),
        np.concatenate([mask.astype(np.float32), extra_mask], 1),
    )


def bag_count(memory, item):
    return sum(
        memory[0xD31F + 2 * i] for i in range(memory[0xD31D]) if memory[0xD31E + 2 * i] == item
    )


def read_labels(env, last_wild_step):
    """Label and validity vectors for the RAM state the current observation shows."""
    m = env.pyboy.memory
    battle = env.read_m(BATTLE_TYPE_ADDRESS)
    in_battle = battle in (WILD, TRAINER)
    mons = env.read_party_mons()
    map_id = env.get_game_coords()[2]
    tracker = env.tracker
    used = tracker.used_regions
    region = tracker.region_map.get(map_id, map_id)
    enemy_max = env.read_hp(ENEMY_MAX_HP)
    values = {
        "encounter_catchable": (
            tracker.encounter_catchable if battle == WILD else region not in used
        ),
        "used_route2": 13 in used,
        "used_forest": 51 in used,
        "used_route22": 33 in used,
        "poisoned": any(mon.status & 8 for mon in mons),
        "owned_gt1": env.read_m(PARTY_SIZE_ADDRESS) + env.read_m(BOX_COUNT_ADDRESS) > 1,
        "party_hp": env.read_hp_fraction(),
        "enemy_hp": env.read_hp(ENEMY_HP) / enemy_max if in_battle and enemy_max else 0.0,
        "balls": min(bag_count(m, POKE_BALL), 10) / 5,
        "antidotes": min(bag_count(m, ANTIDOTE), 10) / 5,
        "event_flags": min(env.get_all_events_reward(), 64) / 32,
        "since_wild_battle": min(env.step_count - last_wild_step, 2048) / 2048,
    }
    valid = dict.fromkeys(NAMES, True)
    valid["encounter_catchable"] = battle == WILD or (not in_battle and map_id in ENCOUNTER_MAPS)
    valid["enemy_hp"] = in_battle and enemy_max > 0
    labels = np.array([float(values[n]) for n in NAMES], dtype=np.float32)
    mask = np.array([float(valid[n]) for n in NAMES], dtype=np.float32)
    return labels, mask


def label_loss(prediction, targets, mask, pos_weight=None, balance_batch=False):
    """Mean over present targets of each target's masked loss.

    ``prediction`` is (..., HEAD_DIM) head output, ``targets``/``mask`` (..., len(TARGETS)).
    Binary targets use BCE (``pos_weight`` per target balances rare positives), regression
    MSE, and categorical cross-entropy with each class present in the batch weighted
    equally (as in Jeffrey's context pilot), so the head cannot win on the majority class.
    ``balance_batch`` balances binary targets the same per-batch way instead of with fixed
    ``pos_weight`` (for online PPO minibatches, where no dataset-wide rate exists).
    """
    prediction = prediction.reshape(-1, prediction.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])
    mask = mask.reshape(-1, mask.shape[-1]) > 0
    losses = []
    for i, (name, kind) in enumerate(TARGETS):
        keep = mask[:, i]
        if not keep.any():
            losses.append(None)
            continue
        out, y = prediction[keep, HEAD_SLICES[name]], targets[keep, i]
        if kind == "binary" and balance_batch:
            each = th.nn.functional.binary_cross_entropy_with_logits(out[:, 0], y, reduction="none")
            loss = th.stack([each[y == c].mean() for c in th.unique(y)]).mean()
        elif kind == "binary":
            weight = None if pos_weight is None else pos_weight[i]
            loss = th.nn.functional.binary_cross_entropy_with_logits(
                out[:, 0], y, pos_weight=weight
            )
        elif kind == "regression":
            loss = ((out[:, 0] - y) ** 2).mean()
        else:
            classes = y.long()
            each = th.nn.functional.cross_entropy(out, classes, reduction="none")
            loss = th.stack([each[classes == c].mean() for c in th.unique(classes)]).mean()
        losses.append(loss)
    present = [loss for loss in losses if loss is not None]
    total = th.stack(present).mean() if present else prediction.sum() * 0
    per_target = th.tensor([np.nan if v is None else v.item() for v in losses])
    return total, per_target


class MemoryModel(th.nn.Module):
    """Label head on an LSTM over frozen policy features, or on the features alone.

    ``recurrent=False`` is the no-memory control: it sees exactly the current input.
    """

    def __init__(self, feature_dim, hidden=256, recurrent=True, mean=None, std=None):
        super().__init__()
        self.recurrent = recurrent
        # The control standardises its input (raw CNN activations differ in scale by
        # orders of magnitude); LSTMs keep raw features, which the policy LSTM expects.
        self.register_buffer("mean", th.zeros(feature_dim) if mean is None else mean)
        self.register_buffer("std", th.ones(feature_dim) if std is None else std)
        self.lstm = th.nn.LSTM(feature_dim, hidden, batch_first=True) if recurrent else None
        self.head = th.nn.Linear(hidden if recurrent else feature_dim, HEAD_DIM)

    def forward(self, features, state=None):
        if not self.recurrent:
            return self.head((features - self.mean) / self.std), None
        out, state = self.lstm(features, state)
        return self.head(out), state

    def load_policy_lstm(self, policy_lstm):
        """Start from a recurrent policy's actor LSTM (same input features)."""
        self.lstm.load_state_dict(policy_lstm.state_dict())


def policy_features(policy, observations):
    """Frozen policy input features (CNN + recent actions) for a list of observations."""
    batch = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
    with th.no_grad():
        tensor, _ = policy.obs_to_tensor(batch)
        return policy.extract_features(tensor).cpu().numpy()


# Start-menu checks. Checked on the real game: the background tile at row 0, column 10
# (wTileMap 0xC3A0) is the start menu's border while it is open and 0x80 on the party
# screen; wCurrentMenuItem 0xCC26 is the start-menu cursor (POKeDEX 0, POKeMON 1, ITEM 2).
MENU_CORNER, START_MENU_TILE, PARTY_SCREEN_TILE = 0xC3A0 + 10, 0x79, 0x80
MENU_CURSOR, JOY_IGNORE = 0xCC26, 0xCD6B
PEEK_TARGETS = {"party": 1, "bag": 2}


def can_peek(env):
    """Overworld with input accepted: the start menu can open."""
    return env.read_m(BATTLE_TYPE_ADDRESS) == 0 and env.read_m(JOY_IGNORE) == 0


def menu_peek(env, kind, last_wild_step):
    """Open the start menu, look at the bag or the party, close it.

    Runs in ``env`` from its current state and returns [(observation, labels, mask)] for
    each action, or None if the menu did not behave as expected (the caller discards it).
    Looking at a menu changes no item, HP or flag, so labels stay true afterwards.
    """
    steps = []

    def act(button):
        obs, _, terminated, truncated, _ = env.step(BUTTONS.index(button))
        steps.append((obs, *read_labels(env, last_wild_step)))
        return terminated or truncated

    if act("start") or env.read_m(MENU_CORNER) != START_MENU_TILE:
        return None
    target = PEEK_TARGETS[kind]
    for _ in range(8):
        cursor = env.read_m(MENU_CURSOR)
        if cursor == target:
            break
        if act("down" if cursor < target else "up"):
            return None
    if env.read_m(MENU_CURSOR) != target or act("A") or act("up"):  # open, then look
        return None
    for _ in range(8):
        if env.read_m(MENU_CORNER) not in (START_MENU_TILE, PARTY_SCREEN_TILE):
            return steps
        if act("B"):
            return None
    return None


def memory_target(env, last_wild_step):
    """All targets and their masks for the current observation, as one flat vector."""
    labels, mask = read_labels(env, last_wild_step)
    battle = np.array([env.read_m(BATTLE_TYPE_ADDRESS)])
    targets, target_mask = targets_from(labels[None], mask[None], battle)
    return np.concatenate([targets[0], target_mask[0]]).astype(np.float32)
