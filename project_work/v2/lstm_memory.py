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
BINARY = np.array([kind == "binary" for _, kind in LABELS])


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


def label_loss(prediction, labels, mask, pos_weight=None):
    """Mean over present labels of each label's masked loss (BCE or MSE).

    ``pos_weight`` (per label, ignored for regression) balances rare binary labels so the
    head cannot score well by always predicting the majority class.
    """
    binary = th.as_tensor(BINARY, device=prediction.device)
    bce = th.nn.functional.binary_cross_entropy_with_logits(
        prediction, labels, reduction="none", pos_weight=pos_weight
    )
    mse = (prediction - labels) ** 2
    counts = mask.sum(dim=tuple(range(mask.dim() - 1)))
    per_label = (th.where(binary, bce, mse) * mask).sum(dim=tuple(range(mask.dim() - 1)))
    per_label = per_label / counts.clamp(min=1)
    present = counts > 0
    loss = per_label[present].mean() if present.any() else prediction.sum() * 0
    return loss, per_label.detach()


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
        self.head = th.nn.Linear(hidden if recurrent else feature_dim, len(LABELS))

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
