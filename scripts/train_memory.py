"""Train and evaluate an LSTM memory on a replayed-episode dataset (supervised, no PPO).

Three models see the same frozen policy features and predict the same RAM labels:

- ``input``: a linear head on the current features only (no memory) - the control;
- ``parent_frozen``: the LSTM parent's actor LSTM, frozen, plus a trained linear head -
  what the parent's memory already holds (a linear probe);
- ``memory``: an LSTM (initialised from the parent's actor LSTM) plus head, both trained.

Episodes are split into train, validation and test sets by episode, stratified by how
they ended, so illegal-catch episodes appear in each. Each model keeps the epoch with
the lowest validation loss (the LSTM can memorise its training episodes). Training is truncated backpropagation through
time: each episode is cut into ``--chunk``-action pieces with the LSTM state carried
(detached) from one piece to the next. A label that ``memory`` predicts on held-out
episodes clearly better than ``input`` is held in memory, not read off the screen.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
from lstm_memory import BINARY, NAMES, WILD, MemoryModel, label_loss
from sb3_contrib import RecurrentPPO

PORTABLE = {"_last_lstm_states": None, "clip_range": 0.2, "lr_schedule": None}


def load_dataset(directory):
    episodes = []
    for path in sorted(p for p in Path(directory).iterdir() if (p / "meta.json").exists()):
        episodes.append(
            {
                "name": path.name,
                "meta": json.loads((path / "meta.json").read_text()),
                "features": np.load(path / "features.npy", mmap_mode="r"),
                "labels": np.load(path / "labels.npy", mmap_mode="r"),
                "mask": np.load(path / "mask.npy", mmap_mode="r"),
                "battle": np.load(path / "battle.npy"),
            }
        )
    if not episodes:
        raise SystemExit(f"No episodes in {directory}")
    return episodes


def split(episodes, test_fraction, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, episode in enumerate(episodes):
        meta = episode["meta"]
        groups[(meta["end_reason"], tuple(meta["violation_reasons"]))].append(i)
    test = set()
    for members in groups.values():
        k = max(1, round(len(members) * test_fraction)) if len(members) > 1 else 0
        test.update(int(i) for i in rng.choice(members, k, replace=False))
    return [e for i, e in enumerate(episodes) if i not in test], [
        e for i, e in enumerate(episodes) if i in test
    ]


def padded(batch, key, dtype=np.float32):
    length = max(len(e[key]) for e in batch)
    shape = (len(batch), length) + batch[0][key].shape[1:]
    out = np.zeros(shape, dtype=dtype)
    for i, episode in enumerate(batch):
        out[i, : len(episode[key])] = episode[key]
    return th.from_numpy(out)


def run_batch(model, batch, chunk, optimizer=None, pos_weight=None):
    """Forward (and optionally train) over a batch of episodes in chunks; return outputs."""
    features = padded(batch, "features")
    labels, mask = padded(batch, "labels"), padded(batch, "mask")
    state, outputs, losses = None, [], []
    for start in range(0, features.shape[1], chunk):
        piece = slice(start, start + chunk)
        prediction, state = model(features[:, piece], state)
        if state is not None:
            state = tuple(s.detach() for s in state)
        loss, _ = label_loss(prediction, labels[:, piece], mask[:, piece], pos_weight)
        if optimizer is not None and mask[:, piece].sum() > 0:
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(loss.item())
        outputs.append(prediction.detach())
    return th.cat(outputs, 1), float(np.mean(losses))


def balance(episodes):
    """Per-label positive weight (negatives / positives) over the training actions."""
    labels = np.concatenate([np.asarray(e["labels"]) for e in episodes])
    mask = np.concatenate([np.asarray(e["mask"]) for e in episodes]) > 0
    weights = np.ones(len(NAMES), dtype=np.float32)
    for i in np.flatnonzero(BINARY):
        y = labels[mask[:, i], i]
        if len(y) and 0 < y.mean() < 1:
            weights[i] = (1 - y.mean()) / y.mean()
    return th.from_numpy(weights)


def feature_stats(episodes):
    total, square, n = 0.0, 0.0, 0
    for episode in episodes:
        x = np.asarray(episode["features"], dtype=np.float64)
        total, square, n = total + x.sum(0), square + (x**2).sum(0), n + len(x)
    mean = total / n
    std = np.sqrt(np.maximum(square / n - mean**2, 0)) + 1e-3
    return th.tensor(mean, dtype=th.float32), th.tensor(std, dtype=th.float32)


def mean_loss(model, episodes, args, pos_weight):
    with th.no_grad():
        return float(
            np.mean(
                [
                    run_batch(
                        model, episodes[i : i + args.batch_episodes], args.chunk, None, pos_weight
                    )[1]
                    for i in range(0, len(episodes), args.batch_episodes)
                ]
            )
        )


def train(model, episodes, validation, args, name, pos_weight):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = th.optim.Adam(parameters, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history, best, best_state = [], np.inf, None
    for epoch in range(args.epochs):
        order = rng.permutation(len(episodes))
        losses = []
        for i in range(0, len(order), args.batch_episodes):
            batch = [episodes[j] for j in order[i : i + args.batch_episodes]]
            _, loss = run_batch(model, batch, args.chunk, optimizer, pos_weight)
            losses.append(loss)
        val = mean_loss(model, validation, args, pos_weight)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val})
        print(json.dumps({"model": name, **history[-1]}), flush=True)
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history, int(min(history, key=lambda h: h["val_loss"])["epoch"])


def score(kind, y, prediction):
    if len(y) == 0:
        return None
    if kind:  # binary: balanced accuracy, chance 0.5
        positive, negative = y == 1, y == 0
        if not positive.any() or not negative.any():
            return None
        hit = prediction > 0
        return float(0.5 * (hit[positive].mean() + (~hit[negative]).mean()))
    variance = ((y - y.mean()) ** 2).sum()  # regression: R^2, 0 = predicting the mean
    return None if variance == 0 else float(1 - ((y - prediction) ** 2).sum() / variance)


def evaluate(model, episodes, args):
    predictions, labels, masks, battles, first_battle = [], [], [], [], []
    with th.no_grad():
        for i in range(0, len(episodes), args.batch_episodes):
            batch = episodes[i : i + args.batch_episodes]
            output, _ = run_batch(model, batch, args.chunk)
            for j, episode in enumerate(batch):
                n = len(episode["labels"])
                predictions.append(output[j, :n].numpy())
                labels.append(np.asarray(episode["labels"]))
                masks.append(np.asarray(episode["mask"]) > 0)
                battle = episode["battle"]
                battles.append(battle)
                entry = (battle == WILD) & np.concatenate([[True], battle[:-1] != WILD])
                first_battle.append(entry)
    prediction, label, mask = map(np.concatenate, (predictions, labels, masks))
    battle, entry = np.concatenate(battles), np.concatenate(first_battle)
    scores = {}
    for i, name in enumerate(NAMES):
        keep = mask[:, i]
        scores[name] = score(BINARY[i], label[keep, i], prediction[keep, i])
    catch = NAMES.index("encounter_catchable")
    # The decision points: every wild-battle action, and the first action of each battle.
    for key, keep in (("catchable_in_wild_battle", battle == WILD), ("catchable_at_entry", entry)):
        keep = keep & mask[:, catch]
        scores[key] = score(True, label[keep, catch], prediction[keep, catch])
        scores[f"{key}_n"] = int(keep.sum())
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-model", type=Path, required=True, help="LSTM parent final.zip")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--batch-episodes", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=["input", "parent_frozen", "memory"])
    args = p.parse_args(argv)
    th.manual_seed(args.seed)
    th.set_num_threads(max(1, th.get_num_threads()))
    episodes = load_dataset(args.dataset)
    train_set, test_set = split(episodes, args.test_fraction, args.seed)
    train_set, validation = split(train_set, args.test_fraction, args.seed + 1)
    parent = RecurrentPPO.load(args.feature_model, device="cpu", custom_objects=PORTABLE)
    feature_dim = episodes[0]["features"].shape[1]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": str(args.dataset),
        "feature_model_sha256": hashlib.sha256(args.feature_model.read_bytes()).hexdigest(),
        "train_episodes": [e["name"] for e in train_set],
        "validation_episodes": [e["name"] for e in validation],
        "test_episodes": [e["name"] for e in test_set],
        "train_actions": int(sum(len(e["labels"]) for e in train_set)),
        "test_actions": int(sum(len(e["labels"]) for e in test_set)),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metric": "balanced accuracy (binary, chance 0.5) / R^2 (regression, 0 = mean)",
        "models": {},
    }
    pos_weight = balance(train_set)
    mean, std = feature_stats(train_set)
    results["positive_weights"] = dict(zip(NAMES, pos_weight.tolist()))
    for name in args.models:
        model = MemoryModel(feature_dim, recurrent=name != "input", mean=mean, std=std)
        if name != "input":
            model.load_policy_lstm(parent.policy.lstm_actor)
        if name == "parent_frozen":
            model.lstm.requires_grad_(False)
        history, best_epoch = train(model, train_set, validation, args, name, pos_weight)
        results["models"][name] = {
            "history": history,
            "best_epoch": best_epoch,
            "test": evaluate(model, test_set, args),
        }
        if name == "memory":
            th.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": list(NAMES),
                    "feature_dim": feature_dim,
                },
                args.output / "memory_model.pt",
            )
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = list(NAMES) + ["catchable_in_wild_battle", "catchable_at_entry"]
    print(f"{'label':26s}" + "".join(f"{m:>15s}" for m in args.models))
    for row in rows:
        cells = [results["models"][m]["test"].get(row) for m in args.models]
        print(f"{row:26s}" + "".join(f"{'-' if v is None else round(v, 3):>15}" for v in cells))


if __name__ == "__main__":
    main()
