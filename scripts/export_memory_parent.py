"""Export a memory-trained model as a plain recurrent parent for later PPO stages.

Input: a context_experiment.py run (context_aux.ContextPPO, usually ``--targets memory``).
Output directory, usable as ``--parent`` by any stage that loads RecurrentPPO:

- ``final.zip``: sb3-contrib RecurrentPPO with the run's CNN, LSTM, policy and value
  weights and the matching Adam state, prediction head removed, original observation
  space. No context_aux code is needed to load it.
- ``manifest.json``: the run's manifest (algorithm, config, ruleset, asset hashes) plus
  provenance of the export.
- ``memory_head.pt``: the removed prediction head and its target layout, so a later
  stage's weights can be re-scored with it to measure how much memory survived.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project_work/v2"))
import lstm_memory
from context_aux import ContextPPO
from sb3_contrib import RecurrentPPO

PORTABLE = {"_last_lstm_states": None, "clip_range": 0.2, "lr_schedule": None}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(run, output, template=None, device="cpu"):
    manifest = json.loads((run / "manifest.json").read_text())
    template = Path(template or manifest["parent_checkpoint"])
    model = ContextPPO.load(run / "final.zip", device=device, custom_objects=PORTABLE)
    plain = RecurrentPPO.load(template, device=device, custom_objects=PORTABLE)
    body = {k: v for k, v in model.policy.state_dict().items() if not k.startswith("context_head.")}
    plain.policy.load_state_dict(body, strict=True)
    # Adam state: keep the continued run's moments for every original parameter (group 0,
    # same order as the parent - checked by context_experiment.convert_parent).
    state = model.policy.optimizer.state_dict()
    head_ids = set(state["param_groups"][1]["params"])
    plain_state = plain.policy.optimizer.state_dict()
    if state["param_groups"][0]["params"] != plain_state["param_groups"][0]["params"]:
        raise ValueError("Parameter order differs from the template parent")
    plain_state["state"] = {k: v for k, v in state["state"].items() if k not in head_ids}
    plain.policy.optimizer.load_state_dict(plain_state)
    plain.num_timesteps = model.num_timesteps
    plain._n_updates = model._n_updates
    output.mkdir(parents=True, exist_ok=False)
    plain.save(output / "final.zip")
    head = {k[len("context_head.") :]: v for k, v in model.policy.state_dict().items()}
    head = {k: v for k, v in head.items() if k in ("weight", "bias")}
    th.save(
        {
            "state_dict": head,
            "context_targets": model.context_targets,
            "targets": list(lstm_memory.TARGET_NAMES),
            "head_slices": {k: [v.start, v.stop] for k, v in lstm_memory.HEAD_SLICES.items()},
        },
        output / "memory_head.pt",
    )
    exported = dict(
        manifest,
        training_class="sb3_contrib.RecurrentPPO",
        exported_from=str(run.resolve()),
        exported_from_sha256=sha256(run / "final.zip"),
        export_template=str(template.resolve()),
        export_note="Prediction head removed; weights, Adam state and timesteps kept.",
    )
    (output / "manifest.json").write_text(json.dumps(exported, indent=2) + "\n")
    return plain


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--template", type=Path, help="Default: the run's parent checkpoint")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    export(args.run, args.output, args.template, args.device)
    print(json.dumps({"exported": str(args.output)}))


if __name__ == "__main__":
    main()
