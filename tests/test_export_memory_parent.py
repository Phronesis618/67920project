import json

import numpy as np
import torch
from context_aux import ContextCallback
from context_experiment import convert_parent
from lstm_memory import HEAD_DIM
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from test_context_aux import TinyPixels
from test_memory_context import MemoryPixels

from scripts import export_memory_parent


def test_export_gives_a_plain_parent_with_the_trained_weights_and_the_head(tmp_path):
    torch.set_num_threads(1)
    base_env, env = DummyVecEnv([TinyPixels]), DummyVecEnv([MemoryPixels])
    try:
        parent = RecurrentPPO(
            "MultiInputLstmPolicy", base_env, n_steps=8, batch_size=8, seed=1, device="cpu"
        )
        parent.save(tmp_path / "parent.zip")
        model = convert_parent(parent, env, seed=1, backprop=True, device="cpu", targets="memory")
        model.learn(16, reset_num_timesteps=False, callback=ContextCallback())
        run = tmp_path / "run"
        run.mkdir()
        model.save(run / "final.zip")
        manifest = {
            "algorithm": "lstm",
            "config": {"observation_mode": "pixels"},
            "ruleset": "test",
            "rom_sha1": "r",
            "state_sha1": "s",
            "parent_checkpoint": str(tmp_path / "parent.zip"),
        }
        (run / "manifest.json").write_text(json.dumps(manifest))

        export_memory_parent.main([str(run), "--output", str(tmp_path / "out")])
        plain = RecurrentPPO.load(tmp_path / "out" / "final.zip", device="cpu")
        assert type(plain.policy).__name__ == "RecurrentMultiInputActorCriticPolicy"
        assert plain.observation_space == parent.observation_space
        assert plain.num_timesteps == model.num_timesteps
        for key, value in plain.policy.state_dict().items():
            torch.testing.assert_close(value, model.policy.state_dict()[key], rtol=0, atol=0)
        obs, _ = TinyPixels().reset()
        assert (
            plain.predict(obs, deterministic=True)[0] == model.predict(obs, deterministic=True)[0]
        )
        # Adam moments of the continued run, not the template's, for the body.
        trained = model.policy.optimizer.state_dict()["state"][0]["exp_avg"]
        np.testing.assert_array_equal(
            plain.policy.optimizer.state_dict()["state"][0]["exp_avg"].numpy(), trained.numpy()
        )
        head = torch.load(tmp_path / "out" / "memory_head.pt")
        assert head["state_dict"]["weight"].shape[0] == HEAD_DIM
        exported = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert exported["rom_sha1"] == "r" and exported["training_class"].endswith("RecurrentPPO")
    finally:
        base_env.close()
        env.close()
