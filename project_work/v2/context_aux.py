"""Recurrent state prediction from pixels, with RAM used only as training targets.

The control learns the same readout with its recurrent features detached. The
treatment also backpropagates that loss into the visual encoder and actor LSTM.
Neither the targets nor the predictions are additional action-policy inputs.
"""

from types import SimpleNamespace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch as th
from poison_curriculum import antidote_count
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F


def context_target(env):
    """Labels for the CURRENT observation, before its action is executed."""
    env = env.unwrapped
    battle = env.read_m(0xD057)
    mons = env.read_party_mons()
    hp = sum(m.hp for m in mons) / max(1, sum(m.max_hp for m in mons))
    return np.array(
        [
            battle if battle in (0, 1, 2) else -1,
            any(m.hp > 0 and m.status & 8 for m in mons),
            np.clip(hp, 0, 1),
            antidote_count(env.pyboy.memory) > 0,
        ],
        dtype=np.float32,
    )


class ContextLabels(gym.Wrapper):
    def step(self, action):
        target = context_target(self.env)
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, dict(info, context_target_before_action=target)


class ContextCallback(BaseCallback):
    def _on_step(self):
        # VecEnv's info belongs to the just-finished action, even on auto-reset.
        self.model.rollout_buffer.pending_targets = np.stack(
            [info["context_target_before_action"] for info in self.locals["infos"]]
        )
        return True


class ContextBuffer(RecurrentDictRolloutBuffer):
    def reset(self):
        super().reset()
        self.context_targets = np.zeros((self.buffer_size, self.n_envs, 4), dtype=np.float32)
        self.pending_targets = None

    def add(self, *args, **kwargs):
        if self.pending_targets is None:
            raise ValueError("ContextCallback must supply pre-action labels")
        self.context_targets[self.pos] = self.pending_targets
        self.pending_targets = None
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.context_targets = self.swap_and_flatten(self.context_targets)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env_change, env=None):
        sample = super()._get_samples(batch_inds, env_change, env)
        targets = self.pad(self.context_targets[batch_inds]).reshape(-1, 4)
        return SimpleNamespace(**sample._asdict(), context_targets=targets)


def balanced_mean(losses, labels):
    """Equal weight per class present in this minibatch; padding already removed."""
    return th.stack([losses[labels == c].mean() for c in th.unique(labels)]).mean()


def prediction_loss(logits, targets, mask):
    logits, targets = logits[mask], targets[mask]
    valid = targets[:, 0] >= 0
    battle = targets[valid, 0].long()
    battle_loss = (
        balanced_mean(F.cross_entropy(logits[valid, :3], battle, reduction="none"), battle)
        if valid.any()
        else logits.sum() * 0
    )
    poison_loss = balanced_mean(
        F.binary_cross_entropy_with_logits(logits[:, 3], targets[:, 1], reduction="none"),
        targets[:, 1],
    )
    hp_loss = F.mse_loss(logits[:, 4].sigmoid(), targets[:, 2])
    item_loss = balanced_mean(
        F.binary_cross_entropy_with_logits(logits[:, 5], targets[:, 3], reduction="none"),
        targets[:, 3],
    )
    return (battle_loss + poison_loss + hp_loss + item_loss) / 4


class ContextPolicy(RecurrentMultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if set(self.observation_space.spaces) != {"screens", "recent_actions"}:
            raise ValueError("Context policy requires pixel/action observations only")
        # Super's optimizer already holds every original parameter, in its old order.
        self.context_head = nn.Linear(self.lstm_output_dim, 6)
        self.optimizer.add_param_group({"params": self.context_head.parameters()})

    def evaluate_with_context(self, obs, actions, lstm_states, episode_starts, detach):
        features = self.extract_features(obs)
        pi_features, vf_features = (
            (features, features) if self.share_features_extractor else features
        )
        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)
        logits = self.context_head(latent_pi.detach() if detach else latent_pi)
        distribution = self._get_action_dist_from_latent(
            self.mlp_extractor.forward_actor(latent_pi)
        )
        values = self.value_net(self.mlp_extractor.forward_critic(latent_vf))
        return values, distribution.log_prob(actions), distribution.entropy(), logits

    def get_distribution(self, obs, lstm_states, episode_starts):
        # Same action computation as the upstream policy; predictions are logging only.
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        latent, states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor
        )
        self.last_context_logits = self.context_head(latent).detach()
        return self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(latent)), states


class ContextPPO(RecurrentPPO):
    policy_aliases: ClassVar = {"MultiInputLstmPolicy": ContextPolicy}

    def __init__(self, *args, context_coef=0.1, context_backprop=True, **kwargs):
        self.context_coef = context_coef
        self.context_backprop = context_backprop
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        if not isinstance(self.policy, ContextPolicy):
            raise TypeError("ContextPPO requires ContextPolicy")
        lstm = self.policy.lstm_actor
        self.rollout_buffer = ContextBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size),
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def train(self):
        # Adapted from sb3-contrib 2.3.0 RecurrentPPO.train (MIT license).
        # Copyright (c) 2020 Stable-Baselines Team; see THIRD_PARTY_LICENSES.md.
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.3.0/sb3_contrib/ppo_recurrent/ppo_recurrent.py
        # Keep PPO/GAE/clipping unchanged; add a separately clipped prediction head.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = (
            self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        )
        metrics = {
            k: []
            for k in (
                "entropy_loss",
                "policy_gradient_loss",
                "value_loss",
                "clip_fraction",
                "context_loss",
            )
        }
        continue_training = True
        base_params = self.policy.optimizer.param_groups[0]["params"]
        head_params = self.policy.optimizer.param_groups[1]["params"]
        for epoch in range(self.n_epochs):
            kls = []
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions.long().flatten()
                mask = data.mask > 1e-8
                values, log_prob, entropy, logits = self.policy.evaluate_with_context(
                    data.observations,
                    actions,
                    data.lstm_states,
                    data.episode_starts,
                    detach=not self.context_backprop,
                )
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8
                    )
                ratio = th.exp(log_prob - data.old_log_prob)
                policy_loss = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                )[mask].mean()
                predicted = (
                    values
                    if clip_vf is None
                    else data.old_values + th.clamp(values - data.old_values, -clip_vf, clip_vf)
                )
                value_loss = ((data.returns - predicted) ** 2)[mask].mean()
                entropy_loss = log_prob[mask].mean() if entropy is None else -entropy[mask].mean()
                aux_loss = prediction_loss(logits, data.context_targets, mask)
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.context_coef * aux_loss
                )
                for name, value in (
                    ("policy_gradient_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy_loss", entropy_loss),
                    ("context_loss", aux_loss),
                ):
                    metrics[name].append(value.item())
                metrics["clip_fraction"].append(
                    ((ratio - 1).abs() > clip_range).float()[mask].mean().item()
                )
                with th.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = ((th.exp(log_ratio) - 1) - log_ratio)[mask].mean().item()
                    kls.append(kl)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                # A detached control readout must not alter the actor's grad clipping.
                th.nn.utils.clip_grad_norm_(base_params, self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(head_params, self.max_grad_norm)
                self.policy.optimizer.step()
            if not continue_training:
                break
        self._n_updates += self.n_epochs
        for name, values_ in metrics.items():
            self.logger.record(f"train/{name}", np.mean(values_))
        self.logger.record("train/approx_kl", np.mean(kls))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record(
            "train/explained_variance",
            explained_variance(
                self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
            ),
        )
        if clip_vf is not None:
            self.logger.record("train/clip_range_vf", clip_vf)
