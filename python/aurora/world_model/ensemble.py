"""Probabilistic Deep Ensemble Dynamics Models for Model-Based RL."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from aurora.nn.layers import MLP, Linear
from aurora.nn.module import Module
from aurora.tensor import Tensor, concat, tensor


class EnsembleMember(Module):
    """Single dynamics ensemble member.

    Given input concatenation [s, a], predicts:
    mean mu and log-variance log_var for target y = [Delta s, r].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int],
        activation: str = "silu",
        log_var_min: float = -10.0,
        log_var_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.backbone = MLP(in_features, hidden_dims, hidden_dims[-1], activation=activation)
        self.mu_head = Linear(hidden_dims[-1], out_features)
        self.log_var_head = Linear(hidden_dims[-1], out_features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feats = self.backbone(x)
        mu = self.mu_head(feats)
        log_var = self.log_var_head(feats).clamp(self.log_var_min, self.log_var_max)
        return mu, log_var


class EnsembleDynamicsModel(Module):
    """Probabilistic Deep Gaussian Ensemble Dynamics Model.

    Parameterizes an ensemble of E independent transition models:
        p_{theta_e}(y | s, a) = N(mu_e(s, a), diag(sigma_e^2(s, a)))
    where y = [s_{t+1} - s_t, r_t] in R^{d_s + 1} (or R^{d_s} if reward is disabled).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "silu",
        log_var_min: float = -10.0,
        log_var_max: float = 2.0,
        predict_reward: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [200, 200, 200]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.predict_reward = predict_reward
        self.out_dim = obs_dim + (1 if predict_reward else 0)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.members: list[EnsembleMember] = []
        for i in range(ensemble_size):
            member = EnsembleMember(
                in_features=obs_dim + action_dim,
                out_features=self.out_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                log_var_min=log_var_min,
                log_var_max=log_var_max,
            )
            setattr(self, f"member_{i}", member)
            self.members.append(member)

    def forward(self, obs: Tensor, action: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """Compute mean and log_var across all ensemble members.

        Returns:
            means: list of E tensors of shape (..., out_dim)
            log_vars: list of E tensors of shape (..., out_dim)
        """
        x = concat([obs, action], axis=-1)
        means: list[Tensor] = []
        log_vars: list[Tensor] = []
        for member in self.members:
            mu, lv = member(x)
            means.append(mu)
            log_vars.append(lv)
        return means, log_vars

    @staticmethod
    def gaussian_nll_loss(
        mean: Tensor,
        log_var: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Heteroscedastic Gaussian Negative Log-Likelihood loss:

        L_NLL = (1 / 2B) * sum [ (y_j - mu_j)^2 / var_j + log(var_j) + log(2*pi) ]
        """
        diff = target - mean
        inv_var = (-log_var).exp()
        elementwise_nll = 0.5 * (diff * diff * inv_var + log_var + math.log(2.0 * math.pi))
        batch_size = mean.shape[0] if mean.ndim > 1 else 1
        return elementwise_nll.sum() / float(batch_size)

    def compute_loss(
        self,
        obs: Tensor,
        action: Tensor,
        next_obs: Tensor,
        reward: Tensor | None = None,
    ) -> Tensor:
        """Compute the sum of Gaussian NLL losses across all ensemble members."""
        delta_s = next_obs - obs
        if self.predict_reward:
            assert reward is not None
            r = reward if reward.ndim > 1 else reward.reshape(-1, 1)
            target = concat([delta_s, r], axis=-1)
        else:
            target = delta_s

        means, log_vars = self.forward(obs, action)
        total_loss = self.gaussian_nll_loss(means[0], log_vars[0], target)
        for e in range(1, self.ensemble_size):
            total_loss = total_loss + self.gaussian_nll_loss(means[e], log_vars[e], target)
        return total_loss

    def predict(
        self,
        obs: Tensor | np.ndarray,
        action: Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict next states, rewards, and variances across all ensemble members.

        Returns:
            next_states: np.ndarray of shape (E, B, obs_dim)
            rewards: np.ndarray of shape (E, B, 1) (or empty if not predict_reward)
            variances: np.ndarray of shape (E, B, out_dim)
        """
        obs_t = obs if isinstance(obs, Tensor) else tensor(obs, requires_grad=False)
        act_t = action if isinstance(action, Tensor) else tensor(action, requires_grad=False)
        if obs_t.ndim == 1:
            obs_t = obs_t.reshape(1, -1)
        if act_t.ndim == 1:
            act_t = act_t.reshape(1, -1)

        means, log_vars = self.forward(obs_t, act_t)

        obs_np = obs_t.numpy()
        next_states_list: list[np.ndarray] = []
        rewards_list: list[np.ndarray] = []
        variances_list: list[np.ndarray] = []

        for e in range(self.ensemble_size):
            mu = means[e].numpy()
            lv = log_vars[e].numpy()
            var = np.exp(lv)

            delta_s = mu[..., : self.obs_dim]
            next_s = obs_np + delta_s
            next_states_list.append(next_s)

            if self.predict_reward:
                r = mu[..., self.obs_dim :]
                rewards_list.append(r)
            variances_list.append(var)

        next_states = np.stack(next_states_list, axis=0)
        rewards = (
            np.stack(rewards_list, axis=0)
            if self.predict_reward
            else np.zeros((self.ensemble_size, obs_np.shape[0], 0))
        )
        variances = np.stack(variances_list, axis=0)
        return next_states, rewards, variances

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        member_idx: int | None = None,
    ) -> tuple[np.ndarray, float | np.ndarray]:
        """Step one transition in the world model using a selected or random member (TS1)."""
        if member_idx is None:
            member_idx = int(np.random.randint(0, self.ensemble_size))

        next_states, rewards, _ = self.predict(obs, action)
        ns = next_states[member_idx]
        rew = rewards[member_idx]

        if obs.ndim == 1:
            return ns[0], float(rew[0, 0]) if self.predict_reward else 0.0
        return ns, rew.squeeze(-1) if self.predict_reward else np.zeros(obs.shape[0])
