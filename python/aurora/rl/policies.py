"""Actor-Critic and Gaussian policy networks for reinforcement learning."""

from __future__ import annotations

import numpy as np

from aurora.distributions.categorical import Categorical
from aurora.distributions.distribution import Distribution
from aurora.distributions.normal import Normal
from aurora.distributions.tanh_normal import TanhNormal
from aurora.nn.layers import MLP, Linear
from aurora.nn.module import Module
from aurora.nn.parameter import Parameter
from aurora.tensor import Tensor, tensor, zeros


class ActorCriticPolicy(Module):
    """Actor-Critic policy network for discrete or continuous on-policy RL (PPO/A2C)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        is_discrete: bool = True,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete

        # Shared or separate backbones
        self.actor_backbone = MLP(obs_dim, hidden_dims, hidden_dims[-1], activation=activation)
        self.critic_backbone = MLP(obs_dim, hidden_dims, hidden_dims[-1], activation=activation)

        if is_discrete:
            self.action_head = Linear(hidden_dims[-1], action_dim)
            self.log_std: Parameter | None = None
        else:
            self.action_head = Linear(hidden_dims[-1], action_dim)
            self.log_std = Parameter(zeros((action_dim,)))

        self.value_head = Linear(hidden_dims[-1], 1)

    def get_distribution(self, obs: Tensor) -> Distribution:
        features = self.actor_backbone(obs)
        if self.is_discrete:
            logits = self.action_head(features)
            return Categorical(logits=logits)
        else:
            loc = self.action_head(features)
            assert self.log_std is not None
            scale = self.log_std.exp()
            return Normal(loc=loc, scale=scale)

    def forward(self, obs: Tensor) -> tuple[Distribution, Tensor]:
        dist = self.get_distribution(obs)
        val_features = self.critic_backbone(obs)
        value = self.value_head(val_features)
        return dist, value

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate log probability, entropy, and value of given actions."""
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value.reshape(-1), log_prob.reshape(-1), entropy.reshape(-1)


class SquashedGaussianActor(Module):
    """Squashed Gaussian policy for off-policy continuous control (SAC)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone = MLP(obs_dim, hidden_dims, hidden_dims[-1], activation=activation)
        self.mu_head = Linear(hidden_dims[-1], action_dim)
        self.log_std_head = Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Compute squashed action and its log probability."""
        features = self.backbone(obs)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        # Clamping log standard deviation for numerical stability
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        scale = log_std.exp()

        dist = TanhNormal(loc=mu, scale=scale)
        if deterministic:
            action = dist.mean
            log_prob = dist.log_prob(action)
        else:
            action, pre_tanh = dist.rsample_with_pre_tanh()
            log_prob = dist.log_prob(action, pre_tanh_value=pre_tanh)

        return action, log_prob

    def act(self, obs: np.ndarray | Tensor, deterministic: bool = False) -> np.ndarray:
        """Sample an action for environment interaction."""
        if not isinstance(obs, Tensor):
            obs_t = tensor(obs, requires_grad=False)
            if obs_t.ndim == 1:
                obs_t = obs_t.reshape(1, -1)
        else:
            obs_t = obs if obs.ndim > 1 else obs.reshape(1, -1)

        action, _ = self.forward(obs_t, deterministic=deterministic)
        act_np = action.numpy()
        return act_np[0] if obs_t.shape[0] == 1 else act_np


class TwinCritic(Module):
    """Twin Q-network architecture for Soft Actor-Critic (SAC)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        in_dim = obs_dim + action_dim
        self.q1 = MLP(in_dim, hidden_dims, 1, activation=activation)
        self.q2 = MLP(in_dim, hidden_dims, 1, activation=activation)

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Concatenate state and action and evaluate twin Q-values."""
        from aurora.tensor import concat

        sa = concat([obs, action], axis=-1)
        v1 = self.q1(sa)
        v2 = self.q2(sa)
        return v1, v2
