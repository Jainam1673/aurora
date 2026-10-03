"""TD-MPC (Hansen et al., 2022) latent trajectory optimization reproduction."""

from __future__ import annotations

import numpy as np
from aurora.nn.layers import MLP
from aurora.nn.module import Module
from aurora.optim.adamw import AdamW
from aurora.tensor import Tensor, concat, tensor


class LatentEncoder(Module):
    """Encodes raw observations into task-oriented latent space."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(obs_dim, hidden_dims, latent_dim, activation=activation)

    def forward(self, obs: Tensor) -> Tensor:
        out: Tensor = self.net(obs)
        return out


class LatentDynamics(Module):
    """Predicts next latent state from current latent state and action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(latent_dim + action_dim, hidden_dims, latent_dim, activation=activation)

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        x = concat([z, action], axis=-1)
        out: Tensor = self.net(x)
        return out


class LatentReward(Module):
    """Predicts scalar reward from latent state and action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(latent_dim + action_dim, hidden_dims, 1, activation=activation)

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        x = concat([z, action], axis=-1)
        out: Tensor = self.net(x)
        return out


class LatentCritic(Module):
    """Twin Q-critic network for latent state-action evaluation."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.q1 = MLP(latent_dim + action_dim, hidden_dims, 1, activation=activation)
        self.q2 = MLP(latent_dim + action_dim, hidden_dims, 1, activation=activation)

    def forward(self, z: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = concat([z, action], axis=-1)
        q1: Tensor = self.q1(x)
        q2: Tensor = self.q2(x)
        return q1, q2


class LatentPolicyPrior(Module):
    """Policy prior predicting actions from latent state."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(latent_dim, hidden_dims, action_dim, activation=activation)

    def forward(self, z: Tensor) -> Tensor:
        # Output bounded in [-1, 1] via tanh
        out: Tensor = self.net(z).tanh()
        return out


class TDMPCAgent(Module):
    """TD-MPC agent combining learned task latent space and CEM trajectory planning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dims: list[int] | None = None,
        gamma: float = 0.99,
        tau: float = 0.01,
        lr: float = 3e-4,
        planning_horizon: int = 5,
        num_samples: int = 50,
        num_elites: int = 10,
        cem_iterations: int = 5,
        alpha_cem: float = 0.5,
        min_std: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.planning_horizon = planning_horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.alpha_cem = alpha_cem
        self.min_std = min_std

        # Online networks
        self.encoder = LatentEncoder(obs_dim, latent_dim, hidden_dims)
        self.dynamics = LatentDynamics(latent_dim, action_dim, hidden_dims)
        self.reward = LatentReward(latent_dim, action_dim, hidden_dims)
        self.critic = LatentCritic(latent_dim, action_dim, hidden_dims)
        self.policy = LatentPolicyPrior(latent_dim, action_dim, hidden_dims)

        # Target critic
        self.target_critic = LatentCritic(latent_dim, action_dim, hidden_dims)
        self._copy_weights(self.critic, self.target_critic)

        # Optimizers
        self.model_optimizer = AdamW(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )
        self.policy_optimizer = AdamW(self.policy.parameters(), lr=lr)

    def _copy_weights(self, src: Module, dst: Module) -> None:
        src_params = src.parameters()
        dst_params = dst.parameters()
        for s, d in zip(src_params, dst_params, strict=True):
            d.data = np.copy(s.data)

    def _polyak_update(self) -> None:
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters(), strict=True):
            tp.data = self.tau * p.data + (1.0 - self.tau) * tp.data

    def evaluate_trajectory(
        self,
        z_init: np.ndarray,
        action_trajectories: np.ndarray,
    ) -> np.ndarray:
        """Evaluate latent trajectory returns for candidate action sequences.

        Args:
            z_init: initial latent state of shape (latent_dim,) or (1, latent_dim)
            action_trajectories: candidate actions of shape (num_samples, horizon, action_dim)

        Returns:
            returns: estimated discounted returns of shape (num_samples,)
        """
        num_samples, horizon, _ = action_trajectories.shape
        z_t = np.repeat(z_init.reshape(1, -1), num_samples, axis=0)  # (N, latent_dim)

        total_returns = np.zeros(num_samples, dtype=np.float32)
        discount = 1.0

        for h in range(horizon):
            a_h = action_trajectories[:, h, :]  # (N, action_dim)
            r_pred = self.reward(tensor(z_t), tensor(a_h)).data.squeeze(-1)  # (N,)
            total_returns += discount * r_pred
            discount *= self.gamma

            z_next = self.dynamics(tensor(z_t), tensor(a_h)).data
            z_t = z_next

        # Terminal value bootstrap
        terminal_actions = self.policy(tensor(z_t)).data
        q1, q2 = self.target_critic(tensor(z_t), tensor(terminal_actions))
        v_terminal = np.minimum(q1.data.squeeze(-1), q2.data.squeeze(-1))
        total_returns += discount * v_terminal

        return total_returns

    def plan(
        self,
        obs: np.ndarray,
        horizon: int | None = None,
        num_samples: int | None = None,
        num_elites: int | None = None,
        iterations: int | None = None,
    ) -> np.ndarray:
        """Plan action using Cross-Entropy Method in latent space.

        Returns:
            best_action: optimal first action of shape (action_dim,)
        """
        h = horizon or self.planning_horizon
        n = num_samples or self.num_samples
        m = num_elites or self.num_elites
        iters = iterations or self.cem_iterations

        z0 = self.encoder(tensor(obs.reshape(1, -1))).data[0]

        # Initialize proposal distribution
        mean = np.zeros((h, self.action_dim), dtype=np.float32)
        std = np.ones((h, self.action_dim), dtype=np.float32)

        for _ in range(iters):
            # Sample candidate action sequences
            noise = np.random.randn(n, h, self.action_dim).astype(np.float32)
            candidates = np.clip(mean[np.newaxis, :, :] + std[np.newaxis, :, :] * noise, -1.0, 1.0)

            # Evaluate candidate trajectories
            returns = self.evaluate_trajectory(z0, candidates)

            # Select top M elites
            elite_indices = np.argsort(returns)[-m:]
            elites = candidates[elite_indices]

            # Fit proposal distribution with momentum
            new_mean = np.mean(elites, axis=0)
            new_std = np.maximum(np.std(elites, axis=0), self.min_std)

            mean = self.alpha_cem * new_mean + (1.0 - self.alpha_cem) * mean
            std = self.alpha_cem * new_std + (1.0 - self.alpha_cem) * std

        return np.asarray(mean[0])

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Update latent model, critic, and policy on batch of transitions."""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Latent representations
        z = self.encoder(obs)
        next_z = self.dynamics(z, actions)

        # Target latent from encoder (stop grad)
        with_no_grad_target_z = self.encoder(next_obs).data

        # Dynamics consistency loss
        dyn_loss = ((next_z - tensor(with_no_grad_target_z)) ** 2).mean()

        # Reward prediction loss
        pred_reward = self.reward(z, actions)
        reward_loss = ((pred_reward - rewards) ** 2).mean()

        # Critic loss with TD-target
        q1, q2 = self.critic(z, actions)
        next_actions = self.policy(tensor(with_no_grad_target_z)).data
        t_q1, t_q2 = self.target_critic(tensor(with_no_grad_target_z), tensor(next_actions))
        target_v = np.minimum(t_q1.data, t_q2.data)
        target_q = rewards.data + self.gamma * (1.0 - dones.data) * target_v

        q1_loss = ((q1 - tensor(target_q)) ** 2).mean()
        q2_loss = ((q2 - tensor(target_q)) ** 2).mean()
        critic_loss = q1_loss + q2_loss

        total_model_loss = dyn_loss + reward_loss + critic_loss

        self.model_optimizer.zero_grad()
        total_model_loss.backward()
        self.model_optimizer.step()

        # Policy loss: maximize Q(z, pi(z))
        pi_action = self.policy(tensor(z.data))
        q_pi, _ = self.critic(tensor(z.data), pi_action)
        policy_loss = -q_pi.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._polyak_update()

        return {
            "dyn_loss": float(dyn_loss.data.item()),
            "reward_loss": float(reward_loss.data.item()),
            "critic_loss": float(critic_loss.data.item()),
            "policy_loss": float(policy_loss.data.item()),
        }
