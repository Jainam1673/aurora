"""Model-Based Policy Optimization (MBPO; Janner et al., 2019) reproduction."""

from __future__ import annotations

import numpy as np
from aurora.optim.adamw import AdamW
from aurora.rl.buffers import ReplayBatch, ReplayBuffer
from aurora.rl.policies import SquashedGaussianActor, TwinCritic
from aurora.rl.sac import SAC
from aurora.tensor import concat
from aurora.world_model.ensemble import EnsembleDynamicsModel


class MBPO:
    """Model-Based Policy Optimization (MBPO).

    Combines deep probabilistic dynamics ensemble rollouts with Soft Actor-Critic (SAC).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 7,
        ensemble_hidden_dims: list[int] | None = None,
        rollout_horizon: int = 1,
        model_ratio: float = 0.5,
        model_lr: float = 1e-3,
        sac_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        env_buffer_capacity: int = 100_000,
        model_buffer_capacity: int = 100_000,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rollout_horizon = rollout_horizon
        self.model_ratio = model_ratio

        # 1. World Model (Ensemble Dynamics)
        self.dynamics = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=ensemble_size,
            hidden_dims=ensemble_hidden_dims,
            predict_reward=True,
        )
        self.dynamics_optimizer = AdamW(self.dynamics.parameters(), lr=model_lr, weight_decay=1e-4)

        # 2. Policy Optimization (Soft Actor-Critic)
        self.actor = SquashedGaussianActor(obs_dim=obs_dim, action_dim=action_dim)
        self.critic = TwinCritic(obs_dim=obs_dim, action_dim=action_dim)
        self.sac = SAC(
            actor=self.actor,
            critic=self.critic,
            lr=sac_lr,
            gamma=gamma,
            tau=tau,
        )

        # 3. Buffers
        self.env_buffer = ReplayBuffer(
            capacity=env_buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
        )
        self.model_buffer = ReplayBuffer(
            capacity=model_buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
        )

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current SAC policy."""
        return self.actor.act(obs, deterministic=deterministic)

    def train_dynamics(self, batch_size: int = 64, num_epochs: int = 5) -> float:
        """Train ensemble dynamics on real experience buffer using Gaussian NLL loss."""
        if len(self.env_buffer) < batch_size:
            return 0.0

        total_loss = 0.0
        steps = 0
        for _ in range(num_epochs):
            batch = self.env_buffer.sample(batch_size)
            self.dynamics_optimizer.zero_grad()
            loss = self.dynamics.compute_loss(
                batch.obs, batch.actions, batch.next_obs, batch.rewards
            )
            loss.backward()
            self.dynamics_optimizer.step()

            total_loss += float(loss.numpy().item())
            steps += 1

        return total_loss / float(max(1, steps))

    def rollout_model(
        self,
        num_rollouts: int = 100,
        horizon: int | None = None,
    ) -> int:
        """Generate k-step branched model rollouts starting from real environment states."""
        if len(self.env_buffer) == 0:
            return 0

        h_steps = horizon if horizon is not None else self.rollout_horizon
        # Sample initial states uniformly from real buffer
        batch = self.env_buffer.sample(num_rollouts)
        curr_states = batch.obs.numpy()

        transitions_added = 0
        for _ in range(h_steps):
            # Query SAC policy for actions across the batch
            actions = np.zeros((num_rollouts, self.action_dim), dtype=np.float64)
            for i in range(num_rollouts):
                actions[i] = self.actor.act(curr_states[i], deterministic=False)

            # Step dynamics ensemble
            next_states, rewards, _ = self.dynamics.predict(curr_states, actions)
            # Sample random member per trajectory (TS1)
            member_indices = np.random.randint(0, self.dynamics.ensemble_size, size=num_rollouts)

            next_s_batch = np.zeros_like(curr_states)
            for i in range(num_rollouts):
                m_idx = member_indices[i]
                next_s = next_states[m_idx, i]
                rew = float(rewards[m_idx, i, 0])
                done = False

                self.model_buffer.add(curr_states[i], actions[i], rew, next_s, done)
                next_s_batch[i] = next_s
                transitions_added += 1

            curr_states = next_s_batch

        return transitions_added

    def sample_mixed_batch(self, batch_size: int = 64) -> ReplayBatch:
        """Sample mini-batch mixing real experience and synthetic model experience."""
        model_size = int(batch_size * self.model_ratio)
        env_size = batch_size - model_size

        if len(self.model_buffer) < model_size or model_size == 0:
            return self.env_buffer.sample(batch_size)

        if len(self.env_buffer) < env_size or env_size == 0:
            return self.model_buffer.sample(batch_size)

        env_batch = self.env_buffer.sample(env_size)
        model_batch = self.model_buffer.sample(model_size)

        return ReplayBatch(
            obs=concat([env_batch.obs, model_batch.obs], axis=0),
            actions=concat([env_batch.actions, model_batch.actions], axis=0),
            rewards=concat([env_batch.rewards, model_batch.rewards], axis=0),
            next_obs=concat([env_batch.next_obs, model_batch.next_obs], axis=0),
            dones=concat([env_batch.dones, model_batch.dones], axis=0),
        )

    def train_policy(self, num_updates: int = 1, batch_size: int = 64) -> dict[str, float]:
        """Perform actor-critic gradient updates using mixed real/synthetic experience."""
        if len(self.env_buffer) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0}

        metrics_sum: dict[str, float] = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": 0.0,
        }

        for _ in range(num_updates):
            mixed_batch = self.sample_mixed_batch(batch_size)
            m = self.sac.train_step(mixed_batch)
            for k, v in m.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        return {k: v / float(num_updates) for k, v in metrics_sum.items()}
