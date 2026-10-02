"""Soft Actor-Critic (SAC) algorithm with automatic entropy tuning."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from aurora.environments.base import Env
from aurora.nn.parameter import Parameter
from aurora.optim.adamw import AdamW
from aurora.optim.optimizer import Optimizer
from aurora.rl.buffers import ReplayBatch, ReplayBuffer
from aurora.rl.policies import SquashedGaussianActor, TwinCritic
from aurora.tensor import Tensor, tensor


class SAC:
    """Soft Actor-Critic (Haarnoja et al., 2018) for continuous control."""

    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: TwinCritic,
        actor_optimizer: Optimizer | None = None,
        critic_optimizer: Optimizer | None = None,
        alpha_optimizer: Optimizer | None = None,
        target_entropy: float | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        initial_alpha: float = 0.2,
        auto_alpha: bool = True,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        self.actor_opt = (
            actor_optimizer if actor_optimizer is not None else AdamW(actor.parameters(), lr=lr)
        )
        self.critic_opt = (
            critic_optimizer if critic_optimizer is not None else AdamW(critic.parameters(), lr=lr)
        )

        # Target critic
        self.critic_target = TwinCritic(
            obs_dim=critic.obs_dim,
            action_dim=critic.action_dim,
            hidden_dims=critic.hidden_dims,
            activation=critic.activation,
        )
        self.critic_target.load_state_dict(critic.state_dict())

        # Target entropy: default -dim(A)
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(actor.action_dim)
        )

        # Temperature parameter alpha = exp(log_alpha)
        init_log_alpha = math.log(initial_alpha)
        self.log_alpha = Parameter(np.array([init_log_alpha], dtype=np.float64))
        if auto_alpha:
            self.alpha_opt = (
                alpha_optimizer if alpha_optimizer is not None else AdamW([self.log_alpha], lr=lr)
            )

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha.numpy()[0]))

    def train_step(
        self,
        batch: dict[str, Tensor] | ReplayBatch | ReplayBuffer,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Perform a single SAC update step on a minibatch of transitions."""
        if isinstance(batch, ReplayBuffer):
            b: dict[str, Tensor] | ReplayBatch = batch.sample(batch_size)
        else:
            b = batch

        obs = b["obs"]
        actions = b["actions"]
        rewards = b["rewards"].reshape(-1, 1)
        next_obs = b["next_obs"]
        dones = b["dones"].reshape(-1, 1)

        alpha = self.alpha

        # 1. Critic Update
        next_actions, next_log_probs = self.actor(next_obs)
        next_log_probs = next_log_probs.reshape(-1, 1)

        q1_target, q2_target = self.critic_target(next_obs, next_actions)
        mask_target_np = (q1_target.numpy() <= q2_target.numpy()).astype(np.float64)
        mask_target = tensor(mask_target_np, requires_grad=False)
        min_q_target = mask_target * q1_target + (1.0 - mask_target) * q2_target

        soft_target_v = min_q_target - next_log_probs * alpha
        target_q_data = rewards.numpy() + self.gamma * (1.0 - dones.numpy()) * soft_target_v.numpy()
        target_q = tensor(target_q_data, requires_grad=False)

        q1, q2 = self.critic(obs, actions)
        diff1 = q1 - target_q
        diff2 = q2 - target_q
        critic_loss = 0.5 * (diff1 * diff1).mean() + 0.5 * (diff2 * diff2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Actor Update
        new_actions, log_probs = self.actor(obs)
        log_probs = log_probs.reshape(-1, 1)
        q1_new, q2_new = self.critic(obs, new_actions)

        mask_new_np = (q1_new.numpy() <= q2_new.numpy()).astype(np.float64)
        mask_new = tensor(mask_new_np, requires_grad=False)
        min_q_new = mask_new * q1_new + (1.0 - mask_new) * q2_new

        actor_loss = (log_probs * alpha - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Temperature Update
        alpha_loss_val = 0.0
        if self.auto_alpha:
            lp_np = log_probs.numpy()
            target_diff = tensor(lp_np + self.target_entropy, requires_grad=False)
            alpha_loss = -(self.log_alpha * target_diff).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = alpha_loss.item()

        # 4. Soft Target Update: theta_target = tau * theta + (1 - tau) * theta_target
        with_target_params = dict(self.critic_target.named_parameters())
        for name, p in self.critic.named_parameters():
            p_target = with_target_params[name]
            p_target.data[:] = self.tau * p.numpy() + (1.0 - self.tau) * p_target.numpy()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": alpha,
        }

    def learn(
        self,
        env: Env,
        total_timesteps: int,
        batch_size: int = 64,
        warmup_steps: int = 1000,
        replay_capacity: int = 100000,
        callback: Callable[[int, list[float]], None] | None = None,
    ) -> list[float]:
        """Execute off-policy learning loop."""
        replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
        )

        obs, _ = env.reset()
        timesteps = 0
        all_rewards: list[float] = []
        curr_ep_reward = 0.0

        while timesteps < total_timesteps:
            # Action selection
            if timesteps < warmup_steps:
                action = env.action_space.sample()
            else:
                obs_t = tensor(obs.reshape(1, -1), requires_grad=False)
                act_t, _ = self.actor(obs_t, deterministic=False)
                action = act_t.numpy()[0]

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            curr_ep_reward += reward

            replay_buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            timesteps += 1

            # Update networks
            if timesteps >= warmup_steps and len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                self.train_step(batch)

            if done:
                all_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
                obs, _ = env.reset()
                if callback is not None:
                    callback(timesteps, all_rewards)

        return all_rewards
