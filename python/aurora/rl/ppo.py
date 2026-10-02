"""Proximal Policy Optimization (PPO) algorithm."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aurora.environments.base import Env
from aurora.optim.adamw import AdamW
from aurora.optim.optimizer import Optimizer, clip_grad_norm
from aurora.rl.buffers import RolloutBuffer
from aurora.rl.policies import ActorCriticPolicy
from aurora.tensor import tensor


class PPO:
    """Proximal Policy Optimization (Schulman et al., 2017) with GAE and clipped objective."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        optimizer: Optimizer | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 128,
        batch_size: int = 32,
        n_epochs: int = 4,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer if optimizer is not None else AdamW(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def collect_rollouts(
        self, env: Env, buffer: RolloutBuffer, current_obs: np.ndarray
    ) -> tuple[np.ndarray, list[float]]:
        """Collect n_steps of environment interaction into the rollout buffer."""
        buffer.reset()
        obs = current_obs
        episode_rewards: list[float] = []
        curr_ep_reward = 0.0

        for _ in range(self.n_steps):
            obs_t = tensor(obs.reshape(1, -1), requires_grad=False)
            dist, val = self.policy(obs_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)

            if self.policy.is_discrete:
                action_env = int(action_t.item())
                action_store = float(action_env)
            else:
                action_env = action_t.numpy()[0]
                action_store = action_env

            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated
            curr_ep_reward += reward

            buffer.add(
                obs=obs,
                action=action_store,
                reward=reward,
                done=done,
                value=val.item(),
                log_prob=log_prob_t.item(),
            )

            if done:
                episode_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap final value for GAE
        last_obs_t = tensor(obs.reshape(1, -1), requires_grad=False)
        _, last_val = self.policy(last_obs_t)
        buffer.compute_returns_and_advantages(
            last_value=last_val.item(),
            done=False,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        return obs, episode_rewards

    def train(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Perform PPO gradient updates across epochs and minibatches."""
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []

        for _ in range(self.n_epochs):
            for (
                batch_obs,
                batch_actions,
                batch_old_log_probs,
                batch_advantages,
                batch_returns,
            ) in buffer.get_generator(self.batch_size):
                values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)

                # 1. Clipped surrogate policy objective
                log_ratio = log_probs - batch_old_log_probs
                ratio = log_ratio.exp()

                surr1 = ratio * batch_advantages
                clamped_ratio = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
                surr2 = clamped_ratio * batch_advantages

                # Minimum of surr1 and surr2 with full autograd gradient routing
                mask_np = (surr1.numpy() <= surr2.numpy()).astype(np.float64)
                mask = tensor(mask_np, requires_grad=False)
                surr_min = mask * surr1 + (1.0 - mask) * surr2
                policy_loss = -surr_min.mean()

                # 2. Value function loss
                val_diff = values - batch_returns
                val_loss = (val_diff * val_diff).mean()

                # 3. Entropy loss (maximize entropy -> minimize negative entropy)
                ent_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * val_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(val_loss.item())
                entropy_losses.append(ent_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(-np.mean(entropy_losses)),
        }

    def learn(
        self,
        env: Env,
        total_timesteps: int,
        callback: Callable[[int, list[float]], None] | None = None,
    ) -> list[float]:
        """Execute on-policy learning loop."""
        obs_shape = env.observation_space.shape
        action_shape = () if self.policy.is_discrete else env.action_space.shape

        buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            obs_shape=obs_shape,
            action_shape=action_shape,
            batch_size=1,
        )

        obs, _ = env.reset()
        timesteps = 0
        all_rewards: list[float] = []

        while timesteps < total_timesteps:
            obs, ep_rewards = self.collect_rollouts(env, buffer, obs)
            timesteps += self.n_steps
            all_rewards.extend(ep_rewards)

            _ = self.train(buffer)
            if callback is not None:
                callback(timesteps, ep_rewards)

        return all_rewards
