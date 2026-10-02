"""Trajectory storage and experience replay buffers for reinforcement learning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from aurora.tensor import Tensor, tensor


@dataclass
class ReplayBatch:
    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    dones: Tensor

    def __getitem__(self, key: str) -> Tensor:
        val = getattr(self, key)
        if not isinstance(val, Tensor):
            raise KeyError(f"Invalid key '{key}'")
        return val


class RolloutBuffer:
    """Storage for on-policy trajectories collected under policy pi_old."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        batch_size: int = 1,
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.batch_size = batch_size

        self.reset()

    def reset(self) -> None:
        self.obs = np.zeros((self.buffer_size, self.batch_size, *self.obs_shape), dtype=np.float64)
        self.actions = np.zeros(
            (self.buffer_size, self.batch_size, *self.action_shape), dtype=np.float64
        )
        self.rewards = np.zeros((self.buffer_size, self.batch_size), dtype=np.float64)
        self.dones = np.zeros((self.buffer_size, self.batch_size), dtype=np.float64)
        self.values = np.zeros((self.buffer_size, self.batch_size), dtype=np.float64)
        self.log_probs = np.zeros((self.buffer_size, self.batch_size), dtype=np.float64)
        self.advantages = np.zeros((self.buffer_size, self.batch_size), dtype=np.float64)
        self.returns = np.zeros((self.buffer_size, self.batch_size), dtype=np.float64)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | int | float,
        reward: float | np.ndarray,
        done: bool | float | np.ndarray,
        value: float | Tensor | np.ndarray,
        log_prob: float | Tensor | np.ndarray,
    ) -> None:
        """Add a step transition into the rollout buffer."""
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full")

        self.obs[self.ptr] = np.reshape(obs, (self.batch_size, *self.obs_shape))
        self.actions[self.ptr] = np.reshape(action, (self.batch_size, *self.action_shape))
        r_val = float(reward.item() if hasattr(reward, "item") else reward)
        d_val = 1.0 if (done.item() if hasattr(done, "item") else done) else 0.0
        self.rewards[self.ptr] = r_val
        self.dones[self.ptr] = d_val

        v_val = float(value.item() if hasattr(value, "item") else value)
        lp_val = float(log_prob.item() if hasattr(log_prob, "item") else log_prob)
        self.values[self.ptr] = v_val
        self.log_probs[self.ptr] = lp_val

        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float | np.ndarray,
        done: bool | np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and target returns."""
        last_val = np.asarray(last_value, dtype=np.float64)
        last_done = np.asarray(done, dtype=np.float64)

        last_gae = np.zeros(self.batch_size, dtype=np.float64)
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_val = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # Alias for convenience
    compute_returns_and_advantage = compute_returns_and_advantages

    def get_generator(
        self, batch_size: int | None = None, minibatch_size: int | None = None
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Yield mini-batches of transitions for PPO training."""
        bs = batch_size if batch_size is not None else minibatch_size
        if bs is None:
            raise ValueError("Must specify batch_size or minibatch_size")
        total_samples = self.buffer_size * self.batch_size
        indices = np.random.permutation(total_samples)

        flat_obs = self.obs.reshape(total_samples, *self.obs_shape)
        flat_actions = self.actions.reshape(total_samples, *self.action_shape)
        flat_log_probs = self.log_probs.reshape(total_samples)
        flat_advantages = self.advantages.reshape(total_samples)
        flat_returns = self.returns.reshape(total_samples)

        # Normalize advantages over the batch
        adv_mean = np.mean(flat_advantages)
        adv_std = np.std(flat_advantages) + 1e-8
        norm_advantages = (flat_advantages - adv_mean) / adv_std

        for start_idx in range(0, total_samples, bs):
            batch_indices = indices[start_idx : start_idx + bs]

            yield (
                tensor(flat_obs[batch_indices], requires_grad=False),
                tensor(flat_actions[batch_indices], requires_grad=False),
                tensor(flat_log_probs[batch_indices], requires_grad=False),
                tensor(norm_advantages[batch_indices], requires_grad=False),
                tensor(flat_returns[batch_indices], requires_grad=False),
            )


class ReplayBuffer:
    """FIFO replay buffer for off-policy deep reinforcement learning (SAC/DQN)."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float64)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float64)
        self.rewards = np.zeros((capacity,), dtype=np.float64)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float64)
        self.dones = np.zeros((capacity,), dtype=np.float64)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Insert a transition (s, a, r, s', d) into the replay buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Uniformly sample a batch of transitions."""
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} samples, fewer than batch_size {batch_size}")

        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            obs=tensor(self.obs[indices], requires_grad=False),
            actions=tensor(self.actions[indices], requires_grad=False),
            rewards=tensor(self.rewards[indices], requires_grad=False),
            next_obs=tensor(self.next_obs[indices], requires_grad=False),
            dones=tensor(self.dones[indices], requires_grad=False),
        )

    def __len__(self) -> int:
        return self.size
