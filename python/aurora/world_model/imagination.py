"""Adaptive Uncertainty-Calibrated Imagination Rollout Engine for AURORA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from aurora.rl.buffers import ReplayBuffer
from aurora.tensor import Tensor, tensor
from aurora.world_model.ensemble import EnsembleDynamicsModel
from aurora.world_model.uncertainty import UncertaintyEstimator


@dataclass(slots=True)
class ImaginationResult:
    """Statistics and trajectories generated during an imagination session."""

    total_transitions: int
    truncated_trajectories: int
    completed_trajectories: int
    mean_horizon: float
    mean_reward: float
    mean_epistemic_uncertainty: float
    transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = field(
        default_factory=list
    )


class ImaginationEngine:
    """Adaptive Uncertainty-Calibrated Rollout Engine.

    Rolls out imagined trajectories using learned ensemble dynamics under an actor policy.
    Compounding model exploitation is bounded by dynamically truncating rollouts when
    the ensemble's epistemic disagreement exceeds tau_threshold:
        max_j U_epistemic,j(s_h, a_h) > tau_threshold
    """

    def __init__(
        self,
        dynamics: EnsembleDynamicsModel,
        policy: Any | Callable[[np.ndarray], np.ndarray],
        max_horizon: int = 15,
        uncertainty_threshold: float = 0.5,
        adaptive_truncation: bool = True,
        sampling_mode: str = "ts1",  # "ts1" (random member per step) or "mean"
    ) -> None:
        self.dynamics = dynamics
        self.policy = policy
        self.max_horizon = max_horizon
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptive_truncation = adaptive_truncation
        self.sampling_mode = sampling_mode

    def _get_action(self, obs: np.ndarray) -> np.ndarray:
        """Query policy for action given observation."""
        if hasattr(self.policy, "act"):
            act = self.policy.act(obs)
            return cast(np.ndarray, act if isinstance(act, np.ndarray) else np.array(act))
        elif callable(self.policy):
            act = self.policy(obs)
            return cast(np.ndarray, act if isinstance(act, np.ndarray) else np.array(act))
        else:
            # Fallback for aurora Module policies: forward pass
            obs_t = tensor(obs, requires_grad=False)
            if obs_t.ndim == 1:
                obs_t = obs_t.reshape(1, -1)
            act_res = self.policy(obs_t)
            act_t = act_res[0] if isinstance(act_res, tuple) else act_res
            if isinstance(act_t, Tensor):
                act_np = act_t.numpy()
                return cast(np.ndarray, act_np[0] if obs.ndim == 1 else act_np)
            return np.array(act_t)

    def rollout_single(
        self,
        initial_state: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]], bool, list[float]]:
        """Perform a single imagination rollout from an initial state.

        Returns:
            transitions: list of (s, a, r, s_next, done)
            truncated: whether rollout was truncated due to high epistemic uncertainty
            uncertainties: list of max epistemic uncertainties at each step
        """
        curr_state = np.array(initial_state, dtype=np.float64)
        transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        uncertainties: list[float] = []
        truncated = False

        for _ in range(self.max_horizon):
            action = self._get_action(curr_state)
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float64)
            if action.ndim == 0:
                action = action.reshape(1)

            # Query ensemble dynamics
            next_states, rewards, vars_all = self.dynamics.predict(curr_state, action)
            # next_states: (E, 1, obs_dim), rewards: (E, 1, 1), vars_all: (E, 1, out_dim)

            # Decompose uncertainty
            # Squeeze batch dimension for decomposition
            means_sq = [next_states[e, 0] for e in range(self.dynamics.ensemble_size)]
            vars_sq = [
                vars_all[e, 0, : self.dynamics.obs_dim] for e in range(self.dynamics.ensemble_size)
            ]
            unc = UncertaintyEstimator.decompose_np(means_sq, vars_sq)

            # Check epistemic disagreement
            max_epistemic = float(unc.disagreement)
            uncertainties.append(max_epistemic)

            if self.adaptive_truncation and max_epistemic > self.uncertainty_threshold:
                # Epistemic disagreement exceeds trust region: truncate imagination
                truncated = True
                break

            # Select next state and reward
            if self.sampling_mode == "ts1":
                member_idx = int(np.random.randint(0, self.dynamics.ensemble_size))
                next_state = next_states[member_idx, 0]
                reward = float(rewards[member_idx, 0, 0]) if self.dynamics.predict_reward else 0.0
            else:
                next_state = unc.mean
                reward = float(np.mean(rewards[:, 0, 0])) if self.dynamics.predict_reward else 0.0

            transitions.append((curr_state.copy(), action.copy(), reward, next_state.copy(), False))
            curr_state = next_state

        return transitions, truncated, uncertainties

    def generate_rollouts(
        self,
        initial_states: np.ndarray,
    ) -> ImaginationResult:
        """Roll out imagination trajectories across a batch of initial real states."""
        all_transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        all_uncertainties: list[float] = []
        total_rewards: list[float] = []
        horizons: list[int] = []
        truncated_count = 0

        states_arr = (
            initial_states if initial_states.ndim > 1 else np.expand_dims(initial_states, axis=0)
        )
        num_trajectories = states_arr.shape[0]

        for i in range(num_trajectories):
            traj, was_truncated, step_unc = self.rollout_single(states_arr[i])
            all_transitions.extend(traj)
            all_uncertainties.extend(step_unc)
            horizons.append(len(traj))
            total_rewards.append(sum(t[2] for t in traj))
            if was_truncated:
                truncated_count += 1

        mean_h = float(np.mean(horizons)) if horizons else 0.0
        mean_r = float(np.mean(total_rewards)) if total_rewards else 0.0
        mean_u = float(np.mean(all_uncertainties)) if all_uncertainties else 0.0

        return ImaginationResult(
            total_transitions=len(all_transitions),
            truncated_trajectories=truncated_count,
            completed_trajectories=num_trajectories - truncated_count,
            mean_horizon=mean_h,
            mean_reward=mean_r,
            mean_epistemic_uncertainty=mean_u,
            transitions=all_transitions,
        )

    def inject_into_buffer(
        self,
        replay_buffer: ReplayBuffer,
        initial_states: np.ndarray,
    ) -> ImaginationResult:
        """Generate rollouts and push all valid synthetic transitions into a replay buffer."""
        result = self.generate_rollouts(initial_states)
        for s, a, r, next_s, done in result.transitions:
            replay_buffer.add(s, a, r, next_s, done)
        return result
