"""Base environment interface for AURORA reinforcement learning."""

from __future__ import annotations

from typing import Any

import numpy as np

from aurora.environments.spaces import Space


class Env:
    """Standardized environment interface compatible with modern RL benchmarks."""

    observation_space: Space
    action_space: Space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state."""
        raise NotImplementedError

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics.

        Returns:
            observation: Agent's observation of the current state.
            reward: Amount of reward returned after previous action.
            terminated: Whether the agent reached a terminal state.
            truncated: Whether a truncation condition outside the MDP is satisfied.
            info: Auxiliary diagnostic information.
        """
        raise NotImplementedError
