"""First-principles implementations of classic control RL environments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aurora.environments.base import Env
from aurora.environments.spaces import Box, Discrete


class CartPole(Env):
    """Cart-Pole balancing environment based on Sutton & Barto (1998)."""

    def __init__(self, max_steps: int = 500) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode: 12 degrees
        self.theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
        self.x_threshold = 2.4
        self.max_steps = max_steps

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float64).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float64).max,
            ],
            dtype=np.float64,
        )
        self.observation_space = Box(-high, high, dtype=np.float64)
        self.action_space = Discrete(2)

        self.state: np.ndarray | None = None
        self._steps_beyond_terminated: int | None = None
        self._step_count = 0
        self._rng = np.random.RandomState()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.state = self._rng.uniform(low=-0.05, high=0.05, size=(4,))
        self._steps_beyond_terminated = None
        self._step_count = 0
        return self.state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "Call reset before using step"
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._step_count += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        truncated = self._step_count >= self.max_steps

        if not terminated:
            reward = 1.0
        elif self._steps_beyond_terminated is None:
            self._steps_beyond_terminated = 0
            reward = 1.0
        else:
            self._steps_beyond_terminated += 1
            reward = 0.0

        return self.state.copy(), reward, terminated, truncated, {}


class Pendulum(Env):
    """Inverted pendulum swing-up and balance with continuous torque control."""

    def __init__(self, max_steps: int = 200) -> None:
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.max_steps = max_steps

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float64)
        self.observation_space = Box(-high, high, dtype=np.float64)
        self.action_space = Box(-self.max_torque, self.max_torque, shape=(1,), dtype=np.float64)

        self.state: np.ndarray | None = None
        self._step_count = 0
        self._rng = np.random.RandomState()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        high_th = math.pi
        high_thdot = 1.0
        theta = self._rng.uniform(low=-high_th, high=high_th)
        theta_dot = self._rng.uniform(low=-high_thdot, high=high_thdot)
        self.state = np.array([theta, theta_dot], dtype=np.float64)
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        th, thdot = self.state
        return np.array([math.cos(th), math.sin(th), thdot], dtype=np.float64)

    def step(
        self, action: float | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "Call reset before step"
        u = float(np.asarray(action).squeeze())
        u = float(np.clip(u, -self.max_torque, self.max_torque))

        th, thdot = self.state
        # Normalized angle in [-pi, pi]
        norm_th = ((th + math.pi) % (2.0 * math.pi)) - math.pi
        costs = norm_th**2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = (
            thdot
            + (3.0 * self.g / (2.0 * self.l) * math.sin(th) + 3.0 / (self.m * self.l**2) * u)
            * self.dt
        )
        newthdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        newth = th + newthdot * self.dt

        self.state = np.array([newth, newthdot], dtype=np.float64)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), -costs, False, truncated, {}
