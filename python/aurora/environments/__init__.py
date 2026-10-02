"""Simulation environments and space definitions for AURORA."""

from aurora.environments.base import Env
from aurora.environments.classic_control import CartPole, Pendulum
from aurora.environments.spaces import Box, Discrete, Space

__all__ = [
    "Box",
    "CartPole",
    "Discrete",
    "Env",
    "Pendulum",
    "Space",
]
