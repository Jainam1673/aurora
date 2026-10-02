"""AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents.

A research-grade platform for uncertainty-aware model-based reinforcement learning.
"""

from aurora import distributions, environments, nn, optim, rl
from aurora.checkpoint import load_checkpoint, save_checkpoint
from aurora.core import AuroraInfo, get_system_info
from aurora.gradcheck import gradcheck
from aurora.tensor import Tensor, arange, ones, randn, tensor, zeros
from aurora.version import __version__

__all__ = [
    "AuroraInfo",
    "Tensor",
    "__version__",
    "arange",
    "distributions",
    "environments",
    "get_system_info",
    "gradcheck",
    "load_checkpoint",
    "nn",
    "ones",
    "optim",
    "randn",
    "rl",
    "save_checkpoint",
    "tensor",
    "zeros",
]
