"""AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents.

A research-grade platform for uncertainty-aware model-based reinforcement learning.
"""

from aurora.core import AuroraInfo, get_system_info
from aurora.version import __version__

__all__ = ["AuroraInfo", "__version__", "get_system_info"]
