"""Base Optimizer abstraction and gradient clipping utilities for AURORA."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from aurora.nn.parameter import Parameter


def clip_grad_norm(
    parameters: Iterable[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clips gradient norm of an iterable of parameters."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(float(np.max(np.abs(g.numpy()))) for g in grads)
    else:
        total_norm_sq = float(sum(float(np.sum(g.numpy() ** norm_type)) for g in grads))
        total_norm = float(total_norm_sq ** (1.0 / norm_type))

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.data *= clip_coef

    return total_norm


def clip_grad_value(parameters: Iterable[Parameter], clip_value: float) -> None:
    """Clips gradient of an iterable of parameters at specified value."""
    for p in parameters:
        if p.grad is not None:
            np.clip(p.grad.numpy(), -clip_value, clip_value, out=p.grad.data)


class Optimizer:
    """Base class for all first-principles optimizers."""

    def __init__(self, params: Iterable[Parameter], defaults: dict[str, Any]) -> None:
        self.params: list[Parameter] = list(params)
        self.defaults = defaults
        self.state: dict[int, dict[str, Any]] = {}
        self.step_count: int = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer state as a dictionary."""
        packed_state: dict[int, dict[str, Any]] = {}
        for idx, p in enumerate(self.params):
            pid = id(p)
            if pid in self.state:
                p_state = self.state[pid]
                serialized: dict[str, Any] = {}
                for k, v in p_state.items():
                    if isinstance(v, np.ndarray):
                        serialized[k] = {
                            "shape": list(v.shape),
                            "data": [float(x) for x in v.flatten()],
                        }
                    else:
                        serialized[k] = v
                packed_state[idx] = serialized

        return {
            "step_count": self.step_count,
            "defaults": self.defaults,
            "state": packed_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state."""
        self.step_count = state_dict["step_count"]
        self.defaults.update(state_dict["defaults"])
        saved_state = state_dict["state"]

        for idx, p in enumerate(self.params):
            if idx in saved_state or str(idx) in saved_state:
                key = idx if idx in saved_state else str(idx)
                entry = saved_state[key]
                p_state: dict[str, Any] = {}
                for k, v in entry.items():
                    if isinstance(v, dict) and "shape" in v and "data" in v:
                        arr = np.array(v["data"], dtype=np.float64)
                        p_state[k] = arr.reshape(tuple(v["shape"]))
                    else:
                        p_state[k] = v
                self.state[id(p)] = p_state
