"""Stochastic Gradient Descent optimizer with momentum and weight decay."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from aurora.nn.parameter import Parameter
from aurora.optim.optimizer import Optimizer


class SGD(Optimizer):
    """SGD optimizer with optional momentum and L2 weight decay."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        defaults: dict[str, Any] = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        self.step_count += 1
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]

        for p in self.params:
            if p.grad is None:
                continue

            grad_data = p.grad.numpy()
            if weight_decay != 0.0:
                grad_data = grad_data + weight_decay * p.numpy()

            if momentum != 0.0:
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {"momentum_buffer": np.zeros_like(grad_data)}
                buf = self.state[pid]["momentum_buffer"]
                buf[:] = momentum * buf + grad_data
                p.data -= lr * buf
            else:
                p.data -= lr * grad_data
