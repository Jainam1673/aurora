"""Adam optimizer implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from aurora.nn.parameter import Parameter
from aurora.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2014) with optional L2 weight decay."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults: dict[str, Any] = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        self.step_count += 1
        lr = self.defaults["lr"]
        beta1 = self.defaults["beta1"]
        beta2 = self.defaults["beta2"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        for p in self.params:
            if p.grad is None:
                continue

            grad_data = p.grad.numpy()
            if weight_decay != 0.0:
                grad_data = grad_data + weight_decay * p.numpy()

            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {
                    "exp_avg": np.zeros_like(grad_data),
                    "exp_avg_sq": np.zeros_like(grad_data),
                }

            exp_avg = self.state[pid]["exp_avg"]
            exp_avg_sq = self.state[pid]["exp_avg_sq"]

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg[:] = beta1 * exp_avg + (1.0 - beta1) * grad_data
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1.0 - beta2) * (grad_data**2)

            # Bias correction
            bias_corr1 = 1.0 - beta1**self.step_count
            bias_corr2 = 1.0 - beta2**self.step_count

            m_hat = exp_avg / bias_corr1
            v_hat = exp_avg_sq / bias_corr2

            # Parameter update
            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
