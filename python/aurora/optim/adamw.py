"""AdamW optimizer with decoupled weight decay."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from aurora.nn.parameter import Parameter
from aurora.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer (Loshchilov & Hutter, 2019) with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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

            # Decoupled weight decay applied directly to parameter
            if weight_decay != 0.0:
                p.data -= lr * weight_decay * p.numpy()

            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {
                    "exp_avg": np.zeros_like(grad_data),
                    "exp_avg_sq": np.zeros_like(grad_data),
                }

            exp_avg = self.state[pid]["exp_avg"]
            exp_avg_sq = self.state[pid]["exp_avg_sq"]

            # Update biased 1st and 2nd moments
            exp_avg[:] = beta1 * exp_avg + (1.0 - beta1) * grad_data
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1.0 - beta2) * (grad_data**2)

            # Bias correction
            bias_corr1 = 1.0 - beta1**self.step_count
            bias_corr2 = 1.0 - beta2**self.step_count

            m_hat = exp_avg / bias_corr1
            v_hat = exp_avg_sq / bias_corr2

            # Parameter update
            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
