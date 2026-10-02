"""Learning rate schedulers for AURORA optimizers."""

from __future__ import annotations

import math

from aurora.optim.optimizer import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_step: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = float(optimizer.defaults["lr"])
        self.last_step = last_step
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.last_step += 1
        lr = self.get_lr()
        self.optimizer.defaults["lr"] = lr


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler."""

    def get_lr(self) -> float:
        return self.base_lr


class LinearWarmupDecayLR(LRScheduler):
    """Linear warmup followed by linear decay scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_step: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_step)

    def get_lr(self) -> float:
        if self.last_step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * float(self.last_step) / float(self.warmup_steps)
        if self.last_step >= self.total_steps:
            return self.min_lr

        decay_ratio = float(self.total_steps - self.last_step) / float(
            self.total_steps - self.warmup_steps
        )
        return self.min_lr + (self.base_lr - self.min_lr) * decay_ratio


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule with optional warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_step: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_step)

    def get_lr(self) -> float:
        if self.last_step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * float(self.last_step) / float(self.warmup_steps)
        if self.last_step >= self.total_steps:
            return self.min_lr

        progress = float(self.last_step - self.warmup_steps) / float(
            self.total_steps - self.warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
