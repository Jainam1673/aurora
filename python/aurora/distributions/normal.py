"""Diagonal Gaussian (Normal) distribution for continuous action spaces."""

from __future__ import annotations

import math

import numpy as np

from aurora.distributions.distribution import Distribution
from aurora.tensor import Tensor, tensor


class Normal(Distribution):
    """Independent diagonal Gaussian distribution."""

    def __init__(self, loc: Tensor, scale: Tensor) -> None:
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample without gradient tracking."""
        shape = sample_shape + self.loc.shape
        eps = np.random.randn(*shape)
        return tensor(self.loc.numpy() + self.scale.numpy() * eps, requires_grad=False)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample using pathwise reparameterization trick: x = loc + scale * eps."""
        shape = sample_shape + self.loc.shape
        eps = tensor(np.random.randn(*shape), requires_grad=False)
        return self.loc + self.scale * eps

    def log_prob(self, value: Tensor, sum_features: bool = True) -> Tensor:
        """Evaluate Gaussian log density."""
        var = self.scale * self.scale
        diff = value - self.loc
        log_scale = self.scale.log()
        c = math.log(2.0 * math.pi)

        # -0.5 * ((x - loc)^2 / scale^2 + 2 * log(scale) + log(2*pi))
        elem_log_p = -0.5 * (diff * diff / var + log_scale * 2.0 + c)
        if sum_features:
            return elem_log_p.sum(axis=-1)
        return elem_log_p

    def entropy(self, sum_features: bool = True) -> Tensor:
        """Evaluate differential entropy: 0.5 * (1 + log(2*pi) + 2 * log(scale))."""
        c = 0.5 * (1.0 + math.log(2.0 * math.pi))
        elem_ent = self.scale.log() + c
        if sum_features:
            return elem_ent.sum(axis=-1)
        return elem_ent

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.scale * self.scale
