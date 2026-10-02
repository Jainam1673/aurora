"""Squashed Gaussian (TanhNormal) distribution for bounded continuous actions."""

from __future__ import annotations

import numpy as np

from aurora.distributions.distribution import Distribution
from aurora.distributions.normal import Normal
from aurora.tensor import Tensor, tensor


class TanhNormal(Distribution):
    """Squashed Normal distribution with tanh bijector for bounded actions in [-1, 1]."""

    def __init__(self, loc: Tensor, scale: Tensor, eps: float = 1e-6) -> None:
        self.loc = loc
        self.scale = scale
        self.normal = Normal(loc, scale)
        self.eps = eps

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample without gradient tracking."""
        u = self.normal.sample(sample_shape)
        return u.tanh()

    def rsample_with_pre_tanh(self, sample_shape: tuple[int, ...] = ()) -> tuple[Tensor, Tensor]:
        """Sample with reparameterization trick, returning squashed action and latent."""
        u = self.normal.rsample(sample_shape)
        a = u.tanh()
        return a, u

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample with reparameterization trick: a = tanh(u)."""
        a, _ = self.rsample_with_pre_tanh(sample_shape)
        return a

    def log_prob(self, value: Tensor, pre_tanh_value: Tensor | None = None) -> Tensor:
        """Evaluate squashed log density using change-of-variables log-determinant."""
        if pre_tanh_value is not None:
            u = pre_tanh_value
        else:
            # Invert tanh: u = arctanh(clip(a, -1+eps, 1-eps))
            clipped = np.clip(value.numpy(), -1.0 + self.eps, 1.0 - self.eps)
            u_np = np.arctanh(clipped)
            u = tensor(u_np, requires_grad=value.requires_grad)

        log_p_gaussian = self.normal.log_prob(u, sum_features=False)

        # Log determinant of Jacobian: log(1 - tanh(u)^2 + eps)
        # Using numerically stable softplus identity: 2 * (log(2) - u - softplus(-2u))
        u_np = u.numpy()
        log_det_np = 2.0 * (np.log(2.0) - u_np - np.logaddexp(0.0, -2.0 * u_np))
        log_det = tensor(log_det_np, requires_grad=u.requires_grad)

        log_p = log_p_gaussian - log_det
        return log_p.sum(axis=-1)

    def entropy(self) -> Tensor:
        raise NotImplementedError(
            "Analytic entropy for TanhNormal is intractable; use sample estimation"
        )

    @property
    def mean(self) -> Tensor:
        """Deterministic squashed mean."""
        return self.loc.tanh()

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError("Analytic variance for TanhNormal is intractable")
