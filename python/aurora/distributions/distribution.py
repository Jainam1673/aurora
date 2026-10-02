"""Base distribution interface for AURORA reinforcement learning."""

from __future__ import annotations

from aurora.tensor import Tensor


class Distribution:
    """Abstract base class for probability distributions."""

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample without gradient tracking."""
        raise NotImplementedError

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample with pathwise gradient propagation (reparameterization trick)."""
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluate log probability density or mass."""
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Evaluate Shannon or differential entropy."""
        raise NotImplementedError

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        """Variance of the distribution."""
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        """Standard deviation of the distribution."""
        return self.variance.sqrt()
