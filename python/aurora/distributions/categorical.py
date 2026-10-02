"""Categorical distribution for discrete action spaces."""

from __future__ import annotations

import numpy as np

from aurora.distributions.distribution import Distribution
from aurora.tensor import Tensor, tensor


class Categorical(Distribution):
    """Categorical distribution parameterized by unnormalized logits or probabilities."""

    def __init__(
        self,
        logits: Tensor | None = None,
        probs: Tensor | None = None,
    ) -> None:
        if (logits is None) == (probs is None):
            raise ValueError("Exactly one of logits or probs must be specified")

        if probs is not None:
            p_data = np.clip(probs.numpy(), 1e-12, 1.0)
            self._logits = tensor(np.log(p_data), requires_grad=probs.requires_grad)
            self._probs = probs
        else:
            assert logits is not None
            self._logits = logits
            self._probs = logits.softmax(axis=-1)

        self._num_classes = self._logits.shape[-1]

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def probs(self) -> Tensor:
        return self._probs

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample discrete category index according to class probabilities."""
        p_np = self._probs.numpy()
        flat_p = p_np.reshape(-1, self._num_classes)
        # Normalize slightly to ensure exact sum to 1 in float64
        flat_p = flat_p / np.sum(flat_p, axis=-1, keepdims=True)

        samples = np.empty(flat_p.shape[0], dtype=np.int64)
        for i in range(flat_p.shape[0]):
            samples[i] = np.random.choice(self._num_classes, p=flat_p[i])

        out_shape = sample_shape + self._probs.shape[:-1]
        return tensor(samples.reshape(out_shape), requires_grad=False)

    def log_prob(self, value: Tensor) -> Tensor:
        """Compute log probability of chosen category index."""
        log_p = self._logits.log_softmax(axis=-1)
        val_np = value.numpy().astype(np.int64)

        # One-hot encoding for differentiable gathering
        one_hot = np.zeros((*val_np.shape, self._num_classes), dtype=np.float64)
        np.put_along_axis(one_hot, np.expand_dims(val_np, -1), 1.0, axis=-1)

        one_hot_t = tensor(one_hot, requires_grad=False)
        selected_log_p = (one_hot_t * log_p).sum(axis=-1)
        return selected_log_p

    def entropy(self) -> Tensor:
        """Compute Shannon entropy: H = - sum(p * log_p)."""
        log_p = self._logits.log_softmax(axis=-1)
        p = self._probs
        return -(p * log_p).sum(axis=-1)

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError("Mean is not uniquely defined for categorical categories")

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError("Variance is not uniquely defined for categorical categories")
