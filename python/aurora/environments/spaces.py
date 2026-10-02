"""State and action space definitions for AURORA environments."""

from __future__ import annotations

from typing import Any

import numpy as np


class Space:
    """Abstract base class for environment observation and action spaces."""

    shape: tuple[int, ...] = ()

    def sample(self) -> Any:
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """Discrete space over integers {0, 1, ..., n - 1}."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"Discrete space size must be positive, got {n}")
        self.n = n
        self.shape: tuple[int, ...] = ()

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))

    def contains(self, x: Any) -> bool:
        if isinstance(x, (int, np.integer)):
            return bool(0 <= x < self.n)
        return False

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box(Space):
    """Continuous space bounded by lower and upper coordinate vectors."""

    def __init__(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float64,
    ) -> None:
        if isinstance(low, (int, float)):
            assert shape is not None, "shape must be specified when low is scalar"
            self.low: np.ndarray = np.full(shape, low, dtype=dtype)
        else:
            self.low = np.asarray(low, dtype=dtype)

        if isinstance(high, (int, float)):
            assert shape is not None, "shape must be specified when high is scalar"
            self.high: np.ndarray = np.full(shape, high, dtype=dtype)
        else:
            self.high = np.asarray(high, dtype=dtype)

        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        # Uniform sampling within bounds (clipping unbounded infinities to reasonable range)
        low_clip = np.where(np.isneginf(self.low), -1e3, self.low)
        high_clip = np.where(np.isposinf(self.high), 1e3, self.high)
        return np.random.uniform(low_clip, high_clip, size=self.shape).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return (
            arr.shape == self.shape
            and bool(np.all(arr >= self.low))
            and bool(np.all(arr <= self.high))
        )

    def __repr__(self) -> str:
        return f"Box(shape={self.shape}, dtype={self.dtype})"
