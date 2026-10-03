"""Performance profiles and probability of improvement estimators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def performance_profile(
    scores: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute empirical performance profile curve across threshold grid.

    tau -> fraction of runs with score >= tau.
    """
    arr = np.asarray(scores, dtype=np.float64)
    thresh = np.asarray(thresholds, dtype=np.float64)

    if arr.size == 0:
        return np.zeros_like(thresh)

    # arr[:, None] >= thresh[None, :]
    passes = arr[:, None] >= thresh[None, :]
    return np.mean(passes, axis=0)


def probability_of_improvement(
    x_scores: Sequence[float] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
) -> float:
    """Compute empirical probability of improvement P(X > Y).

    P(X > Y) = (1 / (Nx * Ny)) * sum_{i, j} [ I(x_i > y_j) + 0.5 * I(x_i == y_j) ]
    """
    x = np.asarray(x_scores, dtype=np.float64).flatten()
    y = np.asarray(y_scores, dtype=np.float64).flatten()

    if x.size == 0 or y.size == 0:
        return 0.5

    # Pairwise comparison matrix
    # shape: (Nx, Ny)
    greater = (x[:, None] > y[None, :]).astype(np.float64)
    equal = (x[:, None] == y[None, :]).astype(np.float64)

    stat = greater + 0.5 * equal
    return float(np.mean(stat))
