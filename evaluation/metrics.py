"""Robust statistical metrics for reinforcement learning evaluation.

Implements Interquartile Mean (IQM), Stratified Bootstrap Confidence Intervals,
and empirical order statistics as recommended by Agarwal et al. (2021).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StatisticalSummary:
    """Summary statistics for empirical evaluation returns."""

    mean: float
    std: float
    median: float
    iqm: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "iqm": self.iqm,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


def compute_iqm(scores: Sequence[float] | np.ndarray) -> float:
    """Compute the Interquartile Mean (IQM) of evaluation scores.

    Trims the lower 25% and upper 25% of scores, returning the mean of the
    central 50% distribution.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.size < 4:
        return float(np.mean(arr))

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    k1 = n // 4
    k2 = n // 4
    trimmed = sorted_arr[k1 : n - k2]
    return float(np.mean(trimmed))


def bootstrap_ci(
    scores: Sequence[float] | np.ndarray,
    estimator: Callable[[np.ndarray], float] = compute_iqm,
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Compute non-parametric percentile bootstrap confidence interval for an estimator.

    Args:
        scores: array of empirical evaluation returns.
        estimator: function computing a scalar summary statistic from a 1D array.
        num_bootstraps: number of bootstrap resamples (default: 2000).
        confidence_level: confidence level in (0, 1) (default: 0.95).
        seed: optional random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) tuple representing the percentile confidence interval.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        val = float(arr[0])
        return val, val

    rng = np.random.RandomState(seed)
    n = len(arr)
    bootstrap_estimates = np.empty(num_bootstraps, dtype=np.float64)

    for b in range(num_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        bootstrap_estimates[b] = estimator(resample)

    bootstrap_estimates.sort()
    alpha = 1.0 - confidence_level
    lower_idx = int(np.floor((alpha / 2.0) * num_bootstraps))
    upper_idx = int(np.ceil((1.0 - alpha / 2.0) * num_bootstraps)) - 1
    lower_idx = max(0, min(num_bootstraps - 1, lower_idx))
    upper_idx = max(0, min(num_bootstraps - 1, upper_idx))

    return float(bootstrap_estimates[lower_idx]), float(bootstrap_estimates[upper_idx])


def compute_statistical_summary(
    scores: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> StatisticalSummary:
    """Compute complete statistical profile including IQM and 95% bootstrap CI."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return StatisticalSummary(
            mean=0.0,
            std=0.0,
            median=0.0,
            iqm=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            min=0.0,
            max=0.0,
            count=0,
        )

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    median_val = float(np.median(arr))
    iqm_val = compute_iqm(arr)
    ci_low, ci_high = bootstrap_ci(
        arr,
        estimator=compute_iqm,
        num_bootstraps=num_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
    )

    return StatisticalSummary(
        mean=mean_val,
        std=std_val,
        median=median_val,
        iqm=iqm_val,
        ci_lower=ci_low,
        ci_upper=ci_high,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        count=int(arr.size),
    )
