"""Statistical hypothesis testing for algorithm comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats as st


@dataclass(frozen=True)
class HypothesisTestResult:
    """Outcome of a two-sample statistical test."""

    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    test_name: str


def welch_t_test(
    x_scores: Sequence[float] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Welch's two-sample t-test (unequal variances assumed).

    Tests H0: mu_x == mu_y against H1: mu_x != mu_y.
    """
    x = np.asarray(x_scores, dtype=np.float64)
    y = np.asarray(y_scores, dtype=np.float64)

    res = st.ttest_ind(x, y, equal_var=False)
    p_val = float(res.pvalue)
    stat = float(res.statistic)

    return HypothesisTestResult(
        statistic=stat,
        p_value=p_val,
        is_significant=bool(p_val < alpha),
        alpha=alpha,
        test_name="Welch t-test",
    )


def mann_whitney_u_test(
    x_scores: Sequence[float] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Mann-Whitney U rank-sum test for non-parametric distribution comparison."""
    x = np.asarray(x_scores, dtype=np.float64)
    y = np.asarray(y_scores, dtype=np.float64)

    res = st.mannwhitneyu(x, y, alternative="two-sided")
    p_val = float(res.pvalue)
    stat = float(res.statistic)

    return HypothesisTestResult(
        statistic=stat,
        p_value=p_val,
        is_significant=bool(p_val < alpha),
        alpha=alpha,
        test_name="Mann-Whitney U",
    )
