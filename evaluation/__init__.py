"""AURORA Scientific Benchmarking and Statistical Evaluation Suite."""

from evaluation.manifest import create_experiment_manifest, save_manifest
from evaluation.metrics import (
    StatisticalSummary,
    bootstrap_ci,
    compute_iqm,
    compute_statistical_summary,
)
from evaluation.plotting import format_ascii_table, plot_performance_profiles
from evaluation.profiles import performance_profile, probability_of_improvement
from evaluation.significance import (
    HypothesisTestResult,
    mann_whitney_u_test,
    welch_t_test,
)

__all__ = [
    "HypothesisTestResult",
    "StatisticalSummary",
    "bootstrap_ci",
    "compute_iqm",
    "compute_statistical_summary",
    "create_experiment_manifest",
    "format_ascii_table",
    "mann_whitney_u_test",
    "performance_profile",
    "plot_performance_profiles",
    "probability_of_improvement",
    "save_manifest",
    "welch_t_test",
]
