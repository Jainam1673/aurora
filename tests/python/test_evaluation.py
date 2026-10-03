"""Unit tests for the scientific benchmarking and statistical evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation import (
    bootstrap_ci,
    compute_iqm,
    compute_statistical_summary,
    create_experiment_manifest,
    mann_whitney_u_test,
    performance_profile,
    probability_of_improvement,
    save_manifest,
    welch_t_test,
)


class TestMetrics:
    """Tests for robust location and dispersion metrics."""

    def test_iqm_outlier_rejection(self) -> None:
        # Array with massive negative and positive outliers
        # Sorted: [-10000, 10, 12, 14, 16, 18, 20, 50000]
        # N=8, trims 2 from each end -> central 4: [12, 14, 16, 18] -> mean = 15.0
        scores = [14.0, 10.0, 50000.0, 18.0, -10000.0, 12.0, 16.0, 20.0]
        iqm = compute_iqm(scores)
        assert pytest.approx(iqm, abs=1e-12) == 15.0

    def test_iqm_edge_cases(self) -> None:
        assert compute_iqm([]) == 0.0
        assert pytest.approx(compute_iqm([5.0]), abs=1e-12) == 5.0
        assert pytest.approx(compute_iqm([2.0, 4.0, 6.0]), abs=1e-12) == 4.0

    def test_bootstrap_ci(self) -> None:
        scores = [10.0, 12.0, 11.0, 13.0, 14.0, 12.5, 11.5, 13.5]
        ci_low, ci_high = bootstrap_ci(scores, num_bootstraps=1000, confidence_level=0.95, seed=42)
        point_iqm = compute_iqm(scores)
        assert ci_low <= point_iqm <= ci_high
        assert ci_low < ci_high

    def test_statistical_summary(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        summary = compute_statistical_summary(scores, seed=42)
        assert summary.count == 8
        assert summary.min == 1.0
        assert summary.max == 8.0
        assert pytest.approx(summary.mean, abs=1e-12) == 4.5
        assert pytest.approx(summary.iqm, abs=1e-12) == 4.5
        assert summary.ci_lower <= summary.iqm <= summary.ci_upper

        d = summary.to_dict()
        assert "iqm" in d
        assert "ci_lower" in d
        assert d["count"] == 8


class TestProfilesAndImprovement:
    """Tests for performance profile curves and probability of improvement."""

    def test_performance_profile(self) -> None:
        scores = [10.0, 20.0, 30.0, 40.0]
        thresholds = [5.0, 15.0, 25.0, 35.0, 45.0]

        prof = performance_profile(scores, thresholds)
        assert prof.shape == (5,)
        assert prof[0] == 1.0
        assert prof[1] == 0.75
        assert prof[2] == 0.50
        assert prof[3] == 0.25
        assert prof[4] == 0.0

    def test_probability_of_improvement(self) -> None:
        # Strictly superior: X = [10, 20], Y = [1, 2]
        assert probability_of_improvement([10.0, 20.0], [1.0, 2.0]) == 1.0

        # Identical
        assert probability_of_improvement([1.0, 2.0], [1.0, 2.0]) == 0.5

        # Partially overlapping with known probability:
        # X = [2, 4], Y = [1, 3] -> (2>1: 1, 2>3: 0, 4>1: 1, 4>3: 1) / 4 = 0.75
        assert probability_of_improvement([2.0, 4.0], [1.0, 3.0]) == 0.75


class TestSignificance:
    """Tests for statistical hypothesis testing."""

    def test_welch_t_test_and_mann_whitney(self) -> None:
        rng = np.random.RandomState(42)
        x = rng.normal(loc=10.0, scale=1.0, size=50)
        y = rng.normal(loc=5.0, scale=1.0, size=50)

        res_t = welch_t_test(x, y)
        assert res_t.is_significant
        assert res_t.p_value < 1e-5

        res_u = mann_whitney_u_test(x, y)
        assert res_u.is_significant
        assert res_u.p_value < 1e-5

        # Same distribution should fail to reject H0
        x_same = rng.normal(loc=5.0, scale=1.0, size=50)
        res_same = welch_t_test(y, x_same)
        assert not res_same.is_significant
        assert res_same.p_value > 0.05


class TestManifest:
    """Tests for immutable experiment manifest generation and persistence."""

    def test_manifest_creation_and_save(self, tmp_path: Path) -> None:
        manifest = create_experiment_manifest(
            experiment_id="test_exp_001",
            environment_name="Pendulum",
            algorithm_name="AURORA",
            hyperparameters={"lr": 3e-4, "horizon_max": 15},
            metrics={"iqm_return": -150.0},
            seed=123,
        )

        assert manifest["manifest_version"] == "1.0.0"
        assert manifest["experiment_id"] == "test_exp_001"
        assert manifest["algorithm"] == "AURORA"
        assert manifest["seed"] == 123
        assert "git" in manifest
        assert "system" in manifest

        out_file = tmp_path / "manifest.json"
        save_manifest(manifest, out_file)
        assert out_file.exists()

        with open(out_file, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["experiment_id"] == "test_exp_001"
        assert loaded["hyperparameters"]["lr"] == 3e-4
