"""Uncertainty-calibrated adaptive horizon scheduling and dynamic blending controllers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class AdaptiveHorizonScheduler:
    """Calculates state-specific imagination horizons based on epistemic uncertainty."""

    def __init__(
        self,
        horizon_max: int = 15,
        horizon_min: int = 1,
        tau_base: float = 0.5,
        kappa: float = 1.0,
        budget_max: float = 2.0,
        gamma: float = 0.99,
    ) -> None:
        self.horizon_max = horizon_max
        self.horizon_min = horizon_min
        self.tau_base = tau_base
        self.kappa = kappa
        self.budget_max = budget_max
        self.gamma = gamma
        self.current_tau = tau_base

    def update_threshold(self, validation_loss: float) -> float:
        """Calibrate disagreement threshold based on dynamics validation error.

        tau_threshold = tau_base * exp(-kappa * min(1.0, max(0.0, val_loss)))
        """
        clamped_loss = min(1.0, max(0.0, float(validation_loss)))
        self.current_tau = self.tau_base * math.exp(-self.kappa * clamped_loss)
        return self.current_tau

    def should_truncate(
        self,
        step: int,
        epistemic_uncertainty: float,
        cumulative_budget: float,
    ) -> tuple[bool, float]:
        """Determine whether an imagined rollout branch should terminate.

        Args:
            step: current imagination step (0-indexed).
            epistemic_uncertainty: scalar disagreement at current step.
            cumulative_budget: accumulated discounted uncertainty along branch.

        Returns:
            truncate: whether to terminate the branch.
            new_budget: updated cumulative discounted uncertainty.
        """
        if step < self.horizon_min:
            new_budget = cumulative_budget + (self.gamma**step) * epistemic_uncertainty
            return False, new_budget

        if step >= self.horizon_max:
            return True, cumulative_budget

        # 1. Peak uncertainty threshold check
        if epistemic_uncertainty > self.current_tau:
            return True, cumulative_budget

        # 2. Cumulative discounted uncertainty budget check
        new_budget = cumulative_budget + (self.gamma**step) * epistemic_uncertainty
        if new_budget > self.budget_max:
            return True, new_budget

        return False, new_budget

    def compute_adaptive_horizon(self, epistemic_trajectory: Sequence[float] | np.ndarray) -> int:
        """Compute the effective rollout horizon H* for a sequence of uncertainties."""
        u_arr = np.asarray(epistemic_trajectory, dtype=np.float64)
        cum_budget = 0.0

        for h, u_val in enumerate(u_arr):
            trunc, cum_budget = self.should_truncate(h, float(u_val), cum_budget)
            if trunc:
                return max(self.horizon_min, h)

        return min(self.horizon_max, len(u_arr))


class DynamicBlendingController:
    """Dynamically regulates the real-to-synthetic experience replay ratio."""

    def __init__(
        self,
        eta_max: float = 0.9,
        eta_min: float = 0.0,
        u_target: float = 0.5,
        momentum: float = 0.8,
    ) -> None:
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.u_target = u_target
        self.momentum = momentum
        self.current_eta = eta_min

    def compute_ratio(self, mean_epistemic_uncertainty: float) -> float:
        """Compute synthetic experience ratio eta_t based on mean model uncertainty.

        eta_raw = eta_max * max(0.0, 1.0 - min(1.0, u_bar / u_target))
        """
        u = max(0.0, float(mean_epistemic_uncertainty))
        scaled_u = min(1.0, u / max(1e-8, self.u_target))
        raw_eta = self.eta_max * (1.0 - scaled_u)
        raw_eta = max(self.eta_min, min(self.eta_max, raw_eta))

        # Exponential moving average smoothing
        self.current_eta = self.momentum * self.current_eta + (1.0 - self.momentum) * raw_eta
        return float(self.current_eta)
