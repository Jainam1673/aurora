"""Uncertainty quantification and decomposition for deep probabilistic ensembles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aurora.tensor import Tensor


@dataclass(slots=True)
class UncertaintyMetrics:
    """Container for decomposed predictive uncertainties."""

    mean: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    total: np.ndarray
    disagreement: np.ndarray  # Max epistemic variance across feature dimensions


class UncertaintyEstimator:
    """Decomposes predictive uncertainty into aleatoric and epistemic components.

    Given ensemble predictions {mu_e, sigma_e^2}_{e=1}^E:
    - Ensemble Mean: bar{mu} = (1 / E) * sum_{e=1}^E mu_e
    - Aleatoric Uncertainty: U_aleatoric = (1 / E) * sum_{e=1}^E sigma_e^2
    - Epistemic Uncertainty: U_epistemic = (1 / E) * sum_{e=1}^E (mu_e - bar{mu})^2
    - Total Predictive Variance: U_total = U_aleatoric + U_epistemic
    """

    @staticmethod
    def decompose_np(
        means: np.ndarray | Sequence[np.ndarray],
        variances: np.ndarray | Sequence[np.ndarray],
    ) -> UncertaintyMetrics:
        """Decompose uncertainties from numpy arrays.

        Args:
            means: array of shape (E, ..., D) or sequence of E arrays of shape (..., D)
            variances: array of shape (E, ..., D) or sequence of E arrays of shape (..., D)
        """
        if isinstance(means, np.ndarray) and means.ndim >= 2:
            stacked_means = means
        else:
            stacked_means = np.stack(list(means), axis=0)

        if isinstance(variances, np.ndarray) and variances.ndim >= 2:
            stacked_vars = variances
        else:
            stacked_vars = np.stack(list(variances), axis=0)

        # Ensemble mean across members (axis 0)
        ens_mean = np.mean(stacked_means, axis=0)

        # Aleatoric uncertainty: expected observation noise
        aleatoric = np.mean(stacked_vars, axis=0)

        # Epistemic uncertainty: variance of predictions across ensemble
        diff = stacked_means - np.expand_dims(ens_mean, axis=0)
        epistemic = np.mean(diff**2, axis=0)

        total = aleatoric + epistemic
        disagreement = np.max(epistemic, axis=-1)

        return UncertaintyMetrics(
            mean=ens_mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            disagreement=disagreement,
        )

    @staticmethod
    def decompose_tensor(
        means: Sequence[Tensor],
        variances: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Tensor-native differentiable decomposition with autograd support.

        Args:
            means: Sequence of E Tensors each of shape (..., D)
            variances: Sequence of E Tensors each of shape (..., D)

        Returns:
            ens_mean: Tensor of shape (..., D)
            aleatoric: Tensor of shape (..., D)
            epistemic: Tensor of shape (..., D)
            total: Tensor of shape (..., D)
        """
        num_models = float(len(means))
        inv_e = 1.0 / num_models

        # 1. Ensemble mean
        sum_m = means[0]
        for m in means[1:]:
            sum_m = sum_m + m
        ens_mean = sum_m * inv_e

        # 2. Aleatoric uncertainty: average of model variances
        sum_v = variances[0]
        for v in variances[1:]:
            sum_v = sum_v + v
        aleatoric = sum_v * inv_e

        # 3. Epistemic uncertainty: average squared deviations
        d0 = means[0] - ens_mean
        sum_sq = d0 * d0
        for m in means[1:]:
            d = m - ens_mean
            sum_sq = sum_sq + (d * d)
        epistemic = sum_sq * inv_e

        # 4. Total predictive variance
        total = aleatoric + epistemic
        return ens_mean, aleatoric, epistemic, total
