# Theory-to-Implementation Traceability Map

**Auditor:** Principal Research Scientist, Skeptical ML Reviewer, Reproducibility Auditor  
**Date:** 2026-10-03

---

## 1. Theory-to-Code Mapping

| Theoretical Concept / Theorem Assumption | Mathematical Formulation | Code Symbol / Implementation | Empirical Validation Script | Audit Status & Traceability |
|---|---|---|---|:---:|
| **Epistemic Disagreement as Discrepancy Bound** | $D_{\text{TV}}(\mathcal{P}, \hat{\mathcal{P}}) \le C_u u_{\text{epi}}(s, a)$ | `EnsembleDynamicsModel.predict()`: `scalar_epi = np.max(np.var(next_states, axis=0), axis=-1)` | **GAP**: No calibration test exists | `UNVERIFIED_ASSUMPTION`: Theoretical assumption is not empirically validated |
| **Validation-Calibrated Dynamic Horizon Threshold** | $\tau_t = \tau_{\text{base}} \exp(-\kappa \mathcal{L}_{\text{val}})$ | `AdaptiveHorizonScheduler.update_threshold()` in Python & C++ | `test_threshold_decay()` in `test_aurora_algorithm.py` | `TRACEABLE`: Matches mathematical definition exactly |
| **State-Specific Horizon Truncation** | $u_{\text{epi}}(s_h, a_h) > \tau_t \lor \sum_{k=0}^h \gamma^k u_{\text{epi}} > B_{\max}$ | `AdaptiveHorizonScheduler.should_truncate()` in Python & C++ | `test_horizon_truncation()` in `test_aurora_algorithm.py` | `TRACEABLE`: Matches equation |
| **Dynamic Synthetic/Real Replay Blending** | $\eta_t = \rho \eta_{t-1} + (1-\rho)\eta_{\max}[1 - \min(1, \frac{\bar{u}}{u_{\text{target}}})]$ | `DynamicBlendingController.compute_ratio()` in Python & C++ | `test_dynamic_blending_ratio()` in `test_aurora_algorithm.py` | `TRACEABLE`: Matches equation |
| **Epistemic Value Regularization / Pessimism** | $\tilde{Q}(s, a) = \min_j Q_j(s, a) - \beta_{\text{pess}} u_{\text{epi}}(s, a)$ | `AURORAAgent.train_policy()` L423; `cpp/src/aurora_algorithm.cpp` L34 | `tests/python/test_aurora_algorithm.py` | `BROKEN_IMPLEMENTATION`: Gradient detached in actor loss; not applied to critic Bellman targets |
| **Monotonic Value Improvement Bound** | $\eta(\pi_{k+1}) \ge \eta(\pi_k) + \mathbb{E}[\tilde{A}^{\pi_k}] - \mathcal{O}(\frac{\epsilon_m H_k^2}{1-\gamma})$ | Paper Appendix A (Proof of Theorem 1) | `experiments/ablation_study.py` | `PARTIALLY_DERIVED`: Proof assumes empirical pessimism is realized during optimization |
| **Interquartile Mean (IQM) Location Estimator** | $\text{IQM}(X) = \frac{1}{0.5 N} \sum_{i = \lfloor 0.25 N \rfloor + 1}^{\lfloor 0.75 N \rfloor} x_{(i)}$ | `evaluation/metrics.py:compute_iqm()`; `cpp/src/statistical_evaluation.cpp:compute_iqm()` | `test_evaluation.py`; `test_evaluation_parity.py` | `TRACEABLE`: Fully verified $< 10^{-10}$ parity |
| **Stratified Bootstrap Confidence Intervals** | Percentile bootstrap over $R=2000$ resamples | `evaluation/metrics.py:bootstrap_ci()`; `cpp/src/statistical_evaluation.cpp:bootstrap_ci()` | `test_evaluation.py`; `test_evaluation_parity.py` | `TRACEABLE`: Fully verified |

---

## 2. Identified Theoretical-Implementation Gaps

1. **Gap in Theorem 1 Implementation:**
   Theorem 1 requires that policy optimization is guided by $\tilde{Q}(s, a)$ such that $\tilde{\eta}(\pi) \le \eta(\pi)$ forms a lower bound. Because the implementation detached `pess_penalty` with `requires_grad=False`, the actor policy gradient was $\nabla_\theta \min_j Q_j(s, a)$ with zero pessimism contribution! Thus, the mathematical premise of Theorem 1 was broken in code.

2. **Assumption 1 Discrepancy:**
   Assumption 1 equates the Total Variation divergence of the transition distribution to the variance of the mean predictions across ensemble members:
   $$D_{\text{TV}}\left( \mathcal{P}(\cdot \mid s, a), \hat{\mathcal{P}}(\cdot \mid s, a) \right) \le C_u \cdot u_{\text{epi}}(s, a)$$
   This assumption requires validation. If the ensemble is uncalibrated or collapses (e.g. all members make the same erroneous prediction), $u_{\text{epi}} \to 0$ while $D_{\text{TV}} \gg 0$, violating the assumption.
