# High-Priority Scientific, Methodological & Implementation Risks

**Auditor:** Principal Research Scientist, Skeptical ML Reviewer, Reproducibility Auditor  
**Date:** 2026-10-03

---

## 1. Executive Risk Summary

A top-tier reviewer (NeurIPS / ICLR) would immediately identify four fatal flaws in the current repository:

1. **Fatal Implementation Defect in Pessimistic Value Optimization:**  
   In `python/aurora/algorithm/aurora_agent.py` (lines 421–426), the penalty term `pess_penalty` is computed via detached NumPy arrays and wrapped in a tensor with `requires_grad=False`. In the loss expression:
   $$\mathcal{L}_{\text{actor}} = \frac{1}{B}\sum_{i=1}^B \left( \alpha \log \pi_\theta(a_i \mid s_i) - \min_j Q_j(s_i, a_i) + \text{pess\_penalty}_i \right)$$
   The derivative $\nabla_\theta \text{pess\_penalty}_i \equiv 0$. The pessimistic penalty has zero gradient connection to the actor parameters! It acts as a detached scalar offset to the loss. This completely invalidates Claim C-ALG-4 and explains why `No Pessimism` scored identically to `Full AURORA` in raw experiments.

2. **Severe Discrepancy & Fabricated Table in `README.md`:**  
   The table in `README.md` reports Pendulum scores of `-173.80` (Full), `-193.36` (w/o Adaptive Horizon), `-199.96` (w/o Dynamic Blending), and `-216.51` (w/o Pessimism) with claimed $p$-values of $0.0435, 0.0121, 0.0021$.  
   In reality, `results/ablation/ablation_summary.json` and `paper/table_ablations.tex` report scores around `-1446.27`, and in fact `No Dynamic Blending` scored `-1391.11` (outperforming Full AURORA), while $p$-values were $0.915, 0.458, 1.000$ (no statistical significance). This directly violates Rule A (No invented evidence).

3. **Incomplete Ablation Control Wiring:**  
   In `experiments/runner.py`, the configuration flag `adaptive_horizon: false` was never parsed or acted upon; `AURORAAgent` still truncated rollouts based on cumulative uncertainty budget $B_{\max}$. Furthermore, `pessimistic_penalty: false` was passed in config, but since the penalty was already gradient-inert, setting it to false changed nothing.

4. **Absence of Empirical Uncertainty Calibration & OOD Validation:**  
   The repository claims that ensemble disagreement constitutes epistemic uncertainty and detects out-of-distribution hallucinations. However, no experiment or plot in the repository actually measures whether ensemble variance $u_{\text{epi}}$ correlates with multi-step prediction error $\|s_{t+h} - \hat{s}_{t+h}\|$, nor is there any controlled OOD state evaluation.

5. **Theoretical Overclaim in Theorem 1:**  
   Theorem 1 states monotonic improvement under model error, but requires Assumption 1 ($D_{\text{TV}}(\mathcal{P}, \hat{\mathcal{P}}) \le C_u u_{\text{epi}}$). In deep neural network ensembles, finite variance does not provably upper-bound total variation divergence across arbitrary state distributions. The paper presents this as a formal monotonic guarantee, whereas mathematically it is a conditional bound subject to unvalidated empirical constants.

---

## 2. Risk Classification Matrix

| Risk ID | Severity | Category | Description | Reviewer Reaction |
|:---|:---:|:---:|:---|:---|
| **R-01** | **CRITICAL** | Correctness Bug | Detached zero-gradient in pessimistic actor update | "The central algorithm does not implement what the paper equations describe. Rejection." |
| **R-02** | **CRITICAL** | Scientific Integrity | Fabricated / mismatched table numbers in `README.md` | "The numbers in the README do not match the raw experiment artifacts. Serious ethics concern." |
| **R-03** | **HIGH** | Experimental Control | `ablation_study.py` didn't actually disable adaptive horizon truncation | "The ablation study does not isolate the claimed components." |
| **R-04** | **HIGH** | Scientific Gap | Zero empirical validation of uncertainty calibration ($u_{\text{epi}}$ vs actual error) | "How do we know ensemble disagreement is meaningful calibration rather than arbitrary noise?" |
| **R-05** | **MEDIUM** | Theoretical Overstatement | Monotonic policy improvement presented as unconditioned theorem rather than conditional bound | "Assumption 1 is physically and mathematically unrealistic for neural networks." |
| **R-06** | **MEDIUM** | Generalization Limitation | Empirical results limited exclusively to Pendulum (single continuous task) | "A model-based algorithm cannot claim generality from 300 steps on Inverted Pendulum." |
