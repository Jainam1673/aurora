# Scientific Gaps & Experimental Shortcomings in AURORA

**Auditor:** Principal Research Scientist, Skeptical ML Reviewer, Reproducibility Auditor  
**Date:** 2026-10-03

---

## 1. Primary Scientific Gaps

### Gap 1: Uncertainty Calibration Is Completely Unmeasured
- **The Claim:** The paper and README assert that epistemic ensemble disagreement $u_{\text{epi}}(s, a)$ accurately quantifies world model prediction failure and detects out-of-distribution transitions.
- **The Reality:** No experiment in the repository measures the empirical correlation between $u_{\text{epi}}$ and ground-truth model error $\|s_{t+h} - \hat{s}_{t+h}\|_2$.
- **Required Fix:** A dedicated calibration experiment measuring:
  1. Scatter plots and binned calibration curves of $u_{\text{epi}}$ vs. ground-truth multi-step prediction error.
  2. Spearman rank correlation coefficient across horizons $h \in \{1, 3, 5, 8\}$.
  3. Calibration error metrics (e.g. Expected Calibration Error or uncertainty-error monotonic rank ordering).

---

### Gap 2: Distribution Shift & OOD Detection Is Not Validated
- **The Claim:** Section 4.2 claims that when transitions enter unfamiliar out-of-distribution states, $\bar{u}_t \gg u_{\text{target}}$, triggering a graceful fallback to model-free learning ($\eta_t \to 0$).
- **The Reality:** No experiment tests whether $u_{\text{epi}}$ actually increases when querying near-OOD and far-OOD states (e.g., perturbing state dimensions outside the training support).
- **Required Fix:** A controlled experiment collecting in-distribution data, perturbing state features into OOD domains, and logging uncertainty distributions to verify if $u_{\text{epi}}(\text{OOD}) \gg u_{\text{epi}}(\text{In-Dist})$.

---

### Gap 3: Controlled Horizon Comparison vs. Fixed Horizons
- **The Claim:** Adaptive Horizon Scheduling outperforms fixed-horizon rollouts.
- **The Reality:** The existing ablation only tested $H=5$ (where truncation was still partially active due to budget checks) against adaptive horizon. It never systematically compared:
  $$H \in \{1, 2, 4, 8, 12, 16\} \quad \text{vs.} \quad \text{Adaptive } H$$
  A skeptical reviewer will ask: *"Does adaptive horizon actually beat an optimal fixed horizon like $H=4$ or $H=6$, or does it simply approximate a good fixed horizon?"*
- **Required Fix:** Systematic sweep comparing fixed horizons $H \in \{1, 2, 4, 8, 12\}$ against Adaptive Horizon on the same training budgets and random seeds.

---

### Gap 4: Controlled Blending Ratio Sweep vs. Fixed $\eta$
- **The Claim:** Dynamic Experience Blending ($\eta_t$) is necessary to protect against model corruption.
- **The Reality:** In the raw ablation results, `No Dynamic Blending` with fixed $\eta = 0.5$ actually achieved a higher return (-1391.11) than Full AURORA (-1446.27)!
- **Required Fix:** Controlled sweep comparing fixed ratios $\eta \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ against adaptive $\eta_t$.

---

### Gap 5: Pessimistic Value Optimization Mechanism & Gradient Flow
- **The Claim:** $\tilde{Q}(s, a) = \min_j Q_j(s, a) - \beta_{\text{pess}} u_{\text{epi}}(s, a)$ regularizes policy optimization.
- **The Reality:**
  1. In `aurora_agent.py`, `pess_penalty` had `requires_grad=False` and zero gradient connection to actor parameters $\theta$.
  2. In the critic Bellman backup, the target value did not include pessimism.
  3. Setting $\beta_{\text{pess}} = 0$ produced identical results down to float precision.
- **Required Fix:**
  - Connect the pessimism penalty properly so that policy optimization actually penalizes actions with high epistemic variance. Specifically:
    Either:
    (a) Backpropagate through the ensemble's epistemic variance $\nabla_a u_{\text{epi}}(s, a)$ with autograd active, OR
    (b) Penalize the synthetic rewards / critic Bellman targets directly (as in MOPO / MOReL: $\tilde{r}(s, a) = r(s, a) - \beta_{\text{pess}} u_{\text{epi}}(s, a)$), which is theoretically justified under the Simulation Lemma!

---

### Gap 6: Single-Environment Limitation & Evaluation Breadth
- **The Claim:** The paper presents general claims about model-based RL, but evaluations were run solely on Pendulum for 300 environment steps.
- **Required Fix:** Verify behavior across multiple environments (e.g. CartPole continuous/discrete, InvertedPendulum) or explicitly document the single-environment scope as a clear limitation in the paper and README.

---

### Gap 7: Mathematical Rigor in Theorem 1 Proof
- **The Claim:** Monotonic policy improvement is mathematically guaranteed.
- **The Reality:** Assumption 1 assumes $D_{\text{TV}}(\mathcal{P}, \hat{\mathcal{P}}) \le C_u u_{\text{epi}}$, which is an asymptotic or heuristic conjecture for finite neural network ensembles, not a proved fact.
- **Required Fix:** Rephrase Theorem 1 as a **Proposition** or **Conditional Monotonic Improvement Bound**, explicitly identifying Assumption 1 as a modeling hypothesis and deriving the precise condition under which the bound holds.
