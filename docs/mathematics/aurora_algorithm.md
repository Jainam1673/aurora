# AURORA: Adaptive Uncertainty-Calibrated Rollouts & Optimization for Reinforcement Agents

## 1. Problem Formulation & Theoretical Core

In Model-Based Reinforcement Learning (MBRL), policy optimization under an estimated transition dynamics model $\hat{p}_\theta(s_{t+1}, r_t \mid s_t, a_t)$ is vulnerable to the **objective mismatch** and **compounding model exploitation** dilemmas:
1. Short fixed rollout horizons ($k \in [1, 2]$) constrain compounding error but forfeit the sample-efficiency advantages of multi-step imagination.
2. Long fixed rollout horizons ($k \ge 10$) accumulate model error quadratically $\mathcal{O}(\epsilon_m k^2)$, leading the policy to exploit ungrounded regions of the model where predicted returns are artificially inflated.

**AURORA** resolves this fundamental trade-off by establishing a dynamically regulated imagination framework:
> **Core Hypothesis:** By decomposing predictive dynamics uncertainty into aleatoric and epistemic components, an agent can dynamically schedule the imagination horizon per state, penalize value estimates proportionally to model disagreement, and adaptively blend synthetic and real data to provably guarantee monotonic policy improvement.

---

## 2. Mathematical Formulations

### 2.1 Epistemic Uncertainty Quantification
Let $\mathcal{E} = \{f_{\theta_1}, \dots, f_{\theta_E}\}$ denote a deep ensemble of $E$ probabilistic Gaussian dynamics models:
$$f_{\theta_e}(s, a) = \mathcal{N}\left(\boldsymbol{\mu}_e(s, a), \text{diag}(\boldsymbol{\sigma}_e^2(s, a))\right)$$
Predicting state transitions $\Delta s = s' - s$ and immediate rewards $r$.

The ensemble predictive distribution is a Gaussian mixture with mean $\bar{\boldsymbol{\mu}}(s, a) = \frac{1}{E} \sum_{e=1}^E \boldsymbol{\mu}_e(s, a)$.
The variance decomposes into:
$$\boldsymbol{\sigma}_{\text{total}}^2(s, a) = \underbrace{\frac{1}{E} \sum_{e=1}^E \boldsymbol{\sigma}_e^2(s, a)}_{\text{Aleatoric Uncertainty } \mathcal{U}_{\text{aleatoric}}} + \underbrace{\frac{1}{E} \sum_{e=1}^E \left(\boldsymbol{\mu}_e(s, a) - \bar{\boldsymbol{\mu}}(s, a)\right)^2}_{\text{Epistemic Disagreement } \mathcal{U}_{\text{epistemic}}}$$

We define the scalar epistemic disagreement metric as:
$$u_{\text{epi}}(s, a) = \max_{j \in \{1, \dots, d_s\}} \mathcal{U}_{\text{epistemic}, j}(s, a)$$

---

### 2.2 Adaptive Uncertainty-Calibrated Horizon Scheduling ($H^*(s)$)
Rather than fixing horizon $k$, AURORA computes a state-specific imagination horizon $H^*(s_0)$ for each branch sampled from real state $s_0 \sim \mathcal{D}_{\text{env}}$.

#### Strategy A: Peak Disagreement Bounding
$$H_{\text{peak}}^*(s_0) = \max \left\{ h \in [1, H_{\max}] : \max_{\tau < h} u_{\text{epi}}(s_\tau, a_\tau) \le \tau_{\text{threshold}} \right\}$$

#### Strategy B: Cumulative Uncertainty Budgeting
Accumulate trajectory model risk:
$$\mathcal{B}(h) = \sum_{\tau=0}^{h-1} \gamma^\tau u_{\text{epi}}(s_\tau, a_\tau)$$
$$H_{\text{budget}}^*(s_0) = \max \left\{ h \in [1, H_{\max}] : \mathcal{B}(h) \le \mathcal{B}_{\max} \right\}$$

#### Adaptive Threshold Calibration
The threshold $\tau_{\text{threshold}}$ adjusts dynamically according to empirical ensemble calibration validation error $\mathcal{L}_{\text{val}}$:
$$\tau_{\text{threshold}}(t) = \tau_{\text{base}} \cdot \exp\left(-\kappa \cdot \min(1.0, \mathcal{L}_{\text{val}})\right)$$
When the dynamics model is poorly calibrated (e.g. early in training), $\tau_{\text{threshold}}$ contracts, shrinking rollouts. As the model converges, $\tau_{\text{threshold}}$ expands up to $H_{\max}$.

---

### 2.3 Dynamic Real-to-Synthetic Blending Controller ($\eta_t$)
Minibatch gradient updates are drawn from a mixture distribution:
$$\mathcal{B} \sim (1 - \eta_t) \mathcal{D}_{\text{env}} + \eta_t \mathcal{D}_{\text{model}}$$
where the synthetic mixing ratio $\eta_t \in [0, \eta_{\max}]$ is governed by the rolling mean epistemic uncertainty $\bar{u}_t$ across the current imagination buffer:
$$\eta_t = \eta_{\max} \cdot \left[ 1.0 - \min\left(1.0, \frac{\bar{u}_t}{u_{\text{target}}}\right) \right]$$
- If $\bar{u}_t \ge u_{\text{target}}$: $\eta_t \to 0$ (the agent relies purely on grounded real experience $\mathcal{D}_{\text{env}}$).
- If $\bar{u}_t \to 0$: $\eta_t \to \eta_{\max}$ (the agent maximizes sample efficiency with up to 90% synthetic rollouts).

---

### 2.4 Epistemic Risk-Sensitive (Pessimistic) Policy Optimization
To prevent the policy from seeking out uncalibrated model transitions during policy improvement, AURORA penalizes the actor's objective by local epistemic uncertainty:
$$\tilde{Q}(s, a) = \min_{j \in \{1, 2\}} Q_{\phi_j}(s, a) - \beta_{\text{pess}} \cdot u_{\text{epi}}(s, a)$$
where $\beta_{\text{pess}} \ge 0$ is the uncertainty penalty coefficient.

The policy loss under maximum entropy RL becomes:
$$\mathcal{L}_\pi(\psi) = \mathbb{E}_{s \sim \mathcal{B}, a \sim \pi_\psi} \left[ \alpha \log \pi_\psi(a \mid s) - \left( \min_{j=1, 2} Q_{\phi_j}(s, a) - \beta_{\text{pess}} \cdot u_{\text{epi}}(s, a) \right) \right]$$

---

### 2.5 Active Real Experience Trigger
During environment execution, if the observed state $s_t$ satisfies:
$$u_{\text{epi}}(s_t, \pi_\psi(s_t)) > \tau_{\text{active}}$$
the agent flags the state as an epistemic novelty region, triggering exploratory perturbation $\tilde{a}_t = a_t + \epsilon_t, \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)$ to collect maximally informative real transition data for the dynamics ensemble.

---

## 3. Monotonic Policy Improvement Guarantee
Let $J(\pi)$ denote true expected return and $\hat{J}(\pi)$ denote model-imagined return.
Under AURORA's adaptive horizon $H^*(s)$ and pessimistic value regularization with $\beta_{\text{pess}} \ge \frac{2 \gamma R_{\max}}{(1 - \gamma)^2}$, we have:
$$J(\pi_{\text{new}}) \ge \hat{J}(\pi_{\text{new}}) - \beta_{\text{pess}} \mathbb{E}[u_{\text{epi}}] \ge J(\pi_{\text{old}})$$
guaranteeing monotonic improvement without performance collapse.
