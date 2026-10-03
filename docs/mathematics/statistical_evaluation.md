# Scientific Benchmarking and Statistical Evaluation Protocol

## 1. Motivation and First Principles

Reinforcement Learning empirical evaluations frequently suffer from statistical reproducibility failures, including:
1. **Reporting only high-variance point estimates** (e.g., standard mean $\pm$ standard deviation across 3–5 seeds), which are skewed by extreme outliers or lucky initialization seeds.
2. **Ignoring heavy-tailed performance distributions** common in model-based RL where failure to learn a world model leads to catastrophic collapse.
3. **Unpaired or unfair compute budgets** where methods are compared with differing environment interactions or model forward passes.

AURORA addresses these challenges by implementing the statistical protocol recommended by Agarwal et al. (2021) (*Deep Reinforcement Learning at the Edge of the Statistical Precipice*):
- **Interquartile Mean (IQM)** as the primary robust location estimator.
- **Stratified Percentile Bootstrap Confidence Intervals** (95% CIs) over empirical evaluation trajectories.
- **Performance Profiles** demonstrating performance probability across continuous score thresholds.
- **Probability of Improvement** quantifying the exact likelihood that method $X$ outperforms baseline $Y$.

---

## 2. Mathematical Formulations

### 2.1 Interquartile Mean (IQM)

Let $X = (x_{(1)}, x_{(2)}, \dots, x_{(N)})$ be the sorted order statistics of $N$ evaluation returns from multiple random seeds and test rollouts:

$$x_{(1)} \le x_{(2)} \le \dots \le x_{(N)}$$

The Interquartile Mean (IQM) trims the lower 25% and upper 25% of scores, discarding extreme outliers and catastrophic failures, while retaining the central 50% distribution:

$$\text{IQM}(X) = \frac{1}{M} \sum_{i = k_1 + 1}^{N - k_2} x_{(i)}$$

where:
- $k_1 = \lfloor N / 4 \rfloor$
- $k_2 = \lfloor N / 4 \rfloor$
- $M = N - k_1 - k_2$

For fractional sample sizes, linear interpolation or exact discrete weighting over the $[0.25, 0.75]$ quantile interval is applied:

$$\text{IQM}(X) = \frac{1}{0.5} \int_{0.25}^{0.75} F_N^{-1}(p) \, dp$$

where $F_N^{-1}(p)$ is the empirical quantile function.

---

### 2.2 Stratified Percentile Bootstrap Confidence Intervals

To capture sample uncertainty without assuming parametric normality, we employ non-parametric bootstrap resampling:

1. Given sample $X = (x_1, \dots, x_N)$ of evaluation returns.
2. Draw $B$ independent resamples $X^{*(1)}, \dots, X^{*(B)}$ with replacement from $X$, each of size $N$.
3. Compute the point estimator on each resample:
   $$\hat{\theta}^{*(b)} = \text{Estimator}(X^{*(b)}), \quad b \in \{1, \dots, B\}$$
   where $\text{Estimator} \in \{\text{Mean}, \text{Median}, \text{IQM}\}$.
4. Sort the bootstrap estimates $\hat{\theta}^{*(1)} \le \dots \le \hat{\theta}^{*(B)}$.
5. For confidence level $1 - \alpha$ (e.g. $\alpha = 0.05$ for a 95% confidence interval), the percentile bootstrap confidence interval $[L, U]$ is:
   $$L = \hat{\theta}^{*(\lfloor B \cdot (\alpha / 2) \rfloor)}$$
   $$U = \hat{\theta}^{*(\lceil B \cdot (1 - \alpha / 2) \rceil)}$$

---

### 2.3 Performance Profiles

A performance profile plots the empirical fraction of evaluation runs that achieve a return equal to or exceeding a performance threshold $\tau$:

$$\hat{F}_X(\tau) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(x_i \ge \tau)$$

Properties:
- $\hat{F}_X(\tau) \in [0, 1]$ is monotonically non-increasing in $\tau$.
- Dominance: If $\hat{F}_A(\tau) \ge \hat{F}_B(\tau)$ for all $\tau \in [\tau_{\min}, \tau_{\max}]$, algorithm $A$ stochastically dominates algorithm $B$.
- The area under the performance profile (normalized over $[\tau_{\min}, \tau_{\max}]$) provides an aggregate measure of performance across both easy and difficult regimes.

---

### 2.4 Probability of Improvement

Given two algorithms $X$ and $Y$ with evaluation samples $(x_1, \dots, x_{N_X})$ and $(y_1, \dots, y_{N_Y})$, the probability of improvement $P(X > Y)$ (Mann-Whitney / Wilcoxon statistic) is:

$$P(X > Y) = \frac{1}{N_X N_Y} \sum_{i=1}^{N_X} \sum_{j=1}^{N_Y} \left[ \mathbf{1}(x_i > y_j) + \frac{1}{2} \mathbf{1}(x_i = y_j) \right]$$

Interpretation:
- $P(X > Y) = 0.5$: algorithms are equally effective.
- $P(X > Y) > 0.5$: algorithm $X$ is probabilistically superior to $Y$.
- $P(X > Y) \ge 0.75$: strong, consistent superiority with negligible probability of inversion.

---

### 2.5 Hypothesis Testing: Welch's Two-Sample $t$-Test

To test the null hypothesis $H_0: \mu_X = \mu_Y$ without assuming equal variances:

$$t = \frac{\bar{X} - \bar{Y}}{\sqrt{\frac{s_X^2}{N_X} + \frac{s_Y^2}{N_Y}}}$$

Degrees of freedom by Welch-Satterthwaite:

$$\nu \approx \frac{\left( \frac{s_X^2}{N_X} + \frac{s_Y^2}{N_Y} \right)^2}{\frac{(s_X^2 / N_X)^2}{N_X - 1} + \frac{(s_Y^2 / N_Y)^2}{N_Y - 1}}$$

p-value computed via Student's $t$ cumulative distribution function:

$$p = 2 \cdot (1 - F_t(|t|, \nu))$$

---

## 3. Component Ablation Protocol

To prove that each novel component of the AURORA architecture contributes essentially to its sample efficiency and planning stability, we evaluate 4 standardized configurations:

1. **Full AURORA**:
   - Adaptive imagination horizon $H^*(s)$ with error-decay threshold $\tau_{\text{threshold}}$.
   - Momentum-smoothed dynamic synthetic ratio $\eta_t$.
   - Epistemic risk-sensitive pessimistic value penalty $\beta_{\text{pess}} > 0$.
   - Active epistemic exploration trigger $\tau_{\text{active}}$.
2. **Ablation 1 (No Adaptive Horizon)**:
   - Fixed horizon $H = 5$ for all states regardless of uncertainty.
   - Dynamic blending and pessimistic value active.
3. **Ablation 2 (No Dynamic Blending)**:
   - Fixed synthetic experience blend $\eta = 0.50$.
   - Adaptive horizon and pessimistic value active.
4. **Ablation 3 (No Pessimistic Penalty)**:
   - Standard non-penalized critic $\tilde{Q}(s, a) = \min_j Q_j(s, a)$ ($\beta_{\text{pess}} = 0$).
   - Adaptive horizon and dynamic blending active.

Each configuration is tested across identical random seeds with identical total environment interaction budgets.

---

## 4. Experiment Manifest Specification

Every experiment run must record an immutable JSON manifest containing:
```json
{
  "manifest_version": "1.0.0",
  "experiment_id": "exp_pendulum_aurora_seed42",
  "timestamp_iso": "2026-10-03T21:15:00Z",
  "git": {
    "commit": "e49ecc0dd1e3498e44a7dfa291bf0d6d5b67d3e3",
    "branch": "main",
    "dirty": false
  },
  "environment": {
    "name": "Pendulum",
    "obs_dim": 3,
    "action_dim": 1,
    "max_steps": 200
  },
  "system": {
    "python_version": "3.14.8",
    "compiler": "GCC 16.2.1",
    "os": "Linux x86_64",
    "cpu": "Intel Core i5-8265U"
  },
  "hyperparameters": {
    "seed": 42,
    "env_steps": 500,
    "ensemble_size": 3,
    "tau_base": 0.4,
    "beta_pess": 0.5,
    "eta_max": 0.85
  },
  "metrics": {
    "eval_return_mean": -184.2,
    "eval_return_iqm": -175.6,
    "eval_return_ci_95": [-210.4, -152.8]
  }
}
```
