# Mathematical Specification: Latent World Models & Uncertainty Calibration

## 1. Overview & Problem Formulation

In Model-Based Reinforcement Learning (MBRL), the agent interacts with an environment governed by transition probability distribution $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ and reward function $\mathcal{R}(s_t, a_t)$. 
A learned world model constructs parametric approximations $\hat{\mathcal{P}}_\theta$ and $\hat{\mathcal{R}}_\theta$ enabling synthetic trajectory generation (imagination) without real environment interaction costs:
$$\hat{\tau} = (\hat{s}_0, \hat{a}_0, \hat{r}_0, \hat{s}_1, \dots, \hat{s}_H)$$

However, standard MBRL suffers from **compounding model error**: small inaccuracies in single-step predictions accumulate exponentially over imagination horizon $H$, leading to catastrophic policy exploitation where the policy learns to seek out flaws in the world model rather than genuine environmental rewards.

AURORA addresses this fundamentally through **probabilistic ensemble dynamics**, **epistemic uncertainty quantification**, and **adaptive rollout horizon calibration**.

---

## 2. Probabilistic Ensemble Dynamics

### 2.1 State-Difference Parameterization
To prevent non-stationary drift and exploit physical continuity, the dynamics model predicts the state residual $\Delta s_t = s_{t+1} - s_t$ and scalar reward $r_t$:
$$\mathbf{y}_t = \begin{bmatrix} s_{t+1} - s_t \\ r_t \end{bmatrix} \in \mathbb{R}^{d_s + 1}$$

### 2.2 Deep Gaussian Ensemble
An ensemble of $E$ independent models $\{ f_{\theta_e} \}_{e=1}^E$ parameterize heteroscedastic diagonal Gaussian distributions:
$$p_{\theta_e}(\mathbf{y} \mid \mathbf{x}) = \mathcal{N}\left( \boldsymbol{\mu}_e(\mathbf{x}), \boldsymbol{\Sigma}_e(\mathbf{x}) \right), \quad \boldsymbol{\Sigma}_e(\mathbf{x}) = \text{diag}\left( \boldsymbol{\sigma}_e^2(\mathbf{x}) \right)$$
where $\mathbf{x} = [\mathbf{s}, \mathbf{a}] \in \mathbb{R}^{d_s + d_a}$.

To ensure numerical stability and prevent variance collapse or explosion:
$$\mathbf{s}_e(\mathbf{x}) = \text{clip}\left( \mathbf{s}_e^{\text{raw}}(\mathbf{x}), \log \sigma_{\min}^2, \log \sigma_{\max}^2 \right)$$
$$\boldsymbol{\sigma}_e^2(\mathbf{x}) = \exp\left( \mathbf{s}_e(\mathbf{x}) \right)$$

### 2.3 Maximum Likelihood Gaussian NLL Loss
Each ensemble member $e \in \{1, \dots, E\}$ is optimized independently on mini-batch transitions via Negative Log-Likelihood:
$$\mathcal{L}_{\text{NLL}}(\theta_e) = \frac{1}{2 B} \sum_{i=1}^B \sum_{j=1}^{d_y} \left[ \frac{(y_{i, j} - \mu_{e, j}(\mathbf{x}_i))^2}{\sigma_{e, j}^2(\mathbf{x}_i)} + \log \sigma_{e, j}^2(\mathbf{x}_i) + \log(2\pi) \right]$$

---

## 3. Uncertainty Decomposition & Quantification

Given ensemble outputs $\{ \boldsymbol{\mu}_e(\mathbf{x}), \boldsymbol{\sigma}_e^2(\mathbf{x}) \}_{e=1}^E$, the predictive distribution is an equally-weighted Gaussian mixture:
$$p(\mathbf{y} \mid \mathbf{x}) = \frac{1}{E} \sum_{e=1}^E \mathcal{N}\left( \boldsymbol{\mu}_e(\mathbf{x}), \boldsymbol{\Sigma}_e(\mathbf{x}) \right)$$

### 3.1 Ensemble Mean
$$\bar{\boldsymbol{\mu}}(\mathbf{x}) = \frac{1}{E} \sum_{e=1}^E \boldsymbol{\mu}_e(\mathbf{x})$$

### 3.2 Aleatoric Uncertainty (Expected Data Noise)
Aleatoric uncertainty measures the irreducible stochasticity of the environment:
$$\mathcal{U}_{\text{aleatoric}}(\mathbf{x}) = \frac{1}{E} \sum_{e=1}^E \boldsymbol{\sigma}_e^2(\mathbf{x})$$

### 3.3 Epistemic Uncertainty (Model Disagreement)
Epistemic uncertainty measures the variance across ensemble model means, reflecting parameter uncertainty due to sparse data:
$$\mathcal{U}_{\text{epistemic}}(\mathbf{x}) = \frac{1}{E} \sum_{e=1}^E \left( \boldsymbol{\mu}_e(\mathbf{x}) - \bar{\boldsymbol{\mu}}(\mathbf{x}) \right)^2$$
(or using sample unbiased estimator with divisor $E - 1$).

### 3.4 Total Predictive Variance
By the law of total variance:
$$\operatorname{Var}[\mathbf{y} \mid \mathbf{x}] = \mathbb{E}[\operatorname{Var}[\mathbf{y} \mid e]] + \operatorname{Var}[\mathbb{E}[\mathbf{y} \mid e]] = \mathcal{U}_{\text{aleatoric}}(\mathbf{x}) + \mathcal{U}_{\text{epistemic}}(\mathbf{x})$$

---

## 4. Recurrent State-Space Model (RSSM)

For partially observable domains, the world model maintains a continuous recurrent state $\mathbf{h}_t \in \mathbb{R}^{d_h}$ and stochastic latent state $\mathbf{z}_t \in \mathbb{R}^{d_z}$.

### 4.1 Transition Structure
- **Deterministic Recurrent State**:
  $$\mathbf{h}_t = \text{GRUCell}\left( \mathbf{h}_{t-1}, [\mathbf{z}_{t-1}, \mathbf{a}_{t-1}] \right)$$
- **Stochastic Prior (Imagination Path)**:
  $$p_\theta(\mathbf{z}_t \mid \mathbf{h}_t) = \mathcal{N}\left( \boldsymbol{\mu}_t^{\text{prior}}(\mathbf{h}_t), \text{diag}((\boldsymbol{\sigma}_t^{\text{prior}}(\mathbf{h}_t))^2) \right)$$
- **Stochastic Posterior (Observation Filtering Path)**:
  $$q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{x}_t) = \mathcal{N}\left( \boldsymbol{\mu}_t^{\text{post}}(\mathbf{h}_t, \mathbf{x}_t), \text{diag}((\boldsymbol{\sigma}_t^{\text{post}}(\mathbf{h}_t, \mathbf{x}_t))^2) \right)$$

### 4.2 Reconstruction Heads & Likelihoods
- Observation predictor: $\hat{\mathbf{x}}_t = g_{\theta_x}(\mathbf{h}_t, \mathbf{z}_t)$
- Reward predictor: $\hat{r}_t = g_{\theta_r}(\mathbf{h}_t, \mathbf{z}_t)$
- Continuation (discount factor) predictor: $\hat{\gamma}_t = \sigma(g_{\theta_\gamma}(\mathbf{h}_t, \mathbf{z}_t))$

### 4.3 Variational Objective & KL Balancing
$$\mathcal{L}_{\text{RSSM}} = \mathbb{E}_{q_\phi} \left[ \frac{1}{2}\|\mathbf{x}_t - \hat{\mathbf{x}}_t\|^2 + \frac{1}{2}(r_t - \hat{r}_t)^2 - \text{BCE}(\text{done}_t, 1 - \hat{\gamma}_t) \right] + \beta_{\text{KL}} \mathcal{L}_{\text{KL}}$$

where KL balancing isolates representation learning from prior dynamics learning:
$$\mathcal{L}_{\text{KL}} = \alpha_{\text{KL}} D_{\text{KL}}\left( q_\phi(\mathbf{z}_t) \parallel \text{stop\_grad}(p_\theta(\mathbf{z}_t)) \right) + (1 - \alpha_{\text{KL}}) D_{\text{KL}}\left( \text{stop\_grad}(q_\phi(\mathbf{z}_t)) \parallel p_\theta(\mathbf{z}_t) \right)$$
with standard $\alpha_{\text{KL}} = 0.8$.

---

## 5. Adaptive Rollout Horizon & Imagination Engine

### 5.1 Dynamic Horizon Truncation
Given an initial state $s_0 \sim \mathcal{D}_{\text{real}}$, a trajectory is unrolled under policy $\pi_\psi(a \mid s)$ for up to $H_{\max}$ steps. At step $h$:
1. Sample action: $\hat{a}_h \sim \pi_\psi(\cdot \mid \hat{s}_h)$
2. Compute ensemble prediction and epistemic disagreement:
   $$u_h = \max_{j} \mathcal{U}_{\text{epistemic}, j}(\hat{s}_h, \hat{a}_h)$$
3. If $u_h > \tau_{\text{threshold}}$, truncate imagination at horizon $H = h$.
4. Otherwise, sample transition $\hat{s}_{h+1} = \hat{s}_h + \boldsymbol{\mu}_e(\hat{s}_h, \hat{a}_h)$ from a randomly selected ensemble member $e \in \{1, \dots, E\}$ (TS1 sampling).

### 5.2 Synthetic Experience Injection
Generated transitions $(\hat{s}_h, \hat{a}_h, \hat{r}_h, \hat{s}_{h+1}, \hat{d}_h)$ are stored in a dedicated synthetic replay buffer and intermixed with real experience at ratio $\eta \in [0, 1]$ during actor-critic updates.
