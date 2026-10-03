# Temporal Difference Model Predictive Control (TD-MPC)

## 1. Purpose & Background
TD-MPC (Hansen et al., 2022) is a model-based reinforcement learning algorithm that performs trajectory planning in a learned, task-oriented latent state space without observation reconstruction. By combining learned latent dynamics, a reward model, terminal Q-value bootstrapping, and Cross-Entropy Method (CEM) trajectory optimization, TD-MPC achieves sample efficiency matching or exceeding pixel/state MBRL approaches with dramatically reduced compute overhead.

## 2. Mathematical Equations

### 2.1 Model Components
1. **Latent Representation**: $z = f_\theta(s)$, where $s \in \mathbb{R}^{d_s}$, $z \in \mathbb{R}^{d_z}$.
2. **Latent Dynamics**: $\hat{z}' = g_\theta(z, a)$, predicting future latent state transitions.
3. **Reward Predictor**: $\hat{r} = R_\theta(z, a)$, predicting instantaneous scalar reward.
4. **Twin Q-Functions**: $Q_{\phi_1}(z, a), Q_{\phi_2}(z, a)$, providing discounted return estimations.
5. **Policy Prior**: $\pi_\psi(z)$, generating candidate actions for terminal value estimation and CEM seeding.

### 2.2 Trajectory Return Objective
For a candidate action sequence $\mathbf{a}_{0:H-1} = (a_0, a_1, \dots, a_{H-1})$ starting from encoded state $z_0 = f_\theta(s)$:
$$\hat{z}_{h+1} = g_\theta(\hat{z}_h, a_h), \quad \hat{r}_h = R_\theta(\hat{z}_h, a_h)$$
$$J(\mathbf{a}_{0:H-1}) = \sum_{h=0}^{H-1} \gamma^h \hat{r}_h + \gamma^H \min_{j \in \{1, 2\}} Q_{\phi_j}(\hat{z}_H, \pi_\psi(\hat{z}_H))$$

### 2.3 Cross-Entropy Method (CEM) Planning
1. Initialize proposal distribution $\mathcal{N}(\boldsymbol{\mu}^{(0)}, (\boldsymbol{\sigma}^{(0)})^2)$ across the horizon $H \times d_a$.
2. For iteration $k = 1 \dots K$:
   - Sample $N$ action candidate sequences: $\mathbf{a}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}^{(k-1)}, (\boldsymbol{\sigma}^{(k-1)})^2)$.
   - Evaluate latent trajectory returns $\{J(\mathbf{a}^{(i)})\}_{i=1}^N$.
   - Select top $M$ elite candidate sequences $\mathcal{E}$.
   - Update proposal parameters with momentum $\alpha_{\text{CEM}}$:
     $$\boldsymbol{\mu}^{(k)} = \alpha_{\text{CEM}} \left( \frac{1}{M} \sum_{e \in \mathcal{E}} \mathbf{a}^{(e)} \right) + (1 - \alpha_{\text{CEM}}) \boldsymbol{\mu}^{(k-1)}$$
     $$\boldsymbol{\sigma}^{(k)} = \alpha_{\text{CEM}} \left( \sqrt{\frac{1}{M} \sum_{e \in \mathcal{E}} (\mathbf{a}^{(e)} - \boldsymbol{\mu}^{(k)})^2 + \epsilon} \right) + (1 - \alpha_{\text{CEM}}) \boldsymbol{\sigma}^{(k-1)}$$
3. Return the first action of the final proposal: $a^* = \boldsymbol{\mu}_0^{(K)}$.

### 2.4 Learning Objectives
- **Reward Loss**: $\mathcal{L}_R(\theta) = \frac{1}{2} (R_\theta(z_t, a_t) - r_t)^2$
- **Dynamics Consistency Loss**: $\mathcal{L}_{\text{dyn}}(\theta) = \frac{1}{2} \| g_\theta(z_t, a_t) - \text{stop\_grad}(f_\theta(s_{t+1})) \|_2^2$
- **Q-Prediction Loss**: $\mathcal{L}_Q(\phi) = \frac{1}{2} (Q_\phi(z_t, a_t) - y_t)^2$ where $y_t = r_t + \gamma (1 - d_t) \min_j Q_{\bar{\phi}_j}(z_{t+1}, \pi(z_{t+1}))$
- **Policy Prior Loss**: $\mathcal{L}_\pi(\psi) = - Q_{\phi_1}(z_t, \pi_\psi(z_t))$

## 3. Algorithm Pseudocode
```text
Initialize encoder f_theta, dynamics g_theta, reward R_theta, critics Q_phi_1, Q_phi_2, policy pi_psi
Initialize replay buffer D
For step t = 1 to T:
    Observe s_t
    z_0 = f_theta(s_t)
    Plan action a_t = CEM_Plan(z_0, g_theta, R_theta, Q_phi, pi_psi, horizon H, iterations K, samples N, elites M)
    Execute a_t in environment, observe r_t, s_{t+1}, d_t
    Store (s_t, a_t, r_t, s_{t+1}, d_t) in D
    If t >= min_train_steps:
        Sample batch B ~ D
        Update model parameters (theta, phi, psi) via combined multi-task gradients
        Polyakt-update target critics Q_bar
```

## 4. Key Design Decisions & Numerical Considerations
- Terminal Q-value bootstrapping accounts for long-term expected returns beyond the finite planning horizon $H$.
- Clamping action sequences to $[-1, 1]$ prevents unconstrained proposal drift.
- Standard deviation floor $\sigma_{\text{min}} > 0$ preserves exploration and prevents degeneracy during CEM optimization.
