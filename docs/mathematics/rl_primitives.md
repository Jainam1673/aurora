# Mathematical Specification: Reinforcement Learning Primitives

This document establishes the mathematical foundations, probabilistic formulations, loss functions, advantage estimators, and algorithm specifications for Reinforcement Learning (RL) in AURORA.

---

## 1. Markov Decision Process (MDP)

An infinite-horizon discounted Markov Decision Process is defined by the 6-tuple:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \rho_0)$$
where:
- $\mathcal{S}$ is the state space (discrete or continuous $\mathbb{R}^{d_s}$).
- $\mathcal{A}$ is the action space (discrete $\{0, \dots, K-1\}$ or continuous $\mathbb{R}^{d_a}$).
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the transition probability distribution $p(s_{t+1} \mid s_t, a_t)$.
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function $r_t = r(s_t, a_t)$.
- $\gamma \in [0, 1)$ is the temporal discount factor.
- $\rho_0 \in \Delta(\mathcal{S})$ is the initial state distribution $s_0 \sim \rho_0$.

The policy $\pi_\theta(a_t \mid s_t)$ specifies a probability distribution over actions given the state. The standard RL objective maximizes expected cumulative discounted return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right], \quad \tau = (s_0, a_0, s_1, a_1, \dots)$$

---

## 2. Policy Distributions

### 2.1 Categorical Distribution (Discrete Actions)
Given unnormalized logits $z \in \mathbb{R}^K$:
$$p_k = \text{softmax}(z)_k = \frac{\exp(z_k - \max_j z_j)}{\sum_{j=1}^K \exp(z_j - \max_j z_j)}$$

- **Log Probability:**
  $$\log \pi(a = k) = z_k - \max_j z_j - \log \sum_{j=1}^K \exp(z_j - \max_j z_j)$$
- **Entropy:**
  $$\mathcal{H}(p) = - \sum_{k=1}^K p_k \log p_k$$

### 2.2 Independent Normal / Diagonal Gaussian (Continuous Actions)
Given mean $\mu \in \mathbb{R}^d$ and log standard deviation $\log \sigma \in \mathbb{R}^d$:
$$\sigma = \exp(\text{clip}(\log \sigma, \log \sigma_{\min}, \log \sigma_{\max}))$$

- **Reparameterized Sampling (Pathwise Gradients):**
  $$\epsilon \sim \mathcal{N}(0, I_d), \quad x = \mu + \sigma \odot \epsilon$$
- **Log Probability:**
  $$\log \pi(x) = -\frac{1}{2} \sum_{i=1}^d \left[ \left(\frac{x_i - \mu_i}{\sigma_i}\right)^2 + 2 \log \sigma_i + \log(2\pi) \right]$$
- **Entropy:**
  $$\mathcal{H}(\pi) = \frac{1}{2} \sum_{i=1}^d \left[ 1 + \log(2\pi) + 2 \log \sigma_i \right]$$

### 2.3 Squashed Gaussian / TanhNormal (Bounded Continuous Actions)
For bounded actions $a \in [-1, 1]^d$, unbounded Gaussian sample $u \sim \mathcal{N}(\mu, \sigma^2)$ is mapped via invertible squashing:
$$a = \tanh(u)$$

By the change-of-variables theorem:
$$p_A(a) = p_U(u) \left| \det \frac{\partial a}{\partial u} \right|^{-1}$$

Since $\frac{da_i}{du_i} = 1 - \tanh^2(u_i) = 1 - a_i^2$, the log Jacobian determinant is diagonal:
$$\log \left| \det \frac{\partial a}{\partial u} \right| = \sum_{i=1}^d \log(1 - a_i^2 + \epsilon)$$

Therefore:
$$\log \pi(a \mid s) = \log p_U(u \mid s) - \sum_{i=1}^d \log(1 - \tanh^2(u_i) + \epsilon)$$
To prevent numerical instability near $a_i \approx \pm 1$, use the identity:
$$\log(1 - \tanh^2(u)) = 2 \left( \log 2 - u - \text{softplus}(-2u) \right)$$

---

## 3. Advantage Estimation & Generalized Advantage Estimation (GAE)

Given trajectory rewards $(r_0, r_1, \dots, r_{T-1})$ and state value estimates $(V(s_0), \dots, V(s_T))$:

### 3.1 Temporal Difference Residual
$$\delta_t^V = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)$$
where $d_t \in \{0, 1\}$ indicates episode termination.

### 3.2 Generalized Advantage Estimation (Schulman et al., 2016)
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + \gamma \lambda (1 - d_t) \hat{A}_{t+1}^{\text{GAE}}$$
Computed backward from $t = T-1$ down to $t = 0$.

Target returns for value function regression:
$$\hat{R}_t = \hat{A}_t^{\text{GAE}} + V(s_t)$$

---

## 4. Proximal Policy Optimization (PPO)

PPO optimizes the clipped surrogate objective over minibatches collected from policy $\pi_{\theta_{\text{old}}}$:

### 4.1 Probability Ratio
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) \right)$$

### 4.2 Clipped Surrogate Loss
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

### 4.3 Value Function Loss (with optional clipping)
$$L^V(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - \hat{R}_t)^2 \right]$$

### 4.4 Total PPO Objective
$$\mathcal{L}^{\text{PPO}}(\theta, \phi) = - L^{\text{CLIP}}(\theta) + c_1 L^V(\phi) - c_2 \hat{\mathbb{E}}_t \left[ \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right]$$
where $c_1 \approx 0.5$, $c_2 \approx 0.01$, and clipping threshold $\epsilon \approx 0.2$.

---

## 5. Soft Actor-Critic (SAC)

SAC optimizes maximum entropy reinforcement learning:
$$J(\pi) = \sum_{t=0}^\infty \mathbb{E}_{(s_t, a_t)} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right]$$

### 5.1 Clipped Double-Q Soft Bellman Residual
Two soft Q-functions $Q_{\theta_1}, Q_{\theta_2}$ and target networks $Q_{\bar{\theta}_1}, Q_{\bar{\theta}_2}$:
$$y_t = r_t + \gamma (1 - d_t) \left( \min_{j=1,2} Q_{\bar{\theta}_j}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi_\phi(\tilde{a}_{t+1} \mid s_{t+1}) \right), \quad \tilde{a}_{t+1} \sim \pi_\phi(\cdot \mid s_{t+1})$$

Critic loss:
$$J_Q(\theta_i) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}, d_t) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_{\theta_i}(s_t, a_t) - y_t \right)^2 \right]$$

### 5.2 Actor Loss (Reparameterized Policy Improvement)
Using squashed sample $\tilde{a}_\phi(s, \epsilon) = \tanh(\mu_\phi(s) + \sigma_\phi(s) \odot \epsilon)$ where $\epsilon \sim \mathcal{N}(0, I)$:
$$J_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \alpha \log \pi_\phi(\tilde{a}_\phi(s, \epsilon) \mid s) - \min_{j=1,2} Q_{\theta_j}(s, \tilde{a}_\phi(s, \epsilon)) \right]$$

### 5.3 Automatic Temperature $\alpha$ Adjustment
Given target entropy $\bar{\mathcal{H}} = - \dim(\mathcal{A})$:
$$J(\alpha) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ -\alpha \left( \log \pi_\phi(a \mid s) + \bar{\mathcal{H}} \right) \right]$$
Optimized via $\log \alpha$ parameterization for positivity guarantee: $\alpha = \exp(\log \alpha)$.

### 5.4 Polyak Target Smoothing
$$\bar{\theta}_j \leftarrow \tau \theta_j + (1 - \tau) \bar{\theta}_j, \quad \tau \in (0, 1] \text{ (typically } 0.005)$$
