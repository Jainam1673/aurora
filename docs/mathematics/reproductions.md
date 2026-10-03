# Mathematical Specification: MBRL Research Lineage & Reproductions

## 1. Overview & Theoretical Foundations

Model-Based Reinforcement Learning (MBRL) algorithms leverage learned environmental transition models to optimize agent policies. Prior research in MBRL broadly bifurcates into three algorithmic paradigms:
1. **Dyna-Style / Hybrid Replay Optimization (e.g., MBPO)**: Generating short synthetic rollouts from an ensemble dynamics model to augment real environment replay buffers for model-free actor-critic updates.
2. **Latent Imagination Actor-Critic (e.g., Dreamer)**: Encoding observations into a latent state space via Recurrent State-Space Models (RSSM) and optimizing actor and value networks purely through imagined multi-step trajectory rollouts.
3. **Latent Trajectory Optimization / Receding Horizon Planning (e.g., TD-MPC, MuZero)**: Utilizing learned latent dynamics and value functions for test-time trajectory shooting, Cross-Entropy Method (CEM) planning, or Monte Carlo Tree Search (MCTS).
4. **Sequence-Modeling / Autoregressive Return Conditioning (e.g., Decision Transformer)**: Casting reinforcement learning as conditional autoregressive sequence modeling over trajectory tokens $(R_t, s_t, a_t)$ using causal Transformers.

AURORA's uncertainty-calibrated imagination architecture builds upon and synthesizes the insights of this lineage.

---

## 2. Model-Based Policy Optimization (MBPO)

### 2.1 Theoretical Guarantee & Monotonic Bound
Janner et al. (2019) establish that the performance gap between the true policy return $\eta[\pi]$ and expected return under the learned model $\hat{\eta}[\pi]$ is bounded by:
$$\eta[\pi] \ge \hat{\eta}[\pi] - \left[ \frac{2 \gamma r_{\max} \epsilon_m}{(1 - \gamma)^2} + \frac{4 r_{\max} \epsilon_\pi}{(1 - \gamma)^2} \right]$$
where $\epsilon_m = \max_t \mathbb{E}_{s \sim \mathcal{D}_{\text{real}}} [D_{\text{TV}}(P(\cdot \mid s, a) \parallel \hat{P}(\cdot \mid s, a))]$ is the maximum single-step model error, and $\epsilon_\pi$ is the policy divergence.

For rollouts of length $k$, compounding error grows as $\mathcal{O}(k \epsilon_m)$, yielding the branching-horizon bound:
$$\eta[\pi] \ge \hat{\eta}[\pi] - 2 r_{\max} \left[ \frac{\gamma^{k+1} \epsilon_\pi}{(1 - \gamma)^2} + \frac{k \epsilon_m}{1 - \gamma} + \frac{\epsilon_\pi}{1 - \gamma} \right]$$

### 2.2 Algorithm Formulation
1. **Real Buffer Collection**: Collect real transitions $\mathcal{D}_{\text{real}} \leftarrow \mathcal{D}_{\text{real}} \cup \{(s_t, a_t, r_t, s_{t+1}, d_t)\}$.
2. **Ensemble Dynamics Training**: Train deep probabilistic ensemble $\{f_{\theta_e}\}_{e=1}^E$ on $\mathcal{D}_{\text{real}}$ via Gaussian NLL loss.
3. **$k$-Step Model Rollouts**: Sample initial states $s \sim \mathcal{D}_{\text{real}}$. For $h = 1 \dots k$:
   - Sample $a \sim \pi_\psi(\cdot \mid s)$
   - Sample $e \sim \text{Uniform}(1..E)$ and predict $s' \sim p_{\theta_e}(s' \mid s, a)$, $r \sim p_{\theta_e}(r \mid s, a)$
   - Store $(s, a, r, s', d)$ in model buffer $\mathcal{D}_{\text{model}}$.
4. **Policy Update**: Sample batch with real-to-synthetic ratio $\eta \in [0, 1]$:
   $$\mathcal{B} \sim (1 - \eta) \mathcal{D}_{\text{real}} + \eta \mathcal{D}_{\text{model}}$$
   Perform SAC actor-critic gradient updates on $\mathcal{B}$.

---

## 3. Latent Imagination: Dreamer / RSSM Lineage

### 3.1 RSSM World Model
As formalized in Milestone 5:
- Deterministic recurrent state: $\mathbf{h}_t = \text{GRUCell}([\mathbf{z}_{t-1}, \mathbf{a}_{t-1}], \mathbf{h}_{t-1})$
- Stochastic state prior: $\mathbf{z}_t \sim p_\theta(\mathbf{z}_t \mid \mathbf{h}_t) = \mathcal{N}(\boldsymbol{\mu}_t^{\text{prior}}, \boldsymbol{\Sigma}_t^{\text{prior}})$
- Stochastic state posterior: $\mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_t^{\text{post}}, \boldsymbol{\Sigma}_t^{\text{post}})$

### 3.2 Pure Latent Imagination
Given starting posterior state $s_0 = (\mathbf{h}_0, \mathbf{z}_0)$, unroll trajectory of horizon $H$ purely in latent space:
$$\hat{\mathbf{a}}_\tau \sim \pi_\psi(\cdot \mid \hat{\mathbf{h}}_\tau, \hat{\mathbf{z}}_\tau)$$
$$\hat{\mathbf{h}}_{\tau+1} = \text{GRUCell}([\hat{\mathbf{z}}_\tau, \hat{\mathbf{a}}_\tau], \hat{\mathbf{h}}_\tau)$$
$$\hat{\mathbf{z}}_{\tau+1} \sim p_\theta(\cdot \mid \hat{\mathbf{h}}_{\tau+1})$$
$$\hat{r}_\tau = g_\theta^r(\hat{\mathbf{h}}_\tau, \hat{\mathbf{z}}_\tau), \quad \hat{\gamma}_\tau = \sigma(g_\theta^\gamma(\hat{\mathbf{h}}_\tau, \hat{\mathbf{z}}_\tau))$$

### 3.3 Latent Generalized Advantage Estimation ($\lambda$-Returns)
The value target $V_t^\lambda$ is computed recursively backwards from horizon $H$:
$$V_H^\lambda = v_\xi(\hat{\mathbf{h}}_H, \hat{\mathbf{z}}_H)$$
$$V_t^\lambda = \hat{r}_t + \gamma \hat{\gamma}_t \left[ (1 - \lambda) v_\xi(\hat{\mathbf{h}}_{t+1}, \hat{\mathbf{z}}_{t+1}) + \lambda V_{t+1}^\lambda \right], \quad t = H-1, \dots, 0$$

### 3.4 Actor-Critic Latent Objectives
- **Critic Loss**:
  $$\mathcal{L}_v(\xi) = \frac{1}{2} \sum_{t=0}^{H-1} \left( v_\xi(\hat{\mathbf{h}}_t, \hat{\mathbf{z}}_t) - \text{stop\_grad}(V_t^\lambda) \right)^2$$
- **Actor Loss**:
  $$\mathcal{L}_\pi(\psi) = - \sum_{t=0}^{H-1} \left( V_t^\lambda + \eta_{\text{ent}} \mathcal{H}(\pi_\psi(\cdot \mid \hat{\mathbf{h}}_t, \hat{\mathbf{z}}_t)) \right)$$

---

## 4. Latent Trajectory Optimization: TD-MPC Lineage

### 4.1 Latent Abstraction Architecture
Unlike Dreamer, TD-MPC (Hansen et al., 2022) does not decode observations $\hat{x}_t \approx x_t$. Instead, it learns a task-oriented latent representation:
1. **Representation**: $z_0 = f_\theta(s_0)$
2. **Latent Dynamics**: $\hat{z}_{t+1} = g_\theta(\hat{z}_t, a_t)$
3. **Reward Model**: $\hat{r}_t = R_\theta(\hat{z}_t, a_t)$
4. **Terminal Q-Value**: $\hat{Q}_t = Q_\theta(\hat{z}_t, a_t)$

### 4.2 Cross-Entropy Method (CEM) Planning
At step $t$, the agent plans an action sequence $\mathbf{a}_{0:H-1}$ of horizon $H$ using the Cross-Entropy Method (CEM):
1. Initialize Gaussian proposal distribution: $\mathcal{N}(\boldsymbol{\mu}_h^{(0)}, \boldsymbol{\Sigma}_h^{(0)})$ for $h = 0 \dots H-1$.
2. For iteration $k = 1 \dots K$:
   - Sample $N$ action candidate sequences: $\mathbf{a}_{0:H-1}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}^{(k-1)}, \boldsymbol{\Sigma}^{(k-1)})$.
   - Predict latent trajectory and accumulated return:
     $$J(\mathbf{a}^{(i)}) = \sum_{h=0}^{H-1} \gamma^h \hat{r}_h^{(i)} + \gamma^H \hat{Q}(\hat{z}_H^{(i)}, \hat{a}_H^{(i)})$$
   - Select top $M$ elite trajectories ($M < N$).
   - Refit proposal distribution to elite mean and covariance:
     $$\boldsymbol{\mu}_h^{(k)} = \alpha_{\text{CEM}} \frac{1}{M} \sum_{m=1}^M \mathbf{a}_{h, m}^* + (1 - \alpha_{\text{CEM}}) \boldsymbol{\mu}_h^{(k-1)}$$
3. Execute the first planned action: $a_t = \boldsymbol{\mu}_0^{(K)}$ (Receding Horizon Control).

---

## 5. Latent Monte Carlo Tree Search: MuZero Lineage

### 5.1 Three-Model Decomposition
- **Representation function**: $s^0 = h_\theta(o_1, \dots, o_t)$
- **Dynamics function**: $r^k, s^k = g_\theta(s^{k-1}, a^k)$
- **Prediction function**: $\mathbf{p}^k, v^k = f_\theta(s^k)$ (policy prior $\mathbf{p}$ and scalar value $v$)

### 5.2 PUCT Search Tree Selection
At search node $s$ selecting action $a$:
$$a^* = \arg\max_a \left[ Q(s, a) + P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log\left( \frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right) \right]$$

### 5.3 Backup & Policy Target
Upon reaching depth $L$, the terminal value $v^L$ is backed up along the search path:
$$Q(s^{k-1}, a^k) \leftarrow \frac{N(s^{k-1}, a^k) Q(s^{k-1}, a^k) + G^k}{N(s^{k-1}, a^k) + 1}, \quad N(s^{k-1}, a^k) \leftarrow N(s^{k-1}, a^k) + 1$$
where $G^k = \sum_{\tau=0}^{L-k-1} \gamma^\tau r^{k+\tau+1} + \gamma^{L-k} v^L$.

---

## 6. Autoregressive RL: Decision Transformer Lineage

### 6.1 Trajectory Sequence Representation
Cast RL as autoregressive sequence modeling. A trajectory of length $K$ is flattened into tokens:
$$\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_K, s_K, a_K \right)$$
where $\hat{R}_t = \sum_{t'=t}^T r_{t'}$ is the Return-to-Go (RTG).

### 6.2 Token Embedding & Causal Attention
- Token projections: $e_R = W_R \hat{R}_t$, $e_s = W_s s_t$, $e_a = W_a a_t$.
- Add learned timestep embeddings: $e_x = e_x + W_{\text{pos}}(t)$.
- Tokens are processed through AURORA's Pre-LN `TransformerDecoder` with causal masking.
- Prediction head projects state token representation $h(s_t)$ to predicted action $\hat{a}_t$.

### 6.3 Objective
$$\mathcal{L}_{\text{DT}}(\theta) = \frac{1}{K} \sum_{t=1}^K \| a_t - \hat{a}_t( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} ) \|_2^2$$
During evaluation, the model is conditioned on desired target return $\hat{R}_1 = R_{\text{target}}$ and generates actions autoregressively while decrementing $\hat{R}_{t+1} = \hat{R}_t - r_t$.
