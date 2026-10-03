# MuZero (Schrittwieser et al., 2020)

## 1. Purpose & Background
MuZero represents the culmination of model-based planning in discrete and continuous domains by learning a model that predicts only quantities relevant to decision-making: the reward, the policy prior, and the value function. It does not attempt to reconstruct the observation space, completely circumventing the visual fidelity bottleneck. Planning is executed using Upper Confidence Bounds for Trees (PUCT) directly in the learned latent space.

## 2. Mathematical Equations

### 2.1 Three-Model Latent Decomposition
1. **Representation function**:
   $$s^0 = h_\theta(o_1, \dots, o_t)$$
   Maps history of observations to root latent state.
2. **Dynamics function**:
   $$r^k, s^k = g_\theta(s^{k-1}, a^k)$$
   Recursively computes predicted immediate reward and next latent state for imagined action $a^k$.
3. **Prediction function**:
   $$\mathbf{p}^k, v^k = f_\theta(s^k)$$
   Computes policy prior distribution $\mathbf{p} \in \Delta^{|\mathcal{A}|}$ and scalar value estimate $v \in \mathbb{R}$.

### 2.2 PUCT Action Selection
At internal search tree node $s$:
$$a^* = \arg\max_a \left[ Q(s, a) + P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log\left( \frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right) \right]$$
where $Q(s, a)$ is normalized mean action-value, $P(s, a)$ is the prior probability from $f_\theta$, and $c_1, c_2$ are exploration constants (e.g. $c_1 = 1.25, c_2 = 19652$).

### 2.3 Search Backup
Upon reaching leaf depth $L$ with evaluation value $v^L$:
$$G^k = \sum_{\tau=0}^{L-k-1} \gamma^\tau r^{k+\tau+1} + \gamma^{L-k} v^L$$
$$N(s^{k-1}, a^k) \leftarrow N(s^{k-1}, a^k) + 1$$
$$Q(s^{k-1}, a^k) \leftarrow \frac{(N-1) Q + G^k}{N}$$

### 2.4 Multi-Task Loss Function
Over unrolled trajectory steps $k = 0 \dots K$:
$$\mathcal{L}(\theta) = \sum_{k=0}^K \left[ \ell_r(r^k, u_{t+k}) + \ell_v(v^k, z_{t+k}) + \ell_p(\mathbf{p}^k, \boldsymbol{\pi}_{t+k}) \right] + c \|\theta\|_2^2$$
where $\boldsymbol{\pi}_{t+k}$ is the MCTS visit count distribution at step $t+k$, and $z_{t+k}$ is the bootstrapped target value.

## 3. Algorithm Pseudocode
```text
Initialize representation h_theta, dynamics g_theta, prediction f_theta
For each environment step t:
    s_0 = h_theta(o_t)
    Initialize MCTS root node at s_0
    For sim = 1 to num_simulations:
        Traverse tree using PUCT selection until unexpanded leaf node s_L
        Expand leaf node: (p_L, v_L) = f_theta(s_L)
        Rollout/step dynamics: s' = g_theta(s_L, a)
        Backup returns along search path to root
    Form search policy pi(a) ~ N(s_0, a)^(1/T)
    Sample action a_t ~ pi
    Execute a_t in environment
```

## 4. Key Design Decisions & Numerical Considerations
- Min-Max Q-value normalization: normalize raw Q-values to $[0, 1]$ using empirical $[\min, \max]$ encountered across the search tree, ensuring balanced exploration across differing return magnitudes.
- Dirichlet noise injection at root node: $\mathbf{p}_{\text{root}} \leftarrow (1 - \epsilon)\mathbf{p}_{\text{root}} + \epsilon \boldsymbol{\eta}$, where $\boldsymbol{\eta} \sim \text{Dir}(\alpha)$, guarantees exploration during training.
