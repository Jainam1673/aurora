# Decision Transformer (Chen et al., 2021)

## 1. Purpose & Background
Decision Transformer (Chen et al., 2021) reframes reinforcement learning as an autoregressive sequence modeling problem. Instead of using conventional dynamic programming or temporal-difference learning (e.g. Q-learning or actor-critic), Decision Transformer outputs actions directly conditioned on desired future returns (Return-to-Go, RTG), past states, and past actions using a causal Pre-LN Transformer architecture.

## 2. Mathematical Equations

### 2.1 Trajectory Sequence Formulation
A trajectory $\tau$ of length $K$ is represented as a sequence of three token types per timestep:
$$\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_K, s_K, a_K \right)$$
where the Return-to-Go (RTG) is defined as:
$$\hat{R}_t = \sum_{t'=t}^T r_{t'}$$

### 2.2 Token Projections & Shared Timestep Embeddings
Each modality is projected to the transformer hidden dimension $d_{\text{model}}$ via a linear layer:
$$e_{R, t} = W_R \hat{R}_t, \quad e_{s, t} = W_s s_t, \quad e_{a, t} = W_a a_t$$
For each timestep $t \in \{1, \dots, K\}$, a learned positional embedding $W_{\text{pos}}(t)$ is added to all three modalities:
$$\tilde{e}_{R, t} = e_{R, t} + W_{\text{pos}}(t)$$
$$\tilde{e}_{s, t} = e_{s, t} + W_{\text{pos}}(t)$$
$$\tilde{e}_{a, t} = e_{a, t} + W_{\text{pos}}(t)$$
The interleaved sequence of length $3K$ (or $3K-1$ if omitting the last action) is processed with causal self-attention.

### 2.3 Causal Attention & Action Prediction
Through $L$ Pre-LN Transformer decoder blocks with causal masking:
$$\mathbf{H} = \text{TransformerBlocks}(\tilde{\mathbf{E}})$$
Actions are predicted from the hidden representations corresponding to state tokens:
$$\hat{a}_t = \tanh(W_{\text{act}} \mathbf{h}_{s, t})$$

### 2.4 Training Loss
The model is trained via mean-squared error on action predictions over offline dataset trajectories:
$$\mathcal{L}_{\text{DT}}(\theta) = \frac{1}{K} \sum_{t=1}^K \| a_t - \hat{a}_t(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1}) \|_2^2$$

### 2.5 Inference & Autoregressive Execution
1. Specify target return $R_{\text{target}}$.
2. Initialize context with $\hat{R}_1 = R_{\text{target}}$ and initial observation $s_1$.
3. At step $t$:
   - Predict $a_t = \text{DT}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$.
   - Execute $a_t$ in environment, observe reward $r_t$ and next state $s_{t+1}$.
   - Compute next RTG: $\hat{R}_{t+1} = \hat{R}_t - r_t$.
   - Append $(a_t, \hat{R}_{t+1}, s_{t+1})$ to context window (sliding window of size $K$).

## 3. Algorithm Pseudocode
```text
Initialize DecisionTransformer(state_dim, act_dim, d_model, n_heads, n_layers, max_ep_len)
Initialize optimizer AdamW
For epoch = 1 to epochs:
    Sample trajectory batch (states, actions, rtgs, timesteps) from offline buffer
    Predict actions a_hat = model(states, actions, rtgs, timesteps)
    Compute loss = MSE(a_hat, actions)
    Backpropagate and update parameters
```

## 4. Key Design Decisions & Numerical Considerations
- Predicting actions from state tokens $h(s_t)$ ensures the action prediction conditions on both the target return $\hat{R}_t$ and current state $s_t$, without information leakage from future transitions.
- Timestep-based positional encoding allows generalization across arbitrary trajectory lengths while preserving episode progress context.
- Tanh action head constrains output predictions to the valid continuous action bounds $[-1, 1]$.
