# Mathematical Specification: Transformer Engine & Attention

This document defines the mathematical formulation, backward gradients, numerical stability considerations, and architectural specifications for the AURORA Transformer Engine in both Python 3.14 and native C++23.

---

## 1. Scaled Dot-Product Attention

Given query $Q$, key $K$, and value $V$ tensors:
- $Q \in \mathbb{R}^{B \times H \times T_q \times d_k}$
- $K \in \mathbb{R}^{B \times H \times T_k \times d_k}$
- $V \in \mathbb{R}^{B \times H \times T_k \times d_v}$

where $B$ is batch size, $H$ is number of attention heads, $T_q$ is query sequence length, $T_k$ is key/value sequence length, and $d_k, d_v$ are head dimensions.

### 1.1 Forward Formulation

1. **Attention Logits:**
   $$S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times T_q \times T_k}$$

2. **Masking:**
   Given an additive mask $M \in \mathbb{R}^{T_q \times T_k}$ (or broadcastable shape):
   $$\tilde{S} = S + M$$
   For causal autoregressive self-attention ($T_q = T_k = T$):
   $$M_{i, j} = \begin{cases} 0, & j \le i \\ -\infty \text{ (practically } -10^9), & j > i \end{cases}$$

3. **Attention Probabilities (Softmax):**
   $$A = \text{softmax}(\tilde{S}, \text{axis}=-1) \in \mathbb{R}^{B \times H \times T_q \times T_k}$$
   For each query position $i$:
   $$A_{b, h, i, j} = \frac{\exp(\tilde{S}_{b, h, i, j} - \max_l \tilde{S}_{b, h, i, l})}{\sum_{k=1}^{T_k} \exp(\tilde{S}_{b, h, i, k} - \max_l \tilde{S}_{b, h, i, l})}$$

4. **Attention Output:**
   $$O = A V \in \mathbb{R}^{B \times H \times T_q \times d_v}$$

---

### 1.2 Analytical Backward Adjoints

Given upstream gradient $\nabla_O = \frac{\partial \mathcal{L}}{\partial O} \in \mathbb{R}^{B \times H \times T_q \times d_v}$:

1. **Gradient w.r.t. Value $V$:**
   $$\nabla_V = A^T \nabla_O \in \mathbb{R}^{B \times H \times T_k \times d_v}$$

2. **Gradient w.r.t. Attention Weights $A$:**
   $$\nabla_A = \nabla_O V^T \in \mathbb{R}^{B \times H \times T_q \times T_k}$$

3. **Gradient w.r.t. Logits $S$ (Softmax VJP):**
   $$\nabla_{\tilde{S}} = A \odot \left( \nabla_A - \sum_{j=1}^{T_k} (\nabla_A)_{b, h, i, j} A_{b, h, i, j} \right)$$
   Since $\tilde{S} = S + M$ and $M$ is non-trainable:
   $$\nabla_S = \nabla_{\tilde{S}}$$

4. **Gradient w.r.t. Query $Q$:**
   $$\nabla_Q = \frac{1}{\sqrt{d_k}} \nabla_S K \in \mathbb{R}^{B \times H \times T_q \times d_k}$$

5. **Gradient w.r.t. Key $K$:**
   $$\nabla_K = \frac{1}{\sqrt{d_k}} \nabla_S^T Q \in \mathbb{R}^{B \times H \times T_k \times d_k}$$

---

## 2. Multi-Head Attention (MHA)

Multi-Head Attention projects the model dimension $D_{\text{model}}$ into $H$ heads, each with dimension $d_k = D_{\text{model}} / H$.

### 2.1 Forward Mechanism

1. **Linear Projections:**
   $$Q_{\text{proj}} = X_q W_Q + b_Q, \quad K_{\text{proj}} = X_k W_K + b_K, \quad V_{\text{proj}} = X_v W_V + b_V$$
   where $W_Q, W_K, W_V \in \mathbb{R}^{D_{\text{model}} \times D_{\text{model}}}$.

2. **Head Splitting & Transposition:**
   $$\text{Reshape: } (B, T, D_{\text{model}}) \to (B, T, H, d_k) \to \text{Transpose: } (B, H, T, d_k)$$

3. **Per-Head Attention:**
   $$O_{\text{heads}} = \text{ScaledDotProductAttention}(Q, K, V, M)$$

4. **Head Concatenation & Out Projection:**
   $$\text{Transpose: } (B, H, T, d_k) \to (B, T, H, d_k) \to \text{Reshape: } (B, T, D_{\text{model}})$$
   $$Y = O_{\text{concat}} W_O + b_O$$
   where $W_O \in \mathbb{R}^{D_{\text{model}} \times D_{\text{model}}}$.

---

## 3. Positional Embeddings

### 3.1 Learned Positional Embeddings
Embedding table $P \in \mathbb{R}^{T_{\max} \times D_{\text{model}}}$ added directly to token/state representations:
$$\tilde{X} = X + P[0:T, :]$$

### 3.2 Rotary Position Embedding (RoPE)
RoPE (Su et al., 2024) encodes relative position by rotating pairs of coordinates in the complex plane:
For $j \in \{0, 1, \dots, d_k/2 - 1\}$:
$$\theta_j = \frac{1}{10000^{2j / d_k}}$$
At position $m$, vector $(x_{2j}, x_{2j+1})$ is rotated by angle $m \theta_j$:
$$\begin{pmatrix} \tilde{x}_{2j} \\ \tilde{x}_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} x_{2j} \\ x_{2j+1} \end{pmatrix}$$

---

## 4. Transformer Block Architecture (Pre-LayerNorm / RMSNorm)

AURORA adopts the Pre-LN / Pre-RMSNorm design for gradient stability across deep horizons:

$$x^{(1)} = x + \text{Dropout}\left(\text{MHA}(\text{Norm}_1(x))\right)$$
$$x^{(2)} = x^{(1)} + \text{Dropout}\left(\text{FFN}(\text{Norm}_2(x^{(1)}))\right)$$

where Feed-Forward Network (FFN) is parameterized as:
$$\text{FFN}(z) = \text{Linear}_2\left(\sigma(\text{Linear}_1(z))\right)$$
with intermediate dimension $D_{\text{ffn}} = 4 \cdot D_{\text{model}}$ and activation $\sigma \in \{\text{GELU}, \text{SiLU}, \text{ReLU}\}$.

---

## 5. Causal Transformer Decoder Stack

A sequence of $L$ stacked transformer blocks for autoregressive sequence prediction:
1. Input tokens or continuous states $S_t \in \mathbb{R}^{B \times T \times D_{\text{in}}}$.
2. Linear input projection or discrete token embedding.
3. Position encoding injection.
4. $L$ consecutive Transformer blocks with causal upper-triangular masking.
5. Final normalization ($\text{LayerNorm}$ or $\text{RMSNorm}$).
6. Prediction head ($\text{Linear}(D_{\text{model}}, D_{\text{out}})$).
