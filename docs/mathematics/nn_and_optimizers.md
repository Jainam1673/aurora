# Mathematical Specification: Neural Network Primitives & Optimizers

This document defines the mathematical formulation, initialization strategies, update rules, and checkpointing specifications for neural network primitives in AURORA (both Python 3.14 and C++23).

---

## 1. Parameter Containers and Layer Primitives

### 1.1 Linear Layer
For input $X \in \mathbb{R}^{B \times D_{\text{in}}}$, weight matrix $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$, and bias $b \in \mathbb{R}^{D_{\text{out}}}$:
$$Y = X W^T + b$$
Gradients:
$$\nabla_X = \nabla_Y W, \quad \nabla_W = \nabla_Y^T X, \quad \nabla_b = \sum_{\text{batch}} \nabla_Y$$

#### Initialization (Kaiming / He Uniform & Normal)
For activation with negative slope $a$ (default $a = 0$ for ReLU/Linear):
$$\sigma = \sqrt{\frac{2}{(1 + a^2) \cdot D_{\text{in}}}}$$
- **He Normal:** $W_{ij} \sim \mathcal{N}(0, \sigma^2)$
- **He Uniform:** $W_{ij} \sim \mathcal{U}(-\sqrt{3}\sigma, \sqrt{3}\sigma)$
- **Bias:** $b_i = 0$

### 1.2 RMSNorm (Root Mean Square Layer Normalization)
RMSNorm reduces computational overhead relative to LayerNorm by enforcing scale invariance without shifting by the mean:
$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}$$
$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)}$$
$$y_i = \gamma_i \hat{x}_i$$

#### Gradient Derivation
Let $g_i = \bar{y}_i \gamma_i$.
$$\frac{\partial \text{RMS}}{\partial x_i} = \frac{x_i}{D \cdot \text{RMS}}$$
$$\bar{x}_i = \frac{g_i}{\text{RMS}} - \frac{x_i}{D \cdot \text{RMS}^3} \sum_{j=1}^D g_j x_j = \frac{1}{\text{RMS}} \left( g_i - \hat{x}_i \cdot \frac{1}{D} \sum_{j=1}^D g_j \hat{x}_j \right)$$
$$\bar{\gamma}_i = \sum_{\text{batch}} \bar{y}_i \hat{x}_i$$

### 1.3 Embedding
For vocabulary of size $V$ and embedding dimension $E$, table $W \in \mathbb{R}^{V \times E}$:
$$\text{Embedding}(i) = W[i, :]$$
Gradient:
$$\bar{W}[i, :] \mathrel{+}= \bar{y}_i$$

### 1.4 Dropout (Inverted)
For retention probability $q = 1 - p \in (0, 1]$:
$$y = \begin{cases} \frac{x \odot m}{q}, & \text{training mode, where } m_i \sim \text{Bernoulli}(q) \\ x, & \text{eval mode} \end{cases}$$
Gradient:
$$\bar{x} = \frac{\bar{y} \odot m}{q}$$

### 1.5 Multi-Layer Perceptron (MLP) & Residual Blocks
- **MLP:** Sequential composition:
  $$\text{MLP}(x) = W_L \left(\sigma_{L-1} \dots \sigma_1(x W_1^T + b_1) \dots \right)^T + b_L$$
- **Residual Block:**
  $$y = x + F(x) \quad \text{or} \quad y = W_{\text{proj}} x + F(x)$$

---

## 2. Optimizers

### 2.1 Stochastic Gradient Descent with Momentum (SGD)
Given learning rate $\eta$, momentum coefficient $\mu \in [0, 1)$, and weight decay $\lambda \ge 0$:
$$g_t = \nabla_\theta \mathcal{L}_t + \lambda \theta_{t-1}$$
$$v_t = \mu v_{t-1} + g_t$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

### 2.2 Adam
Given $\beta_1, \beta_2 \in [0, 1)$, $\epsilon > 0$, learning rate $\eta$, and L2 penalty $\lambda$:
$$g_t = \nabla_\theta \mathcal{L}_t + \lambda \theta_{t-1}$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 2.3 AdamW (Decoupled Weight Decay)
In AdamW (Loshchilov & Hutter, 2019), weight decay is applied directly to the weights, uncoupled from the moving gradient moments:
$$g_t = \nabla_\theta \mathcal{L}_t$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 2.4 Gradient Norm Clipping
To stabilize training against gradient explosions:
$$\|g\|_2 = \sqrt{\sum_{p \in \text{Params}} \sum_i g_{p, i}^2}$$
If $\|g\|_2 > C$:
$$g_p \leftarrow g_p \cdot \frac{C}{\|g\|_2 + 10^{-6}}$$

### 2.5 Learning Rate Schedulers
1. **Linear Warmup and Linear Decay:**
   $$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, & t < T_{\text{warmup}} \\ \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{T_{\max} - t}{T_{\max} - T_{\text{warmup}}}, & t \ge T_{\text{warmup}} \end{cases}$$

2. **Cosine Annealing with Warmup:**
   $$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, & t < T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}}\right)\right), & t \ge T_{\text{warmup}} \end{cases}$$

---

## 3. Checkpoint Exchange Format

The canonical checkpoint format is JSON-encoded for auditability and exact numerical interchange between Python and C++23:

```json
{
  "aurora_version": "0.1.0",
  "metadata": {
    "framework": "AURORA-M2",
    "timestamp": 1727914000,
    "step": 1000
  },
  "model_state_dict": {
    "linear1.weight": {
      "shape": [4, 8],
      "data": [...]
    },
    "linear1.bias": {
      "shape": [4],
      "data": [...]
    }
  },
  "optimizer_state_dict": {
    "type": "AdamW",
    "step": 1000,
    "lr": 0.0003,
    "param_groups": [
      {
        "lr": 0.0003,
        "weight_decay": 0.01,
        "params": ["linear1.weight", "linear1.bias"]
      }
    ]
  }
}
```
