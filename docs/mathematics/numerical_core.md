# Mathematical Specification: AURORA Numerical Core & Autograd Tape

This document defines the mathematical, algorithmic, and numerical standards for the tensor engine and reverse-mode automatic differentiation tape in both Python and C++23.

---

## 1. Tensor Memory Layout, Striding, and Indexing

Let a tensor $T$ have rank $d \in \mathbb{N}$ and shape $(n_0, n_1, \dots, n_{d-1})$.

### 1.1 Contiguous Striding (Row-Major / C-Order)
For a contiguous tensor in row-major layout, the strides $(s_0, s_1, \dots, s_{d-1})$ satisfy:
$$s_{d-1} = 1$$
$$s_k = \prod_{j=k+1}^{d-1} n_j = s_{k+1} \cdot n_{k+1}, \quad \text{for } 0 \le k < d-1$$

The total number of elements is:
$$N = \prod_{k=0}^{d-1} n_k$$

For a multi-index $(i_0, i_1, \dots, i_{d-1})$ with $0 \le i_k < n_k$, the linear memory offset is:
$$\text{offset} = \sum_{k=0}^{d-1} i_k \cdot s_k$$

### 1.2 Broadcasting Rules
Two shapes $A = (a_0, \dots, a_{d_A-1})$ and $B = (b_0, \dots, b_{d_B-1})$ are broadcast-compatible if, starting from the trailing dimensions:
- For each dimension pair $(a_i, b_j)$, either $a_i = b_j$, $a_i = 1$, or $b_j = 1$.
- Missing leading dimensions in the lower-rank tensor are prepended with dimension 1.

The resulting broadcast shape has dimension:
$$c_k = \max(a_k, b_k)$$

During backpropagation, any gradient tensor $\nabla_{\text{broadcast}}$ accumulated on a broadcasted operand must be summed across dimensions where the original shape had size 1 or was prepended.

---

## 2. Reverse-Mode Automatic Differentiation (Vector-Jacobian Products)

Reverse-mode autograd evaluates derivatives of a scalar loss $\mathcal{L} \in \mathbb{R}$ with respect to all intermediate variables $x_i$ using Vector-Jacobian Products (VJPs).

Let node $y$ be computed as $y = f(x_1, \dots, x_m)$. Given the upstream adjoint:
$$\bar{y} = \frac{\partial \mathcal{L}}{\partial y}$$
the adjoints for inputs $x_i$ are accumulated as:
$$\bar{x}_i \mathrel{+}= \bar{y} \cdot \frac{\partial y}{\partial x_i}$$

### 2.1 Elementary Arithmetic Operations

1. **Addition:** $y = x_1 + x_2$
   $$\bar{x}_1 = \bar{y}, \quad \bar{x}_2 = \bar{y}$$

2. **Subtraction:** $y = x_1 - x_2$
   $$\bar{x}_1 = \bar{y}, \quad \bar{x}_2 = -\bar{y}$$

3. **Hadamard Multiplication:** $y = x_1 \odot x_2$
   $$\bar{x}_1 = \bar{y} \odot x_2, \quad \bar{x}_2 = \bar{y} \odot x_1$$

4. **Division:** $y = \frac{x_1}{x_2}$
   $$\bar{x}_1 = \frac{\bar{y}}{x_2}, \quad \bar{x}_2 = -\bar{y} \odot \frac{x_1}{x_2^2} = -\frac{\bar{y} \odot y}{x_2}$$

### 2.2 Matrix Multiplication
Let $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $C = A B \in \mathbb{R}^{M \times N}$.
Given upstream gradient $\bar{C} \in \mathbb{R}^{M \times N}$:
$$\bar{A} = \bar{C} B^T, \quad \bar{B} = A^T \bar{C}$$

For batched matrix multiplication with batch dimensions $B_{0}, \dots, B_{k-1}$, the rule applies independently per 2D slice.

### 2.3 Reductions

1. **Sum:** $y = \sum_{i} x_i$
   $$\bar{x}_i = \bar{y}$$

2. **Mean:** $y = \frac{1}{N} \sum_{i=1}^N x_i$
   $$\bar{x}_i = \frac{1}{N} \bar{y}$$

### 2.4 Unary Nonlinearities

1. **Exponential:** $y = \exp(x)$
   $$\bar{x} = \bar{y} \odot \exp(x) = \bar{y} \odot y$$

2. **Natural Logarithm:** $y = \log(x)$ (for $x > 0$)
   $$\bar{x} = \frac{\bar{y}}{x}$$

3. **Square Root:** $y = \sqrt{x}$ (for $x > 0$)
   $$\bar{x} = \frac{\bar{y}}{2 \sqrt{x}} = \frac{\bar{y}}{2 y}$$

4. **ReLU:** $y = \max(0, x)$
   $$\bar{x} = \bar{y} \odot \mathbb{I}(x > 0)$$
   *(Subgradient at $x = 0$ is set to 0 by convention).*

5. **GELU (Gaussian Error Linear Unit):**
   Exact formulation:
   $$\text{GELU}(x) = x \Phi(x) = \frac{x}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
   Derivative:
   $$\frac{d}{dx} \text{GELU}(x) = \Phi(x) + x \phi(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] + \frac{x}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$$
   $$\bar{x} = \bar{y} \odot \frac{d}{dx} \text{GELU}(x)$$

6. **SiLU (Swish-1):**
   $$y = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
   Derivative:
   $$\frac{dy}{dx} = \sigma(x) + x \sigma(x)(1 - \sigma(x)) = \sigma(x) [1 + x(1 - \sigma(x))]$$
   $$\bar{x} = \bar{y} \odot \sigma(x) [1 + x(1 - \sigma(x))]$$

### 2.5 Normalizations and Probability Distributions

1. **Softmax:**
   For a vector $x \in \mathbb{R}^K$, with $s_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$:
   $$\frac{\partial s_i}{\partial x_j} = s_i (\delta_{ij} - s_j)$$
   Given upstream gradient $\bar{s} \in \mathbb{R}^K$:
   $$\bar{x}_i = s_i \left( \bar{s}_i - \sum_{j=1}^K \bar{s}_j s_j \right) = s_i (\bar{s}_i - \bar{s}^T s)$$

2. **Log-Softmax:**
   $$y_i = \log s_i = x_i - \max(x) - \log \left(\sum_j e^{x_j - \max(x)}\right)$$
   Given upstream gradient $\bar{y} \in \mathbb{R}^K$:
   $$\bar{x}_i = \bar{y}_i - s_i \left(\sum_{j=1}^K \bar{y}_j\right)$$

3. **Layer Normalization:**
   For $x \in \mathbb{R}^D$, scale $\gamma \in \mathbb{R}^D$, bias $\beta \in \mathbb{R}^D$, and $\epsilon > 0$:
   $$\mu = \frac{1}{D} \sum_{j=1}^D x_j, \quad \sigma^2 = \frac{1}{D} \sum_{j=1}^D (x_j - \mu)^2, \quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   $$y_i = \gamma_i \hat{x}_i + \beta_i$$
   Gradients with respect to parameters:
   $$\bar{\gamma}_i = \sum \bar{y}_i \hat{x}_i, \quad \bar{\beta}_i = \sum \bar{y}_i$$
   Gradient with respect to input $x$:
   Let $\sigma_{\epsilon} = \sqrt{\sigma^2 + \epsilon}$. Then:
   $$\bar{x}_i = \frac{1}{D \sigma_{\epsilon}} \left[ D \bar{y}_i \gamma_i - \sum_{j=1}^D \bar{y}_j \gamma_j - \hat{x}_i \sum_{j=1}^D \bar{y}_j \gamma_j \hat{x}_j \right]$$

---

## 3. Finite-Difference Numerical Gradient Checking

To guarantee absolute mathematical correctness before any performance optimization, all operations must pass two-sided finite-difference checking:

$$\left(\nabla_{\text{num}} f(x)\right)_i = \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$
where $e_i$ is the $i$-th standard basis vector and $\epsilon = 10^{-6}$ (for double precision float64) or $\epsilon = 10^{-3}$ (for float32).

### Error Metrics
For analytical gradient $g_{\text{analytical}}$ and numerical gradient $g_{\text{numerical}}$:

1. **Max Absolute Error:**
   $$\text{MaxAbsErr} = \max_i |g_{\text{analytical}, i} - g_{\text{numerical}, i}|$$

2. **Relative Error:**
   $$\text{RelErr}_i = \frac{|g_{\text{analytical}, i} - g_{\text{numerical}, i}|}{\max(|g_{\text{analytical}, i}|, |g_{\text{numerical}, i}|) + \delta}$$
   where $\delta = 10^{-7}$ prevents division by zero in flat regions.

### Acceptance Criterion
For float64 reference testing:
$$\text{RelErr} < 10^{-5} \quad \text{or} \quad \text{MaxAbsErr} < 10^{-6}$$
