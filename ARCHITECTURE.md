# AURORA System Architecture

## 1. Overview and Core Philosophy

AURORA (**Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents**) is structured as a dual-language, layered research platform:

- **Python (3.14):** Optimizes for research velocity, rapid algorithmic hypothesis formulation, statistical evaluation, rich visualization, and reference numerical correctness. Managed exclusively via `uv`.
- **C++ (C++23):** Optimizes for systems efficiency, deterministic low-overhead execution, high-throughput environment simulation, high-frequency planning (e.g., CEM/MCTS rollouts), and production deployment.

Python and C++ are **scientific peers**. Where mathematical semantics overlap, cross-language numerical parity is strictly enforced via golden vectors and automated property tests.

```text
+-------------------------------------------------------------------------------+
|                             RESEARCH & HYPOTHESIS                             |
|       Theory & Mathematical Formulation  <--->  Empirical Benchmarking        |
+-------------------------------------------------------------------------------+
                                        │
                                        ▼
+-----------------------------------+       +-----------------------------------+
|       PYTHON RESEARCH LAYER       |       |       NATIVE C++23 LAYER          |
|         (Python 3.14 + uv)        |       |        (CMake + Ninja)            |
|                                   |       |                                   |
| • High-level experiment orchestration|    | • Low-latency model rollout engine|
| • Uncertainty calibration analysis|       | • Real-time planning (CEM, MCTS)  |
| • Statistical evaluation & IQM    |       | • Cache-friendly tensor memory    |
| • Publication figure pipelines    |       | • Modular CUDA acceleration       |
| • PyTorch & Gymnasium ecosystem   |       | • Zero-overhead C++23 abstractions|
+-----------------------------------+       +-----------------------------------+
                  ▲                                           ▲
                  │                                           │
                  └───────────────[ PARITY LAYER ]────────────┘
                              Golden Test Vectors
                         Tolerance-verified Invariants
                           Shared Checkpoint Format
```

---

## 2. Layered Subsystems

AURORA is structured into 11 decoupled subsystems:

### Subsystem 1: Tensor and Autograd Foundation
- **Python:** Custom tensor reference implementation with explicit reverse-mode autograd tape alongside PyTorch reference backends.
- **C++:** Contiguous multidimensional array abstraction (`aurora::Tensor`) with strided views, explicit memory ownership, CPU SIMD vectorization, and modular CUDA kernels.
- **Validation:** Finite-difference gradient checking across all forward/backward operators.

### Subsystem 2: Neural Network Primitives
- Modules: `Linear`, `MLP`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, activations (`ReLU`, `GELU`, `SiLU`, `Softmax`, `LogSoftmax`), residual connections.
- Parameter containers, state dict serialization, and optimizers (`SGD`, `Adam`, `AdamW` with decoupled weight decay).

### Subsystem 3: Transformer Engine
- Scaled dot-product attention, multi-head causal self-attention, and stacked Transformer blocks.
- Autoregressive sequence modeling for sequential world models and Decision Transformers.

### Subsystem 4: Environment & Trajectory Abstraction
- Unified `StepResult(observation, reward, terminated, truncated, info)` abstraction.
- High-throughput vectorized environment execution and replay buffer management.

### Subsystem 5: World Model Architecture
- Latent dynamics pipeline:
  $$\text{Observation } o_t \xrightarrow{\text{Encoder}} z_t \xrightarrow[\text{Dynamics}]{a_t} \hat{z}_{t+1} \xrightarrow{\text{Heads}} (\hat{r}_t, \hat{\gamma}_t, \hat{o}_t)$$
- Interchangeable dynamics models: MLP, recurrent (RSSM), and Transformer.

### Subsystem 6: Uncertainty Estimation & Calibration
- Deep ensemble dynamics models: $\{f_1, \dots, f_K\}$.
- Disagreement metric:
  $$U_t = \frac{1}{K-1} \sum_{k=1}^K \|\hat{z}_{t+1}^{(k)} - \bar{z}_{t+1}\|^2$$
- Calibration measurement: rank correlation and expected calibration error comparing $U_t$ against cumulative rollout drift $\|z_{t+h} - \hat{z}_{t+h}\|$.

### Subsystem 7: The AURORA Adaptive Imagination Engine
- Dynamically selects rollout horizon $H_t = f(U_t, D_t, S_t, C_t)$ based on epistemic uncertainty $U_t$, novelty $D_t$, value sensitivity $S_t$, and compute budget $C_t$.
- Pessimistic value penalization: $\tilde{V}(s, a) = \mu_V(s, a) - \beta \sigma_V(s, a)$.
- Active data acquisition triggered when uncertainty exceeds safety thresholds.

### Subsystem 8: Planners
- Interchangeable trajectory optimizers: Actor rollouts, Random Shooting, Cross-Entropy Method (CEM), and tree search.

### Subsystem 9: Reinforcement Learning Algorithms
- Core RL algorithms implemented from first principles: Bandits, TD($\lambda$), Q-Learning, PPO, SAC.

### Subsystem 10: Research Lineage & Reproductions
- Faithful reproductions with documented deviations: World Models, PlaNet, Dreamer (v1/v2/v3), TD-MPC/TD-MPC2, MuZero, Decision Transformer.

### Subsystem 11: Statistical Evaluation & Reporting
- Interquartile Mean (IQM), stratified bootstrap confidence intervals, performance profiles, and automated LaTeX table/figure generation.

---

## 3. Parity and Verification Protocol

```text
[ Mathematical Specification ]
              │
              ├──> [ Python Reference Implementation ] ──┐
              │                                          ├──> [ Numerical Parity Comparison ]
              └──> [ C++23 Native Implementation ]     ──┘      (max_abs_err < tol)
```

- Every primitive must pass unit tests, property-based tests (Hypothesis), and cross-language numerical tests before downstream usage.
