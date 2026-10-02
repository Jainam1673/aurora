# AURORA Research & Engineering Roadmap

This roadmap outlines the sequenced progression of AURORA from bootstrap to publication-grade research artifacts.

---

## Milestone Overview

| Milestone | Subsystem / Focus | Status | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **M0** | Repository Bootstrap & Systems Foundation | **Completed** | `uv`, CMake C++23, CI, dual-language smoke tests |
| **M1** | Numerical Core & Autograd Tape | **Completed** | `Tensor`, reverse-mode autograd, gradient checks |
| **M2** | Neural Network Primitives & Optimizers | Queued | `Linear`, `MLP`, `LayerNorm`, `AdamW`, checkpointing |
| **M3** | Transformer Engine & Attention | Queued | Causal MHA, Stacked Transformer, token/sec benchmarks |
| **M4** | Reinforcement Learning Primitives | Queued | Bandits, TD($\lambda$), GAE, PPO, SAC |
| **M5** | Latent World Models & Imagination | Queued | Encoder, latent dynamics (MLP/RSSM/Transformer) |
| **M6** | MBRL Research Reproduction Suite | Queued | PlaNet, DreamerV1-V3, TD-MPC2, MuZero |
| **M7** | Uncertainty Estimation & Calibration | Queued | Ensembles, calibration curves, error rank correlation |
| **M8** | AURORA Algorithm Implementation | Queued | Adaptive horizon $H_t$, pessimistic planning, active data |
| **M9** | Benchmarking & Systems Analysis | Queued | Multi-seed IQM, ablation studies, throughput profiling |
| **M10** | Publication-Grade Paper & Artifacts | Queued | LaTeX paper, appendices, reproducible manifests |

---

## Detailed Milestone Specifications

### M0: Repository Bootstrap & Systems Foundation
- **Goal:** Establish a rock-solid dual-language environment.
- **Deliverables:**
  - Python 3.14 managed by `uv` with reproducible `uv.lock`.
  - C++23 native build system managed by CMake and Ninja with GCC and Clang support.
  - Automated CI workflow for linting, testing, and formatting.
  - Core living documentation: `STATUS.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `TODO.md`.
  - Smoke tests in Python and C++ demonstrating verified builds and executions.
- **Acceptance Criteria:** `uv run pytest` and `ctest --preset debug` pass cleanly without warnings.

### M1: Numerical Core & Autograd Tape
- **Goal:** Implement tensor abstractions and reverse-mode automatic differentiation from first principles.
- **Deliverables:**
  - N-dimensional tensor with striding, views, and broadcasting.
  - Primitive arithmetic and matrix ops with finite-difference gradient checks.
  - Numerical parity tests between Python and C++ implementations.
- **Acceptance Criteria:** Finite-difference gradient error $< 10^{-5}$ across all operators.

### M2: Neural Network Primitives & Optimizers
- **Goal:** Build foundational neural network layers and training loops.
- **Deliverables:**
  - `Linear`, `LayerNorm`, `RMSNorm`, activations, parameter containers.
  - `SGD`, `Adam`, `AdamW` with decoupled weight decay.
  - Standardized serialization format for cross-language weight loading.
- **Acceptance Criteria:** Deterministic toy optimization parity between Python and C++.

### M3: Transformer Engine
- **Goal:** High-performance, mathematically verified causal Transformer blocks.
- **Deliverables:**
  - Scaled dot-product attention, multi-head attention with causal masking.
  - Autoregressive sequence prediction test harness.
  - Throughput benchmarks (latency, memory, FLOPs estimate).
- **Acceptance Criteria:** Exact parity on attention masks and weights against PyTorch reference.

### M4: RL Foundation
- **Goal:** Establish core model-free RL algorithms on Tier-A benchmark environments.
- **Deliverables:**
  - Bandits, tabular Q-learning, SARSA, Actor-Critic, PPO, SAC.
  - Verified learning curves on CartPole, MountainCar, Acrobot.
- **Acceptance Criteria:** PPO and SAC reach standard solve thresholds across 5 independent seeds.

### M5: Latent World Models & Imagination
- **Goal:** Action-conditioned latent dynamics prediction and rollout generation.
- **Deliverables:**
  - Observation encoder, transition predictor, reward predictor, continuation head.
  - Multi-step latent rollout engine supporting branching and configurable horizons.
- **Acceptance Criteria:** Multi-step rollouts generate valid state predictions and loss convergence.

### M6: MBRL Research Reproduction Suite
- **Goal:** Reproduce conceptual foundations of leading MBRL architectures.
- **Deliverables:**
  - Reproductions of Dreamer (latent imagination), TD-MPC (latent trajectory optimization), MuZero (MCTS in latent space), Decision Transformer.
  - Documented deviations and compute conditions in `reproductions/`.
- **Acceptance Criteria:** Reproductions match published conceptual dynamics and score baselines.

### M7: Uncertainty Module & Calibration
- **Goal:** Quantify and calibrate epistemic and aleatoric world model uncertainty.
- **Deliverables:**
  - Deep ensemble dynamics models.
  - Disagreement metrics and uncertainty calibration benchmarking suite.
  - Empirical evaluation of $U_t$ vs. cumulative trajectory error $\|z_{t+h} - \hat{z}_{t+h}\|$.
- **Acceptance Criteria:** AUROC and Spearman rank correlation demonstrate statistically significant predictive capability.

### M8: AURORA Algorithm
- **Goal:** Novel adaptive uncertainty-calibrated imagination and planning algorithm.
- **Deliverables:**
  - Dynamic horizon controller $H_t = f(U_t, D_t, S_t, C_t)$.
  - Pessimistic planning value correction: $\tilde{V} = \mu_V - \beta \sigma_V$.
  - Active real experience acquisition switch based on uncertainty bounds.
- **Acceptance Criteria:** Controlled comparison showing sample-efficiency gain over fixed horizons $H \in \{1, 3, 5, 10, 20\}$.

### M9: Scientific Benchmarking & Systems Analysis
- **Goal:** Rigorous statistical evaluation across multi-tier environments.
- **Deliverables:**
  - Multi-seed runs (IQM, bootstrap confidence intervals, performance profiles).
  - Systems throughput breakdown: env steps/s, world model steps/s, planning decisions/s.
  - Ablation suite (AURORA full vs. -uncertainty, -adaptive horizon, -pessimism, -active data).
- **Acceptance Criteria:** Statistically grounded comparisons under fair interaction and compute budgets.

### M10: Research Paper & Publication Artifacts
- **Goal:** Complete, publication-ready research paper package.
- **Deliverables:**
  - Full LaTeX manuscript with sections 1-18 and comprehensive appendices.
  - Automated figure and table generation scripts pulling directly from raw results.
  - Reproducibility checklist and one-click reproduction scripts.
- **Acceptance Criteria:** Standalone compilation of camera-ready PDF and release bundle.
