# AURORA Research & Engineering Roadmap

This roadmap outlines the sequenced progression of AURORA from bootstrap to publication-grade research artifacts.

---

## Milestone Overview

| Milestone | Subsystem / Focus | Status | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **M0** | Repository Bootstrap & Systems Foundation | **Completed** | `uv`, CMake C++23, CI, dual-language smoke tests |
| **M1** | Numerical Core & Autograd Tape | **Completed** | `Tensor`, reverse-mode autograd, gradient checks |
| **M2** | Neural Network Primitives & Optimizers | **Completed** | `Linear`, `MLP`, `LayerNorm`, `AdamW`, checkpointing |
| **M3** | Transformer Engine & Attention | **Completed** | Causal MHA, Stacked Transformer, parity & benchmarks |
| **M4** | Reinforcement Learning Primitives | **Completed** | Bandits, TD($\lambda$), GAE, PPO, SAC |
| **M5** | Latent World Models & Imagination | **Completed** | Ensemble dynamics, uncertainty, RSSM, rollout engine |
| **M6** | MBRL Research Reproduction Suite | **Completed** | MBPO, Dreamer, TD-MPC, MuZero, Decision Transformer |
| **M7** | AURORA Novel Algorithm | **Active** | Adaptive horizon $H^*(s)$, dynamic blending, calibrated RL |
| **M8** | Uncertainty Calibration & Active Acquisition | Queued | Deep ensemble calibration, error rank correlation |
| **M9** | Scientific Benchmarking & Systems Analysis | Queued | Multi-seed IQM, ablation studies, throughput profiling |
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

### M4: RL Foundation (Completed)
- **Goal:** Establish core model-free RL algorithms, distributions, and native simulation environments.
- **Deliverables:**
  - Policy distributions: Categorical, Normal, and TanhNormal (Squashed Gaussian).
  - Trajectory buffers: RolloutBuffer with GAE-$\lambda$ and ReplayBuffer.
  - Native physics environments: CartPole and Pendulum in Python and C++23.
  - Core RL algorithms: PPO (clipped objective) and SAC (twin Q-critics, auto $\alpha$).
  - Full dual-language parity on distributions and GAE ($< 10^{-10}$ error).
- **Acceptance Criteria:** Dual-language unit tests and parity tests pass with 100% success rate.

### M5: Latent World Models & Uncertainty Calibration (Completed)
- **Goal:** Probabilistic ensemble dynamics, epistemic/aleatoric uncertainty quantification, and adaptive imagination rollout engine.
- **Deliverables:**
  - State-difference heteroscedastic Gaussian ensemble dynamics model (`EnsembleDynamicsModel`, `EnsembleDynamics`).
  - Differentiable Gaussian NLL loss with exact dual-language numerical parity.
  - Epistemic and aleatoric uncertainty decomposition module (`UncertaintyEstimator`, `decompose_uncertainty`).
  - Adaptive uncertainty-calibrated rollout engine (`ImaginationEngine`) with dynamic truncation thresholding.
  - Recurrent State-Space Model (`RSSM`, `GRUCell`) with multi-head decoders and $\alpha$-balanced KL divergence.
  - Dual-language parity test suite verifying $< 10^{-10}$ error across all world model operations.
- **Acceptance Criteria:** 100% test pass rate across pytest (96/96) and GoogleTest (56/56 on GCC and Clang).

### M6: MBRL Research Reproduction Suite (Completed)
- **Goal:** Reproduce conceptual foundations of leading MBRL architectures from first principles.
- **Deliverables:**
  - Full mathematical derivations in `docs/mathematics/reproductions.md`.
  - MBPO: $k$-step branched rollouts with hybrid real/synthetic replay buffer and SAC.
  - Dreamer: RSSM latent imagination, analytical $\lambda$-returns, latent actor-critic.
  - TD-MPC: non-reconstructive task latent dynamics, CEM trajectory optimization, terminal Q bootstrapping.
  - MuZero: representation/dynamics/prediction decomposition, PUCT MCTS with min-max normalization.
  - Decision Transformer: Return-to-Go (RTG) conditioned autoregressive sequence modeling.
  - C++23 native peers: `compute_lambda_returns`, `CEMPlanner`, `PUCTPlanner`, and `MBPOBufferManager`.
  - Autograd and tensor core extensions (`PowBackward`, `__pow__`, `stack()`, and `dim` argument support).
  - Cross-language numerical parity suite verifying $< 10^{-10}$ error.
- **Acceptance Criteria:** 100% test pass rate across pytest (103/103) and GoogleTest (60/60 on GCC and Clang).

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
