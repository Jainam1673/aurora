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
| **M7** | AURORA Novel Algorithm | **Completed** | Adaptive horizon $H^*(s)$, dynamic blending, calibrated RL |
| **M8** | Scientific Benchmarking & Statistical Evaluation | **Completed** | Multi-seed IQM, ablation studies, throughput profiling |
| **M9** | Systems Performance & Native Scaling | **Active** | Profiling, SIMD vectorization, memory optimization |
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

### M7: Novel AURORA Algorithm (Completed)
- **Goal:** Design, implement, and validate the novel AURORA model-based reinforcement learning algorithm.
- **Deliverables:**
  - Mathematical formulation (`docs/mathematics/aurora_algorithm.md`) and algorithm specification (`python/aurora/algorithm/algorithm.md`).
  - `AdaptiveHorizonScheduler`: validation-error decay thresholding and cumulative discounted uncertainty budgeting.
  - `DynamicBlendingController`: momentum-smoothed synthetic-to-real replay modulation.
  - `AURORAAgent`: integrated world model, adaptive rollouts, hybrid buffer sampling, and pessimistic actor updates.
  - C++23 native peers in `cpp/include/aurora/aurora_algorithm.hpp` and `cpp/src/aurora_algorithm.cpp`.
  - Parity test suite (`tests/parity/test_aurora_parity.py`) verifying $< 10^{-10}$ error.
  - Benchmark harness (`benchmarks/benchmark_aurora.py`) evaluating AURORA vs. MBPO vs. SAC.
- **Acceptance Criteria:** 100% test pass rate across pytest (112/112) and GoogleTest (66/66 on GCC and Clang).

### M8: Scientific Benchmarking & Statistical Evaluation (Completed)
- **Goal:** Rigorous statistical evaluation across multi-tier environments and ablation studies.
- **Deliverables:**
  - Formal protocol in `docs/mathematics/statistical_evaluation.md`.
  - Python evaluation module (`evaluation/metrics.py`, `profiles.py`, `significance.py`, `manifest.py`, `plotting.py`).
  - Declarative experiment configs in `configs/` (`pendulum_aurora`, `pendulum_mbpo`, ablations).
  - Reproducible multi-seed runner (`experiments/runner.py`) and ablation study suite (`experiments/ablation_study.py`).
  - Native C++23 peers in `cpp/include/aurora/statistical_evaluation.hpp` and `cpp/src/statistical_evaluation.cpp`.
  - Parity test suite (`tests/parity/test_evaluation_parity.py`) verifying $< 10^{-10}$ error.
  - Automated visualization (`results/ablation/performance_profiles.png`).
- **Acceptance Criteria:** 100% test pass rate across pytest (123/123) and GoogleTest (70/70 on GCC and Clang).

### M9: Systems Performance & Native Scaling (Active)
- **Goal:** Systems profiling, simulation throughput maximization, and native C++23 execution scaling.
- **Deliverables:**
  - Detailed flamegraph and bottleneck profiling (environment step, dynamics inference, policy update).
  - Trajectory throughput benchmarks (steps/sec) comparing Python and C++23 native implementations.
  - Native memory locality and SIMD vectorization optimizations.
- **Acceptance Criteria:** Quantifiable speedup and latency reduction without compromising numerical parity.

### M10: Research Paper & Publication Artifacts
- **Goal:** Complete, publication-ready research paper package.
- **Deliverables:**
  - Full LaTeX manuscript with sections 1-18 and comprehensive appendices.
  - Automated figure and table generation scripts pulling directly from raw results.
  - Reproducibility checklist and one-click reproduction scripts.
- **Acceptance Criteria:** Standalone compilation of camera-ready PDF and release bundle.
