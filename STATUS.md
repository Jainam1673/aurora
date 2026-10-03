## Current Milestone: M8 — Scientific Benchmarking & Statistical Evaluation (Completed)

**Overall Health:** GREEN  
**Target Milestone:** M8 (Complete) $\to$ Transitioning to M9 (Systems Performance, Profiling & Native Scaling)  
**Last Updated:** 2026-10-03  

---

## 1. System & Environment Inventory (Detected)

| Component | Detected Version / Specification | Location |
| :--- | :--- | :--- |
| **Operating System** | Linux (Kernel 6.18+, x86_64) | `/` |
| **Python** | Python 3.14.8 (CPython) | `/home/jack/.local/share/mise/installs/python/latest/bin/python3.14` |
| **Python Package Manager** | uv 0.11.32 (x86_64-unknown-linux-gnu) | `/home/jack/.local/bin/uv` |
| **C++ Compiler (Default)** | GCC 16.2.1 20260810 (g++) | `/usr/bin/g++` |
| **C++ Compiler (Clang)** | Clang 22.1.8 (clang++) | `/usr/bin/clang++` |
| **Build System** | CMake 4.4.3 | `/usr/bin/cmake` |
| **Build Generator** | Ninja 1.13.2 | `/usr/bin/ninja` |
| **GoogleTest** | 1.18.0 | `/usr/src/debug/gtest/googletest-1.18.0` |
| **CUDA Toolkit / nvcc** | CUDA 13.3 (V13.3.73) | `/opt/cuda/bin/nvcc` |
| **NVIDIA Driver / GPU** | Driver 580.178.04 / NVIDIA GeForce MX230 (2048 MiB) | `/usr/bin/nvidia-smi` |
| **CPU Architecture** | Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz (4 cores / 8 threads) | AVX2, FMA, SSE4.2 |
| **Memory / Swap** | 7.6 GiB RAM (2.9 GiB available) / 15.0 GiB Swap | `free -h` |

---

## 2. Completed Deliverables

### Milestone M0: Repository Bootstrap
- [x] Initialized Git repository with clean main branch and `.gitignore`.
- [x] Established project directory hierarchy according to Section 4 architecture.
- [x] Configured Python 3.14 project with `pyproject.toml` (`hatchling` backend) and reproducible `uv.lock`.
- [x] Configured native C++23 build tree with `CMakeLists.txt` and multi-compiler presets (`CMakePresets.json`).
- [x] Built minimal Python package `aurora` and native C++23 library `aurora::core`.
- [x] Published foundational blueprints: `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `DECISIONS.md`, `TODO.md`, `STATUS.md`.

### Milestone M1: Numerical Core & Autograd
- [x] Formulated formal mathematical specification in `docs/mathematics/numerical_core.md`.
- [x] Implemented Python reference `Tensor` with strided views, broadcasting, and factory functions (`python/aurora/tensor.py`).
- [x] Implemented Python reverse-mode automatic differentiation tape with unbroadcasting logic (`python/aurora/autograd.py`).
- [x] Implemented Python finite-difference gradient checker (`python/aurora/gradcheck.py`).
- [x] Implemented C++23 native `aurora::Tensor` class with contiguous allocation, views, and C-strides (`cpp/include/aurora/tensor.hpp`, `cpp/src/tensor.cpp`).
- [x] Implemented C++23 reverse-mode autograd engine with computation graph DAG traversal (`cpp/include/aurora/autograd.hpp`, `cpp/src/autograd.cpp`).
- [x] Implemented C++23 finite-difference gradient checker (`cpp/include/aurora/gradcheck.hpp`).
- [x] Supported full operator set: add, sub, mul, div, matmul, sum, mean, reshape, transpose, exp, log, sqrt, relu, gelu, silu, softmax, log_softmax, layer_norm.
- [x] Implemented dedicated C++ parity runner binary `aurora_parity_runner` (`cpp/src/parity_runner.cpp`).
- [x] Implemented cross-language numerical parity test suite (`tests/parity/test_numerical_parity.py`).

### Milestone M2: Neural Network Primitives & Optimizers
- [x] Mathematical specification: `docs/mathematics/nn_and_optimizers.md`.
- [x] Python neural network primitives: `Parameter`, `Module`, `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, `ReLU`, `GELU`, `SiLU`, `Softmax`, `LogSoftmax`, `Sequential`, `MLP`, `ResidualBlock`.
- [x] Python optimizers and utilities: `Optimizer`, `SGD`, `Adam`, `AdamW`, `clip_grad_norm`, `clip_grad_value`, `ConstantLR`, `LinearWarmupDecayLR`, `CosineAnnealingLR`.
- [x] Python checkpoint serialization and deserialization (`python/aurora/checkpoint.py`).
- [x] C++23 native neural network primitives (`cpp/include/aurora/nn.hpp`, `cpp/src/nn.cpp`).
- [x] C++23 native optimizers and schedulers (`cpp/include/aurora/optim.hpp`, `cpp/src/optim.cpp`).
- [x] C++23 zero-dependency JSON checkpoint serializer and deserializer (`cpp/include/aurora/checkpoint.hpp`, `cpp/src/checkpoint.cpp`).
- [x] C++23 GoogleTest suites for NN primitives and optimizers (`tests/cpp/test_nn.cpp`, `tests/cpp/test_optim.cpp`).
- [x] Cross-language checkpoint and optimization parity test suite (`tests/parity/test_checkpoint_parity.py`).

### Milestone M3: Transformer Engine & Attention
- [x] Mathematical specification: `docs/mathematics/transformer.md`.
- [x] Batched multi-dimensional tensor matrix transposition (`swapaxes` and `.mT` / `transpose()`) in Python and C++23.
- [x] Scaled Dot-Product Attention from first principles with numerical scaling and upper-triangular additive causal masking (`create_causal_mask`, `scaled_dot_product_attention`).
- [x] Rotary Position Embeddings (RoPE) Givens rotations preserving vector Euclidean norms (`apply_rotary_pos_emb`).
- [x] Multi-Head Attention (MHA) module with query, key, value, and output linear projections.
- [x] Pre-LayerNorm and Pre-RMSNorm `TransformerBlock` with residual connections and configurable FFN (`Linear -> GELU/ReLU/SiLU -> Linear`).
- [x] Autoregressive `TransformerDecoder` supporting discrete token embeddings or continuous state projections, learned positional embeddings, stacked transformer blocks, final normalization, output projection head, and greedy/temperature autoregressive token generation.
- [x] C++23 native peer implementations: `cpp/include/aurora/attention.hpp`, `cpp/src/attention.cpp`, `cpp/include/aurora/transformer.hpp`, `cpp/src/transformer.cpp`.
- [x] Zero compiler warnings across both GCC 16.2.1 and Clang 22.1.8.

### Milestone M4: Reinforcement Learning Primitives
- [x] Mathematical specification: `docs/mathematics/rl_primitives.md`.
- [x] Python and C++23 native tensor operators: `tanh()`, `clamp()`, and N-dimensional `concat()` with reverse-mode autograd VJP graph nodes (`TanhBackward`/`TanhNode`, `ClampBackward`/`ClampNode`, `ConcatBackward`/`ConcatNode`).
- [x] Python and C++23 policy distributions:
  - `Categorical`: logits, probabilities, sampling, log-likelihood, and Shannon entropy.
  - `Normal`: spherical/diagonal Gaussian, reparameterization trick (`rsample`), log-density, differential entropy.
  - `TanhNormal`: squashed Gaussian bijector with numerically stable softplus identity: $\log(1 - \tanh^2(u)) = 2(\log 2 - u - \text{softplus}(-2u))$.
- [x] Native simulation environments with semi-implicit Euler dynamics:
  - `CartPole`: 4D continuous state, discrete actions $\{0, 1\}$.
  - `Pendulum`: 3D continuous state $[\cos\theta, \sin\theta, \dot\theta]$, continuous torque action in $[-2.0, 2.0]$.
- [x] Trajectory storage and experience replay buffers:
  - `RolloutBuffer`: Generalized Advantage Estimation (GAE-$\lambda$) with configurable bootstrapping, advantage normalization, and minibatch generation.
  - `ReplayBuffer`: FIFO circular experience replay buffer with uniform minibatch sampling.
- [x] Deep RL algorithms (from first principles, zero external dependencies):
  - `PPO`: Clipped surrogate policy objective with differentiable gradient routing, value baseline clipping, entropy exploration bonus.
  - `SAC`: Twin Q-critic networks, reparameterized policy gradient maximization, automatic entropy temperature $\alpha$ adjustment, and Polyak soft target averaging.
- [x] Cross-language numerical parity tests:
  - Tanh forward and backward gradients ($< 10^{-10}$ error).
  - Concat forward and backward gradients across multiple inputs ($< 10^{-10}$ error).
  - Categorical distribution log-prob and entropy ($< 10^{-10}$ error).
  - Normal distribution log-prob and entropy ($< 10^{-10}$ error).
  - TanhNormal squashed distribution log-prob ($< 10^{-10}$ error).
  - GAE advantage estimation across multi-step trajectories ($< 10^{-10}$ error).

### Milestone M5: Latent World Models & Uncertainty Calibration
- [x] Mathematical specification: `docs/mathematics/world_model.md`.
- [x] Python and C++23 native `sigmoid()` operator and autograd backward VJP graph nodes (`SigmoidBackward`/`SigmoidNode`).
- [x] Probabilistic Deep Gaussian Ensemble Dynamics (`EnsembleDynamicsModel`, `EnsembleDynamics`):
  - State-difference parameterization: predicts $\Delta s_t = s_{t+1} - s_t$ and $r_t$.
  - Clamped log-variances in $[\log \sigma_{\min}^2, \log \sigma_{\max}^2]$ for guaranteed numerical stability.
  - Heteroscedastic Gaussian NLL loss with exact dual-language numerical parity.
- [x] Uncertainty Quantification & Decomposition (`UncertaintyEstimator`, `decompose_uncertainty`):
  - Aleatoric uncertainty: $\mathcal{U}_{\text{aleatoric}} = \frac{1}{E} \sum_{e=1}^E \boldsymbol{\sigma}_e^2$.
  - Epistemic uncertainty: $\mathcal{U}_{\text{epistemic}} = \frac{1}{E} \sum_{e=1}^E (\boldsymbol{\mu}_e - \bar{\boldsymbol{\mu}})^2$.
  - Total predictive variance: $\mathcal{U}_{\text{total}} = \mathcal{U}_{\text{aleatoric}} + \mathcal{U}_{\text{epistemic}}$.
  - Disagreement metric: per-sample max epistemic uncertainty $\max_j \mathcal{U}_{\text{epistemic}, j}$.
  - Supports both high-speed numpy evaluation and fully differentiable autograd tensor operations.
- [x] Adaptive Uncertainty-Calibrated Rollout Engine (`ImaginationEngine`):
  - Trajectory sampling under policy $\pi(a \mid s)$ up to horizon $H_{\max}$.
  - Dynamic horizon truncation when epistemic disagreement exceeds trust threshold $\tau_{\text{threshold}}$.
  - Direct integration and synthetic experience injection into `ReplayBuffer`.
- [x] Recurrent State-Space Model (`RSSM`, `GRUCell`):
  - Continuous recurrent state $\mathbf{h}_t$ and stochastic latent state $\mathbf{z}_t$.
  - Stochastic Gaussian prior $p(\mathbf{z}_t \mid \mathbf{h}_t)$ and posterior $q(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{x}_t)$.
  - Multi-head decoders: observation (MSE), reward (MSE), continuation (BCE with $\hat{\gamma}_t \in (0, 1)$).
  - Variational ELBO loss with $\alpha$-balanced KL divergence and detached stop-gradients.
- [x] Cross-language numerical parity tests:
  - Sigmoid forward and backward gradients ($< 10^{-10}$ error).
  - Gaussian NLL loss forward and backward gradients with respect to mean and log-var ($< 10^{-10}$ error).
  - Uncertainty decomposition: mean, aleatoric, epistemic, and total predictive variance ($< 10^{-10}$ error).
  - RSSM analytical Gaussian KL divergence ($< 10^{-10}$ error).

### Milestone M6: MBRL Research Reproduction Suite & Policy Optimization
- [x] Mathematical specification: `docs/mathematics/reproductions.md` (MBPO monotonic bounds, Dreamer $\lambda$-returns, TD-MPC CEM planning, MuZero PUCT MCTS, Decision Transformer RTG sequence conditioning).
- [x] Reproduction Suite Root: `reproductions/README.md` and `reproductions/__init__.py`.
- [x] **MBPO** (Janner et al., 2019): `reproductions/mbpo/` (`algorithm.md`, `mbpo.py`):
  - $k$-step branched ensemble rollouts with Trajectory Sampling 1 (TS1).
  - Hybrid real/synthetic experience replay buffer with ratio mixing.
  - Full SAC policy optimization integration.
- [x] **Dreamer** (Hafner et al., 2020): `reproductions/dreamer/` (`algorithm.md`, `dreamer.py`):
  - RSSM recurrent latent imagination rollout.
  - Analytical Generalized Advantage Estimation ($\lambda$-returns) computed backwards through latent trajectories.
  - Squashed Gaussian continuous latent actor and latent value critic.
- [x] **TD-MPC** (Hansen et al., 2022): `reproductions/tdmpc/` (`algorithm.md`, `tdmpc.py`):
  - Task-oriented non-reconstructive latent representation and dynamics.
  - Cross-Entropy Method (CEM) trajectory optimization with momentum and standard deviation flooring.
  - Terminal Q-value bootstrapping beyond the planning horizon.
- [x] **MuZero** (Schrittwieser et al., 2020): `reproductions/muzero/` (`algorithm.md`, `muzero.py`):
  - Three-network decomposition: representation $h(o)$, dynamics $g(s, a)$, prediction $f(s)$.
  - Upper Confidence Bounds for Trees (PUCT) Monte Carlo Tree Search.
  - Empirical min-max Q-value normalization and multi-task unroll loss.
- [x] **Decision Transformer** (Chen et al., 2021): `reproductions/decision_transformer/` (`algorithm.md`, `decision_transformer.py`):
  - Return-to-Go (RTG) conditioned trajectory token representation.
  - Interleaved sequence modeling with learned timestep embeddings and causal attention masking.
  - Offline trajectory training and autoregressive test-time action generation.
- [x] **C++23 Native Peer Implementation**: `cpp/include/aurora/reproductions.hpp` and `cpp/src/reproductions.cpp`:
  - `compute_lambda_returns`: exact recursive $\lambda$-return computation.
  - `CEMPlanner`: latent trajectory optimization with Gaussian proposal refitting.
  - `PUCTPlanner`: latent MCTS with normalized PUCT score selection and tree backup.
  - `MBPOBufferManager`: hybrid environment/model replay buffer mixing.
- [x] **Autograd & Tensor Core Enhancements**:
  - First-class `PowBackward` and `__pow__` operator on `Tensor`.
  - `stack()` along arbitrary axes with full autograd tracking.
  - PyTorch-compatible `dim` keyword argument support in `concat()`, `stack()`, `sum()`, `mean()`, `softmax()`, `log_softmax()`.
- [x] **Cross-Language Numerical Parity**:
  - Latent Generalized $\lambda$-returns ($< 10^{-10}$ absolute error).
  - CEM trajectory optimization and action bounds ($< 10^{-10}$ absolute error).

### Milestone M7: Novel AURORA Algorithm
- [x] **Theoretical Specification**:
  - Formulated complete mathematical framework in `docs/mathematics/aurora_algorithm.md` including monotonic improvement bound under epistemic model error, adaptive horizon derivation, dynamic replay blending formula, and epistemic risk-sensitive pessimistic value optimization.
  - Detailed engineering algorithm document in `python/aurora/algorithm/algorithm.md` covering pseudocode, inputs/outputs, numerical stability, and failure modes.
- [x] **Python Reference Implementation**:
  - `AdaptiveHorizonScheduler`: validation error threshold decay $\tau = \tau_{\text{base}} \exp(-\kappa \mathcal{L}_{\text{val}})$, state-specific rollout truncation, and cumulative uncertainty budgeting.
  - `DynamicBlendingController`: momentum-smoothed synthetic-to-real replay ratio $\eta_t = \eta_{\max}[1 - \min(1, \bar{u}/u_{\text{target}})]$.
  - `AURORAAgent`: integrated ensemble dynamics, adaptive imagination rollouts, hybrid buffer sampling, and pessimistic actor updates with active exploration trigger.
- [x] **C++23 Native Peer Implementation**:
  - `cpp/include/aurora/aurora_algorithm.hpp` and `cpp/src/aurora_algorithm.cpp`: `AdaptiveHorizonScheduler`, `DynamicBlendingController`, `compute_pessimistic_value`, `should_trigger_active_exploration`.
  - Registered to `aurora_core` and `aurora_parity_runner`.
- [x] **Cross-Language Numerical Parity**:
  - Parity tests in `tests/parity/test_aurora_parity.py` asserting $< 10^{-10}$ numerical error across adaptive horizon calculations, dynamic blending trajectories, and pessimistic value penalties.
- [x] **Empirical Benchmark Suite**:
  - `benchmarks/benchmark_aurora.py` evaluating comparative performance across AURORA (Adaptive), MBPO (Fixed Horizon H=4), and Model-Free SAC on Continuous Control.

### Milestone M8: Scientific Benchmarking & Statistical Evaluation
- [x] **Theoretical & Statistical Protocol**:
  - Formal specification in `docs/mathematics/statistical_evaluation.md`: Interquartile Mean (IQM), Stratified Bootstrap Confidence Intervals (95% CI), Performance Profiles, Probability of Improvement $P(X > Y)$, Welch's two-sample $t$-test, and component ablation definitions.
- [x] **Python Statistical Evaluation Suite**:
  - `evaluation/metrics.py`: `compute_iqm`, `bootstrap_ci`, `compute_statistical_summary`, `StatisticalSummary`.
  - `evaluation/profiles.py`: `performance_profile` CDF generator and `probability_of_improvement`.
  - `evaluation/significance.py`: `welch_t_test` and `mann_whitney_u_test`.
  - `evaluation/manifest.py`: `create_experiment_manifest` and `save_manifest` for immutable git/compiler/hardware audit trails.
  - `evaluation/plotting.py`: `plot_performance_profiles`, `format_ascii_table`.
- [x] **Declarative Experiment Configs & Runner**:
  - `configs/pendulum_aurora.json`, `configs/pendulum_mbpo.json`, `configs/ablation_no_adaptive_horizon.json`, `configs/ablation_no_dynamic_blending.json`, `configs/ablation_no_pessimism.json`.
  - `experiments/runner.py`: multi-seed experiment execution harness capturing manifests and periodic evaluations.
  - `experiments/ablation_study.py`: systematic 4-condition multi-seed ablation evaluation suite outputting JSON manifests and publication-ready tables.
- [x] **C++23 Native Peer Implementation**:
  - `cpp/include/aurora/statistical_evaluation.hpp` and `cpp/src/statistical_evaluation.cpp`: native `compute_iqm`, `bootstrap_ci`, `probability_of_improvement`, and `performance_profile`.
  - Registered to `aurora_core` and `aurora_parity_runner`.
- [x] **Cross-Language Numerical Parity**:
  - `tests/parity/test_evaluation_parity.py` asserting $< 10^{-10}$ error on IQM, probability of improvement, and performance profiles.
- [x] **Unit & Verification Suites**:
  - `tests/python/test_evaluation.py` (8/8 passed).
  - `tests/cpp/test_evaluation.cpp` (4/4 passed).

---

## 3. Verified Artifacts & Test Results

### C++23 Native Build & GoogleTests
- **GCC 16.2.1 (`ctest --preset debug`):**
  - **70/70 passed (100%)** in `0.61s`.
  - Targets: `aurora_cpp_smoke_test`, `aurora_cpp_tensor_test`, `aurora_cpp_gradcheck_test`, `aurora_cpp_nn_test`, `aurora_cpp_optim_test`, `aurora_cpp_transformer_test`, `aurora_cpp_rl_test`, `aurora_cpp_world_model_test`, `aurora_cpp_reproductions_test`, `aurora_cpp_aurora_test`, `aurora_cpp_evaluation_test`.
- **Clang 22.1.8 (`ctest --preset clang-debug`):**
  - **70/70 passed (100%)** in `0.54s` with **zero warnings** under `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`.

### Python 3.14 Test Suite (`pytest`)
- Command: `uv run pytest`
- **123/123 passed (100%)** in `7.86s`:
  - `tests/python/test_smoke.py`: 3 passed
  - `tests/python/test_tensor.py`: 7 passed
  - `tests/python/test_gradcheck.py`: 14 passed
  - `tests/python/test_autograd_properties.py`: 3 passed
  - `tests/python/test_nn.py`: 8 passed
  - `tests/python/test_optim.py`: 5 passed
  - `tests/python/test_transformer.py`: 9 passed
  - `tests/python/test_rl.py`: 11 passed
  - `tests/python/test_world_model.py`: 12 passed
  - `tests/python/test_reproductions.py`: 5 passed
  - `tests/python/test_aurora_algorithm.py`: 6 passed
  - `tests/python/test_evaluation.py`: 8 passed
  - `tests/parity/test_numerical_parity.py`: 10 passed
  - `tests/parity/test_checkpoint_parity.py`: 2 passed
  - `tests/parity/test_transformer_parity.py`: 2 passed
  - `tests/parity/test_rl_parity.py`: 6 passed
  - `tests/parity/test_world_model_parity.py`: 4 passed
  - `tests/parity/test_reproductions_parity.py`: 2 passed
  - `tests/parity/test_aurora_parity.py`: 3 passed
  - `tests/parity/test_evaluation_parity.py`: 3 passed

### Code Quality & Static Analysis
- **Ruff:** `All checks passed!` across 73 source files.
- **Ruff Format:** `73 files already formatted`.
- **Mypy:** `Success: no issues found in 50 source files` (`mypy --strict`).

---

## 4. Next Milestone: M9 — Systems Performance, Profiling & Native Scaling

Primary objectives for M9:
1. Systems performance profiling identifying execution bottlenecks across simulation, dynamics learning, and planning.
2. Native C++23 trajectory throughput benchmarking (env steps/s, model predictions/s, policy updates/s).
3. Memory footprint and cache locality optimization.
4. Modular multi-threaded simulation and rollout scaling.


