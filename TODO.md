# AURORA Engineering & Research Backlog (TODO)

## Milestone 0: Repository Bootstrap (Current)
- [x] Inspect host environment toolchains (Python, GCC, Clang, CMake, Ninja, CUDA, CPU/GPU).
- [x] Establish directory hierarchy separating core, native, tests, research, and papers.
- [x] Configure `pyproject.toml` targeting Python 3.14 with `hatchling` backend and `uv.lock`.
- [x] Create C++23 native build system (`CMakeLists.txt`, `CMakePresets.json`).
- [x] Implement minimal Python package (`python/aurora/`).
- [x] Implement minimal C++23 library (`aurora_core`) and executable (`aurora_cli`).
- [x] Implement Python smoke tests (`tests/python/test_smoke.py`).
- [x] Implement C++ smoke tests with GoogleTest (`tests/cpp/test_smoke.cpp`).
- [x] Configure GitHub Actions CI workflow for linting, building, and testing (`.github/workflows/ci.yml`).
- [x] Establish living governance docs: `STATUS.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `TODO.md`.

## Milestone 1: Numerical Core & Autograd (Completed)
- [x] Design Python `Tensor` reference class with storage, shape, and strides.
- [x] Implement C++23 `aurora::Tensor` class with contiguous allocation and strided views.
- [x] Implement fundamental tensor operations: add, sub, mul, div, matmul, transpose, reshape, reductions.
- [x] Implement reverse-mode autograd tape in Python and C++.
- [x] Implement finite-difference gradient checking utility with tight numerical tolerances.
- [x] Build cross-language numerical parity test harness (`tests/parity/test_numerical_parity.py`).

## Milestone 2: Neural Network Primitives & Optimizers (Completed)
- [x] Formulate formal mathematical specifications (`docs/mathematics/nn_and_optimizers.md`).
- [x] Implement `Module`, `Parameter`, and state dictionary serialization in Python and C++23.
- [x] Implement `Linear`, `MLP`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, and activations (`ReLU`, `GELU`, `SiLU`, `Softmax`, `LogSoftmax`).
- [x] Implement `SGD`, `Adam`, `AdamW` optimizers with decoupled weight decay, learning rate schedulers, and gradient clipping.
- [x] Implement language-agnostic JSON checkpoint exchange format with exact IEEE-754 precision.
- [x] Build cross-language checkpoint and optimization parity test suite (`tests/parity/test_checkpoint_parity.py`).

## Milestone 3: Transformer Engine (Completed)
- [x] Formulate mathematical specifications (`docs/mathematics/transformer.md`).
- [x] Implement batched tensor matrix transpose (`swapaxes`, `.mT` / `transpose()`) in Python and C++23.
- [x] Implement Scaled Dot-Product Attention from first principles with causal masking.
- [x] Implement Rotary Positional Embeddings (RoPE) and learned positional embeddings.
- [x] Implement Multi-Head Attention (MHA) with Q, K, V, and out projections.
- [x] Implement Pre-LayerNorm / Pre-RMSNorm `TransformerBlock` and autoregressive `TransformerDecoder`.
- [x] Implement C++23 native peer implementations (`attention.hpp`/`.cpp`, `transformer.hpp`/`.cpp`).
- [x] Build cross-language attention and transformer block optimization parity tests (`tests/parity/test_transformer_parity.py`).
- [x] Create benchmark suite for attention throughput and generation latency (`benchmarks/benchmark_transformer.py`).

## Milestone 4: RL Foundation (Completed)
- [x] Mathematical specification: `docs/mathematics/rl_primitives.md`.
- [x] Implement policy distributions in Python and C++23: `Categorical`, `Normal`, `TanhNormal` (Squashed Gaussian).
- [x] Implement native classical control simulation environments: `CartPole` and `Pendulum`.
- [x] Implement `RolloutBuffer` with Generalized Advantage Estimation (GAE-$\lambda$) and `ReplayBuffer`.
- [x] Implement `ActorCriticPolicy`, `SquashedGaussianActor`, and `TwinCritic` architectures.
- [x] Implement `PPO` (clipped surrogate objective) and `SAC` (twin critics, auto-alpha) with autograd gradient routing.
- [x] Build dual-language numerical parity tests for distributions, GAE, and autograd tensor operators (`tests/parity/test_rl_parity.py`).
- [x] Validate 100% test pass rate across Python (`tests/python/test_rl.py`) and C++23 (`tests/cpp/test_rl.cpp`).

## Milestone 5: Latent World Models & Uncertainty Calibration (Completed)
- [x] Mathematical specification: `docs/mathematics/world_model.md`.
- [x] Implement deep Gaussian ensemble dynamics (`EnsembleDynamicsModel`, `EnsembleDynamics`) with state-difference formulation.
- [x] Implement heteroscedastic Gaussian NLL loss with exact dual-language numerical parity.
- [x] Implement epistemic and aleatoric uncertainty decomposition (`UncertaintyEstimator`, `decompose_uncertainty`).
- [x] Implement adaptive uncertainty-calibrated imagination engine (`ImaginationEngine`) with dynamic truncation thresholding.
- [x] Implement Recurrent State-Space Model (`RSSM`, `GRUCell`) with variational ELBO and $\alpha$-balanced KL divergence.
- [x] Implement first-class `sigmoid()` operator and autograd backward VJP nodes across Python and C++23.
- [x] Build dual-language numerical parity tests asserting $< 10^{-10}$ error (`tests/parity/test_world_model_parity.py`).
- [x] Validate 100% test pass rate across Python (`tests/python/test_world_model.py`) and C++23 (`tests/cpp/test_world_model.cpp` on GCC and Clang).

## Milestone 6: MBRL Research Reproduction Suite & Policy Optimization (Completed)
- [x] Comprehensive mathematical specification in `docs/mathematics/reproductions.md`.
- [x] Conceptual reproduction of MBPO in `reproductions/mbpo/` (`algorithm.md`, `mbpo.py`).
- [x] Conceptual reproduction of Dreamer in `reproductions/dreamer/` (`algorithm.md`, `dreamer.py`).
- [x] Conceptual reproduction of TD-MPC in `reproductions/tdmpc/` (`algorithm.md`, `tdmpc.py`).
- [x] Conceptual reproduction of MuZero in `reproductions/muzero/` (`algorithm.md`, `muzero.py`).
- [x] Conceptual reproduction of Decision Transformer in `reproductions/decision_transformer/` (`algorithm.md`, `decision_transformer.py`).
- [x] Native C++23 peer implementations: `cpp/include/aurora/reproductions.hpp` and `cpp/src/reproductions.cpp`.
- [x] First-class `PowBackward`, `__pow__`, `stack()`, and `dim` keyword argument support in autograd and tensor core.
- [x] Dual-language numerical parity tests asserting $< 10^{-10}$ error (`tests/parity/test_reproductions_parity.py`).
- [x] Validate 100% test pass rate across Python (`tests/python/test_reproductions.py`) and C++23 (`tests/cpp/test_reproductions.cpp` on GCC and Clang).

## Milestone 7: Novel AURORA Algorithm (Completed)
- [x] Mathematical specification: `docs/mathematics/aurora_algorithm.md` (monotonic improvement bounds, adaptive horizon, dynamic blending, pessimistic optimization).
- [x] Engineering specification: `python/aurora/algorithm/algorithm.md` (pseudocode, I/O, numerical stability, failure modes).
- [x] Implement `AdaptiveHorizonScheduler` in Python with validation error decay thresholding and cumulative budget tracking (`python/aurora/algorithm/scheduler.py`).
- [x] Implement `DynamicBlendingController` in Python with momentum-smoothed synthetic ratio modulation (`python/aurora/algorithm/scheduler.py`).
- [x] Implement `AURORAAgent` in Python with integrated dynamics, adaptive imagination rollouts, hybrid buffer sampling, and pessimistic actor updates (`python/aurora/algorithm/aurora_agent.py`).
- [x] Implement C++23 native peers: `cpp/include/aurora/aurora_algorithm.hpp` and `cpp/src/aurora_algorithm.cpp`.
- [x] Cross-language numerical parity tests asserting $< 10^{-10}$ error (`tests/parity/test_aurora_parity.py`).
- [x] Unit and integration test suites in Python (`tests/python/test_aurora_algorithm.py`) and C++ GoogleTest (`tests/cpp/test_aurora_algorithm.cpp`).
- [x] Empirical benchmark harness comparing AURORA vs. MBPO vs. SAC (`benchmarks/benchmark_aurora.py`).
- [x] Validate 100% test pass rate across Python (112/112) and C++23 (66/66 on GCC and Clang).

## Milestone 8: Scientific Benchmarking & Statistical Evaluation
- [ ] Multi-seed statistical benchmark protocol with aggregate performance profiles (IQM, bootstrap CIs).
- [ ] Systematic ablation studies (dissecting adaptive horizon, dynamic blending, and pessimistic penalty).
- [ ] Benchmark suite across classic control and continuous state tasks.
- [ ] Experiment logging, automated visualization, and manifest tracking infrastructure.

## Milestone 9: Systems Performance & Native Scaling
- [ ] Systems performance profiling (env vs. model vs. planner bottlenecks).
- [ ] Native C++23 SIMD vectorization and cache locality optimizations.
- [ ] Trajectory throughput benchmarks (env steps/s, model steps/s, planning decisions/s).

## Milestone 10: Research Paper & Publication Package
- [ ] Write publication-ready LaTeX manuscript in `paper/`.
- [ ] Automate figure and table generation pipelines from raw experiment data.
- [ ] Assemble reproducibility checklist and self-contained replication scripts.
