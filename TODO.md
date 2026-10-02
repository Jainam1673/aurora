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

## Milestone 4: RL Foundation
- [ ] Define standardized `Environment` interface in Python and C++.
- [ ] Implement Multi-Armed Bandits, TD(0), TD($\lambda$), SARSA, and Q-learning.
- [ ] Implement Policy Gradient, REINFORCE, Actor-Critic, GAE, PPO, and SAC.
- [ ] Benchmark and plot verified learning curves on CartPole and continuous control toys.

## Milestone 5: Latent World Model
- [ ] Implement observation encoders and latent transition models.
- [ ] Implement reward and continuation prediction heads.
- [ ] Build multi-step latent imagination engine with configurable branching.
- [ ] Support interchangeable dynamics: MLP, RSSM, and Transformer.

## Milestone 6: Research Lineage Reproductions
- [ ] Implement conceptual reproduction of Dreamer / RSSM in `reproductions/dreamer/`.
- [ ] Implement conceptual reproduction of TD-MPC in `reproductions/tdmpc/`.
- [ ] Implement conceptual reproduction of MuZero latent search in `reproductions/muzero/`.
- [ ] Document precise scope, architectural differences, and reproduction baselines.

## Milestone 7: Uncertainty Module & Calibration
- [ ] Implement deep ensemble dynamics models $\{f_1, \dots, f_K\}$.
- [ ] Implement latent disagreement and reward uncertainty metrics.
- [ ] Build calibration benchmark measuring uncertainty $U_t$ vs. empirical future error $\|z_{t+h} - \hat{z}_{t+h}\|$.
- [ ] Compute Spearman rank correlation and AUROC for rollout failure prediction.

## Milestone 8: AURORA Algorithm
- [ ] Implement adaptive horizon function $H_t = f(U_t, D_t, S_t, C_t)$.
- [ ] Implement pessimistic value penalization $\tilde{V} = \mu_V - \beta \sigma_V$.
- [ ] Implement active real-data collection trigger.
- [ ] Run benchmark comparisons against fixed horizons $H \in \{1, 3, 5, 10, 20\}$.

## Milestone 9: Scientific Benchmarking
- [ ] Run multi-seed evaluation with stratified bootstrap confidence intervals.
- [ ] Perform full ablation study (full AURORA vs. component ablations).
- [ ] Conduct systems performance profiling (env vs. model vs. planner bottlenecks).

## Milestone 10: Research Paper & Publication Package
- [ ] Write publication-ready LaTeX manuscript in `paper/`.
- [ ] Automate figure and table generation pipelines from raw experiment data.
- [ ] Assemble reproducibility checklist and self-contained replication scripts.
