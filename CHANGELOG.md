# Changelog

All notable changes to the AURORA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-03

### Added
- **M0 Repository Bootstrap**:
  - Python 3.14 package configuration with `uv` (`pyproject.toml`, `uv.lock`).
  - C++23 native build system using CMake and Ninja (`CMakeLists.txt`, `CMakePresets.json`).
  - Architectural blueprints: `ARCHITECTURE.md`, `ROADMAP.md`, `STATUS.md`, `DECISIONS.md`, `TODO.md`.
  - Python smoke test infrastructure (`tests/python/test_smoke.py`).
  - C++ smoke test infrastructure with GoogleTest (`tests/cpp/test_smoke.cpp`).
  - Continuous Integration configuration (`.github/workflows/ci.yml`).
  - Core directory structure separating research, reproductions, native systems, and paper artifacts.
- **M1 Numerical Core & Autograd**:
  - Python reference `Tensor` with broadcasting, views, and reverse-mode autograd (`python/aurora/tensor.py`, `autograd.py`).
  - C++23 native `aurora::Tensor` with row-major memory allocation, strided views, and DAG autograd (`cpp/include/aurora/tensor.hpp`, `cpp/src/tensor.cpp`, `autograd.cpp`).
  - Primitives: add, sub, mul, div, matmul, sum, mean, reshape, transpose, exp, log, sqrt, relu, gelu, silu, softmax, log_softmax, layer_norm.
  - Finite-difference gradient checkers in Python and C++ (`python/aurora/gradcheck.py`, `cpp/include/aurora/gradcheck.hpp`).
  - Cross-language numerical parity suite (`tests/parity/test_numerical_parity.py`) with C++ parity binary (`aurora_parity_runner`).
  - Mathematical specification in `docs/mathematics/numerical_core.md`.
- **M2 Neural Network Primitives & Optimizers**:
  - Python neural network primitives: `Parameter`, `Module`, `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, `Sequential`, `MLP`, `ResidualBlock`.
  - Python optimizers and schedulers: `Optimizer`, `SGD`, `Adam`, `AdamW`, `clip_grad_norm`, `clip_grad_value`, `ConstantLR`, `LinearWarmupDecayLR`, `CosineAnnealingLR`.
  - C++23 native neural network primitives: `Parameter`, `Module`, `Linear`, `Embedding` (with `EmbeddingNode` autograd backwards), `LayerNorm`, `RMSNorm`, `Dropout`, `Sequential`, `MLP`, `ResidualBlock`.
  - C++23 native optimizers and schedulers: `Optimizer`, `SGD` (with momentum), `Adam`, `AdamW` (with decoupled weight decay), `clip_grad_norm`, `clip_grad_value`, `ConstantLR`, `LinearWarmupDecayLR`, `CosineAnnealingLR`.
  - Zero-dependency JSON checkpoint serializer and deserializer with exact IEEE-754 17-digit precision (`save_checkpoint`, `load_checkpoint`).
  - C++ GoogleTest suites: `tests/cpp/test_nn.cpp`, `tests/cpp/test_optim.cpp`.
  - Cross-language optimization parity suite: `tests/parity/test_checkpoint_parity.py`, verifying $< 10^{-10}$ error lockstep between Python and C++ across forward, autograd backward, and AdamW updates.
  - Mathematical specification in `docs/mathematics/nn_and_optimizers.md`.
  - ADR-006: Language-Agnostic JSON Checkpoint Exchange Format & Parameter Registration Order.
- **M3 Transformer Engine & Attention**:
  - Mathematical specification: `docs/mathematics/transformer.md`.
  - Python & C++ batched matrix transposition: `swapaxes` and `.mT` / `transpose()`.
  - Scaled Dot-Product Attention from first principles with numerical scaling and upper-triangular additive causal masking (`create_causal_mask`, `scaled_dot_product_attention`).
  - Rotary Position Embeddings (`apply_rotary_pos_emb`) preserving vector Euclidean norms.
  - Multi-Head Attention (`MultiHeadAttention`) with linear projections for queries, keys, values, and outputs.
  - `TransformerBlock` with Pre-LayerNorm / Pre-RMSNorm, residual connections, and configurable FFN (`Linear -> GELU/ReLU/SiLU -> Linear`).
  - `TransformerDecoder` autoregressive sequence model with token embeddings or continuous projection, learned positional embeddings, stacked transformer blocks, final normalization, prediction head, and autoregressive `generate()`.
  - C++23 native implementations (`cpp/include/aurora/attention.hpp`, `cpp/src/attention.cpp`, `cpp/include/aurora/transformer.hpp`, `cpp/src/transformer.cpp`).
  - Python unit test suite (`tests/python/test_transformer.py`: 9/9 passed, 63/63 repo total).
  - C++23 GoogleTest suite (`tests/cpp/test_transformer.cpp`: 8/8 passed, 40/40 repo total on GCC and Clang).
  - Cross-language numerical parity suite (`tests/parity/test_transformer_parity.py`) verifying $< 10^{-10}$ error on attention forward/backward and multi-parameter transformer block optimization.
  - Benchmark suite (`benchmarks/benchmark_transformer.py`).
- **M4 Reinforcement Learning Primitives**:
  - Mathematical specification in `docs/mathematics/rl_primitives.md`.
  - Python & C++23 native tensor operators: `tanh()`, `clamp()`, and N-dimensional `concat()` with reverse-mode autograd VJP graph nodes (`TanhBackward`/`TanhNode`, `ClampBackward`/`ClampNode`, `ConcatBackward`/`ConcatNode`).
  - Python & C++23 policy distributions: `Categorical`, `Normal`, and `TanhNormal` (Squashed Gaussian with stable softplus change-of-variables log determinant).
  - Native simulation environments in Python and C++23: `CartPole` and `Pendulum` with semi-implicit Euler dynamics.
  - Trajectory storage and buffers: `RolloutBuffer` with Generalized Advantage Estimation (GAE-$\lambda$) and `ReplayBuffer` circular experience replay with uniform sampling.
  - Deep RL policy architectures: `ActorCriticPolicy`, `SquashedGaussianActor`, `TwinCritic`.
  - Model-free RL algorithms: `PPO` (clipped surrogate loss, value loss, entropy bonus) and `SAC` (twin Q-critics, reparameterized policy improvement, automatic entropy temperature $\alpha$).
  - Python unit test suite (`tests/python/test_rl.py`: 11/11 passed, 80/80 repo total).
  - C++23 GoogleTest suite (`tests/cpp/test_rl.cpp`: 7/7 passed, 47/47 repo total on GCC and Clang).
  - Cross-language numerical parity suite (`tests/parity/test_rl_parity.py`: 6/6 passed) verifying $< 10^{-10}$ error on `tanh`, `concat`, `Categorical`, `Normal`, `TanhNormal`, and GAE advantage calculations.
  - **M5 Latent World Models & Uncertainty Calibration**:
  - Mathematical specification in `docs/mathematics/world_model.md`.
  - Python & C++23 native `sigmoid()` operator and autograd backward VJP graph nodes (`SigmoidBackward`/`SigmoidNode`).
  - Python & C++23 deep probabilistic Gaussian ensemble dynamics (`EnsembleDynamicsModel`, `EnsembleDynamics`) with state-difference target formulation ($\Delta s_t, r_t$) and clamped log-variances.
  - Heteroscedastic Gaussian Negative Log-Likelihood loss with exact dual-language numerical parity.
  - Uncertainty decomposition module (`UncertaintyEstimator`, `decompose_uncertainty`) computing aleatoric uncertainty, epistemic disagreement, and total predictive variance.
  - Adaptive uncertainty-calibrated imagination rollout engine (`ImaginationEngine`) with dynamic truncation thresholding preventing compounding model exploitation.
  - Recurrent State-Space Model (`RSSM`, `GRUCell`) with variational ELBO and $\alpha$-balanced KL divergence with stop-gradients.
  - Python unit test suite (`tests/python/test_world_model.py`: 12/12 passed, 96/96 repo total).
  - C++23 GoogleTest suite (`tests/cpp/test_world_model.cpp`: 9/9 passed, 56/56 repo total on GCC and Clang).
  - Cross-language numerical parity suite (`tests/parity/test_world_model_parity.py`: 4/4 passed, 24/24 parity total) asserting $< 10^{-10}$ error on `sigmoid`, Gaussian NLL loss, uncertainty metrics, and analytical KL divergence.
  - ADR-009: Latent World Models, Deep Probabilistic Ensembles, and Uncertainty Calibration.
