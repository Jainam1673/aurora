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
