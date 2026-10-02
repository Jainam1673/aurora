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

## Milestone 1: Numerical Core & Autograd
- [ ] Design Python `Tensor` reference class with storage, shape, and strides.
- [ ] Implement C++23 `aurora::Tensor` class with contiguous allocation and strided views.
- [ ] Implement fundamental tensor operations: add, sub, mul, div, matmul, transpose, reshape, reductions.
- [ ] Implement reverse-mode autograd tape in Python and C++.
- [ ] Implement finite-difference gradient checking utility with tight numerical tolerances.
- [ ] Build cross-language numerical parity test harness (`tests/parity/test_tensor_parity.py`).

## Milestone 2: Neural Network Primitives
- [ ] Implement `Module`, `Parameter`, and state dictionary serialization.
- [ ] Implement `Linear`, `MLP`, `Embedding`, `LayerNorm`, `RMSNorm`, and activations (`ReLU`, `GELU`, `SiLU`).
- [ ] Implement `SGD`, `Adam`, `AdamW` optimizers with weight decay.
- [ ] Implement deterministic cross-language checkpoint exchange format (JSON/binary).

## Milestone 3: Transformer Engine
- [ ] Implement Scaled Dot-Product Attention from first principles.
- [ ] Implement Multi-Head Attention with causal autoregressive masking.
- [ ] Implement Transformer Encoder / Decoder blocks and stacked architecture.
- [ ] Create benchmark suite comparing Python and C++ attention throughput (tokens/sec, memory).

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
