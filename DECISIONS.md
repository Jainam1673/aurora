# Architecture Decision Records (ADRs)

This document records key architectural, scientific, and engineering decisions made during the lifecycle of the AURORA project.

---

## ADR-001: Target Python 3.14 and Mandate `uv` for Environment Management
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Reproducibility in ML research often fails due to divergent package versions, non-deterministic solver algorithms, and implicit global environments. Modern Python 3.14 offers enhanced runtime performance and typing features.
- **Decision:** Target Python 3.14 exclusively. Mandate `uv` (`pyproject.toml` and `uv.lock`) as the single source of truth for dependencies. No `pip install` or divergent requirement files.
- **Consequences:** Reproducible, lightning-fast virtual environments and deterministic dependency locking across all research runs.

---

## ADR-002: Target Native C++23 with CMake and Ninja
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Model-based RL rollouts and online trajectory optimization (e.g. CEM, MCTS) require high computational throughput and predictable memory locality. C++23 provides standard library advancements (e.g., `std::println`, `std::expected`, `std::mdspan`, concepts).
- **Decision:** Build the native layer targeting C++23 using CMake (minimum 3.28) and Ninja generator. Support both GCC (14+) and Clang (18+).
- **Consequences:** Eliminates boilerplate, guarantees strict adherence to modern language features, and ensures high simulation throughput.

---

## ADR-003: Python and C++ as Peer Systems with Parity Testing
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Many RL frameworks relegate C++ to an unmaintained backend or treat Python as an unoptimized throwaway script.
- **Decision:** Python and C++ are scientific peers. Python is optimized for exploratory velocity, statistical analysis, and reference mathematics; C++ is optimized for memory efficiency, low latency, and throughput. A dedicated test suite (`tests/parity/`) validates numerical equality across both implementations against golden vectors.
- **Consequences:** Requires double-implementation discipline for shared primitives, but provides unprecedented validation confidence and prevents subtle numerical bugs.

---

## ADR-004: Modular and Optional CUDA Acceleration
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Requiring CUDA from day one complicates CI, CPU-only verification, and portability. However, GPU acceleration is indispensable for large-scale world model training.
- **Decision:** Decouple CUDA into a modular backend (`cpp/cuda/`) gated by `AURORA_ENABLE_CUDA`. Implement and rigorously test all algorithms on the CPU reference implementation before adding GPU kernels.
- **Consequences:** Zero blocking dependency on specific GPU hardware during foundational math validation; clean progressive enhancement path.

---

## ADR-005: Declarative Experiment Manifests and Statistical Metrics
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** RL literature frequently suffers from reporting single lucky seeds or non-robust $mean \pm std$ statistics that hide outliers.
- **Decision:** All experiments must be configured via declarative YAML files and output immutable metadata manifests (commit, seed, compiler, hardware, lockfile). Evaluation must compute Interquartile Mean (IQM), bootstrap confidence intervals, and full distribution plots.
- **Consequences:** Guarantees publication-ready statistical rigor and auditability of all empirical claims.

---

## ADR-006: Language-Agnostic JSON Checkpoint Exchange Format & Parameter Registration Order
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Cross-language numerical parity between Python 3.14 and C++23 native systems requires exact state serialization and deserialization of neural network parameters, optimizer states (such as AdamW first and second moments), and metadata without binary platform endianness or compiler-specific struct layout discrepancies.
- **Decision:** Use a human-auditable JSON checkpoint format with IEEE-754 17-digit precision (`std::setprecision(17)`). Maintain strict parameter registration order in C++ via `std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>` matching Python's insertion-ordered dictionary, and adopt standard moment names (`exp_avg`, `exp_avg_sq`).
- **Consequences:** Completely zero-dependency checkpoint parser in C++, guaranteed $< 10^{-10}$ floating-point parity between Python and C++ model weights and optimizer buffers across multiple consecutive optimization steps.

---

## ADR-007: Pre-LayerNorm Transformer Architecture & Multi-Dimensional Matrix Transposition
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Transformer architectures in model-based RL (decision transformers, trajectory transformers, world model dynamics) require stable gradient propagation across long horizons and efficient batched multi-head attention. Traditional Post-LN transformers suffer from unstable initialization and gradient vanishing, while multi-head tensor operations require swapping arbitrary axes in 4D tensors $(B, H, T, d_k)$.
- **Decision:**
  1. Standardize on the Pre-LN / Pre-RMSNorm residual architecture: $x_{l+1} = x_l + \text{MHA}(\text{Norm}(x_l))$, ensuring clean residual gradient highways without auxiliary warmup tricks.
  2. Implement `swapaxes(axis1, axis2)` and batched matrix transposition `.mT` (swapping only the final two dimensions $[-1, -2]$) in both Python and C++23 native `Tensor`.
  3. Ensure autograd topological tape execution correctly handles non-contiguous strided views via automatic contiguous reconciliation during matmul and reductions.
  4. Enforce strict submodule registration order matching dataflow order (`norm1`, `attn`, `norm2`, `linear1`, `linear2`) to ensure identical checkpoint serialization and optimization parity.
- **Consequences:** Guaranteed $< 10^{-10}$ mathematical parity across Python and C++ for both non-causal and causal Multi-Head Attention, stable multi-layer autoregressive rollouts, and seamless serialization.

---

## ADR-008: Reinforcement Learning Primitives, Policy Distributions, and Dual-Language Physics Environments
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Milestone 4 requires foundational reinforcement learning primitives: action distributions for discrete and continuous control, experience buffers (GAE rollout buffer and off-policy replay buffer), native physics simulation environments (CartPole and Pendulum), and reference algorithms (PPO and SAC). Key challenges include:
  1. Differentiable routing of min/max operators in clipped surrogate objectives (PPO) and double Q-learning (SAC) without breaking autograd graph connectivity.
  2. Numerical stability in change-of-variables log-determinant for Squashed Gaussians ($\text{TanhNormal}$) as actions approach boundaries $\pm 1$.
  3. Ensuring exact numerical equivalence between Python and C++ native physics simulation environments.
- **Decision:**
  1. Implement `TanhNormal` utilizing the softplus identity $\log(1 - \tanh^2(u)) = 2(\log 2 - u - \text{softplus}(-2u))$ to eliminate catastrophic numerical cancellation.
  2. Implement `tanh()`, `clamp()`, and N-dimensional `concat()` with explicit backward VJP graph nodes in both Python and C++23 native autograd engines.
  3. Route clipped surrogate losses and minimum twin critic evaluations through differentiable boolean masks to maintain uninterrupted pathwise gradient propagation.
  4. Implement deterministic classical control environments (`CartPole`, `Pendulum`) natively in C++23 and Python with matched semi-implicit Euler dynamics.
- **Consequences:** Eliminates external RL library dependencies, provides independent C++ simulation throughput with zero Python GIL overhead, and delivers $< 10^{-10}$ numerical parity across distribution densities and GAE advantage estimates.

