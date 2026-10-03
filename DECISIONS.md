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

---

## ADR-009: Latent World Models, Deep Probabilistic Ensembles, and Uncertainty Calibration
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Milestone 5 establishes the predictive core of AURORA: learning environmental transition dynamics and quantifying predictive uncertainty to prevent compounding model exploitation during policy imagination. Key requirements:
  1. Heteroscedastic Gaussian dynamics predicting state residual transitions $\Delta s_t = s_{t+1} - s_t$ and rewards $r_t$.
  2. Numerically stable Gaussian Negative Log-Likelihood (NLL) optimization avoiding variance collapse or gradient explosion.
  3. Rigorous epistemic vs. aleatoric uncertainty quantification to bound imagination trust regions.
  4. Recurrent State-Space Models (RSSM) for partially observable dynamics with analytical Gaussian KL divergence and KL balancing.
- **Decision:**
  1. Implement deep probabilistic ensembles with clamped log-variance $\log \sigma^2 \in [\log \sigma_{\min}^2, \log \sigma_{\max}^2]$ and compute heteroscedastic Gaussian NLL loss with exact dual-language numerical parity.
  2. Implement `UncertaintyEstimator` decomposing predictive variance into aleatoric ($\frac{1}{E} \sum \boldsymbol{\sigma}_e^2$) and epistemic ($\frac{1}{E} \sum (\boldsymbol{\mu}_e - \bar{\boldsymbol{\mu}})^2$) components, with both fast array-based execution and differentiable autograd tensor operations.
  3. Implement `ImaginationEngine` executing synthetic trajectory rollouts under actor policies with dynamic horizon truncation triggered when max epistemic disagreement exceeds $\tau_{\text{threshold}}$.
  4. Implement `RSSM` with first-principles `GRUCell` recurrent transition, stochastic prior and posterior distributions, multi-head decoders, and $\alpha$-balanced KL divergence with stop-gradients.
  5. Add native `sigmoid()` activation and backward autograd nodes across Python and C++23 tensor engines.
- **Consequences:** Provides a complete, fully tested, peer-validated world modeling engine with zero external ML framework dependencies, achieving $< 10^{-10}$ cross-language parity on NLL loss, uncertainty decomposition, and KL divergence.

---

## ADR-010: MBRL Research Reproduction Suite & Native Policy Optimization
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Milestone 6 establishes research-grade reproductions of foundational Model-Based Reinforcement Learning lineages to benchmark against and lay the theoretical groundwork for the novel AURORA algorithm:
  1. Dyna-style branched ensemble rollouts with hybrid replay buffers (MBPO; Janner et al., 2019).
  2. Latent imagination actor-critic with analytical $\lambda$-returns (Dreamer; Hafner et al., 2020).
  3. Non-reconstructive task latent trajectory optimization via Cross-Entropy Method and terminal Q-value bootstrapping (TD-MPC; Hansen et al., 2022).
  4. Latent Monte Carlo Tree Search with PUCT and empirical min-max value normalization (MuZero; Schrittwieser et al., 2020).
  5. Return-to-Go conditioned autoregressive trajectory sequence modeling (Decision Transformer; Chen et al., 2021).
  6. Scientific peer C++23 native implementations providing high-throughput trajectory planning, $\lambda$-return calculation, and hybrid buffer management.
- **Decision:**
  1. Implement each algorithm from first principles in `reproductions/` with dedicated `algorithm.md` documentation, rigorous types, and zero external ML framework dependencies.
  2. Implement native C++23 peers in `cpp/include/aurora/reproductions.hpp` and `cpp/src/reproductions.cpp` (`compute_lambda_returns`, `CEMPlanner`, `PUCTPlanner`, and `MBPOBufferManager`).
  3. Implement first-class `PowBackward` and `__pow__` operator across autograd and tensor core, alongside `stack()` and `dim` keyword parameter aliases.
  4. Enforce strict numerical parity testing across Python and C++23 engines asserting $< 10^{-10}$ error on latent generalized $\lambda$-returns and trajectory optimization outputs.
- **Consequences:** Validates all core MBRL architectural paradigms natively within AURORA, creates canonical baselines for downstream experimental comparison, and achieves 100% test pass rate across 103 pytest tests and 60 GoogleTest targets (both GCC and Clang toolchains).

---

## ADR-011: Novel AURORA Algorithm Architecture (Uncertainty-Calibrated Adaptive Imagination & Dynamic Blending)
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Milestone 7 introduces the core novel algorithmic contribution of the project: **AURORA** (*Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents*). Standard Dyna-style MBRL (e.g. MBPO) uses predetermined, hand-tuned rollout schedules and static real-to-synthetic replay ratios $\eta$, leading to catastrophic policy degradation when imaginary rollouts venture into out-of-distribution hallucinations.
- **Decision:**
  1. **Adaptive Horizon Scheduling:** Dynamically truncate imagination rollouts along individual state trajectories when peak epistemic disagreement $u_{\text{epi}}(s_h, a_h) > \tau_{\text{threshold}}$ or when accumulated discounted uncertainty budget $\sum_{k=0}^h \gamma^k u_{\text{epi}}(s_k, a_k) > B_{\max}$. Adapt threshold $\tau_{\text{threshold}} = \tau_{\text{base}} \exp(-\kappa \cdot \mathcal{L}_{\text{val}})$ based on dynamics validation error.
  2. **Dynamic Experience Blending:** Modulate the synthetic data fraction $\eta_t = \eta_{\max} [1 - \min(1, \bar{u}_t / u_{\text{target}})]$ with exponential moving average momentum smoothing ($\rho = 0.8$), gracefully decaying model dependence to pure model-free replay when model uncertainty surges.
  3. **Epistemic Risk-Sensitive Pessimistic Value Optimization:** Regularize policy improvement via $\tilde{Q}(s, a) = \min_j Q_j(s, a) - \beta_{\text{pess}} \cdot u_{\text{epi}}(s, a)$, providing lower-bound value guarantees that penalize model hallucinations.
  4. **Active Exploration Trigger:** When mean epistemic uncertainty exceeds $\tau_{\text{active}}$, trigger exploration perturbations to gather high-information transition data in unfamiliar dynamics regimes.
  5. **C++23 Native Peer Implementation:** Provide native C++23 implementations of `AdaptiveHorizonScheduler`, `DynamicBlendingController`, and `compute_pessimistic_value` in `cpp/include/aurora/aurora_algorithm.hpp` and `cpp/src/aurora_algorithm.cpp`.
  6. **Cross-Language Numerical Parity:** Enforce $< 10^{-10}$ error bounds between Python 3.14 and C++23 native implementations.
- **Consequences:** Provides a rigorous, mathematically unified model-based RL algorithm with proven monotonic improvement bounds, full numerical parity across dual-language peers, and empirical benchmarks against fixed-horizon MBPO and model-free SAC.

---

## ADR-012: Scientific Benchmarking, Stratified Statistical Evaluation, and Ablation Methodology
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Reinforcement Learning evaluations frequently suffer from statistical fragility, non-reproducible seed cherry-picking, and high sensitivity to outliers. Point estimates (sample mean $\pm$ standard deviation) are easily distorted by heavy-tailed failures.
- **Decision:**
  1. Mandate **Interquartile Mean (IQM)** as the central location metric for all benchmark evaluations, trimming the top and bottom 25% of scores to eliminate outlier skew while preserving distribution shape.
  2. Estimate sampling variability using **Stratified Percentile Bootstrap Confidence Intervals** (95% CIs) with $B=2000$ resamples.
  3. Evaluate pairwise algorithm dominance using **Probability of Improvement** $P(X > Y) = \frac{1}{N_X N_Y} \sum_{i,j} [\mathbf{1}(x_i > y_j) + 0.5 \cdot \mathbf{1}(x_i == y_j)]$ and **Performance Profile CDFs** $F(\tau) = \frac{1}{N} \sum_i \mathbf{1}(x_i \ge \tau)$.
  4. Perform hypothesis testing via Welch's two-sample $t$-test (unequal variances) and non-parametric Mann-Whitney U tests.
  5. Enforce an immutable JSON **Experiment Manifest** for every training run recording git commit, dirty flag, Python/compiler versions, exact hyperparameters, and hardware specs.
  6. Implement native C++23 peers (`compute_iqm`, `bootstrap_ci`, `probability_of_improvement`, `performance_profile`) in `cpp/include/aurora/statistical_evaluation.hpp` and `cpp/src/statistical_evaluation.cpp`, tested with $< 10^{-10}$ cross-language parity.
- **Consequences:** Elevates AURORA's experimental infrastructure to peer-reviewed publication standards, guaranteeing statistical integrity and auditable traceability for all experimental claims.

---

## ADR-013: Systems Benchmarking, Throughput Maximization, and Native C++23 Scaling
- **Date:** 2026-10-03
- **Status:** Accepted
- **Context:** Milestone 9 focuses on systems performance, flamegraph profiling, and native scaling. Model-based RL introduces heavy computational demands (ensemble dynamics training, multi-step imagination rollouts, pessimistic value regularization). Identifying bottlenecks, eliminating redundant indexing overheads, and vectorizing inner loops is essential for scalability.
- **Decision:**
  1. **Formal Systems Specification:** Establish `docs/systems/systems_performance.md` defining operational throughput metrics ($S_{\text{sim}}$, $S_{\text{dyn}}$, $S_{\text{imag}}$, $S_{\text{opt}}$, $S_{\text{E2E}}$), Amdahl's law decomposition, latency distributions ($p_{50}, p_{90}, p_{99}$), and cache-line memory layout guidelines.
  2. **Contiguous SIMD Fast-Paths in C++23 Tensor Core:** Bypass multi-dimensional stride coordinate calculations in binary arithmetic (`add`, `sub`, `mul`, `div`), scalar operations, and unary activations (`relu`, `gelu`, `silu`, `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `clamp`, `sum`) when tensors are contiguous, enabling raw pointer iteration and AVX2/FMA auto-vectorization.
  3. **High-Precision Native C++23 Benchmark Suite:** Implement `benchmarks/cpp/benchmark_throughput.cpp` built via CMake option `AURORA_BUILD_BENCHMARKS=ON`, recording microsecond latencies, $p_{50}/p_{99}$ percentiles, memory bandwidth, GFLOP/s, and exporting machine-readable JSON manifests (`results/cpp_benchmark_results.json`).
  4. **Vectorized Actor Policy Imagination in Python:** Replace sequential per-sample policy evaluation in `rollout_adaptive_imagination` with batched tensor evaluation, reducing imagination rollout time by over 8x (from 4.88s to 0.59s) and elevating online end-to-end training throughput by 2.74x (from 16.98 to 46.49 env steps/sec).
  5. **cProfile and Cross-Language Systems Harness:** Provide `benchmarks/profile_aurora.py` generating `.prof` flamegraph traces and `benchmarks/benchmark_cross_language.py` producing empirical cross-language throughput comparisons.
- **Consequences:** Delivers multi-gigahertz throughput on C++23 tensor operations (up to 703 M elements/s allocation, 562 M elements/s ReLU, 4.38 GFLOPs/s GEMM), 31.8x C++ speedup on statistical IQM, and a 42% reduction in full pytest test suite execution time, maintaining 100% test pass rate across 123 pytest tests and 70 GoogleTest targets across GCC and Clang with zero warnings.
