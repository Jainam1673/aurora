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
