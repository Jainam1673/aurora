# AURORA

**Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents**

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-green.svg)](https://docs.python.org/3.14/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

AURORA is a research-grade, reproducible reinforcement-learning research platform and flagship research project investigating adaptive, uncertainty-aware imagination and planning in model-based RL.

---

## Central Research Question

> **Can an RL agent improve sample efficiency and planning reliability by dynamically deciding how much to trust its learned world model, how far to imagine, and when to collect real experience?**

---

## Core Pillars & First Principles

1. **Correctness before performance:** Mathematical reference $\to$ unit tests $\to$ property tests $\to$ golden numerical tests $\to$ cross-language parity $\to$ integration tests $\to$ optimization.
2. **Research before novelty:** Every addition addresses an explicit scientific question and hypothesis.
3. **Never hide uncertainty:** Instrument, test, calibrate, and document uncertainty; never fabricate or cherry-pick.
4. **Preserve reproducibility:** Track commit, dependencies, compiler, seeds, hyperparameters, and compute budgets.
5. **Python and C++ are peers:**
   - **Python (3.14):** Research velocity, interactive exploration, visualizations, reference correctness.
   - **C++ (C++23):** Systems quality, memory efficiency, high-throughput simulation, planning, and deployment.
6. **Modern Package Management:** Strict reliance on `uv` for Python environments and dependencies.

---

## Repository Structure

```text
AURORA/
├── README.md               # Project overview and instructions
├── LICENSE                 # Apache 2.0 License
├── CITATION.cff            # Research citation metadata
├── CHANGELOG.md            # Version and progress log
├── pyproject.toml          # Python project specification (Python 3.14)
├── uv.lock                 # Reproducible Python dependency lockfile
├── CMakeLists.txt          # Root CMake specification (C++23)
├── CMakePresets.json       # Standardized CMake build/test presets
├── STATUS.md               # Living implementation status
├── ROADMAP.md              # Research roadmap and milestone tracking
├── ARCHITECTURE.md         # System design and component interactions
├── DECISIONS.md            # Architecture Decision Records (ADRs)
├── TODO.md                 # Detailed action items and backlog
├── python/
│   └── aurora/             # Python reference and research package
├── cpp/
│   ├── include/aurora/     # C++23 public header-only and interface library
│   ├── src/                # C++23 implementation sources and binaries
│   └── cuda/               # Modular CUDA acceleration kernels
├── tests/
│   ├── python/             # Pytest / Hypothesis unit and property tests
│   ├── cpp/                # GoogleTest unit and integration tests
│   └── parity/             # Cross-language numerical parity test vectors
├── benchmarks/             # Benchmarks (Python & Google Benchmark)
├── environments/           # Environment adapters and wrappers
├── datasets/               # Trajectory and offline dataset utilities
├── experiments/            # Declarative experiment configs and execution scripts
├── configs/                # Shared model and training YAML configurations
├── evaluation/             # Statistical evaluation, bootstrap CI, performance profiles
├── reproductions/          # Reference lineage (Dreamer, PlaNet, TD-MPC, MuZero, etc.)
├── theory/                 # Mathematical proofs, derivations, error bounds
├── paper/                  # Publication-ready LaTeX source, figures, and appendices
├── docs/                   # Architectural notes, tutorials, algorithm specs
├── scripts/                # Development, parity verification, and analysis scripts
├── docker/                 # Containerized reproducibility environments
└── results/                # Structured experiment outputs, checkpoints, and logs
```

---

## Quickstart

### Prerequisites

- **Python:** 3.14+ (managed via `uv`)
- **C++ Compiler:** GCC 14+ or Clang 18+ supporting C++23
- **Build System:** CMake 3.28+ and Ninja
- **Optional:** CUDA Toolkit 12.0+ for GPU acceleration

### Python Setup (uv)

```bash
# Sync local virtual environment with locked dependencies
uv sync --extra dev

# Run Python smoke test
uv run pytest tests/python

# Run type checker and linter
uv run mypy python tests/python
uv run ruff check python tests/python
```

### C++23 Build (CMake & Ninja)

```bash
# Configure using CMake Presets
cmake --preset debug
# or cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build library, executables, and tests
cmake --build --preset debug

# Run C++ tests
ctest --preset debug
```

---

## Milestone Roadmap

- [x] **M0: Repository Bootstrap** — uv, CMake, C++23, Python 3.14, CI, docs, smoke tests
- [ ] **M1: Numerical Core** — Tensor abstractions, autograd, gradient checks, parity tests
- [ ] **M2: Neural Network Core** — Modules, optimizers, checkpoint exchange
- [ ] **M3: Transformer Engine** — Scaled dot-product, causal attention, benchmarks
- [ ] **M4: RL Foundation** — Bandits, TD($\lambda$), PPO, SAC, toy environments
- [ ] **M5: World Model** — Latent dynamics, encoder/decoder, RSSM, rollout engine
- [ ] **M6: Reproduction Suite** — Dreamer lineage, TD-MPC, MuZero, Decision Transformer
- [ ] **M7: Uncertainty Module** — Ensembles, calibration, epistemic vs. aleatoric estimation
- [ ] **M8: AURORA Algorithm** — Adaptive horizon, pessimistic planning, active data collection
- [ ] **M9: Benchmark & Systems Suite** — Multi-seed evaluation, IQM bootstrap, throughput profiling
- [ ] **M10: Research Paper** — Publication-grade paper, formal theory, camera-ready reproducibility artifacts

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
