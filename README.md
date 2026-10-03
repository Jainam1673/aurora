# AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-green.svg)](https://docs.python.org/3.14/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/pytest-123%2F123%20passed-brightgreen.svg)](tests/python/)
[![Tests: GoogleTest](https://img.shields.io/badge/GoogleTest-70%2F70%20passed-brightgreen.svg)](tests/cpp/)
[![Parity: Lockstep](https://img.shields.io/badge/Cross--Language%20Parity-%3C%2010%5E%7B--10%7D-success.svg)](tests/parity/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-One--Click%20Verified-blue.svg)](paper/REPRODUCIBILITY.md)

**AURORA** is a research-grade, zero-dependency model-based reinforcement learning (MBRL) platform engineered from mathematical first principles. Targeting **Python 3.14** and **C++23** as scientific peers, AURORA resolves the fundamental challenge of compounding model errors and out-of-distribution policy degradation through adaptive, uncertainty-calibrated imagination and risk-sensitive planning.

---

## Central Research Question & Motivation

> **How can an RL agent eliminate compounding model error and catastrophic hallucination by dynamically deciding how far to imagine, how much synthetic data to blend, and when to regularize policy updates?**

Standard Dyna-style MBRL algorithms (e.g., MBPO) rely on hand-tuned, fixed rollout length schedules ($H$) and static real-to-synthetic replay ratios ($\eta$). When an ensemble dynamics model ventures into out-of-distribution state spaces, imaginary rollouts generate delusional trajectories that destabilize actor-critic optimization.

AURORA addresses this via a mathematically grounded triad:
1. **Adaptive Horizon Scheduling ($H_{\text{adaptive}}$):** Truncates individual synthetic rollouts whenever epistemic ensemble variance exceeds dynamic thresholds calibrated against validation error ($\tau = \tau_{\text{base}} e^{-\kappa \mathcal{L}_{\text{val}}}$) or when cumulative discounted uncertainty exceeds a budget ($B_{\max}$).
2. **Dynamic Experience Blending ($\eta_t$):** Modulates the synthetic-to-real replay ratio using momentum-smoothed epistemic disagreement ($\eta_t = \eta_{\max}[1 - \min(1, \bar{u}/u_{\text{target}})]$), gracefully falling back to model-free replay when model uncertainty surges.
3. **Epistemic Risk-Sensitive Pessimistic Value Optimization ($\tilde{Q}$):** Penalizes policy optimization in uncertain state-action regions ($\tilde{Q}(s, a) = \min_j Q_j(s, a) - \beta_{\text{pess}} u_{\text{epi}}(s, a)$), providing a provable lower-bound on return under model discrepancy.

**Theoretical Guarantee:** We prove that AURORA achieves monotonic policy improvement under model error through a telescopic value expansion via the Simulation Lemma (see [Paper Appendix A](paper/main.tex)).

---

## Research Paper, Figures & Reproducibility Suite

AURORA includes a complete, publication-ready research paper package adhering to top-tier machine learning conference standards (NeurIPS / ICLR / ICML):

| Research Artifact | Location | Description |
|---|---|---|
| **LaTeX Manuscript** | [`paper/main.tex`](paper/main.tex) | 12-section camera-ready manuscript with Appendices A–F |
| **BibTeX Citations** | [`paper/references.bib`](paper/references.bib) | Complete bibliography of foundational MBRL and statistical papers |
| **Reproducibility Guide** | [`paper/REPRODUCIBILITY.md`](paper/REPRODUCIBILITY.md) | Official NeurIPS/ICLR reproducibility checklist, seeds, and environment audit |
| **One-Click Script** | [`scripts/reproduce_all.sh`](scripts/reproduce_all.sh) | Executable bash runner orchestrating the complete reproduction pipeline |
| **Python Runner** | [`scripts/reproduce_all.py`](scripts/reproduce_all.py) | Parametric benchmark runner, compiler, and SHA-256 verifier |
| **Ablation Table** | [`paper/table_ablations.tex`](paper/table_ablations.tex) | Systematic component ablation metrics with 95% Bootstrap CIs |
| **C++ Systems Table** | [`paper/table_systems.tex`](paper/table_systems.tex) | Native C++23 throughput and microsecond latency percentiles |
| **Cross-Language Table** | [`paper/table_cross_language.tex`](paper/table_cross_language.tex) | Empirical throughput parity and C++ speedups over Python 3.14 |
| **Performance Profiles** | [`paper/fig_performance_profiles.pdf`](paper/fig_performance_profiles.pdf) | Stratified score distribution CDFs across evaluation thresholds |
| **Systems Breakdown** | [`paper/fig_systems_breakdown.pdf`](paper/fig_systems_breakdown.pdf) | Amdahl execution time share breakdown (Amdahl bottleneck profile) |
| **Checksum Manifest** | [`paper/manifest_checksums.json`](paper/manifest_checksums.json) | Immutable cryptographic SHA-256 hashes of all reproduced artifacts |

---

## One-Click Reproduction

You can replicate all empirical results, microbenchmarks, figures, and LaTeX tables in under **45 seconds**:

```bash
# Clone the repository
git clone https://github.com/Jainam1673/aurora.git
cd aurora

# Quick full reproduction (C++ benchmarks, Python profiling, cross-language parity, figures & tables)
./scripts/reproduce_all.sh --quick
```

Or directly via `uv`:
```bash
uv run python scripts/reproduce_all.py --quick
```

To recompile all publication tables (`table_*.tex`) and vector figures (`fig_*.pdf`, `fig_*.png`) from existing raw manifests without re-executing benchmarks:
```bash
uv run python scripts/reproduce_all.py --skip-benchmarks
```

A consolidated reproduction audit is automatically generated and verified at [`results/reproduction_report.json`](results/reproduction_report.json).

---

## Empirical Benchmark & Ablation Results

Evaluations follow the rigorous statistical evaluation methodology proposed by Agarwal et al. (2021) using **Interquartile Mean (IQM)** with 25% outlier trimming, $B=2,000$ stratified bootstrap resamples (95% CIs), and Welch's $t$-tests:

### Continuous Control Component Ablation Study (Pendulum)

| Method / Variant | IQM $\uparrow$ | 95% Bootstrap CI | Mean $\pm$ Std | Median | $P(\text{Full} > \text{Var})$ | Welch $p$-value |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AURORA (Full Model)** | **-173.80** | **[-184.28, -164.71]** | **-174.96 $\pm$ 11.23** | **-174.19** | — | — |
| w/o Adaptive Horizon ($H=4$) | -193.36 | [-207.29, -180.89] | -194.22 $\pm$ 15.35 | -193.68 | 0.88 | 0.0435 |
| w/o Dynamic Blending ($\eta=0.5$) | -199.96 | [-214.34, -187.69] | -201.21 $\pm$ 16.71 | -200.75 | 0.94 | 0.0121 |
| w/o Pessimistic Penalty ($\beta=0$) | -216.51 | [-234.34, -200.98] | -218.06 $\pm$ 19.34 | -217.42 | 0.98 | 0.0021 |

*All results derive directly from raw experiment manifests in [`results/ablation/`](results/ablation/).*

---

## Dual-Peer Systems Architecture (Python 3.14 & C++23)

AURORA contains **zero external machine learning framework dependencies** (no PyTorch, TensorFlow, or JAX). All algorithms, tensors, automatic differentiation engines, neural modules, and replay buffers are implemented from scratch in both languages:

```
                            AURORA ARCHITECTURE
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         RESEARCH API (Python 3.14)                     │
 │  • Dynamic DAG Tape Autograd           • Modular Actor-Critic Agents   │
 │  • Probabilistic Dynamics Ensembles    • Agarwaal et al. Metrics       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Numerical Parity (< 10⁻¹⁰)
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │                         SYSTEMS ENGINE (C++23)                         │
 │  • Row-Major SIMD Contiguous Paths     • Cache-friendly i-k-j GEMMs    │
 │  • Heterogeneous Graph VJP Engine      • Low-Latency Active Planners   │
 └────────────────────────────────────────────────────────────────────────┘
```

### Native C++23 Engine Throughput Highlights
*Measured on Intel Core i5-8265U (AVX2 / FMA) via [`aurora_benchmark_throughput`](benchmarks/cpp/benchmark_throughput.cpp):*
- **Contiguous Allocation & Fill:** `1,069.70 M elements/s` (Mean latency: `467.42 μs`)
- **Contiguous Elementwise Add:** `486.57 M elements/s` (Mean latency: `1.03 ms`)
- **Contiguous ReLU Activation:** `287.52 M elements/s` (Mean latency: `1.74 ms`)
- **GEMM Compute Throughput (64x64x64):** `3.86 GFLOPs/s` (Mean latency: `135.87 μs`)
- **Dynamics Ensemble Forward (B=64, E=5):** `40,526.62 transitions/s`
- **Statistical IQM Throughput:** `67.92 M samples/s` (**31.8x speedup** over Python reference)
- **Bootstrap CI Throughput (R=1000):** `16.18 M resamples/s` (**6.9x speedup** over Python reference)

---

## Quickstart & Developer Guide

### Prerequisites
- **Python:** 3.14+ (managed via [`uv`](https://github.com/astral-sh/uv))
- **C++ Compiler:** GCC 14+ (`g++`) or Clang 18+ (`clang++`) supporting C++23
- **Build System:** CMake 3.28+ and Ninja
- **Hardware Support:** x86_64 CPU (AVX2/FMA recommended)

### 1. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/Jainam1673/aurora.git
cd aurora

# Install locked dependencies into local virtual environment
uv sync --extra dev

# Run complete Python test suite (123/123 tests)
uv run pytest

# Run code quality, formatting, and static typing checks
uv run ruff check .
uv run ruff format --check .
uv run mypy python evaluation benchmarks scripts paper
```

### 2. C++23 Native Engine Setup
```bash
# Configure debug and release presets with CMake & Ninja
cmake --preset debug
cmake --preset release

# Build complete C++ target tree
cmake --build --preset release

# Execute GoogleTest test suite (70/70 targets passing with zero warnings)
ctest --test-dir build/release --output-on-failure

# Execute native high-precision systems benchmark binary
./build/release/aurora_benchmark_throughput --json results/cpp_benchmark_results.json
```

### 3. Verify Cross-Language Numerical Parity
AURORA asserts that the C++23 engine and Python 3.14 reference match to within floating-point precision ($< 10^{-10}$ absolute error):
```bash
uv run pytest tests/parity
```

---

## Repository Hierarchy

```text
aurora/
├── README.md                           # Main researcher documentation and system summary
├── LICENSE                             # Apache 2.0 open-source license
├── CITATION.cff                        # Machine-readable research citation metadata
├── CHANGELOG.md                        # Milestone release log and detailed version history
├── STATUS.md                           # Living implementation status and test summaries
├── ROADMAP.md                          # Full milestone trajectory (M0 through M10)
├── ARCHITECTURE.md                     # Deep architectural specifications and data flow
├── DECISIONS.md                        # Architecture Decision Records (ADR-001 to ADR-014)
├── TODO.md                             # Comprehensive task and backlog register
│
├── paper/                              # Publication Paper Package (NeurIPS / ICLR Ready)
│   ├── main.tex                        # Full LaTeX academic manuscript with Appendices A-F
│   ├── references.bib                  # BibTeX bibliography database
│   ├── REPRODUCIBILITY.md              # Conference reproducibility checklist & guidelines
│   ├── generate_figures_and_tables.py  # Automated figure & table generation script
│   ├── manifest_checksums.json         # SHA-256 cryptographic hashes of all paper artifacts
│   ├── fig_performance_profiles.pdf    # Vector performance profile CDF
│   ├── fig_systems_breakdown.pdf       # Amdahl systems time breakdown
│   ├── table_ablations.tex             # Component ablation LaTeX table
│   ├── table_systems.tex               # C++23 microbenchmarks LaTeX table
│   └── table_cross_language.tex        # Cross-language speedup LaTeX table
│
├── scripts/                            # Scientific Reproducibility & Orchestration
│   ├── reproduce_all.sh                # Executable one-click shell wrapper
│   └── reproduce_all.py                # End-to-end Python reproducibility pipeline
│
├── python/aurora/                      # Python 3.14 Reference Research Package
│   ├── tensor.py                       # Pure tensor engine with strided views & broadcasting
│   ├── autograd.py                     # Reverse-mode dynamic computation graph autograd
│   ├── gradcheck.py                    # Finite-difference gradient numerical checker
│   ├── checkpoint.py                   # Zero-dependency IEEE-754 JSON state serialization
│   ├── nn/                             # Modules (Linear, LayerNorm, RMSNorm, Dropout, MLP)
│   ├── optim/                          # Optimizers (SGD, Adam, AdamW, Learning Rate Schedulers)
│   ├── attention/                      # Attention mechanisms (Causal Scaled Dot-Product, RoPE)
│   ├── transformer/                    # Pre-Norm TransformerDecoder autoregressive models
│   ├── rl/                             # Policy distributions (TanhNormal), GAE RolloutBuffer, ReplayBuffer
│   ├── world_model/                    # Probabilistic Dynamics Ensembles, RSSM, ImaginationEngine
│   ├── reproductions/                  # Baseline implementations (MBPO, PETS, Dreamer, SAC, PPO)
│   └── algorithm/                      # Core AURORA Agent, Adaptive Horizons, Dynamic Blending
│
├── cpp/                                # C++23 Native High-Throughput Systems Core
│   ├── include/aurora/                 # Public C++23 header interfaces
│   │   ├── tensor.hpp                  # Tensor memory layouts, views, and SIMD fast paths
│   │   ├── autograd.hpp                # Tape-based computational graph DAG engine
│   │   ├── nn.hpp                      # Modular neural primitives & parameter registries
│   │   ├── optim.hpp                   # Vectorized AdamW, SGD, and learning rate schedulers
│   │   ├── attention.hpp               # C++23 causal scaled dot-product attention
│   │   ├── transformer.hpp             # Transformer decoders with rotary embeddings
│   │   ├── rl.hpp                      # Squashed distributions and circular replay buffers
│   │   ├── world_model.hpp             # High-throughput dynamics ensembles & rollouts
│   │   ├── reproductions.hpp           # Native MBPO hybrid buffers and planners
│   │   ├── aurora_algorithm.hpp        # Native AURORA adaptive horizon & blending engines
│   │   └── statistical_evaluation.hpp  # High-speed IQM & bootstrap confidence intervals
│   └── src/                            # Implementation sources & benchmark binaries
│
├── tests/                              # Comprehensive Verification Suites
│   ├── python/                         # Pytest unit tests (123 tests total)
│   ├── cpp/                            # GoogleTest suites across GCC and Clang (70 targets)
│   └── parity/                         # Cross-language numerical lockstep verification (< 10⁻¹⁰)
│
├── benchmarks/                         # Systems & Algorithm Performance Suites
│   ├── cpp/benchmark_throughput.cpp    # High-precision C++23 microbenchmark engine
│   ├── profile_aurora.py               # Monotonic profiler generating .prof flamegraph traces
│   └── benchmark_cross_language.py     # Python vs. C++ parity & speedup benchmark
│
├── evaluation/                         # Statistical Evaluation Infrastructure (Agarwal et al.)
│   ├── metrics.py                      # IQM, bootstrap confidence intervals, statistical summaries
│   ├── profiles.py                     # Performance profiles and probability of improvement
│   ├── significance.py                 # Welch's t-test and Mann-Whitney U test
│   ├── manifest.py                     # Immutable JSON experiment manifest generator
│   └── plotting.py                     # Performance curve and score distribution plotter
│
├── experiments/                        # Experiment Runners & Multi-Seed Ablation Studies
│   ├── runner.py                       # Multi-seed deterministic trial runner
│   └── ablation_study.py               # Systematic component ablation runner
│
├── configs/                            # Declarative JSON Experiment Configurations
└── results/                            # Raw JSON Manifests, Traces, and Benchmark Outputs
```

---

## Roadmap & Milestone Status

All 11 milestones of the AURORA project have been executed to 100% completion:

- [x] **M0: Repository Bootstrap** — Toolchains, C++23 presets, Python 3.14 `uv`, governance blueprints.
- [x] **M1: Numerical Core & Autograd** — Dual-peer `Tensor`, reverse-mode autograd, gradcheck, $< 10^{-10}$ parity.
- [x] **M2: NN Primitives & Optimizers** — `Linear`, `LayerNorm`, `RMSNorm`, `AdamW`, zero-dependency JSON serialization.
- [x] **M3: Transformer Sequence Engine** — Rotary Position Embeddings (RoPE), causal multi-head attention, decoder blocks.
- [x] **M4: RL Primitives & Physics** — `TanhNormal`, CartPole & Pendulum physics, GAE buffers, PPO & SAC agents.
- [x] **M5: Deep Dynamics & Uncertainty** — Gaussian NLL dynamics ensembles, epistemic disagreement, RSSM cell.
- [x] **M6: MBPO & Dyna Reproductions** — Vectorized hybrid replay sampling, truncated rollouts, cross-language parity.
- [x] **M7: The AURORA Algorithm** — Adaptive horizon truncation, dynamic experience blending, pessimistic value optimization.
- [x] **M8: Scientific Benchmarking** — Agarwal et al. IQM, 95% bootstrap CIs, Welch $t$-tests, immutable run manifests.
- [x] **M9: Systems Performance & Native Scaling** — SIMD contiguous fast-paths, 4.38 GFLOPs/s GEMM, 8.2x rollout speedup.
- [x] **M10: Research Paper & Reproducibility** — 12-section LaTeX paper, Appendices A–F, automated figures & tables, one-click runner.

---

## Citation

If you use AURORA in your academic research, please cite our manuscript using the following BibTeX entry:

```bibtex
@article{aurora2026,
  title     = {AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents},
  author    = {AURORA Research Lab},
  journal   = {arXiv preprint},
  year      = {2026},
  url       = {https://github.com/Jainam1673/aurora}
}
```

---

## License

AURORA is open-source software licensed under the [Apache License, Version 2.0](LICENSE).
