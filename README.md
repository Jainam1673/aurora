# AURORA
### Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-green.svg)](https://docs.python.org/3.14/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![CMake](https://img.shields.io/badge/build-CMake%203.28%2B-blue)](https://cmake.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/python/)
[![Tests: GoogleTest](https://img.shields.io/badge/GoogleTest-passing-brightgreen.svg)](tests/cpp/)
[![Parity: Numerical](https://img.shields.io/badge/Cross--Language%20Parity-%3C%201e--10-success.svg)](tests/parity/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-one--click%20pipeline-blue.svg)](paper/REPRODUCIBILITY.md)

AURORA is a research-grade model-based reinforcement learning (MBRL) platform engineered from first principles without external deep learning frameworks. Implemented as dual scientific peers in **Python 3.14** and **C++23**, the system investigates uncertainty-aware imagination, adaptive rollout depth, and dynamic replay blending to mitigate compounding errors in learned dynamics models.

---

## Overview

Model-based reinforcement learning algorithms often improve sample efficiency by training policies on synthetic transitions imagined by a learned world model. In practice, however, learned dynamics models degrade when predicting out-of-distribution transitions, causing imaginary trajectories to compound errors and corrupt policy optimization.

AURORA investigates whether an agent can dynamically regulate its reliance on a learned world model by quantifying epistemic uncertainty across an ensemble of probabilistic dynamics models. The platform couples:
1. **Adaptive Horizon Scheduling**, which dynamically truncates individual rollout trajectories when model uncertainty exceeds validation-calibrated thresholds.
2. **Dynamic Experience Blending**, which modulates the ratio of synthetic to real environment transitions in policy updates based on recent model disagreement.
3. **Pessimistic Value Regularization**, which discounts value targets in state-action regions where dynamics variance is high.

The codebase includes complete, independent implementations of tensors, reverse-mode automatic differentiation, neural networks, policy optimizers, probabilistic dynamics ensembles, and statistical evaluation tools in both Python and C++23.

---

## Research Question

> **Can an RL agent improve sample efficiency and policy stability by dynamically controlling its imagination horizon, synthetic/real experience mixture, and risk sensitivity using epistemic uncertainty from an ensemble world model?**

Standard Dyna-style algorithms (such as MBPO) rely on predetermined, fixed rollout schedules ($H$) and static real-to-synthetic replay fractions ($\eta$). When an ensemble dynamics model extrapolates into unfamiliar state spaces, static schedules continue generating uncalibrated synthetic data. AURORA investigates whether conditioning rollout length and replay mixture on instantaneous and cumulative model uncertainty provides superior sample efficiency and robustness.

---

## What AURORA Does

```text
Real Environment
       │
       ▼
 Collect Real Transitions (D_env)
       │
       ▼
 Train Ensemble World Model (f_θ)
       │
       ▼
 Quantify Epistemic Disagreement: u_epi(s, a)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 Horizon Truncation               Dynamic Blending Ratio
 u_epi > τ_t  ==> Stop branch     η_t = f(mean u_epi)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
            Imagined Replay (D_model)
                        │
                        ▼
            Actor-Critic Optimization
      (Policy Updates with Pessimistic Value)
```

---

## Why This Problem Matters

Model-based RL algorithms promise orders-of-magnitude sample efficiency gains over model-free counterparts. However, their practical deployment has historically been impeded by:
- **Compounding Simulation Error:** Prediction errors multiply over multi-step rollouts, creating catastrophic hallucinations.
- **Objective Inconsistency:** Policies exploit model inaccuracies ("model exploitation"), achieving high synthetic reward while failing in the true environment.
- **Fixed Hyperparameter Sensitivity:** The optimal imagination horizon $H$ and replay ratio $\eta$ vary substantially across environments and training phases, requiring expensive manual tuning.

Addressing these issues through dynamic uncertainty calibration is essential for reliable, sample-efficient reinforcement learning.

---

## Key Contributions

### Algorithmic Mechanisms
1. **Adaptive Horizon Scheduling:** Individual imagination trajectories are terminated when instantaneous epistemic disagreement exceeds a dynamic validation-decayed threshold $\tau_t$ or when the cumulative discounted uncertainty budget is exhausted ($B_{\max}$).
2. **Dynamic Experience Blending:** The synthetic-to-real replay sampling ratio $\eta_t$ decays gracefully toward zero when recent rollout uncertainty surges, falling back to model-free learning when the model is untrusted.
3. **Pessimistic Value Regularization:** Policy evaluation incorporates an epistemic variance penalty $\tilde{Q}(s, a) = \min_j Q_j(s, a) - \beta u(s, a)$ to discourage optimistic extrapolation.

### Infrastructure & Systems
4. **Dual-Peer C++23 / Python 3.14 Architecture:** Complete from-scratch mathematical cores in both languages with zero third-party ML framework dependencies (no PyTorch, TensorFlow, or JAX).
5. **Rigorous Evaluation Infrastructure:** Implementation of the Agarwal et al. (2021) statistical evaluation protocol, including Interquartile Mean (IQM), stratified bootstrap confidence intervals ($R = 2000$), probability of improvement, and automated JSON run manifests.
6. **One-Click Reproducibility Pipeline:** Standalone automation scripts verifying hardware environment, building native binaries, compiling figures/tables, and checking SHA-256 artifact checksums.

---

## Architecture

```text
                                  AURORA
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       Python 3.14 Reference                   C++23 Systems Core
      (Research & Prototyping)             (Throughput & Native Scale)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                             Shared Semantics
                     (Numerical Parity < 1e-10)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  Numerical Core               World Model               RL & Planning
• Strided Tensor Core        • Deep Dynamics Ensemble  • Squashed Gaussian Actor
• DAG Reverse Autograd       • Epistemic Variance      • Twin Critic
• AdamW & Schedulers         • Imagination Engine      • Circular Replay Buffers
        │                            │                            │
        └────────────────────────────┼────────────────────────────┘
                                     │
                                AURORA Agent
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
         Adaptive Horizon Scheduler      Dynamic Blending Controller
        (Validation-Decayed Threshold)   (Uncertainty-Modulated Ratio)
```

---

## Algorithm Formulation

### 1. Dynamics Ensemble & Uncertainty Decomposition

Given continuous state $s$ and action $a$, an ensemble of $E$ probabilistic dynamics models predicts next-state transitions and scalar rewards:

$$
\hat{s}_{t+1} = s_t + \mu_i(s_t, a_t), \quad \hat{r}_t = r_i(s_t, a_t)
$$

Total predictive variance across ensemble members decomposes into aleatoric (data noise) and epistemic (model disagreement) components:

$$
\sigma_{\text{tot}}^2(s, a) = \sigma_{\text{ale}}^2(s, a) + \sigma_{\text{epi}}^2(s, a)
$$

where aleatoric variance is the mean predicted variance across ensemble members:

$$
\sigma_{\text{ale}}^2(s, a) = \frac{1}{E} \sum_{i=1}^E \sigma_i^2(s, a)
$$

and epistemic variance is the disagreement across ensemble predictions:

$$
\sigma_{\text{epi}}^2(s, a) = \frac{1}{E} \sum_{i=1}^E \left(\mu_i(s, a) - \bar{\mu}(s, a)\right)^2, \quad \text{with} \quad \bar{\mu}(s, a) = \frac{1}{E}\sum_{i=1}^E \mu_i(s, a)
$$

The scalar epistemic disagreement $u_{\text{epi}}(s, a)$ used for control decisions is evaluated as the maximum dimension-wise variance across state dimensions $k \in \{1, \dots, d_s\}$:

$$
u_{\text{epi}}(s, a) = \max_{k \in \{1, \dots, d_s\}} \frac{1}{E} \sum_{i=1}^E \left(\mu_k^{(i)}(s, a) - \bar{\mu}_k(s, a)\right)^2
$$

### 2. Adaptive Horizon Truncation

A rollout trajectory branch terminates at step $h$ when either instantaneous epistemic uncertainty exceeds the dynamic threshold or the cumulative discounted uncertainty budget is exhausted:

$$
u_{\text{epi}}(s_h, a_h) > \tau_t \quad \text{or} \quad \sum_{k=0}^h \gamma^k u_{\text{epi}}(s_k, a_k) > B_{\max}
$$

where the threshold $\tau_t$ adapts dynamically to model validation loss $\mathcal{L}_{\text{val}}$:

$$
\tau_t = \tau_{\text{base}} \exp\left(-\kappa \cdot \mathcal{L}_{\text{val}}\right)
$$

### 3. Dynamic Experience Blending

The synthetic replay fraction $\eta_t \in [0, \eta_{\max}]$ is dynamically computed from recent mean imagination uncertainty $\bar{u}_t$ and smoothed with momentum parameter $\rho \in [0, 1)$:

$$
\eta_t^{\text{raw}} = \eta_{\max} \left[1 - \min\left(1, \frac{\bar{u}_t}{u_{\text{target}}}\right)\right]
$$

$$
\eta_t = \rho \eta_{t-1} + (1 - \rho)\eta_t^{\text{raw}}
$$

where $\bar{u}_t$ is the mean epistemic uncertainty across all steps and branches of the recent imagination batch:

$$
\bar{u}_t = \frac{1}{B} \sum_{b=1}^B \frac{1}{H_b} \sum_{h=0}^{H_b-1} u_{\text{epi}}(s_h^{(b)}, a_h^{(b)})
$$

### 4. Pessimistic Value Formulation

To prevent policy exploitation of optimistic model errors, policy optimization evaluates actions against a penalized lower-bound critic:

$$
\tilde{Q}(s, a) = \min_{j \in \{1, 2\}} Q_{\psi_j}(s, a) - \beta_{\text{pess}} \cdot u_{\text{epi}}(s, a)
$$

with the actor objective:

$$
\mathcal{J}(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi} \left[ \tilde{Q}(s, a) - \alpha \log \pi_\phi(a \mid s) \right]
$$

---

## Theoretical Analysis

The accompanying manuscript (`paper/main.tex`, Section 5 and Appendix A) presents a model-error analysis based on the Simulation Lemma. It formalizes conditions under which bounding accumulated rollout discrepancy via dynamic truncation and pessimistic regularizers supports monotonic policy improvement bounds under bounded model error:

$$
D_{\text{TV}}(\mathcal{P}, \hat{\mathcal{P}}) \le C_u \cdot u_{\text{epi}}(s, a)
$$

> [!NOTE]
> The theoretical guarantee relies on the assumption that ensemble epistemic variance upper-bounds total variation divergence between true and learned dynamics. In finite deep neural networks, this represents an empirical modeling hypothesis rather than an unconditional mathematical fact.

---

## Dual-Peer Implementation

AURORA implements both a Python 3.14 reference library and a C++23 native systems core:

| Subsystem | Python 3.14 Reference | C++23 Systems Core | Verified Numerical Parity |
|---|:---:|:---:|:---:|
| **Tensor Abstraction** | Strided NumPy backend | Contiguous heap memory, views, C-strides | < 1e-10 |
| **Automatic Differentiation** | Dynamic tape DAG VJP | Polymorphic DAG graph nodes | < 1e-10 |
| **Neural Primitives** | `Linear`, `LayerNorm`, `RMSNorm` | `Linear`, `LayerNorm`, `RMSNorm` | < 1e-10 |
| **Optimizers** | `SGD`, `Adam`, `AdamW`, Schedulers | Vectorized `SGD`, `AdamW`, Schedulers | < 1e-10 |
| **Sequence Architecture** | Scaled Dot-Product, RoPE, Decoder | C++23 Attention, RoPE, Decoder | < 1e-10 |
| **Reinforcement Learning** | `Normal`, `TanhNormal`, GAE buffer | `Normal`, `TanhNormal`, GAE buffer | < 1e-10 |
| **Dynamics Ensemble** | Gaussian NLL, TS1 Trajectory Sampling | Parallel ensemble inference | < 1e-10 |
| **AURORA Controllers** | Adaptive Horizon & Blending | `AdaptiveHorizonScheduler`, `DynamicBlending` | < 1e-10 |
| **Statistical Evaluation** | IQM, Bootstrap CIs, Profiles | High-speed IQM & Bootstrap CI kernels | < 1e-10 |

---

## Verified Benchmarks & Results

All empirical numbers below originate from actual generated manifests and benchmark logs stored in `results/`.

### 1. Component Ablation Study (Continuous Control — Pendulum)
*Evaluated over multiple random seeds (30 evaluation episodes per condition, 300 environment steps) from `results/ablation/ablation_summary.json`:*

| Method / Variant | IQM Return (↑) | 95% Bootstrap CI | Mean ± Std | P(Full > Variant) | Welch p-value |
|:---|---:|:---:|:---:|:---:|:---:|
| **AURORA (Full)** | **-1446.27** | [-1593.41, -1298.99] | -1452.09 ± 321.20 | — | — |
| w/o Adaptive Horizon (H = 4) | -1462.02 | [-1586.36, -1337.21] | -1460.44 ± 282.23 | 0.50 | 0.9151 |
| w/o Dynamic Blending (η = 0.5) | -1391.11 | [-1527.22, -1266.92] | -1393.35 ± 286.89 | 0.44 | 0.4581 |
| w/o Pessimistic Penalty (β = 0) | -1446.27 | [-1593.41, -1298.99] | -1452.09 ± 321.20 | 0.50 | 1.0000 |

> [!IMPORTANT]
> **Audit Finding on Ablation Results:** As documented in our scientific audit (`research_audit/FINAL_AUDIT.md`), at short training horizons (300 steps), differences between variants are not statistically significant ($p > 0.45$). In particular, `No Dynamic Blending` with fixed $\eta = 0.5$ performed competitively with Full AURORA, and `No Pessimism` scored identically due to a gradient detachment in the actor loss. We explicitly document this rather than reporting synthetic numbers.

### 2. C++23 Native Systems Throughput
*Measured on an Intel Core i5-8265U CPU (AVX2/FMA enabled) with GCC 16.2.1 (`-O3 -march=native`) via `benchmarks/cpp/benchmark_throughput.cpp`:*

| Subsystem | Workload Description | Mean Latency | Median (p50) | Throughput |
|---|---|---:|---:|---:|
| **Tensor Memory** | Contiguous Allocation & Fill | 437.45 µs | 369.02 µs | 1.14 G elements/s |
| **Tensor Compute** | Contiguous Elementwise Add | 1447.76 µs | 1181.02 µs | 345.36 M elements/s |
| **Tensor Activation** | Contiguous ReLU Activation | 1126.00 µs | 943.22 µs | 444.05 M elements/s |
| **GEMM Compute** | Matmul 64 × 64 × 64 | 161.44 µs | 157.34 µs | 3.25 GFLOP/s |
| **GEMM Compute** | Matmul 128 × 128 × 128 | 1534.16 µs | 1331.47 µs | 2.73 GFLOP/s |
| **Dynamics Ensemble** | Forward Ensemble (B = 64, E = 5) | 8383.29 µs | 8220.90 µs | 38,171 transitions/s |
| **Statistical Kernel** | IQM Evaluation (N = 100) | 0.80 µs | 0.79 µs | 125.67 M samples/s |
| **Statistical Kernel** | Bootstrap CI (N = 100, R = 1000) | 4255.60 µs | 4219.30 µs | 23.50 M resamples/s |

### 3. Cross-Language Parity & Speedup Comparison
*Comparing Python 3.14 reference vs. C++23 native implementation (`results/cross_language_comparison.json`):*

| Computational Workload | Unit | Python Mean (µs) | C++23 Mean (µs) | C++ Speedup |
|---|---|---:|---:|---:|
| **Statistical IQM (N = 100)** | samples/s | 32.75 | 0.80 | **41.16×** |
| **Bootstrap CI (N = 100, R = 1000)** | resamples/s | 57,671.15 | 4,255.60 | **13.55×** |
| **Contiguous Allocation & Fill** | elements/s | 316.61 | 437.45 | 0.72× |
| **GEMM Matmul (64 × 64 × 64)** | FLOP/s | 48.15 | 161.44 | 0.30× (vs BLAS) |

*(Note: Python linear algebra calls out to optimized NumPy C/OpenBLAS routines, while custom statistical and rollout loops benefit heavily from C++23 native compilation.)*

---

## One-Click Reproducibility

AURORA provides an automated replication pipeline that checks the host environment, executes benchmarks, compiles publication tables and figures, and verifies SHA-256 artifact hashes.

### Quick Validation (< 45 seconds)
Executes C++ native benchmarks, Python profiler smoke run, cross-language comparison, and artifact compilation:
```bash
# Clone the repository
git clone https://github.com/Jainam1673/aurora.git
cd aurora

# Sync dependencies using uv
uv sync --extra dev

# Run automated one-click reproduction pipeline
./scripts/reproduce_all.sh --quick
```
Or directly with Python:
```bash
uv run python scripts/reproduce_all.py --quick
```

### Artifact Compilation From Existing Manifests (< 5 seconds)
Recompiles all publication LaTeX tables (`paper/table_*.tex`) and vector figures (`paper/fig_*.pdf`, `fig_*.png`) from raw results without re-running compute benchmarks:
```bash
uv run python scripts/reproduce_all.py --skip-benchmarks
```

A complete reproduction summary is exported to [`results/reproduction_report.json`](results/reproduction_report.json), and output hashes are verified against [`paper/manifest_checksums.json`](paper/manifest_checksums.json).

---

## Quick Start & Development

### Prerequisites
- **Python:** 3.14+ managed via [`uv`](https://github.com/astral-sh/uv)
- **C++ Compiler:** GCC with C++23 support (`g++ >= 14`) or Clang with C++23 support (`clang++ >= 18`)
- **Build System:** CMake 3.28+ and Ninja

### Python Development
```bash
# Install development environment
uv sync --extra dev

# Run unit and integration tests (123 tests)
uv run pytest

# Run numerical parity suite
uv run pytest tests/parity

# Run static type checking and formatting
uv run ruff check .
uv run ruff format --check .
uv run mypy python evaluation benchmarks scripts paper
```

### C++23 Native Build & Tests
```bash
# Configure debug and release builds with CMake Presets
cmake --preset debug
cmake --preset release

# Build release targets
cmake --build --preset release

# Run GoogleTest suite (70 test targets)
ctest --test-dir build/release --output-on-failure

# Execute the native systems benchmark binary
./build/release/aurora_benchmark_throughput --json results/cpp_benchmark_results.json
```

---

## Repository Structure

```text
aurora/
├── python/aurora/            # Python 3.14 reference package
│   ├── tensor.py             # Pure tensor engine with views & broadcasting
│   ├── autograd.py           # Reverse-mode dynamic computation graph
│   ├── nn/                   # Modules (Linear, Normalization, Dropout)
│   ├── optim/                # Optimizers (AdamW, SGD, Schedulers)
│   ├── attention/            # Causal attention & Rotary Embeddings
│   ├── transformer/          # Pre-Norm TransformerDecoder models
│   ├── rl/                   # Policy distributions, GAE rollout buffer
│   ├── world_model/          # Ensemble dynamics & imagination engine
│   └── algorithm/            # AURORA Agent, Adaptive Horizon & Blending
│
├── cpp/                      # C++23 native systems core
│   ├── include/aurora/       # Public C++23 headers
│   └── src/                  # Implementations and benchmark binaries
│
├── tests/                    # Comprehensive test infrastructure
│   ├── python/               # Pytest unit and integration tests (123 tests)
│   ├── cpp/                  # GoogleTest C++ test targets (70 tests)
│   └── parity/               # Cross-language numerical lockstep tests (< 10⁻¹⁰)
│
├── benchmarks/               # Systems throughput and cross-language harnesses
├── evaluation/               # Statistical evaluation protocols (Agarwal et al. 2021)
├── experiments/              # Multi-seed runners and ablation execution scripts
├── configs/                  # Declarative experiment JSON configurations
├── results/                  # Raw JSON manifests, traces, and benchmark logs
├── paper/                    # Publication manuscript (LaTeX, BibTeX, figures, tables)
├── research_audit/           # Formal scientific audit and claim-evidence matrix
└── scripts/                  # One-click reproduction automation scripts
```

For deeper architectural specifications and decision rationale, see [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`DECISIONS.md`](DECISIONS.md).

---

## Known Limitations & Failure Modes

In the spirit of transparent open science, we explicitly document the current empirical and algorithmic boundaries of AURORA:

1. **Short-Horizon Statistical Significance:** At short training horizons (300 environment steps on Inverted Pendulum), variance between seeds remains high, and differences between full AURORA and its ablations do not yet achieve statistical significance ($p > 0.05$).
2. **Actor Pessimism Gradient Flow:** In the current implementation, the pessimism penalty `\beta_{\text{pess}} \cdot u_{\text{epi}}(s, a)` is evaluated as a detached constant in the actor loss. Connecting autograd gradients or applying pessimism directly to reward targets (as in MOPO) is required for `\beta_{\text{pess}}` to influence policy improvement dynamically.
3. **Task Scope:** Systematic benchmarking in the current release is focused on continuous control dynamics (Pendulum). Evaluating across visual latent states (e.g. via RSSM) and contact-rich environments represents active future research.
4. **Computational Cost of Ensembles:** Training and evaluating an ensemble of $E = 5$ deep neural networks scales computational cost linearly compared to single-model methods.

For detailed audit findings, see [`research_audit/FINAL_AUDIT.md`](research_audit/FINAL_AUDIT.md).

---

## Research Paper

The formal academic manuscript and supplementary material are available in the [`paper/`](paper/) directory:
- [`paper/main.tex`](paper/main.tex): Full LaTeX source including Sections 1–12 and Appendices A–F.
- [`paper/references.bib`](paper/references.bib): BibTeX database of related literature.
- [`paper/REPRODUCIBILITY.md`](paper/REPRODUCIBILITY.md): Official conference reproducibility checklist and environment audit.

---

## Citation

If you use AURORA in your research or reference its from-scratch systems implementation, please cite:

```bibtex
@article{aurora2026,
  title     = {AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents},
  author    = {AURORA Research Lab},
  journal   = {Preprint},
  year      = {2026},
  url       = {https://github.com/Jainam1673/aurora}
}
```

---

## License

AURORA is open-source software licensed under the [Apache License, Version 2.0](LICENSE).
