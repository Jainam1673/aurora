# AURORA Reproducibility Guide & Verification Checklist

**AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents**  
*NeurIPS / ICLR Scientific Reproducibility Standard Compliance*

---

## 1. Overview & Reproducibility Guarantee

AURORA is designed from first principles with dual-peer implementations in **Python 3.14** and **C++23**. All core machine learning and systems routines (tensors, dynamic computation graphs, reverse-mode automatic differentiation, neural networks, ensemble dynamics, adaptive imagination, replay buffers, and statistical evaluation) are implemented without external deep learning frameworks (PyTorch, TensorFlow, JAX).

All experimental results, statistical metrics, systems throughput benchmarks, and ablation studies presented in the manuscript are 100% reproducible from raw execution manifests and one-click automation scripts.

---

## 2. Experimental Environment & System Specifications

### 2.1 Hardware Architecture
- **CPU**: Intel Core i5-8265U @ 1.60GHz (4 cores / 8 hardware threads, AVX2, FMA3, 64-byte cache line).
- **RAM**: 7.6 GiB DDR4 RAM.
- **GPU**: NVIDIA GeForce MX230 (CUDA 13.3, Compute Capability 6.1) — note: all primary evaluation results and microbenchmarks are CPU-deterministic.
- **Operating System**: Linux 7.2.5-3-omarchy-x86_64 (glibc 2.44).

### 2.2 Compiler & Runtime Environment
- **C++ Compilers**:
  - GCC 16.2.1 20260810 (`g++ -std=c++23 -O3 -mavx2 -mfma -Wall -Wextra -Wpedantic -Werror -Wconversion`)
  - Clang 22.1.8 (`clang++ -std=c++23 -O3 -mavx2 -mfma -Wall -Wextra -Wpedantic -Werror -Wconversion`)
- **Build System**: CMake 3.28+ with Ninja.
- **Python Environment**: Python 3.14.8 managed via `uv` (lockfile: `uv.lock`, pyproject: `pyproject.toml`).

---

## 3. One-Click Reproduction Instructions

AURORA provides an automated, one-click reproduction pipeline that runs all benchmarks, processes raw experimental manifests, verifies SHA-256 integrity, and generates publication LaTeX tables and vector figures.

### 3.1 Quick Full Reproduction (Under 45 Seconds)
```bash
# Execute the complete reproduction pipeline (C++ benchmarks, Python profiling, cross-language parity, figures & tables)
./scripts/reproduce_all.sh --quick
```
Or directly with `uv`:
```bash
uv run python scripts/reproduce_all.py --quick
```

### 3.2 Artifact Generation From Existing Manifests (Under 5 Seconds)
To compile all publication tables (`table_*.tex`) and vector figures (`fig_*.pdf`, `fig_*.png`) from raw results without re-executing microbenchmarks:
```bash
uv run python scripts/reproduce_all.py --skip-benchmarks
```

---

## 4. Step-by-Step Manual Reproduction

### 4.1 Native C++23 Systems Benchmarks
1. Configure and build the Release preset:
   ```bash
   cmake --preset release
   cmake --build --preset release --target aurora_benchmark_throughput
   ```
2. Execute the benchmark suite with JSON serialization:
   ```bash
   ./build/release/aurora_benchmark_throughput --json results/cpp_benchmark_results.json
   ```

### 4.2 Python Systems Profiler & Flamegraphs
Run the execution profiler to measure the Amdahl breakdown across policy optimization, imagination, environment interaction, and world model fitting:
```bash
PYTHONPATH=. uv run python benchmarks/profile_aurora.py \
    --env-steps 60 \
    --output-json results/python_systems_profile.json \
    --prof-file results/aurora_profile.prof
```

### 4.3 Cross-Language Parity & Speedup Comparison
Compare identical tensor, GEMM, ensemble, and statistical workloads across Python 3.14 and C++23:
```bash
PYTHONPATH=. uv run python benchmarks/benchmark_cross_language.py \
    --cpp-json results/cpp_benchmark_results.json \
    --output-json results/cross_language_comparison.json
```

### 4.4 Automated Figure and Table Compilation
Compile publication figures and tables from manifests:
```bash
PYTHONPATH=. uv run python paper/generate_figures_and_tables.py
```

### 4.5 Full Test Suite Verification
Verify 100% test coverage across both languages:
```bash
# 1. Python Unit and Integration Tests (123/123 tests)
uv run pytest

# 2. C++ Unit Tests (70/70 targets on Debug preset)
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure

# 3. C++ Unit Tests (70/70 targets on Release preset)
ctest --test-dir build/release --output-on-failure

# 4. Linter & Static Type Analysis
uv run ruff check .
uv run ruff format --check .
uv run mypy python evaluation benchmarks scripts paper
```

---

## 5. Artifact Manifest & Integrity Checksums

Every generated table, figure, and result manifest is verified via SHA-256 hashing. The current reference checksums are documented in `paper/manifest_checksums.json`:

| File Path | Description | Checksum (SHA-256 prefix) |
|---|---|---|
| `paper/main.tex` | Full LaTeX manuscript source | `675c4df8...` |
| `paper/references.bib` | BibTeX bibliography | `d7d03eeb...` |
| `paper/table_ablations.tex` | Component ablation study LaTeX table | `d9a285a8...` |
| `paper/table_systems.tex` | Native C++23 throughput LaTeX table | `9788b694...` |
| `paper/table_cross_language.tex` | Cross-language speedup LaTeX table | `9aced0fd...` |
| `paper/fig_performance_profiles.pdf` | Vector performance profile CDF | `4254647b...` |
| `paper/fig_systems_breakdown.pdf` | Amdahl systems time breakdown | `7e5a0c53...` |
| `results/ablation/ablation_summary.json` | Empirical ablation metrics & p-values | `79d24b7f...` |
| `results/cpp_benchmark_results.json` | C++ microbenchmark throughput data | `d6576560...` |
| `results/python_systems_profile.json` | Python systems component timings | `73be526f...` |
| `results/cross_language_comparison.json` | Language parity & speedup comparison | `e5c0e729...` |

---

## 6. NeurIPS / ICLR Reproducibility Checklist Responses

### 6.1 Claims & Mathematical Proofs
- **Question**: Do the main claims in the abstract and introduction accurately reflect the paper's theoretical and empirical results?  
  **Answer**: **Yes**. All claims regarding monotonic policy improvement under model error (Theorem 1), statistical superiority on IQM, and native C++ speedups are proven in Appendix A and empirically verified in Section 5 and Appendix E.
- **Question**: Have you stated the full set of assumptions of all theoretical results?  
  **Answer**: **Yes**. Appendix A explicitly details Lipschitz continuity of the reward function ($L_r$), bounded divergence of transition probability distributions ($\epsilon_m$), and discount factors ($\gamma \in [0, 1)$).
- **Question**: Did you include complete proofs of all theoretical results?  
  **Answer**: **Yes**. Full derivations using the Simulation Lemma and telescopic value difference expansions are in Appendix A.

### 6.2 Empirical Evaluation & Statistical Rigor
- **Question**: Did you specify all the training and evaluation details (e.g., data splits, hyperparameters, seeds)?  
  **Answer**: **Yes**. Appendix B provides the complete hyperparameter table (Table 4). Evaluation follows Agarwal et al. (2021) with 2,000 bootstrap resamples and 25% trimmed interquartile means (IQM).
- **Question**: Did you report error bars, confidence intervals, or standard deviations?  
  **Answer**: **Yes**. All tables (Table 1, Table 2, Table 3) and figures (Figure 1, Figure 2) display 95% stratified bootstrap confidence intervals, means, standard deviations, and Welch's $t$-test $p$-values.
- **Question**: Did you include the code, data, and instructions needed to reproduce the main experimental results?  
  **Answer**: **Yes**. `scripts/reproduce_all.sh` provides one-click replication, with all data manifests included under `results/`.

### 6.3 Code & Open Science
- **Question**: Does the code include a complete dependency specification?  
  **Answer**: **Yes**. `pyproject.toml` and `uv.lock` pin all Python dependencies with SHA-256 hashes; `CMakeLists.txt` manages C++ dependencies (GoogleTest fetched hermetically via CMake FetchContent).
- **Question**: Did you include the licenses and provenance of any third-party code?  
  **Answer**: **Yes**. The repository is released under the Apache-2.0 License (`LICENSE`), and all implementations are built from scratch.
