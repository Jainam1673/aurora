# Reproduction Pipeline Audit & Verification Status

**Auditor:** Principal Research Scientist, Skeptical ML Reviewer, Reproducibility Auditor  
**Date:** 2026-10-03  
**Commit Audited:** `cfdb849` / `170951b`

---

## 1. Reproduction Pipeline Audit

We executed the official reproduction scripts:
- `scripts/reproduce_all.sh --quick`
- `scripts/reproduce_all.py --quick`
- `scripts/reproduce_all.py --skip-benchmarks`

### Step-by-Step Stage Verification

| Pipeline Stage | Executed Command | Runtime | Exit Code | Artifacts Produced | Verification Result |
|---|---|:---:|:---:|---|:---:|
| **1. Environment Inspection** | Python `platform`, `subprocess` | 0.01s | 0 | Environment JSON | Pass |
| **2. C++23 Native Benchmark** | `./build/release/aurora_benchmark_throughput --json results/cpp_benchmark_results.json` | 26.62s | 0 | `results/cpp_benchmark_results.json` | Pass (Values match systems table) |
| **3. Python Profiler** | `benchmarks/profile_aurora.py --env-steps 30` | 1.22s | 0 | `results/python_systems_profile.json`, `results/aurora_profile.prof` | Pass |
| **4. Cross-Language Parity** | `benchmarks/benchmark_cross_language.py` | 3.69s | 0 | `results/cross_language_comparison.json` | Pass |
| **5. Figures & Tables** | `paper/generate_figures_and_tables.py` | 3.47s | 0 | `table_*.tex`, `fig_*.pdf`, `fig_*.png` | Pass (Script runs cleanly) |
| **6. Checksum Verification** | `compute_sha256()` | 0.01s | 0 | `paper/manifest_checksums.json`, `results/reproduction_report.json` | Pass |

---

## 2. Discrepancies Uncovered During Reproduction Audit

While the pipeline runs without execution errors, the audit uncovered major data discrepancies between stages:

1. **Table Data vs README Discrepancy:**
   - `paper/generate_figures_and_tables.py` reads `results/ablation/ablation_summary.json` and produces `paper/table_ablations.tex` with IQM values $\approx -1446.27$.
   - The root `README.md` contained a manually written table displaying IQM values of $-173.80 \dots -216.51$.
   - **Finding:** The reproduction script does NOT update the README table, leading to a direct contradiction between the paper/manifest and the README.

2. **Ablation Study Was Not Part of Default Reproduction:**
   - `scripts/reproduce_all.py` compiles figures and tables from existing `results/ablation/` data, but does not re-run `experiments/ablation_study.py` (which takes several minutes).
   - If `results/ablation/` is deleted, `reproduce_all.py` fails at stage 4 because `ablation_summary.json` is missing.

3. **Checksum Invalidation:**
   - Any modification to `generate_figures_and_tables.py` or re-running of benchmarks slightly updates timings in `results/cpp_benchmark_results.json`, immediately invalidating the recorded SHA-256 hashes in `paper/manifest_checksums.json`.

---

## 3. Reproduction Audit Verdict

- **Automated Execution:** Fully functional and reproducible from clean command invocation.
- **Scientific Reproducibility of Claims:** **FAILED** due to:
  1. The gradient detachment bug in the pessimistic value optimization.
  2. Mismatch between raw ablation returns (-1446) and README reported numbers (-173).
  3. Lack of statistical significance in actual experiment manifests.
