# Baseline Fairness & Experimental Control Audit

**Auditor:** Principal Research Scientist, Skeptical ML Reviewer, Reproducibility Auditor  
**Date:** 2026-10-03

---

## 1. Audit of Baseline Comparisons

AURORA is compared against:
1. **Model-Based Policy Optimization (MBPO; Janner et al., 2019)**
2. **Soft Actor-Critic (SAC; Haarnoja et al., 2018)**
3. **Component Ablations (w/o Adaptive Horizon, w/o Dynamic Blending, w/o Pessimism)**

### Evaluation Criteria Checklist

| Fairness Dimension | Assessment | Audit Details | Verdict |
|---|:---:|---|:---:|
| **Environment & Dynamics** | Identical | All methods evaluated on identical `Pendulum` continuous physics implementation. | **FAIR** |
| **Observation & Action Preprocessing** | Identical | Raw continuous observations (angle $\cos, \sin$, angular velocity $\dot{\theta}$) and action bounds $[-2.0, 2.0]$. | **FAIR** |
| **Interaction Budget** | Identical | 300 environment interaction steps with 50 warmup steps across all runs. | **FAIR** |
| **Evaluation Protocol** | Identical | Deterministic policy evaluation over identical evaluation seeds (`seed * 100 + ep`). | **FAIR** |
| **Network Architectures** | Comparable | Same actor/critic MLP widths `[48, 48]` and ensemble widths `[48, 48]`. | **FAIR** |
| **Optimizer & Hyperparameters** | Matched | AdamW with identical learning rates ($3\times 10^{-4}$ actor/critic, $1\times 10^{-3}$ model). | **FAIR** |
| **Replay Buffer Capacity** | Matched | Real buffer: 10,000; Model buffer: 20,000. | **FAIR** |
| **Ablation Control Integrity** | **FAILED** | In `runner.py`, `adaptive_horizon: false` was not wired to bypass budget checks; `pessimistic_penalty: false` changed nothing because the gradient was already 0. | **UNFAIR / DEFECTIVE** |
| **Benchmark Sample Size** | **UNDERPOWERED** | 3 seeds per condition ($N=30$ total evaluation episodes). Welch tests between conditions yielded $p > 0.45$. | **UNDERPOWERED** |

---

## 2. Recommendations for Fair & Defensible Baselines

1. **Fix Ablation Isolation:** Ensure that `adaptive_horizon: false` enforces a strict fixed horizon $H$ with no threshold or budget truncation.
2. **Increase Seed Count:** Increase evaluation from 3 seeds to at least 5–10 seeds with sufficient environment steps (e.g. 500–1000 steps) to allow asymptotic policy convergence and detect statistically significant differences.
3. **Report Exact Numbers:** Synchronize all tables and README documentation with raw JSON manifests.
