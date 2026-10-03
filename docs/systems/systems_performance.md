# AURORA Systems Performance, Profiling & Native Scaling Specification

**Project:** AURORA (*Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents*)  
**Module:** Systems Performance & Native Scaling (`docs/systems/systems_performance.md`)  
**Target Platforms:** Python 3.14 (Reference) & C++23 (Native High-Throughput Engine)  
**Hardware Baseline:** Intel Core i5-8265U (4C/8T, AVX2, FMA, 64-byte cache line), NVIDIA GeForce MX230 (CUDA 13.3 Compute 6.1)

---

## 1. Executive Summary & Purpose

Reinforcement learning with learned world models introduces significant computational demands beyond standard model-free algorithms. In **AURORA**, the agent continually executes five distinct computational workloads:
1. **Environment Interaction:** Simulating physical transition dynamics in real environments.
2. **World Model Training:** Backpropagating Gaussian Negative Log-Likelihood (NLL) gradients through an ensemble of deep neural dynamics models.
3. **Adaptive Uncertainty Imagination:** Rolling out synthetic trajectories through the dynamics ensemble, computing epistemic disagreement $\sigma_{\text{epi}}^2(s, a)$ at every step, and dynamically adjusting rollout horizon $H_t$ and synthetic blending ratio $\eta_t$.
4. **Policy & Value Optimization:** Optimizing twin pessimistic soft actor-critic objectives over hybrid minibatches ($D_{\text{env}} \cup D_{\text{model}}$).
5. **Statistical Evaluation & Monitoring:** Computing Interquartile Mean (IQM) and bootstrap confidence intervals over empirical training metrics.

This specification formalizes the systems performance targets, memory hierarchy constraints, SIMD vectorization criteria, microbenchmarking protocols, and cross-language throughput scaling for AURORA.

---

## 2. Mathematical Formulations of Systems Throughput

### 2.1 Operational Throughput Metrics

We define five canonical throughput metrics to quantify performance across the reinforcement learning pipeline:

#### 1. Environment Simulation Throughput ($S_{\text{sim}}$)
Measures the physical or simulated environment stepping rate:
$$S_{\text{sim}} = \frac{N_{\text{env\_steps}}}{\Delta t_{\text{sim}}} \quad [\text{steps / second}]$$

#### 2. Model Forward & Disagreement Prediction Throughput ($S_{\text{dyn}}$)
Measures the number of ensemble transition evaluations executed per unit time:
$$S_{\text{dyn}} = \frac{E \cdot B \cdot H}{\Delta t_{\text{dyn}}} \quad [\text{ensemble transitions / second}]$$
where $E$ is the ensemble size (typically $E \in \{3, 5, 7\}$), $B$ is the batch size, and $H$ is the rollout horizon.

#### 3. Adaptive Imagination Rollout Throughput ($S_{\text{imag}}$)
Measures the generation and epistemic filtering rate of synthetic experience:
$$S_{\text{imag}} = \frac{\sum_{i=1}^B H_i}{\Delta t_{\text{imag}}} \quad [\text{synthetic transitions / second}]$$
where $H_i \le H_{\max}$ is the dynamically truncated horizon for seed state $i$ based on uncertainty threshold $\tau_t$.

#### 4. Policy & Critic Optimization Throughput ($S_{\text{opt}}$)
Measures the gradient update step throughput over minibatch batches:
$$S_{\text{opt}} = \frac{N_{\text{grad\_updates}} \cdot B_{\text{batch}}}{\Delta t_{\text{opt}}} \quad [\text{sample transitions trained / second}]$$

#### 5. End-to-End System Throughput ($S_{\text{E2E}}$)
Measures real-time environment interaction steps completed per total wall-clock second during online training:
$$S_{\text{E2E}} = \frac{N_{\text{env\_steps}}}{\Delta t_{\text{total}}} \quad [\text{env steps / wall-clock second}]$$

---

## 3. Systems Decomposition & Latency Distributions

### 3.1 Amdahl's Law Decomposition

The total wall-clock execution time $T_{\text{total}}$ of an online training epoch is partitioned as:
$$T_{\text{total}} = T_{\text{env}} + T_{\text{dyn\_train}} + T_{\text{imag}} + T_{\text{policy\_opt}} + T_{\text{eval}} + T_{\text{overhead}}$$

The fraction of execution time consumed by module $m$ is:
$$\phi_m = \frac{T_m}{T_{\text{total}}}$$

In model-based RL with ensemble dynamics and actor-critic optimization, the theoretical bottleneck profiles typically satisfy:
- Policy Optimization ($\phi_{\text{policy\_opt}}$): $40\% - 55\%$
- World Model Training ($\phi_{\text{dyn\_train}}$): $25\% - 35\%$
- Adaptive Imagination ($\phi_{\text{imag}}$): $10\% - 20\%$
- Environment Simulation ($\phi_{\text{env}}$): $3\% - 8\%$
- Logging & Statistical Evaluation ($\phi_{\text{eval}}$): $< 2\%$

### 3.2 Latency Percentiles & Tail Analysis

For real-time and interactive deployment, mean execution time is insufficient. Systems must report percentiles of per-step latency:
- $p_{50}$ (Median latency): Representative per-step execution time.
- $p_{90}$ (90th percentile): Upper bound under normal scheduling load.
- $p_{99}$ (Tail latency): Impact of GC pauses, cache misses, dynamic reallocation, or OS scheduling jitter.

---

## 4. Native C++23 Memory Hierarchy & SIMD Optimization

### 4.1 Memory Layout & Cache Line Alignment

1. **Contiguous Row-Major Storage:**
   All `aurora::Tensor` instances maintain standard C-contiguous row-major strides:
   $$\text{stride}[d] = \prod_{k=d+1}^{D-1} \text{shape}[k]$$
2. **Direct Fast-Path for Contiguous Tensors:**
   When both operands in binary arithmetic (`add`, `sub`, `mul`, `div`) share identical shapes and contiguous layouts, coordinate-stride indexing $(d / \text{stride}[k] \pmod \cdot)$ must be bypassed in favor of raw pointer iteration:
   ```cpp
   if (shape_ == other->shape() && is_contiguous() && other->is_contiguous()) {
       const double* a_ptr = data();
       const double* b_ptr = other->data();
       double* out_ptr = out_data.data();
       #pragma GCC ivdep
       for (size_t i = 0; i < total; ++i) {
           out_ptr[i] = a_ptr[i] + b_ptr[i];
       }
   }
   ```
3. **Cache-Friendly Matmul (GEMM):**
   Matrix multiplication computes $C_{M \times N} = A_{M \times K} B_{K \times N}$.
   Standard $i-j-k$ loop order incurs severe cache misses when accessing $B$ column-wise. AURORA enforces cache-friendly $i-k-j$ loop order:
   ```cpp
   for (size_t i = 0; i < M; ++i) {
       for (size_t k = 0; k < K; ++k) {
           double a_ik = ptr_a[i * K + k];
           for (size_t j = 0; j < N; ++j) {
               ptr_c[i * N + j] += a_ik * ptr_b[k * N + j];
           }
       }
   }
   ```
   In the inner $j$ loop, both `ptr_b[k * N + j]` and `ptr_c[i * N + j]` enjoy unit-stride contiguous sequential access, enabling hardware prefetchers and AVX2 SIMD fused multiply-add (`vfmadd231pd`).

### 4.2 Arithmetic Intensity & Roofline Model

Arithmetic intensity is defined as:
$$\mathcal{I} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{DRAM Memory Traffic (Bytes)}}$$

For matrix multiplication $M \times K \times N$:
$$\text{FLOPs} = 2 M K N$$
$$\text{Bytes} = (M K + K N + M N) \times 8 \quad (\text{for float64})$$

For large inner dimensions ($K, N \gg 1$), $\mathcal{I} \approx \frac{2 K N}{8(K + N)} \propto \mathcal{O}(\min(K, N))$, moving the computation from memory-bandwidth bound to compute bound.

---

## 5. Microbenchmarking Protocol & Rigor

All systems benchmarks must adhere to the following empirical protocol:
1. **Warmup Phase:** Minimum 10 iterations prior to recording timings to warm CPU instruction and data caches, eliminate lazy allocations, and trigger branch predictors.
2. **Measurement Phase:** Minimum 50 to 100 trials, recording high-resolution monotonic timestamps (`std::chrono::steady_clock` in C++, `time.perf_counter_ns()` in Python).
3. **Outlier Filtering:** Calculate Interquartile Mean (IQM) and 95% bootstrap confidence intervals across trials.
4. **Reproducibility Header:** Record commit hash, compiler flags (`-O3 -march=native`), CPU frequency, and memory footprint.
