"""AURORA Cross-Language Systems Parity & Throughput Comparison Suite.

Executes identical computational workloads in Python 3.14 and loads C++23 native benchmarks
to produce rigorous scientific throughput comparisons and speedup metrics:
- Low-level tensor allocation and elementwise operations
- GEMM compute throughput (FLOPs/s)
- Dynamics ensemble forward evaluation (transitions/s)
- Synthetic imagination rollout generation (transitions/s)
- Statistical IQM and bootstrap confidence interval computation (resamples/s)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from aurora.tensor import tensor
from aurora.world_model.ensemble import EnsembleDynamicsModel

from evaluation.metrics import bootstrap_ci, compute_iqm


def run_python_benchmarks(num_trials: int = 20) -> dict[str, dict[str, Any]]:
    """Execute corresponding benchmark suite in Python."""
    results: dict[str, dict[str, Any]] = {}

    # 1. Tensor Alloc & Elementwise (500k elements)
    n_elem = 500_000
    shape = (500, 1000)

    # Allocation & Fill
    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        _ = tensor(np.zeros(shape, dtype=np.float64))
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Contiguous Allocation & Fill"] = {
        "mean_us": mean_us,
        "throughput": (n_elem / (mean_us * 1e-6)),
        "unit": "elements/s",
    }

    # Elementwise Add
    t1_np = np.ones(shape, dtype=np.float64)
    t2_np = np.ones(shape, dtype=np.float64)
    t1_t = tensor(t1_np)
    t2_t = tensor(t2_np)

    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        _ = t1_t + t2_t
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Contiguous Elementwise Add"] = {
        "mean_us": mean_us,
        "throughput": (n_elem / (mean_us * 1e-6)),
        "unit": "elements/s",
    }

    # Elementwise Mul
    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        _ = t1_t * t2_t
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Contiguous Elementwise Mul"] = {
        "mean_us": mean_us,
        "throughput": (n_elem / (mean_us * 1e-6)),
        "unit": "elements/s",
    }

    # ReLU Activation
    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        _ = t1_t.relu()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Contiguous ReLU Activation"] = {
        "mean_us": mean_us,
        "throughput": (n_elem / (mean_us * 1e-6)),
        "unit": "elements/s",
    }

    # Full Sum
    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        _ = t1_t.sum()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Full Scalar Sum Reduction"] = {
        "mean_us": mean_us,
        "throughput": (n_elem / (mean_us * 1e-6)),
        "unit": "elements/s",
    }

    # 2. GEMM Matrix Multiplication (64, 128, 256)
    for sz in [64, 128, 256]:
        a = tensor(np.ones((sz, sz), dtype=np.float64))
        b = tensor(np.ones((sz, sz), dtype=np.float64))
        flops = 2 * sz * sz * sz

        times = []
        for _ in range(15):
            t0 = time.perf_counter_ns()
            _ = a @ b
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0)
        mean_us = float(np.mean(times))
        results[f"Matmul {sz}x{sz}x{sz}"] = {
            "mean_us": mean_us,
            "throughput": (flops / (mean_us * 1e-6)),
            "unit": "FLOPs/s",
        }

    # 3. Dynamics Ensemble Forward (B=64, E=5)
    dyn = EnsembleDynamicsModel(
        obs_dim=4, action_dim=1, ensemble_size=5, hidden_dims=[64, 64], activation="silu"
    )
    obs = tensor(np.zeros((64, 4), dtype=np.float64))
    act = tensor(np.zeros((64, 1), dtype=np.float64))

    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        _ = dyn.forward(obs, act)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Forward Ensemble (B=64)"] = {
        "mean_us": mean_us,
        "throughput": ((64 * 5) / (mean_us * 1e-6)),
        "unit": "transitions/s",
    }

    # 4. Statistical Evaluation (IQM & Bootstrap CI)
    scores = np.array([float((i % 30) * 1.5) for i in range(100)])

    times = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        _ = compute_iqm(scores)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["IQM (N=100)"] = {
        "mean_us": mean_us,
        "throughput": (100.0 / (mean_us * 1e-6)),
        "unit": "samples/s",
    }

    times = []
    for _ in range(15):
        t0 = time.perf_counter_ns()
        _ = bootstrap_ci(scores, num_bootstraps=1000, confidence_level=0.95)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    mean_us = float(np.mean(times))
    results["Bootstrap CI (N=100, R=1000)"] = {
        "mean_us": mean_us,
        "throughput": ((100.0 * 1000.0) / (mean_us * 1e-6)),
        "unit": "resamples/s",
    }

    return results


def compare_systems(
    cpp_json_path: str = "results/cpp_benchmark_results.json",
    output_comparison_json: str = "results/cross_language_comparison.json",
) -> dict[str, Any]:
    """Compare Python and C++23 native systems throughput."""
    cpp_file = Path(cpp_json_path)
    if not cpp_file.exists():
        raise FileNotFoundError(
            f"C++ benchmark results not found at {cpp_json_path}. "
            "Run aurora_benchmark_throughput first."
        )

    with open(cpp_file) as f:
        cpp_data = json.load(f)

    cpp_map = {item["name"]: item for item in cpp_data.get("benchmarks", [])}

    print("Running Python benchmark workloads for parity comparison...")
    py_results = run_python_benchmarks()

    comparisons = []
    for name, py_res in py_results.items():
        if name in cpp_map:
            cpp_res = cpp_map[name]
            speedup = cpp_res["throughput"] / py_res["throughput"]
            comparisons.append(
                {
                    "benchmark": name,
                    "unit": py_res["unit"],
                    "python_mean_us": py_res["mean_us"],
                    "cpp_mean_us": cpp_res["mean_us"],
                    "python_throughput": py_res["throughput"],
                    "cpp_throughput": cpp_res["throughput"],
                    "speedup_cpp_vs_python": float(speedup),
                }
            )

    # Print Table
    print("\n" + "=" * 100)
    print("                     AURORA CROSS-LANGUAGE SYSTEMS BENCHMARK (Python vs C++23)      ")
    print("=" * 100)
    header = (
        f"{'Benchmark':<32} {'Unit':<15} {'Python Lat (us)':<16} "
        f"{'C++23 Lat (us)':<16} {'Speedup (x)':<12}"
    )
    print(header)
    print("-" * 100)
    for c in comparisons:
        row = (
            f"{c['benchmark']:<32} {c['unit']:<15} {c['python_mean_us']:<16.2f} "
            f"{c['cpp_mean_us']:<16.2f} {c['speedup_cpp_vs_python']:<12.2f}"
        )
        print(row)
    print("=" * 100 + "\n")

    out_data = {"comparisons": comparisons}
    out_path = Path(output_comparison_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Cross-language systems comparison exported to: {output_comparison_json}")

    return out_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AURORA Cross-Language Systems Performance Comparison"
    )
    parser.add_argument(
        "--cpp-json",
        type=str,
        default="results/cpp_benchmark_results.json",
        help="Path to C++ benchmark results JSON",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/cross_language_comparison.json",
        help="Path for comparative output JSON",
    )
    args = parser.parse_args()

    compare_systems(cpp_json_path=args.cpp_json, output_comparison_json=args.output_json)


if __name__ == "__main__":
    main()
