"""Cross-Language Numerical Parity Tests for Statistical Evaluation Module.

Verifies strict numerical equivalence (< 1e-10 error) between Python and C++23 native
implementations of:
1. Interquartile Mean (IQM)
2. Probability of Improvement P(X > Y)
3. Performance Profile Curve F(tau)
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.tensor import Tensor, tensor

from evaluation import compute_iqm, performance_profile, probability_of_improvement

PARITY_BIN_CANDIDATES = [
    Path(__file__).parents[2] / "build" / "aurora_parity_runner",
    Path(__file__).parents[2] / "build" / "debug" / "aurora_parity_runner",
]


def get_parity_bin() -> Path:
    for candidate in PARITY_BIN_CANDIDATES:
        if candidate.exists():
            return candidate
    pytest.skip("C++ parity runner binary not found.")
    return Path()


def run_cpp_parity(
    op: str,
    inputs: Sequence[Tensor],
    axis: int | None = None,
    keepdims: bool = False,
    eps: float = 1e-5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Execute operation in C++ native engine via aurora_parity_runner."""
    bin_path = get_parity_bin()

    payload = [f"OP: {op}"]
    if axis is not None:
        payload.append(f"AXIS: {axis}")
    payload.append(f"KEEPDIMS: {1 if keepdims else 0}")
    payload.append(f"EPS: {eps}")

    for inp in inputs:
        payload.append("TENSOR:")
        shape_str = f"{len(inp.shape)} " + " ".join(str(s) for s in inp.shape)
        payload.append(shape_str)
        payload.append(f"{1 if inp.requires_grad else 0}")
        flat_data = " ".join(f"{x:.17g}" for x in inp.numpy().flatten())
        payload.append(flat_data)

    payload.append("END")
    input_str = "\n".join(payload) + "\n"

    proc = subprocess.run(
        [str(bin_path)],
        input=input_str,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = proc.stdout.strip().split("\n")
    idx = 0
    assert lines[idx] == "STATUS: OK"
    idx += 1

    # Parse output shape
    out_shape_parts = lines[idx].split()[1:]
    rank = int(out_shape_parts[0])
    out_shape = tuple(int(x) for x in out_shape_parts[1 : 1 + rank])
    idx += 1

    # Parse output data
    out_data_parts = lines[idx].split()[1:]
    out_count = int(out_data_parts[0])
    out_data = np.array([float(x) for x in out_data_parts[1 : 1 + out_count]]).reshape(out_shape)
    idx += 1

    # Parse grads count
    grads_count = int(lines[idx].split()[1])
    idx += 1

    grads: list[np.ndarray] = []
    for _ in range(grads_count):
        g_shape_parts = lines[idx].split()[1:]
        g_rank = int(g_shape_parts[0])
        g_shape = tuple(int(x) for x in g_shape_parts[1 : 1 + g_rank])
        idx += 1

        g_data_parts = lines[idx].split()[1:]
        g_count = int(g_data_parts[0])
        g_data = np.array([float(x) for x in g_data_parts[1 : 1 + g_count]]).reshape(g_shape)
        idx += 1
        grads.append(g_data)

    return out_data, grads


def test_iqm_parity() -> None:
    """Assert < 1e-10 difference on IQM calculation between Python and C++23."""
    rng = np.random.RandomState(42)

    cases = [
        [14.0, 10.0, 50000.0, 18.0, -10000.0, 12.0, 16.0, 20.0],
        rng.randn(25).tolist(),
        rng.uniform(-500.0, 500.0, size=50).tolist(),
        [1.0, 2.0, 3.0],
    ]

    for scores in cases:
        py_iqm = compute_iqm(scores)
        inp_t = tensor(np.array(scores, dtype=np.float64))

        cpp_out, _ = run_cpp_parity("eval_iqm", [inp_t])
        cpp_iqm = float(cpp_out.item())

        err = abs(py_iqm - cpp_iqm)
        assert err < 1e-10, f"IQM parity mismatch: py={py_iqm}, cpp={cpp_iqm}, err={err}"


def test_probability_of_improvement_parity() -> None:
    """Assert < 1e-10 difference on P(X > Y) between Python and C++23."""
    rng = np.random.RandomState(123)

    cases = [
        (rng.randn(20).tolist(), rng.randn(20).tolist()),
        ([10.0, 20.0, 30.0], [5.0, 15.0, 25.0]),
        ([2.0, 4.0], [1.0, 3.0]),
    ]

    for x, y in cases:
        py_p = probability_of_improvement(x, y)
        x_t = tensor(np.array(x, dtype=np.float64))
        y_t = tensor(np.array(y, dtype=np.float64))

        cpp_out, _ = run_cpp_parity("eval_probability_of_improvement", [x_t, y_t])
        cpp_p = float(cpp_out.item())

        err = abs(py_p - cpp_p)
        assert err < 1e-10, f"P(X > Y) parity mismatch: py={py_p}, cpp={cpp_p}, err={err}"


def test_performance_profile_parity() -> None:
    """Assert < 1e-10 difference on performance profile between Python and C++23."""
    rng = np.random.RandomState(999)
    scores = rng.uniform(-100.0, 100.0, size=40).tolist()
    thresholds = np.linspace(-120.0, 120.0, 15).tolist()

    py_prof = performance_profile(scores, thresholds)
    s_t = tensor(np.array(scores, dtype=np.float64))
    th_t = tensor(np.array(thresholds, dtype=np.float64))

    cpp_out, _ = run_cpp_parity("eval_performance_profile", [s_t, th_t])
    cpp_prof = cpp_out.flatten()

    max_err = float(np.max(np.abs(py_prof - cpp_prof)))
    assert max_err < 1e-10, f"Performance profile error {max_err} exceeds 1e-10"
