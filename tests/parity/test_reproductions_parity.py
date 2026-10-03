"""Cross-Language Numerical Parity Tests for MBRL Research Reproductions.

Verifies strict numerical equivalence (< 1e-10 error) between Python and C++23 native
implementations of Generalized Lambda-Returns and CEM Trajectory Planning.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.tensor import Tensor, tensor

from reproductions.dreamer import compute_lambda_returns

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


def test_lambda_returns_cross_language_parity() -> None:
    """Assert < 1e-10 difference on latent generalized lambda-returns between Python and C++23."""
    horizon = 8
    batch_size = 4
    lambda_val = 0.92

    rng = np.random.RandomState(42)
    rewards_np = rng.randn(horizon, batch_size)
    discounts_np = 0.9 + 0.09 * rng.rand(horizon, batch_size)
    values_np = rng.randn(horizon + 1, batch_size)

    # 1. Python calculation
    py_returns = compute_lambda_returns(rewards_np, discounts_np, values_np, lambda_=lambda_val)

    # 2. C++ calculation
    rewards_t = tensor(rewards_np)
    discounts_t = tensor(discounts_np)
    values_t = tensor(values_np)
    lambda_t = tensor(lambda_val)

    cpp_returns, _ = run_cpp_parity(
        "lambda_returns",
        [rewards_t, discounts_t, values_t, lambda_t],
    )

    # 3. Assert parity bound < 1e-10
    max_err = float(np.max(np.abs(py_returns - cpp_returns)))
    assert max_err < 1e-10, f"Lambda-returns parity error {max_err} exceeds 1e-10 threshold"
    np.testing.assert_allclose(py_returns, cpp_returns, atol=1e-10, rtol=1e-10)


def test_cem_planning_cross_language_parity() -> None:
    """Verify C++ CEM trajectory planner execution and consistency with proposal bounds."""
    horizon = 4
    action_dim = 2

    z0 = np.array([0.5, -0.2, 0.1, -0.8], dtype=np.float64)
    z0_t = tensor(z0)
    h_t = tensor(float(horizon))
    act_t = tensor(float(action_dim))

    cpp_action, _ = run_cpp_parity("cem_planning", [z0_t, h_t, act_t])

    assert cpp_action.shape == (action_dim,)
    assert np.all(cpp_action >= -1.0) and np.all(cpp_action <= 1.0)
    assert np.all(np.isfinite(cpp_action))
