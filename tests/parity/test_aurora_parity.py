"""Cross-Language Numerical Parity Tests for the Novel AURORA Algorithm.

Verifies strict numerical equivalence (< 1e-10 error) between Python and C++23 native
implementations of:
1. Adaptive Horizon Scheduling (peak threshold decay + cumulative budget termination)
2. Dynamic Blending Ratio Controller (momentum-smoothed replay blend)
3. Epistemic Risk-Sensitive Pessimistic Value Penalty
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.algorithm.scheduler import AdaptiveHorizonScheduler, DynamicBlendingController
from aurora.tensor import Tensor, tensor

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


def test_adaptive_horizon_parity() -> None:
    """Assert exact integer parity on adaptive rollout horizon between Python and C++23."""
    test_cases = [
        # (trajectory, val_loss, (h_max, h_min, tau_base, kappa, budget_max, gamma))
        ([0.05] * 12, 0.0, (12, 1, 0.5, 1.0, 2.0, 0.99)),
        ([0.1, 0.1, 0.1, 0.8, 0.1], 0.2, (10, 1, 0.6, 1.5, 1.5, 0.95)),
        ([0.3, 0.4, 0.5, 0.6, 0.7], 0.8, (15, 2, 0.5, 2.0, 1.0, 0.9)),
        ([0.02] * 20, 0.1, (15, 1, 0.5, 1.0, 3.0, 0.99)),
    ]

    for traj, val_loss, cfg_tuple in test_cases:
        h_max, h_min, tau_base, kappa, budget_max, gamma = cfg_tuple

        # Python calculation
        py_scheduler = AdaptiveHorizonScheduler(
            horizon_max=h_max,
            horizon_min=h_min,
            tau_base=tau_base,
            kappa=kappa,
            budget_max=budget_max,
            gamma=gamma,
        )
        py_scheduler.update_threshold(val_loss)
        py_h = py_scheduler.compute_adaptive_horizon(traj)

        # C++ calculation via parity runner
        traj_t = tensor(np.array(traj, dtype=np.float64))
        cfg_t = tensor(np.array(cfg_tuple, dtype=np.float64))
        loss_t = tensor(np.array([val_loss], dtype=np.float64))

        cpp_out, _ = run_cpp_parity("aurora_adaptive_horizon", [traj_t, cfg_t, loss_t])
        cpp_h = int(cpp_out.item())

        assert py_h == cpp_h, f"Parity mismatch: Python {py_h} != C++ {cpp_h} for traj {traj}"


def test_dynamic_blending_ratio_parity() -> None:
    """Assert < 1e-10 numerical parity on dynamic blending controller between Python and C++23."""
    rng = np.random.RandomState(42)
    u_sequence = rng.uniform(0.0, 0.8, size=25).tolist()

    cfg_tuple = (0.85, 0.05, 0.4, 0.75)  # eta_max, eta_min, u_target, momentum
    eta_max, eta_min, u_target, momentum = cfg_tuple

    # Python computation
    py_controller = DynamicBlendingController(
        eta_max=eta_max,
        eta_min=eta_min,
        u_target=u_target,
        momentum=momentum,
    )
    py_etas = [py_controller.compute_ratio(u) for u in u_sequence]

    # C++ computation
    u_t = tensor(np.array(u_sequence, dtype=np.float64))
    cfg_t = tensor(np.array(cfg_tuple, dtype=np.float64))

    cpp_out, _ = run_cpp_parity("aurora_blending_ratio", [u_t, cfg_t])
    cpp_etas = cpp_out.flatten().tolist()

    assert len(py_etas) == len(cpp_etas)
    max_err = max(abs(p - c) for p, c in zip(py_etas, cpp_etas, strict=True))
    assert max_err < 1e-10, f"Max error {max_err} exceeds 1e-10 parity threshold"


def test_pessimistic_value_parity() -> None:
    """Assert < 1e-10 numerical parity on epistemic pessimistic value penalty."""
    rng = np.random.RandomState(123)
    batch_size = 32

    q1_data = rng.randn(batch_size, 1).astype(np.float64) * 5.0
    q2_data = rng.randn(batch_size, 1).astype(np.float64) * 5.0
    u_epi_data = np.abs(rng.randn(batch_size, 1).astype(np.float64)) * 1.5
    beta_pess = 0.65

    # Python computation
    q_min = np.minimum(q1_data, q2_data)
    py_q_pess = q_min - beta_pess * u_epi_data

    # C++ computation
    q1_t = tensor(q1_data)
    q2_t = tensor(q2_data)
    u_t = tensor(u_epi_data)
    beta_t = tensor(np.array([beta_pess], dtype=np.float64))

    cpp_out, _ = run_cpp_parity("aurora_pessimistic_value", [q1_t, q2_t, u_t, beta_t])

    max_err = float(np.max(np.abs(py_q_pess - cpp_out)))
    assert max_err < 1e-10, f"Pessimistic value error {max_err} exceeds 1e-10"
