"""Cross-Language Numerical Parity Test Suite for World Models and Uncertainty Calibration.

Verifies strict mathematical equivalence (< 1e-10 error) between Python
and C++23 native implementations of Sigmoid, Gaussian NLL Loss, Uncertainty Decomposition,
and RSSM Analytical KL Divergence.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.tensor import Tensor, tensor
from aurora.world_model.ensemble import EnsembleDynamicsModel
from aurora.world_model.rssm import RSSM
from aurora.world_model.uncertainty import UncertaintyEstimator

PARITY_BIN = Path(__file__).parents[2] / "build" / "debug" / "aurora_parity_runner"


def run_cpp_parity(
    op: str,
    inputs: Sequence[Tensor],
    axis: int | None = None,
    keepdims: bool = False,
    eps: float = 1e-5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Execute operation in C++ native engine via aurora_parity_runner."""
    if not PARITY_BIN.exists():
        pytest.skip(f"C++ parity runner binary not found at {PARITY_BIN}.")

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
        [str(PARITY_BIN)],
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


class TestWorldModelNumericalParity:
    def test_sigmoid_parity(self) -> None:
        np.random.seed(42)
        x_data = np.random.randn(3, 4)
        x_py = tensor(x_data, requires_grad=True)

        y_py = x_py.sigmoid()
        loss_py = y_py.sum()
        loss_py.backward()

        y_cpp, grads_cpp = run_cpp_parity("sigmoid", [x_py])
        assert x_py.grad is not None
        fwd_err = float(np.max(np.abs(y_py.numpy() - y_cpp)))
        bwd_err = float(np.max(np.abs(x_py.grad.numpy() - grads_cpp[0])))

        assert fwd_err < 1e-10, f"Sigmoid forward parity violated: err={fwd_err}"
        assert bwd_err < 1e-10, f"Sigmoid backward parity violated: err={bwd_err}"

    def test_gaussian_nll_loss_parity(self) -> None:
        np.random.seed(123)
        mean_data = np.random.randn(4, 3)
        log_var_data = np.random.uniform(-3.0, 1.0, size=(4, 3))
        target_data = np.random.randn(4, 3)

        mean_py = tensor(mean_data, requires_grad=True)
        log_var_py = tensor(log_var_data, requires_grad=True)
        target_py = tensor(target_data, requires_grad=False)

        loss_py = EnsembleDynamicsModel.gaussian_nll_loss(mean_py, log_var_py, target_py)
        loss_py.backward()

        loss_cpp, grads_cpp = run_cpp_parity("gaussian_nll_loss", [mean_py, log_var_py, target_py])

        fwd_err = float(np.abs(loss_py.numpy().item() - loss_cpp.item()))
        assert mean_py.grad is not None
        assert log_var_py.grad is not None
        mean_grad_err = float(np.max(np.abs(mean_py.grad.numpy() - grads_cpp[0])))
        log_var_grad_err = float(np.max(np.abs(log_var_py.grad.numpy() - grads_cpp[1])))

        assert fwd_err < 1e-10, f"Gaussian NLL loss fwd parity violated: err={fwd_err}"
        assert mean_grad_err < 1e-10, f"Gaussian NLL mean grad parity violated: err={mean_grad_err}"
        assert (
            log_var_grad_err < 1e-10
        ), f"Gaussian NLL log_var grad parity violated: err={log_var_grad_err}"

    def test_uncertainty_decomposition_parity(self) -> None:
        np.random.seed(999)
        num_models = 4
        m_list = [np.random.randn(2, 3) for _ in range(num_models)]
        v_list = [np.random.uniform(0.1, 2.0, size=(2, 3)) for _ in range(num_models)]

        m_tensors = [tensor(m, requires_grad=True) for m in m_list]
        v_tensors = [tensor(v, requires_grad=True) for v in v_list]

        ens_mean_py, aleatoric_py, epistemic_py, total_py = UncertaintyEstimator.decompose_tensor(
            m_tensors, v_tensors
        )

        # Check mean parity (axis=0)
        mean_cpp, _ = run_cpp_parity("uncertainty_decompose", m_tensors + v_tensors, axis=0)
        mean_err = float(np.max(np.abs(ens_mean_py.numpy() - mean_cpp)))
        assert mean_err < 1e-10, f"Ensemble mean parity violated: err={mean_err}"

        # Check aleatoric parity (axis=1)
        al_cpp, _ = run_cpp_parity("uncertainty_decompose", m_tensors + v_tensors, axis=1)
        al_err = float(np.max(np.abs(aleatoric_py.numpy() - al_cpp)))
        assert al_err < 1e-10, f"Aleatoric uncertainty parity violated: err={al_err}"

        # Check epistemic parity (axis=2)
        ep_cpp, _ = run_cpp_parity("uncertainty_decompose", m_tensors + v_tensors, axis=2)
        ep_err = float(np.max(np.abs(epistemic_py.numpy() - ep_cpp)))
        assert ep_err < 1e-10, f"Epistemic uncertainty parity violated: err={ep_err}"

        # Check total variance parity (axis=3)
        tot_cpp, _ = run_cpp_parity("uncertainty_decompose", m_tensors + v_tensors, axis=3)
        tot_err = float(np.max(np.abs(total_py.numpy() - tot_cpp)))
        assert tot_err < 1e-10, f"Total predictive variance parity violated: err={tot_err}"

    def test_rssm_kl_divergence_parity(self) -> None:
        np.random.seed(777)
        mu_q = tensor(np.random.randn(3, 4), requires_grad=False)
        log_std_q = tensor(np.random.uniform(-1.0, 1.0, size=(3, 4)), requires_grad=False)
        mu_p = tensor(np.random.randn(3, 4), requires_grad=False)
        log_std_p = tensor(np.random.uniform(-1.0, 1.0, size=(3, 4)), requires_grad=False)

        kl_py = RSSM.kl_divergence(mu_q, log_std_q, mu_p, log_std_p)
        kl_cpp, _ = run_cpp_parity("rssm_kl_divergence", [mu_q, log_std_q, mu_p, log_std_p])

        kl_err = float(np.max(np.abs(kl_py.numpy() - kl_cpp)))
        assert kl_err < 1e-10, f"RSSM KL divergence parity violated: err={kl_err}"
