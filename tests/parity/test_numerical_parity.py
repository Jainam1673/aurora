"""Cross-Language Numerical Parity Test Suite for AURORA.

Verifies strict mathematical equivalence between Python reference autograd tape
and C++23 native autograd engine.
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.tensor import Tensor, randn, tensor

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
        flat_data = " ".join(f"{x:.17g}" for x in inp.data.flatten())
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


def assert_parity(
    py_out: Tensor,
    py_inputs: Sequence[Tensor],
    cpp_out: np.ndarray,
    cpp_grads: Sequence[np.ndarray],
    atol: float = 1e-10,
    rtol: float = 1e-7,
) -> None:
    """Verify numerical parity for forward output and input gradients."""
    py_out_np = py_out.numpy()
    max_abs_err = float(np.max(np.abs(py_out_np - cpp_out)))
    mean_abs_err = float(np.mean(np.abs(py_out_np - cpp_out)))
    denom = np.maximum(np.abs(py_out_np), np.abs(cpp_out)) + 1e-8
    rel_err = float(np.max(np.abs(py_out_np - cpp_out) / denom))

    assert np.allclose(py_out_np, cpp_out, atol=atol, rtol=rtol), (
        f"Forward parity mismatch: MaxAbsErr={max_abs_err:.2e}, "
        f"MeanAbsErr={mean_abs_err:.2e}, RelErr={rel_err:.2e}"
    )

    # Check gradients if backward was performed
    for i, (py_inp, cpp_grad) in enumerate(zip(py_inputs, cpp_grads, strict=False)):
        if py_inp.requires_grad:
            assert py_inp.grad is not None
            py_g = py_inp.grad.numpy()
            g_max_abs = float(np.max(np.abs(py_g - cpp_grad)))
            g_denom = np.maximum(np.abs(py_g), np.abs(cpp_grad)) + 1e-8
            g_rel = float(np.max(np.abs(py_g - cpp_grad) / g_denom))
            assert np.allclose(py_g, cpp_grad, atol=atol, rtol=rtol), (
                f"Gradient mismatch on input {i}: MaxAbsErr={g_max_abs:.2e}, RelErr={g_rel:.2e}"
            )


def test_parity_add_broadcasted() -> None:
    np.random.seed(42)
    a = randn(2, 3, requires_grad=True)
    b = randn(3, requires_grad=True)

    py_out = a + b
    py_out.sum().backward()

    cpp_out, cpp_grads = run_cpp_parity("add", [a, b])
    assert_parity(py_out, [a, b], cpp_out, cpp_grads)


def test_parity_sub() -> None:
    np.random.seed(43)
    a = randn(3, 4, requires_grad=True)
    b = randn(3, 4, requires_grad=True)

    py_out = a - b
    py_out.sum().backward()

    cpp_out, cpp_grads = run_cpp_parity("sub", [a, b])
    assert_parity(py_out, [a, b], cpp_out, cpp_grads)


def test_parity_mul_broadcasted() -> None:
    np.random.seed(44)
    a = randn(2, 3, requires_grad=True)
    b = randn(1, 3, requires_grad=True)

    py_out = a * b
    py_out.sum().backward()

    cpp_out, cpp_grads = run_cpp_parity("mul", [a, b])
    assert_parity(py_out, [a, b], cpp_out, cpp_grads)


def test_parity_div() -> None:
    a = tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    b = tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=True)

    py_out = a / b
    py_out.sum().backward()

    cpp_out, cpp_grads = run_cpp_parity("div", [a, b])
    assert_parity(py_out, [a, b], cpp_out, cpp_grads)


def test_parity_matmul() -> None:
    np.random.seed(45)
    a = randn(3, 4, requires_grad=True)
    b = randn(4, 2, requires_grad=True)

    py_out = a @ b
    py_out.sum().backward()

    cpp_out, cpp_grads = run_cpp_parity("matmul", [a, b])
    assert_parity(py_out, [a, b], cpp_out, cpp_grads)


def test_parity_sum_and_mean() -> None:
    np.random.seed(46)
    x = randn(3, 4, requires_grad=True)

    # Full sum
    py_s = x.sum()
    py_s.backward()
    cpp_out, cpp_grads = run_cpp_parity("sum", [x])
    assert_parity(py_s, [x], cpp_out, cpp_grads)

    # Axis mean
    x.zero_grad()
    py_m = x.mean(axis=1, keepdims=False)
    py_m.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("mean", [x], axis=1, keepdims=False)
    assert_parity(py_m, [x], cpp_out, cpp_grads)


def test_parity_exp_log_sqrt() -> None:
    x = tensor([1.2, 2.5, 3.1, 0.8], requires_grad=True)

    py_exp = x.exp()
    py_exp.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("exp", [x])
    assert_parity(py_exp, [x], cpp_out, cpp_grads)

    x.zero_grad()
    py_log = x.log()
    py_log.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("log", [x])
    assert_parity(py_log, [x], cpp_out, cpp_grads)

    x.zero_grad()
    py_sqrt = x.sqrt()
    py_sqrt.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("sqrt", [x])
    assert_parity(py_sqrt, [x], cpp_out, cpp_grads)


def test_parity_relu_gelu_silu() -> None:
    x = tensor([-2.5, -1.0, 0.5, 1.8, 3.2], requires_grad=True)

    py_relu = x.relu()
    py_relu.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("relu", [x])
    assert_parity(py_relu, [x], cpp_out, cpp_grads)

    x.zero_grad()
    py_gelu = x.gelu()
    py_gelu.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("gelu", [x])
    assert_parity(py_gelu, [x], cpp_out, cpp_grads)

    x.zero_grad()
    py_silu = x.silu()
    py_silu.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("silu", [x])
    assert_parity(py_silu, [x], cpp_out, cpp_grads)


def test_parity_softmax_and_log_softmax() -> None:
    np.random.seed(47)
    x = randn(2, 4, requires_grad=True)

    py_sm = x.softmax(axis=-1)
    py_sm.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("softmax", [x], axis=-1)
    assert_parity(py_sm, [x], cpp_out, cpp_grads)

    x.zero_grad()
    py_lsm = x.log_softmax(axis=-1)
    py_lsm.sum().backward()
    cpp_out, cpp_grads = run_cpp_parity("log_softmax", [x], axis=-1)
    assert_parity(py_lsm, [x], cpp_out, cpp_grads)


def test_parity_layer_norm() -> None:
    np.random.seed(48)
    x = randn(2, 4, requires_grad=True)
    gamma = randn(4, requires_grad=True)
    beta = randn(4, requires_grad=True)

    py_ln = x.layer_norm(gamma=gamma, beta=beta)
    py_ln.sum().backward()

    cpp_out, cpp_grads = run_cpp_parity("layer_norm", [x, gamma, beta])
    assert_parity(py_ln, [x, gamma, beta], cpp_out, cpp_grads)
