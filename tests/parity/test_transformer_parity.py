"""Cross-Language Attention & Transformer Parity Test Suite for AURORA.

Verifies bidirectional numerical equivalence (< 1e-10 error) between Python
and C++23 native implementations of Scaled Dot-Product Attention, causal masking,
autograd VJPs, and multi-step Transformer Block optimization.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.checkpoint import load_checkpoint, save_checkpoint
from aurora.nn.attention import create_causal_mask, scaled_dot_product_attention
from aurora.nn.module import Module
from aurora.nn.parameter import Parameter
from aurora.nn.transformer import TransformerBlock
from aurora.optim.adamw import AdamW
from aurora.tensor import Tensor, tensor

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
    out_shape = tuple(int(x) for x in out_shape_parts[1 : rank + 1])
    idx += 1

    # Parse output data
    out_data_parts = lines[idx].split()[1:]
    out_data = np.array([float(x) for x in out_data_parts[1:]], dtype=np.float64).reshape(out_shape)
    idx += 1

    # Parse grads
    grads_count = int(lines[idx].split()[1])
    idx += 1
    grads: list[np.ndarray] = []
    for _ in range(grads_count):
        g_shape_parts = lines[idx].split()[1:]
        g_rank = int(g_shape_parts[0])
        g_shape = tuple(int(x) for x in g_shape_parts[1 : g_rank + 1])
        idx += 1

        g_data_parts = lines[idx].split()[1:]
        g_data = np.array([float(x) for x in g_data_parts[1:]], dtype=np.float64).reshape(g_shape)
        idx += 1
        grads.append(g_data)

    return out_data, grads


def test_scaled_dot_product_attention_parity() -> None:
    """Verify Scaled Dot-Product Attention forward and backward numerical parity (< 1e-10)."""
    rng = np.random.RandomState(42)
    b, h, t, d_k = 2, 2, 4, 8

    q_data = rng.randn(b, h, t, d_k).astype(np.float64)
    k_data = rng.randn(b, h, t, d_k).astype(np.float64)
    v_data = rng.randn(b, h, t, d_k).astype(np.float64)

    # 1. Non-causal attention
    q_py = tensor(q_data, requires_grad=True)
    k_py = tensor(k_data, requires_grad=True)
    v_py = tensor(v_data, requires_grad=True)

    out_py, _ = scaled_dot_product_attention(q_py, k_py, v_py)
    loss_py = out_py.sum()
    loss_py.backward()

    q_cpp = tensor(q_data, requires_grad=True)
    k_cpp = tensor(k_data, requires_grad=True)
    v_cpp = tensor(v_data, requires_grad=True)

    out_cpp, grads_cpp = run_cpp_parity("attention", [q_cpp, k_cpp, v_cpp])

    # Assert forward parity
    np.testing.assert_allclose(out_py.numpy(), out_cpp, atol=1e-10, rtol=1e-10)

    # Assert backward gradient parity
    assert q_py.grad is not None
    assert k_py.grad is not None
    assert v_py.grad is not None
    np.testing.assert_allclose(q_py.grad.numpy(), grads_cpp[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(k_py.grad.numpy(), grads_cpp[1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(v_py.grad.numpy(), grads_cpp[2], atol=1e-10, rtol=1e-10)

    # 2. Causal attention with mask
    mask_py = create_causal_mask(t)
    q_py_c = tensor(q_data, requires_grad=True)
    k_py_c = tensor(k_data, requires_grad=True)
    v_py_c = tensor(v_data, requires_grad=True)

    out_py_c, _ = scaled_dot_product_attention(q_py_c, k_py_c, v_py_c, mask=mask_py)
    loss_py_c = out_py_c.sum()
    loss_py_c.backward()

    mask_cpp = tensor(mask_py.numpy(), requires_grad=False)
    q_cpp_c = tensor(q_data, requires_grad=True)
    k_cpp_c = tensor(k_data, requires_grad=True)
    v_cpp_c = tensor(v_data, requires_grad=True)

    out_cpp_c, grads_cpp_c = run_cpp_parity("attention", [q_cpp_c, k_cpp_c, v_cpp_c, mask_cpp])

    np.testing.assert_allclose(out_py_c.numpy(), out_cpp_c, atol=1e-10, rtol=1e-10)
    assert q_py_c.grad is not None
    assert k_py_c.grad is not None
    assert v_py_c.grad is not None
    np.testing.assert_allclose(q_py_c.grad.numpy(), grads_cpp_c[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(k_py_c.grad.numpy(), grads_cpp_c[1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(v_py_c.grad.numpy(), grads_cpp_c[2], atol=1e-10, rtol=1e-10)


def test_transformer_block_optimization_step_parity() -> None:
    """Verify Transformer Block forward, autograd, and AdamW parity in checkpoint lockstep."""
    if not PARITY_BIN.exists():
        pytest.skip(f"Parity runner binary not found at {PARITY_BIN}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_ckpt_path = Path(tmpdir) / "in_ckpt.json"
        x_ckpt_path = Path(tmpdir) / "x_ckpt.json"
        cpp_out_ckpt_path = Path(tmpdir) / "cpp_out_ckpt.json"

        d_model = 16
        num_heads = 4
        d_ff = 32

        # 1. Initialize Python TransformerBlock and AdamW optimizer
        block_py = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            norm_type="layernorm",
            activation="gelu",
            dropout=0.0,
            bias=True,
        )
        opt_py = AdamW(
            block_py.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

        # Set deterministic weights
        rng = np.random.RandomState(123)
        for _, p in block_py.named_parameters():
            p.data = rng.randn(*p.shape).astype(np.float64) * 0.1

        # Save initial checkpoint
        save_checkpoint(in_ckpt_path, block_py, opt_py, {"init": "true"})

        # Input X: shape (2, 3, 16)
        x_data = rng.randn(2, 3, d_model).astype(np.float64)
        x_tensor = tensor(x_data, requires_grad=False)

        class DummyWrapper(Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = Parameter(x_data, requires_grad=False)

        dummy = DummyWrapper()
        save_checkpoint(x_ckpt_path, dummy)

        # 2. Run C++ transformer step
        cmd = [
            str(PARITY_BIN),
            "--transformer-step",
            str(in_ckpt_path),
            str(x_ckpt_path),
            str(cpp_out_ckpt_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "STATUS: OK" in proc.stdout

        # 3. Perform exact same step in Python
        opt_py.zero_grad()
        out_py, _ = block_py.forward_with_attention(x_tensor, mask=None, is_causal=True)
        loss_py = out_py.sum()
        loss_py.backward()
        opt_py.step()

        # 4. Load C++ checkpoint and verify parameter and moment parity
        cpp_loaded = load_checkpoint(cpp_out_ckpt_path)
        assert cpp_loaded["metadata"]["step"] == "1"
        assert cpp_loaded["optimizer_state_dict"]["step_count"] == 1

        py_params = dict(block_py.named_parameters())
        cpp_params = cpp_loaded["model_state_dict"]

        for name, p in py_params.items():
            assert name in cpp_params, f"Parameter '{name}' missing from C++ state dict"
            c_data = np.array(cpp_params[name]["data"]).reshape(cpp_params[name]["shape"])
            np.testing.assert_allclose(
                p.numpy(),
                c_data,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Discrepancy in updated parameter '{name}'",
            )

        # Verify AdamW momentum buffers
        cpp_opt_state = cpp_loaded["optimizer_state_dict"]["state"]

        for idx, (_, p) in enumerate(py_params.items()):
            key = str(idx)
            if key not in cpp_opt_state and idx in cpp_opt_state:
                key = idx  # type: ignore[assignment]
            assert key in cpp_opt_state, (
                f"Optimizer state for param {key} missing from C++ state dict"
            )

            cpp_m = np.array(cpp_opt_state[key]["exp_avg"]["data"]).reshape(
                cpp_opt_state[key]["exp_avg"]["shape"]
            )
            cpp_v = np.array(cpp_opt_state[key]["exp_avg_sq"]["data"]).reshape(
                cpp_opt_state[key]["exp_avg_sq"]["shape"]
            )

            py_m = opt_py.state[id(p)]["exp_avg"]
            py_v = opt_py.state[id(p)]["exp_avg_sq"]

            np.testing.assert_allclose(
                py_m,
                cpp_m,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Discrepancy in AdamW exp_avg for param index {idx}",
            )
            np.testing.assert_allclose(
                py_v,
                cpp_v,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Discrepancy in AdamW exp_avg_sq for param index {idx}",
            )
