"""Cross-Language Checkpoint and Optimization Parity Test Suite for AURORA.

Verifies end-to-end mathematical equivalence between Python and C++23 native
model layers, autograd backward pass, AdamW decoupled weight decay updates,
and bidirectional checkpoint JSON serialization/deserialization.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from aurora.checkpoint import load_checkpoint, save_checkpoint
from aurora.nn.layers import MLP
from aurora.nn.module import Module
from aurora.optim.adamw import AdamW
from aurora.tensor import tensor

PARITY_BIN = Path(__file__).parents[2] / "build" / "debug" / "aurora_parity_runner"


def test_cross_language_checkpoint_and_optimization_parity() -> None:
    """Verify bidirectional checkpoint loading, forward pass, autograd, and AdamW parity."""
    if not PARITY_BIN.exists():
        pytest.skip(f"Parity runner binary not found at {PARITY_BIN}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_ckpt_path = Path(tmpdir) / "in_ckpt.json"
        x_ckpt_path = Path(tmpdir) / "x_ckpt.json"
        cpp_out_ckpt_path = Path(tmpdir) / "cpp_out_ckpt.json"
        py_out_ckpt_path = Path(tmpdir) / "py_out_ckpt.json"

        # 1. Initialize Python model and optimizer with deterministic values
        model_py = MLP(4, [8], 2, activation="relu", dropout=0.0)
        opt_py = AdamW(
            model_py.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

        # Set deterministic weights
        rng = np.random.RandomState(42)
        for _, p in model_py.named_parameters():
            p.data = rng.randn(*p.shape).astype(np.float64) * 0.5

        # 2. Save checkpoint from Python
        save_checkpoint(in_ckpt_path, model_py, opt_py, {"init": "true"})

        # Save input X into a checkpoint
        x_data = rng.randn(3, 4).astype(np.float64)
        x_tensor = tensor(x_data, requires_grad=False)

        class DummyWrapper(Module):
            def __init__(self) -> None:
                super().__init__()
                from aurora.nn.parameter import Parameter

                self.x = Parameter(x_data, requires_grad=False)

        dummy = DummyWrapper()
        save_checkpoint(x_ckpt_path, dummy)

        # 3. Run C++ checkpoint step via aurora_parity_runner
        cmd = [
            str(PARITY_BIN),
            "--checkpoint-step",
            str(in_ckpt_path),
            str(x_ckpt_path),
            str(cpp_out_ckpt_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "STATUS: OK" in proc.stdout

        # 4. Perform the exact same step in Python
        opt_py.zero_grad()
        pred_py = model_py(x_tensor)
        loss_py = pred_py.sum()
        loss_py.backward()
        opt_py.step()

        save_checkpoint(py_out_ckpt_path, model_py, opt_py, {"step": "1"})

        # 5. Load C++ checkpoint into Python and compare
        cpp_loaded = load_checkpoint(cpp_out_ckpt_path)

        assert cpp_loaded["metadata"]["step"] == "1"
        assert cpp_loaded["optimizer_state_dict"]["step_count"] == 1

        py_params = dict(model_py.named_parameters())
        cpp_params = cpp_loaded["model_state_dict"]

        # Check every weight and bias matches with < 1e-10 difference!
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

        # Check optimizer state moments exp_avg and exp_avg_sq match with < 1e-10 difference!
        cpp_opt_state = cpp_loaded["optimizer_state_dict"]["state"]
        for idx, p in enumerate(model_py.parameters()):
            key = str(idx)
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
                err_msg=f"Moment exp_avg mismatch for param {idx}",
            )
            np.testing.assert_allclose(
                py_v,
                cpp_v,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Moment exp_avg_sq mismatch for param {idx}",
            )


def test_multi_step_optimization_parity() -> None:
    """Verify multiple consecutive steps of forward/backward/AdamW stay in mathematical lockstep."""
    if not PARITY_BIN.exists():
        pytest.skip(f"Parity runner binary not found at {PARITY_BIN}")

    with tempfile.TemporaryDirectory() as tmpdir:
        current_ckpt = Path(tmpdir) / "ckpt_step0.json"

        # Initialize Python model and AdamW
        model_py = MLP(4, [8], 2, activation="relu", dropout=0.0)
        opt_py = AdamW(
            model_py.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

        rng = np.random.RandomState(1337)
        for _, p in model_py.named_parameters():
            p.data = rng.randn(*p.shape).astype(np.float64) * 0.2

        save_checkpoint(current_ckpt, model_py, opt_py, {"step": "0"})

        # Run 3 consecutive steps through C++ and Python
        for step in range(1, 4):
            x_step_path = Path(tmpdir) / f"x_step_{step}.json"
            next_ckpt = Path(tmpdir) / f"ckpt_step_{step}.json"

            x_batch = rng.randn(4, 4).astype(np.float64)
            x_t = tensor(x_batch, requires_grad=False)

            class Wrapper(Module):
                def __init__(self, data: np.ndarray) -> None:
                    super().__init__()
                    from aurora.nn.parameter import Parameter

                    self.x = Parameter(data, requires_grad=False)

            save_checkpoint(x_step_path, Wrapper(x_batch))

            # Step in C++
            cmd = [
                str(PARITY_BIN),
                "--checkpoint-step",
                str(current_ckpt),
                str(x_step_path),
                str(next_ckpt),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Step in Python
            opt_py.zero_grad()
            pred = model_py(x_t)
            loss = pred.sum()
            loss.backward()
            opt_py.step()

            # Verify parity after this step
            cpp_ckpt = load_checkpoint(next_ckpt)
            for name, p in model_py.named_parameters():
                c_data = np.array(cpp_ckpt["model_state_dict"][name]["data"]).reshape(p.shape)
                np.testing.assert_allclose(p.numpy(), c_data, rtol=1e-10, atol=1e-10)

            current_ckpt = next_ckpt
