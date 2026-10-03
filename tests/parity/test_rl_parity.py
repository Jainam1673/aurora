"""Cross-Language Numerical Parity Test Suite for RL Primitives.

Verifies strict mathematical equivalence (< 1e-10 error) between Python
and C++23 native implementations of Tanh, Concat, and Policy Distributions.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from aurora.distributions.categorical import Categorical
from aurora.distributions.normal import Normal
from aurora.distributions.tanh_normal import TanhNormal
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


class TestRLNumericalParity:
    def test_tanh_parity(self) -> None:
        np.random.seed(42)
        x_data = np.random.randn(3, 4)
        x_py = tensor(x_data, requires_grad=True)

        y_py = x_py.tanh()
        loss_py = y_py.sum()
        loss_py.backward()

        y_cpp, grads_cpp = run_cpp_parity("tanh", [x_py])
        assert x_py.grad is not None
        fwd_err = float(np.max(np.abs(y_py.numpy() - y_cpp)))
        bwd_err = float(np.max(np.abs(x_py.grad.numpy() - grads_cpp[0])))

        assert fwd_err < 1e-10, f"Tanh forward parity violated: err={fwd_err}"
        assert bwd_err < 1e-10, f"Tanh backward parity violated: err={bwd_err}"

    def test_concat_parity(self) -> None:
        np.random.seed(123)
        a_data = np.random.randn(2, 3)
        b_data = np.random.randn(2, 4)

        a_py = tensor(a_data, requires_grad=True)
        b_py = tensor(b_data, requires_grad=True)

        from aurora.tensor import concat

        out_py = concat([a_py, b_py], axis=-1)
        loss = out_py.sum()
        loss.backward()

        out_cpp, grads_cpp = run_cpp_parity("concat", [a_py, b_py], axis=-1)

        assert a_py.grad is not None
        assert b_py.grad is not None
        fwd_err = float(np.max(np.abs(out_py.numpy() - out_cpp)))
        bwd_err_a = float(np.max(np.abs(a_py.grad.numpy() - grads_cpp[0])))
        bwd_err_b = float(np.max(np.abs(b_py.grad.numpy() - grads_cpp[1])))

        assert fwd_err < 1e-10, f"Concat forward parity violated: err={fwd_err}"
        assert bwd_err_a < 1e-10, f"Concat grad A parity violated: err={bwd_err_a}"
        assert bwd_err_b < 1e-10, f"Concat grad B parity violated: err={bwd_err_b}"

    def test_categorical_distribution_parity(self) -> None:
        logits_data = np.array([[1.0, 2.0, -1.0, 0.5], [-0.5, 0.0, 1.5, 2.5]])
        actions_data = np.array([1.0, 3.0])

        logits_py = tensor(logits_data, requires_grad=False)
        actions_py = tensor(actions_data, requires_grad=False)

        dist_py = Categorical(logits=logits_py)
        lp_py = dist_py.log_prob(actions_py)
        ent_py = dist_py.entropy()

        lp_cpp, _ = run_cpp_parity("categorical_log_prob", [logits_py, actions_py])
        ent_cpp, _ = run_cpp_parity("categorical_entropy", [logits_py])

        lp_err = float(np.max(np.abs(lp_py.numpy() - lp_cpp)))
        ent_err = float(np.max(np.abs(ent_py.numpy() - ent_cpp)))

        assert lp_err < 1e-10, f"Categorical log_prob parity violated: err={lp_err}"
        assert ent_err < 1e-10, f"Categorical entropy parity violated: err={ent_err}"

    def test_normal_distribution_parity(self) -> None:
        loc_data = np.array([[0.0, 1.0], [2.0, -1.0]])
        scale_data = np.array([[1.0, 2.0], [0.5, 1.5]])
        val_data = np.array([[0.2, 0.8], [1.8, -0.7]])

        loc_py = tensor(loc_data, requires_grad=False)
        scale_py = tensor(scale_data, requires_grad=False)
        val_py = tensor(val_data, requires_grad=False)

        dist_py = Normal(loc=loc_py, scale=scale_py)
        lp_py = dist_py.log_prob(val_py)
        ent_py = dist_py.entropy()

        lp_cpp, _ = run_cpp_parity("normal_log_prob", [loc_py, scale_py, val_py])
        ent_cpp, _ = run_cpp_parity("normal_entropy", [loc_py, scale_py])

        lp_err = float(np.max(np.abs(lp_py.numpy() - lp_cpp)))
        ent_err = float(np.max(np.abs(ent_py.numpy() - ent_cpp)))

        assert lp_err < 1e-10, f"Normal log_prob parity violated: err={lp_err}"
        assert ent_err < 1e-10, f"Normal entropy parity violated: err={ent_err}"

    def test_tanh_normal_distribution_parity(self) -> None:
        loc_data = np.array([[0.0, 0.5], [-0.5, 0.2]])
        scale_data = np.array([[1.0, 1.2], [0.8, 1.5]])
        act_data = np.array([[0.3, -0.4], [-0.2, 0.7]])

        loc_py = tensor(loc_data, requires_grad=False)
        scale_py = tensor(scale_data, requires_grad=False)
        act_py = tensor(act_data, requires_grad=False)

        dist_py = TanhNormal(loc=loc_py, scale=scale_py)
        lp_py = dist_py.log_prob(act_py)

        lp_cpp, _ = run_cpp_parity("tanh_normal_log_prob", [loc_py, scale_py, act_py])

        lp_err = float(np.max(np.abs(lp_py.numpy() - lp_cpp)))
        assert lp_err < 1e-10, f"TanhNormal log_prob parity violated: err={lp_err}"

    def test_gae_parity(self) -> None:
        from aurora.rl.buffers import RolloutBuffer

        np.random.seed(42)
        t_steps = 10
        rewards_data = np.random.randn(t_steps)
        values_data = np.random.randn(t_steps)
        dones_data = np.zeros(t_steps, dtype=np.float64)
        dones_data[4] = 1.0  # Terminal state mid-rollout
        last_v = float(np.random.randn())
        last_d = 0.0

        # Python GAE
        buf_py = RolloutBuffer(buffer_size=t_steps, obs_shape=(1,), action_shape=(1,), batch_size=1)
        for t in range(t_steps):
            buf_py.add(
                np.array([0.0]),
                np.array([0.0]),
                rewards_data[t],
                bool(dones_data[t]),
                values_data[t],
                0.0,
            )
        buf_py.compute_returns_and_advantages(last_v, bool(last_d), gamma=0.99, gae_lambda=0.95)
        adv_py = buf_py.advantages.flatten()

        # C++ GAE
        r_t = tensor(rewards_data, requires_grad=False)
        v_t = tensor(values_data, requires_grad=False)
        d_t = tensor(dones_data, requires_grad=False)
        lv_t = tensor([last_v], requires_grad=False)
        ld_t = tensor([last_d], requires_grad=False)
        gam_t = tensor([0.99], requires_grad=False)
        lam_t = tensor([0.95], requires_grad=False)

        adv_cpp, _ = run_cpp_parity("gae", [r_t, v_t, d_t, lv_t, ld_t, gam_t, lam_t])

        max_err = float(np.max(np.abs(adv_py - adv_cpp.flatten())))
        assert max_err < 1e-10, f"GAE computation parity violated: err={max_err}"
