"""Finite-difference numerical gradient checking utility for AURORA."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from aurora.tensor import Tensor


def gradcheck(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Validate analytical gradients against two-sided finite differences.

    Args:
        func: Callable taking inputs and returning a scalar Tensor.
        inputs: List of input Tensors with requires_grad=True.
        eps: Finite difference perturbation step size.
        atol: Absolute error tolerance.
        rtol: Relative error tolerance.

    Returns:
        True if all gradients match within tolerance.

    Raises:
        AssertionError: If analytical and numerical gradients mismatch.
    """
    # 1. Compute analytical gradients via autograd
    # Zero all previous grads
    for inp in inputs:
        inp.zero_grad()

    out = func(*inputs)
    if out.size != 1:
        raise ValueError(f"gradcheck func must produce a scalar output, got shape {out.shape}")

    out.backward()

    # 2. Compute numerical gradients via central finite difference
    for idx, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue

        if inp.grad is None:
            raise AssertionError(f"Input {idx} has requires_grad=True but its grad is None")

        analytical_grad = inp.grad.numpy().copy()
        numerical_grad = np.zeros_like(analytical_grad)

        orig_data = inp.data.copy()
        it = np.nditer(orig_data, flags=["multi_index"])

        while not it.finished:
            multi_idx = it.multi_index

            # f(x + eps)
            inp.data[multi_idx] = orig_data[multi_idx] + eps
            out_pos = func(*inputs).item()

            # f(x - eps)
            inp.data[multi_idx] = orig_data[multi_idx] - eps
            out_neg = func(*inputs).item()

            # Central difference
            numerical_grad[multi_idx] = (out_pos - out_neg) / (2.0 * eps)

            # Restore original data
            inp.data[multi_idx] = orig_data[multi_idx]

            it.iternext()

        # Compare analytical vs numerical
        diff = np.abs(analytical_grad - numerical_grad)
        denom = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad)) + 1e-8
        rel_diff = diff / denom

        max_abs = float(np.max(diff))
        max_rel = float(np.max(rel_diff))

        mismatch_mask = (diff > atol) & (rel_diff > rtol)
        if np.any(mismatch_mask):
            bad_indices = np.argwhere(mismatch_mask)
            first_bad = tuple(bad_indices[0])
            msg = (
                f"Gradient check failed for input {idx} at index {first_bad}:\n"
                f"  Analytical: {analytical_grad[first_bad]:.8e}\n"
                f"  Numerical:  {numerical_grad[first_bad]:.8e}\n"
                f"  Abs Error:  {diff[first_bad]:.8e} (tolerance: {atol})\n"
                f"  Rel Error:  {rel_diff[first_bad]:.8e} (tolerance: {rtol})\n"
                f"  Max Abs:    {max_abs:.8e}\n"
                f"  Max Rel:    {max_rel:.8e}"
            )
            raise AssertionError(msg)

    return True
