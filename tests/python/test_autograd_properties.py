"""Property-based tests for AURORA autograd and tensor engine using Hypothesis."""

import hypothesis.strategies as st
import numpy as np
from aurora.tensor import tensor
from hypothesis import given, settings


@settings(max_examples=50)
@given(
    st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    )
)
def test_softmax_sums_to_one(vals: list[float]) -> None:
    t = tensor(vals)
    s = t.softmax(axis=-1)
    total = float(s.sum().item())
    assert np.isclose(total, 1.0, atol=1e-6)


@settings(max_examples=50)
@given(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_addition_commutativity_and_gradient(val1: float, val2: float) -> None:
    x1 = tensor([val1], requires_grad=True)
    y1 = tensor([val2], requires_grad=True)
    z1 = x1 + y1
    z1.backward()

    x2 = tensor([val1], requires_grad=True)
    y2 = tensor([val2], requires_grad=True)
    z2 = y2 + x2
    z2.backward()

    assert np.isclose(z1.item(), z2.item())
    assert x1.grad is not None and x2.grad is not None
    assert np.isclose(x1.grad.item(), x2.grad.item())
    assert y1.grad is not None and y2.grad is not None
    assert np.isclose(y1.grad.item(), y2.grad.item())


@settings(max_examples=30)
@given(
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=2, max_value=6),
)
def test_transpose_involution(rows: int, cols: int) -> None:
    arr = np.random.randn(rows, cols)
    t = tensor(arr)
    t_tt = t.transpose().transpose()
    assert t_tt.shape == t.shape
    assert np.allclose(t_tt.numpy(), t.numpy())
