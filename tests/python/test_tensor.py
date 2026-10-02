"""Unit tests for AURORA Tensor data structure and forward operations."""

import numpy as np
from aurora.tensor import arange, ones, randn, tensor, zeros


def test_tensor_creation() -> None:
    t = tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.ndim == 1
    assert t.size == 3
    assert t.dtype == np.float64
    assert not t.requires_grad
    assert np.allclose(t.numpy(), [1.0, 2.0, 3.0])


def test_zeros_ones_randn() -> None:
    z = zeros((2, 3))
    assert z.shape == (2, 3)
    assert np.all(z.numpy() == 0.0)

    o = ones((3, 2))
    assert o.shape == (3, 2)
    assert np.all(o.numpy() == 1.0)

    r = randn(4, 4)
    assert r.shape == (4, 4)


def test_reshape_and_transpose() -> None:
    a = arange(0.0, 6.0).reshape(2, 3)
    assert a.shape == (2, 3)

    a_t = a.transpose()
    assert a_t.shape == (3, 2)
    assert a_t.numpy()[0, 1] == a.numpy()[1, 0]


def test_arithmetic_elementwise() -> None:
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    y = tensor([[5.0, 6.0], [7.0, 8.0]])

    add_res = x + y
    assert np.allclose(add_res.numpy(), [[6.0, 8.0], [10.0, 12.0]])

    sub_res = x - y
    assert np.allclose(sub_res.numpy(), [[-4.0, -4.0], [-4.0, -4.0]])

    mul_res = x * y
    assert np.allclose(mul_res.numpy(), [[5.0, 12.0], [21.0, 32.0]])

    div_res = y / x
    assert np.allclose(div_res.numpy(), [[5.0, 3.0], [7.0 / 3.0, 2.0]])


def test_broadcasting_arithmetic() -> None:
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    y = tensor([10.0, 20.0, 30.0])  # (3,)

    res = x + y
    assert res.shape == (2, 3)
    assert np.allclose(res.numpy(), [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])


def test_matmul_2d_and_batched() -> None:
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    assert np.allclose(c.numpy(), [[19.0, 22.0], [43.0, 50.0]])

    # Batched matmul (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
    b_a = ones((2, 2, 3))
    b_b = ones((2, 3, 2))
    b_c = b_a @ b_b
    assert b_c.shape == (2, 2, 2)
    assert np.allclose(b_c.numpy(), 3.0)


def test_reductions() -> None:
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert x.sum().item() == 10.0
    assert x.mean().item() == 2.5

    sum_axis0 = x.sum(axis=0)
    assert np.allclose(sum_axis0.numpy(), [4.0, 6.0])

    sum_axis1_keepdims = x.sum(axis=1, keepdims=True)
    assert sum_axis1_keepdims.shape == (2, 1)
    assert np.allclose(sum_axis1_keepdims.numpy(), [[3.0], [7.0]])
