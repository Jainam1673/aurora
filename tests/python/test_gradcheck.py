"""Finite-difference gradient checks for all AURORA autograd primitives."""

from aurora.gradcheck import gradcheck
from aurora.tensor import randn, tensor


def test_gradcheck_add() -> None:
    # Vector
    a = tensor([1.5, 2.5, -3.2], requires_grad=True)
    b = tensor([0.5, -1.2, 4.0], requires_grad=True)
    assert gradcheck(lambda x, y: (x + y).sum(), [a, b])

    # Broadcasted (2, 3) + (3,)
    x = randn(2, 3, requires_grad=True)
    y = randn(3, requires_grad=True)
    assert gradcheck(lambda a, b: (a + b).sum(), [x, y])


def test_gradcheck_sub() -> None:
    a = randn(3, 4, requires_grad=True)
    b = randn(3, 4, requires_grad=True)
    assert gradcheck(lambda x, y: (x - y).sum(), [a, b])


def test_gradcheck_mul() -> None:
    a = randn(2, 3, requires_grad=True)
    b = randn(2, 3, requires_grad=True)
    assert gradcheck(lambda x, y: (x * y).sum(), [a, b])

    # Broadcasted
    c = randn(1, 3, requires_grad=True)
    assert gradcheck(lambda x, y: (x * y).sum(), [a, c])


def test_gradcheck_div() -> None:
    a = tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    b = tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=True)
    assert gradcheck(lambda x, y: (x / y).sum(), [a, b])


def test_gradcheck_matmul() -> None:
    # 2D matmul (3, 4) @ (4, 2)
    a = randn(3, 4, requires_grad=True)
    b = randn(4, 2, requires_grad=True)
    assert gradcheck(lambda x, y: (x @ y).sum(), [a, b])

    # Batched matmul (2, 3, 4) @ (2, 4, 2)
    ba = randn(2, 3, 4, requires_grad=True)
    bb = randn(2, 4, 2, requires_grad=True)
    assert gradcheck(lambda x, y: (x @ y).sum(), [ba, bb])


def test_gradcheck_sum_mean() -> None:
    x = randn(3, 4, requires_grad=True)
    assert gradcheck(lambda a: a.sum(), [x])
    assert gradcheck(lambda a: a.sum(axis=0).sum(), [x])
    assert gradcheck(lambda a: a.sum(axis=1, keepdims=True).sum(), [x])

    assert gradcheck(lambda a: a.mean(), [x])
    assert gradcheck(lambda a: a.mean(axis=0).sum(), [x])
    assert gradcheck(lambda a: a.mean(axis=1, keepdims=True).sum(), [x])


def test_gradcheck_reshape_transpose() -> None:
    x = randn(2, 3, 4, requires_grad=True)
    assert gradcheck(lambda a: a.reshape(6, 4).sum(), [x])
    assert gradcheck(lambda a: a.transpose(1, 0, 2).sum(), [x])


def test_gradcheck_exp_log_sqrt() -> None:
    # Keep strictly positive for log and sqrt
    x = tensor([1.2, 2.5, 3.1, 0.8], requires_grad=True)
    assert gradcheck(lambda a: a.exp().sum(), [x])
    assert gradcheck(lambda a: a.log().sum(), [x])
    assert gradcheck(lambda a: a.sqrt().sum(), [x])


def test_gradcheck_relu() -> None:
    # Avoid testing at exactly 0.0 where subgradient is non-smooth
    x = tensor([-2.5, -1.0, 0.5, 1.8, 3.2], requires_grad=True)
    assert gradcheck(lambda a: a.relu().sum(), [x])


def test_gradcheck_gelu() -> None:
    x = tensor([-1.5, -0.5, 0.0, 0.8, 2.1], requires_grad=True)
    assert gradcheck(lambda a: a.gelu().sum(), [x])


def test_gradcheck_silu() -> None:
    x = tensor([-2.0, -0.5, 0.0, 1.0, 2.5], requires_grad=True)
    assert gradcheck(lambda a: a.silu().sum(), [x])


def test_gradcheck_softmax() -> None:
    x = randn(2, 4, requires_grad=True)
    # Test scalar reduction of weighted softmax
    w = randn(2, 4)
    assert gradcheck(lambda a: (a.softmax(axis=-1) * w).sum(), [x])


def test_gradcheck_log_softmax() -> None:
    x = randn(3, 5, requires_grad=True)
    w = randn(3, 5)
    assert gradcheck(lambda a: (a.log_softmax(axis=-1) * w).sum(), [x])


def test_gradcheck_layer_norm() -> None:
    x = randn(2, 4, requires_grad=True)
    gamma = randn(4, requires_grad=True)
    beta = randn(4, requires_grad=True)
    assert gradcheck(lambda a, g, b: a.layer_norm(gamma=g, beta=b).sum(), [x, gamma, beta])
