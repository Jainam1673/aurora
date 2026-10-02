"""Unit and integration tests for AURORA neural network layers and modules."""

import numpy as np
from aurora.nn.layers import (
    MLP,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    ResidualBlock,
    RMSNorm,
)
from aurora.tensor import ones, randn


def test_linear_forward_backward() -> None:
    np.random.seed(42)
    lin = Linear(4, 2, bias=True)
    x = randn(3, 4, requires_grad=True)

    y = lin(x)
    assert y.shape == (3, 2)

    loss = y.sum()
    loss.backward()

    assert lin.weight.grad is not None
    assert lin.bias is not None and lin.bias.grad is not None
    assert x.grad is not None
    assert lin.weight.grad.shape == (2, 4)
    assert lin.bias.grad.shape == (2,)


def test_embedding() -> None:
    emb = Embedding(num_embeddings=10, embedding_dim=8)
    indices = [1, 2, 1]  # duplicate index to test gradient accumulation
    out = emb(indices)
    assert out.shape == (3, 8)

    loss = out.sum()
    loss.backward()

    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == (10, 8)
    # Index 1 was queried twice, so gradient should be 2.0
    assert np.allclose(emb.weight.grad.numpy()[1], 2.0)
    # Index 2 was queried once, so gradient should be 1.0
    assert np.allclose(emb.weight.grad.numpy()[2], 1.0)
    # Unqueried indices should have 0.0 gradient
    assert np.allclose(emb.weight.grad.numpy()[0], 0.0)


def test_rmsnorm() -> None:
    norm = RMSNorm(dim=4)
    x = randn(2, 4, requires_grad=True)
    y = norm(x)
    assert y.shape == (2, 4)

    y.sum().backward()
    assert norm.weight.grad is not None
    assert x.grad is not None


def test_layernorm_module() -> None:
    ln = LayerNorm(normalized_shape=4)
    x = randn(2, 4, requires_grad=True)
    y = ln(x)
    assert y.shape == (2, 4)

    y.sum().backward()
    assert ln.weight is not None and ln.weight.grad is not None
    assert ln.bias is not None and ln.bias.grad is not None
    assert x.grad is not None


def test_dropout_train_vs_eval() -> None:
    drop = Dropout(p=0.5)
    x = ones((100, 100))

    drop.train()
    y_train = drop(x)
    # In training, ~50% elements should be 0.0 and rest scaled by 2.0
    zeros_ratio = np.mean(y_train.numpy() == 0.0)
    assert 0.4 < zeros_ratio < 0.6
    assert np.allclose(y_train.numpy().mean(), 1.0, atol=0.1)

    drop.eval()
    y_eval = drop(x)
    assert np.allclose(y_eval.numpy(), x.numpy())


def test_mlp_and_sequential() -> None:
    mlp = MLP(in_features=8, hidden_dims=[16, 16], out_features=4, activation="gelu")
    x = randn(5, 8, requires_grad=True)
    y = mlp(x)
    assert y.shape == (5, 4)

    y.sum().backward()
    assert x.grad is not None
    assert len(mlp.parameters()) == 6  # 3 linear layers * (weight + bias)


def test_residual_block() -> None:
    block = Linear(4, 4)
    res = ResidualBlock(block)
    x = ones((2, 4))
    y = res(x)
    assert y.shape == (2, 4)


def test_module_state_dict_load_and_save() -> None:
    lin1 = Linear(4, 2)
    state = lin1.state_dict()

    lin2 = Linear(4, 2)
    # Modify lin2
    lin2.weight.data.fill(0.0)
    assert not np.allclose(lin1.weight.numpy(), lin2.weight.numpy())

    lin2.load_state_dict(state)
    assert np.allclose(lin1.weight.numpy(), lin2.weight.numpy())
    if lin1.bias is not None and lin2.bias is not None:
        assert np.allclose(lin1.bias.numpy(), lin2.bias.numpy())
