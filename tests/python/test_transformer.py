"""Tests for attention mechanisms and Transformer architectures."""

from __future__ import annotations

import numpy as np
from aurora.nn.attention import (
    MultiHeadAttention,
    apply_rotary_pos_emb,
    create_causal_mask,
    scaled_dot_product_attention,
)
from aurora.nn.transformer import (
    TransformerBlock,
    TransformerDecoder,
)
from aurora.optim.adamw import AdamW
from aurora.tensor import tensor


def test_create_causal_mask() -> None:
    mask = create_causal_mask(4)
    data = mask.numpy()
    assert data.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert data[i, j] == 0.0
            else:
                assert data[i, j] == -1e9


def test_scaled_dot_product_attention_shapes_and_values() -> None:
    b, h, t, d_k = 2, 3, 5, 8
    q = tensor(np.random.randn(b, h, t, d_k), requires_grad=True)
    k = tensor(np.random.randn(b, h, t, d_k), requires_grad=True)
    v = tensor(np.random.randn(b, h, t, d_k), requires_grad=True)

    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, h, t, d_k)
    assert weights.shape == (b, h, t, t)

    # Weights must sum to 1 along the key dimension
    w_sum = np.sum(weights.numpy(), axis=-1)
    np.testing.assert_allclose(w_sum, np.ones_like(w_sum), atol=1e-6)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not np.isnan(q.grad.numpy()).any()
    assert not np.isnan(k.grad.numpy()).any()
    assert not np.isnan(v.grad.numpy()).any()


def test_scaled_dot_product_attention_causal_mask() -> None:
    t = 4
    d_k = 4
    q = tensor(np.random.randn(1, 1, t, d_k))
    k = tensor(np.random.randn(1, 1, t, d_k))
    v = tensor(np.random.randn(1, 1, t, d_k))
    mask = create_causal_mask(t)

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    w = weights.numpy()[0, 0]
    for i in range(t):
        for j in range(t):
            if j > i:
                # Masked positions must have essentially 0 attention probability
                assert w[i, j] < 1e-6


def test_rotary_pos_emb_preserves_norm() -> None:
    # RoPE applies 2D planar rotations, which must strictly preserve vector Euclidean norms
    seq_len = 6
    d_k = 8
    x_data = np.random.randn(2, seq_len, d_k)
    x = tensor(x_data)
    x_rot = apply_rotary_pos_emb(x, seq_dim=1)

    norm_orig = np.linalg.norm(x.numpy(), axis=-1)
    norm_rot = np.linalg.norm(x_rot.numpy(), axis=-1)
    np.testing.assert_allclose(norm_rot, norm_orig, atol=1e-9)


def test_multi_head_attention_forward_backward() -> None:
    b, t, d_model = 2, 4, 16
    num_heads = 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, bias=True)

    x = tensor(np.random.randn(b, t, d_model), requires_grad=True)
    out, weights = mha(x, is_causal=True)

    assert out.shape == (b, t, d_model)
    assert weights.shape == (b, num_heads, t, t)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    for p in mha.parameters():
        assert p.grad is not None
        assert not np.isnan(p.grad.numpy()).any()


def test_transformer_block_forward_backward() -> None:
    b, t, d_model = 2, 4, 16
    num_heads = 2
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=32,
        norm_type="layernorm",
        activation="gelu",
    )

    x = tensor(np.random.randn(b, t, d_model), requires_grad=True)
    out, attn_weights = block.forward_with_attention(x, is_causal=True)

    assert out.shape == (b, t, d_model)
    assert attn_weights.shape == (b, num_heads, t, t)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    for p in block.parameters():
        assert p.grad is not None
        assert not np.isnan(p.grad.numpy()).any()


def test_transformer_block_rmsnorm() -> None:
    b, t, d_model = 2, 4, 16
    block = TransformerBlock(
        d_model=d_model,
        num_heads=2,
        d_ff=32,
        norm_type="rmsnorm",
        activation="silu",
    )
    x = tensor(np.random.randn(b, t, d_model))
    out = block(x)
    assert out.shape == (b, t, d_model)


def test_transformer_decoder_discrete_and_continuous() -> None:
    # 1. Discrete token decoder
    vocab_size = 20
    d_model = 16
    decoder_discrete = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    tokens = np.array([[1, 4, 9, 2], [0, 3, 5, 8]])
    logits = decoder_discrete(tokens)
    assert logits.shape == (2, 4, vocab_size)

    # Autoregressive generation
    prompt = [1, 2]
    generated = decoder_discrete.generate(prompt, max_new_tokens=3, temperature=0.0)
    assert len(generated) == 5
    assert list(generated[:2]) == prompt

    # 2. Continuous input decoder (for world model trajectories)
    in_dim = 8
    out_dim = 8
    decoder_continuous = TransformerDecoder(
        in_dim=in_dim,
        out_dim=out_dim,
        d_model=d_model,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    states = tensor(np.random.randn(2, 6, in_dim))
    pred_states = decoder_continuous(states)
    assert pred_states.shape == (2, 6, out_dim)


def test_transformer_decoder_optimization_step() -> None:
    # Test end-to-end training step on discrete sequence modeling
    np.random.seed(42)
    vocab_size = 10
    d_model = 16
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    opt = AdamW(decoder.parameters(), lr=1e-3)

    tokens = np.array([[1, 2, 3, 4, 5]])
    target = np.array([[2, 3, 4, 5, 6]])

    # Forward
    logits = decoder(tokens)
    # Simple MSE-like loss on logits for test purpose
    target_one_hot = np.zeros_like(logits.numpy())
    for t_idx, val in enumerate(target[0]):
        target_one_hot[0, t_idx, val] = 1.0

    target_t = tensor(target_one_hot)
    diff = logits - target_t
    loss = (diff * diff).mean()

    initial_loss = loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Second step should reduce loss
    logits2 = decoder(tokens)
    diff2 = logits2 - target_t
    loss2 = (diff2 * diff2).mean()

    assert loss2.item() < initial_loss
