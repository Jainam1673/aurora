"""Attention mechanisms and Multi-Head Attention for AURORA."""

from __future__ import annotations

import math

import numpy as np

from aurora.nn.layers import Linear
from aurora.nn.module import Module
from aurora.tensor import Tensor, tensor


def create_causal_mask(seq_len: int) -> Tensor:
    """Create additive upper-triangular causal attention mask.

    Positions where j > i receive -1e9, while valid positions j <= i receive 0.0.
    Shape: (seq_len, seq_len).
    """
    mask_data = np.full((seq_len, seq_len), -1e9, dtype=np.float64)
    # Lower triangular including diagonal is 0.0
    tril_indices = np.tril_indices(seq_len)
    mask_data[tril_indices] = 0.0
    return tensor(mask_data, requires_grad=False)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Scaled Dot-Product Attention from first principles.

    Args:
        q: Query tensor of shape (..., T_q, d_k)
        k: Key tensor of shape (..., T_k, d_k)
        v: Value tensor of shape (..., T_k, d_v)
        mask: Optional additive attention mask broadcastable to (..., T_q, T_k)
        scale: Scaling factor. Defaults to 1 / sqrt(d_k)

    Returns:
        output: Attended values of shape (..., T_q, d_v)
        attention_weights: Normalized attention weights of shape (..., T_q, T_k)
    """
    d_k = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(float(d_k))

    # Logits: S = (Q @ K^T) * scale
    k_t = k.mT
    scores = (q @ k_t) * scale

    if mask is not None:
        scores = scores + mask

    # Attention weights: Softmax over the key sequence dimension
    attn_weights = scores.softmax(axis=-1)

    # Output: O = A @ V
    output = attn_weights @ v
    return output, attn_weights


def apply_rotary_pos_emb(x: Tensor, seq_dim: int = -2) -> Tensor:
    """Apply Rotary Position Embeddings (RoPE) to tensor coordinates.

    Rotates adjacent pairs of feature dimensions according to their position index.
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Feature dimension must be even for RoPE, got {d}")

    seq_len = x.shape[seq_dim]
    positions = np.arange(seq_len, dtype=np.float64)
    dim_indices = np.arange(0, d, 2, dtype=np.float64)
    inv_freq = 1.0 / (10000.0 ** (dim_indices / d))

    # Outer product: (seq_len, d // 2)
    angles = np.outer(positions, inv_freq)
    sin = np.sin(angles)
    cos = np.cos(angles)

    x_np = x.numpy()
    x_rot = np.zeros_like(x_np)

    # Slice coordinates
    x1 = x_np[..., 0::2]
    x2 = x_np[..., 1::2]

    # Expand sin and cos to match x dimensions
    expand_dims = [1] * x.ndim
    expand_dims[seq_dim] = seq_len
    expand_dims[-1] = d // 2

    sin_exp = sin.reshape(expand_dims)
    cos_exp = cos.reshape(expand_dims)

    x_rot[..., 0::2] = x1 * cos_exp - x2 * sin_exp
    x_rot[..., 1::2] = x1 * sin_exp + x2 * cos_exp

    return tensor(x_rot, requires_grad=x.requires_grad)


class MultiHeadAttention(Module):
    """Multi-Head Attention (MHA) block from first principles."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(float(self.d_k))

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Compute multi-head attention.

        Args:
            q: Query sequence tensor of shape (B, T_q, d_model)
            k: Key sequence tensor (defaults to q for self-attention)
            v: Value sequence tensor (defaults to k for self-attention)
            mask: Optional additive mask
            is_causal: If True, applies lower-triangular causal mask

        Returns:
            out: Projected output of shape (B, T_q, d_model)
            attn_weights: Attention weights of shape (B, num_heads, T_q, T_k)
        """
        if k is None:
            k = q
        if v is None:
            v = k

        b, t_q, _ = q.shape
        _, t_k, _ = k.shape

        # 1. Linear projections
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # 2. Reshape and transpose to (B, num_heads, T, d_k)
        q_heads = q_proj.reshape(b, t_q, self.num_heads, self.d_k).swapaxes(1, 2)
        k_heads = k_proj.reshape(b, t_k, self.num_heads, self.d_k).swapaxes(1, 2)
        v_heads = v_proj.reshape(b, t_k, self.num_heads, self.d_k).swapaxes(1, 2)

        # 3. Handle causal masking
        attn_mask = mask
        if is_causal:
            causal_m = create_causal_mask(t_q)
            attn_mask = causal_m if mask is None else mask + causal_m

        # 4. Scaled dot-product attention
        out_heads, attn_weights = scaled_dot_product_attention(
            q_heads, k_heads, v_heads, mask=attn_mask, scale=self.scale
        )

        # 5. Concatenate heads and out project: (B, num_heads, T_q, d_k) -> (B, T_q, d_model)
        out_concat = out_heads.swapaxes(1, 2).reshape(b, t_q, self.d_model)
        out = self.out_proj(out_concat)

        return out, attn_weights
