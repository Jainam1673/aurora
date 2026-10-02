"""Transformer architecture and autoregressive decoders for AURORA."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from aurora.nn.attention import MultiHeadAttention
from aurora.nn.layers import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
    ReLU,
    RMSNorm,
    SiLU,
)
from aurora.nn.module import Module
from aurora.tensor import Tensor, tensor


class TransformerBlock(Module):
    """Transformer block with Pre-LayerNorm/RMSNorm, Multi-Head Attention, and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.norm_type = norm_type.lower()
        self.activation_name = activation.lower()
        self.dropout_p = dropout

        # Pre-attention normalization
        if self.norm_type == "layernorm":
            self.norm1: Module = LayerNorm(d_model)
        elif self.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Multi-Head Attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        self.dropout1 = Dropout(dropout) if dropout > 0.0 else None

        # Pre-FFN normalization
        if self.norm_type == "layernorm":
            self.norm2: Module = LayerNorm(d_model)
        elif self.norm_type == "rmsnorm":
            self.norm2 = RMSNorm(d_model)

        # Feed-Forward Network (FFN)
        self.linear1 = Linear(d_model, self.d_ff, bias=bias)
        match self.activation_name:
            case "gelu":
                self.act: Module = GELU()
            case "relu":
                self.act = ReLU()
            case "silu":
                self.act = SiLU()
            case _:
                raise ValueError(f"Unknown activation: {activation}")
        self.linear2 = Linear(self.d_ff, d_model, bias=bias)
        self.dropout2 = Dropout(dropout) if dropout > 0.0 else None

    def forward_with_attention(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning both transformed representations and attention weights."""
        # Pre-LN / RMSNorm + Self-Attention + Residual
        normed = cast(Tensor, self.norm1(x))
        attn_out, attn_weights = self.attn(
            normed, mask=mask, is_causal=is_causal
        )
        if self.dropout1 is not None:
            attn_out = self.dropout1(attn_out)
        x = x + attn_out

        # Pre-LN / RMSNorm + FFN + Residual
        normed2 = cast(Tensor, self.norm2(x))
        ffn_h = self.act(self.linear1(normed2))
        ffn_out = cast(Tensor, self.linear2(ffn_h))
        if self.dropout2 is not None:
            ffn_out = self.dropout2(ffn_out)
        x = x + ffn_out

        return x, attn_weights

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass through transformer block."""
        out, _ = self.forward_with_attention(x, mask=mask, is_causal=is_causal)
        return out


class TransformerDecoder(Module):
    """Autoregressive Transformer Decoder stack for world modeling and sequential RL."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int | None = None,
        in_dim: int | None = None,
        out_dim: int | None = None,
        max_seq_len: int = 1024,
        d_ff: int | None = None,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size is None and in_dim is None:
            raise ValueError("Either vocab_size or in_dim must be provided")

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.in_dim = in_dim
        self.max_seq_len = max_seq_len

        # Token embedding or continuous state projection
        if vocab_size is not None:
            self.token_emb: Module | None = Embedding(vocab_size, d_model)
            self.input_proj: Module | None = None
        else:
            self.token_emb = None
            assert in_dim is not None
            self.input_proj = Linear(in_dim, d_model, bias=bias)

        # Learned positional embeddings
        self.pos_emb = Embedding(max_seq_len, d_model)
        self.drop = Dropout(dropout) if dropout > 0.0 else None

        # Transformer blocks
        blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(num_layers)
        ]
        self.blocks = ModuleList(blocks)

        # Final normalization
        norm_t = norm_type.lower()
        if norm_t == "layernorm":
            self.final_norm: Module = LayerNorm(d_model)
        elif norm_t == "rmsnorm":
            self.final_norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Prediction head
        if out_dim is not None:
            target_out = out_dim
        elif vocab_size is not None:
            target_out = vocab_size
        elif in_dim is not None:
            target_out = in_dim
        else:
            target_out = d_model

        self.head = Linear(d_model, target_out, bias=bias)

    def forward(
        self,
        x: Tensor | np.ndarray | Sequence[Any],
        mask: Tensor | None = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass through decoder stack.

        Args:
            x: Discrete token indices of shape (B, T) or continuous tensor (B, T, in_dim).
            mask: Optional attention mask.
            is_causal: Whether to apply causal autoregressive masking (default: True).
        """
        if self.token_emb is not None:
            h = cast(Tensor, self.token_emb(x))
        else:
            assert self.input_proj is not None
            if not isinstance(x, Tensor):
                x = tensor(np.asarray(x, dtype=np.float64))
            h = cast(Tensor, self.input_proj(x))

        _b, t, _d = h.shape
        if t > self.max_seq_len:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len {self.max_seq_len}"
            )

        # Positional encoding injection
        positions = np.arange(t, dtype=np.int64)
        pos_enc = cast(Tensor, self.pos_emb(positions))
        h = h + pos_enc

        if self.drop is not None:
            h = self.drop(h)

        # Pass through transformer blocks
        for block in self.blocks:
            h = cast(TransformerBlock, block)(h, mask=mask, is_causal=is_causal)

        # Final norm and linear prediction head
        h = cast(Tensor, self.final_norm(h))
        logits = cast(Tensor, self.head(h))
        return logits

    def generate(
        self,
        prompt_tokens: Sequence[int] | np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Autoregressively generate new discrete tokens."""
        if self.token_emb is None:
            raise RuntimeError("generate() is only supported for discrete token models")

        tokens = np.array(prompt_tokens, dtype=np.int64).reshape(1, -1)
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            if tokens.shape[1] <= self.max_seq_len:
                cond_tokens = tokens
            else:
                cond_tokens = tokens[:, -self.max_seq_len:]
            logits = self.forward(cond_tokens, is_causal=True)
            # Take logits at last position: shape (1, vocab_size)
            last_logits = logits.numpy()[:, -1, :]
            if temperature > 0.0:
                scaled_logits = last_logits / temperature
                # Softmax
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                next_token = np.random.choice(probs.shape[-1], p=probs[0])
            else:
                next_token = int(np.argmax(last_logits, axis=-1)[0])

            tokens = np.concatenate([tokens, np.array([[next_token]], dtype=np.int64)], axis=1)

        return np.asarray(tokens[0])

