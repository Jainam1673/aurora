"""Benchmark harness for Scaled Dot-Product Attention, MHA, and Transformer Decoders."""

from __future__ import annotations

import time

import numpy as np

from aurora.nn.attention import MultiHeadAttention, scaled_dot_product_attention
from aurora.nn.transformer import TransformerBlock, TransformerDecoder
from aurora.tensor import tensor


def benchmark_scaled_dot_product_attention() -> None:
    print("\n--- Benchmarking Scaled Dot-Product Attention ---")
    b, h, d_k = 4, 8, 64
    seq_lens = [64, 128, 256, 512]
    num_warmup = 3
    num_iters = 10

    for t in seq_lens:
        q = tensor(np.random.randn(b, h, t, d_k), requires_grad=True)
        k = tensor(np.random.randn(b, h, t, d_k), requires_grad=True)
        v = tensor(np.random.randn(b, h, t, d_k), requires_grad=True)

        # Warmup
        for _ in range(num_warmup):
            out, _ = scaled_dot_product_attention(q, k, v)
            loss = out.sum()
            loss.backward()
            q.zero_grad()
            k.zero_grad()
            v.zero_grad()

        # Timed forward + backward
        t0 = time.perf_counter()
        for _ in range(num_iters):
            out, _ = scaled_dot_product_attention(q, k, v)
            loss = out.sum()
            loss.backward()
            q.zero_grad()
            k.zero_grad()
            v.zero_grad()
        t1 = time.perf_counter()

        elapsed_per_iter = (t1 - t0) / num_iters
        tokens_per_sec = (b * t) / elapsed_per_iter
        print(
            f"Seq len {t:3d}: {elapsed_per_iter * 1000.0:7.2f} ms/iter | "
            f"Throughput: {tokens_per_sec:9.1f} tokens/s"
        )


def benchmark_transformer_block() -> None:
    print("\n--- Benchmarking Transformer Block (Pre-LN, 4 heads, d_model=128) ---")
    b, d_model, num_heads = 4, 128, 4
    seq_lens = [32, 64, 128]
    num_warmup = 2
    num_iters = 5

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=256,
        norm_type="layernorm",
        activation="gelu",
    )

    for t in seq_lens:
        x = tensor(np.random.randn(b, t, d_model), requires_grad=True)

        for _ in range(num_warmup):
            out = block(x, is_causal=True)
            loss = out.sum()
            loss.backward()
            block.zero_grad()
            x.zero_grad()

        t0 = time.perf_counter()
        for _ in range(num_iters):
            out = block(x, is_causal=True)
            loss = out.sum()
            loss.backward()
            block.zero_grad()
            x.zero_grad()
        t1 = time.perf_counter()

        elapsed_per_iter = (t1 - t0) / num_iters
        tokens_per_sec = (b * t) / elapsed_per_iter
        print(
            f"Seq len {t:3d}: {elapsed_per_iter * 1000.0:7.2f} ms/iter | "
            f"Throughput: {tokens_per_sec:9.1f} tokens/s"
        )


def benchmark_transformer_decoder_generation() -> None:
    print("\n--- Benchmarking Transformer Decoder Generation ---")
    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 4

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=128,
    )
    prompt = [1, 2, 3, 4]
    max_new_tokens = 20

    t0 = time.perf_counter()
    tokens = decoder.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tok_per_sec = max_new_tokens / elapsed
    print(f"Generated {len(tokens)} tokens in {elapsed * 1000.0:.2f} ms ({tok_per_sec:.1f} tokens/s)")


if __name__ == "__main__":
    benchmark_scaled_dot_product_attention()
    benchmark_transformer_block()
    benchmark_transformer_decoder_generation()
