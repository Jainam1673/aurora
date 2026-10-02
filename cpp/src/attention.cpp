#include "aurora/attention.hpp"

#include <cmath>
#include <format>
#include <stdexcept>
#include <vector>

namespace aurora {

std::shared_ptr<Tensor> create_causal_mask(size_t seq_len) {
    std::vector<double> mask_data(seq_len * seq_len, 0.0);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            if (j > i) {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
    }
    return std::make_shared<Tensor>(std::vector<size_t>{seq_len, seq_len}, std::move(mask_data), false);
}

std::shared_ptr<Tensor> apply_rotary_pos_emb(const std::shared_ptr<Tensor>& x, int seq_dim) {
    size_t ndim = x->ndim();
    int s_dim = (seq_dim >= 0) ? seq_dim : static_cast<int>(ndim) + seq_dim;
    if (s_dim < 0 || s_dim >= static_cast<int>(ndim)) {
        throw std::out_of_range("seq_dim out of range for apply_rotary_pos_emb");
    }

    size_t d = x->shape().back();
    if (d % 2 != 0) {
        throw std::invalid_argument(std::format("Feature dimension must be even for RoPE, got {}", d));
    }

    size_t seq_len = x->shape()[static_cast<size_t>(s_dim)];
    size_t half_d = d / 2;

    // Angles: theta_j = 1.0 / (10000.0 ^ (2j / d))
    std::vector<double> inv_freq(half_d);
    for (size_t j = 0; j < half_d; ++j) {
        inv_freq[j] = 1.0 / std::pow(10000.0, static_cast<double>(2 * j) / static_cast<double>(d));
    }

    // Outer product angles: shape (seq_len, half_d)
    std::vector<double> sin_table(seq_len * half_d);
    std::vector<double> cos_table(seq_len * half_d);
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t j = 0; j < half_d; ++j) {
            double angle = static_cast<double>(pos) * inv_freq[j];
            sin_table[pos * half_d + j] = std::sin(angle);
            cos_table[pos * half_d + j] = std::cos(angle);
        }
    }

    auto contig = x->contiguous();
    std::vector<double> out_data(x->numel());
    const double* src = contig->data();

    // Iterate through elements and apply 2D Givens rotations
    size_t outer_size = 1;
    for (int i = 0; i < s_dim; ++i) outer_size *= x->shape()[static_cast<size_t>(i)];
    size_t inner_size = 1;
    for (size_t i = static_cast<size_t>(s_dim) + 1; i < ndim - 1; ++i) inner_size *= x->shape()[i];

    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            for (size_t in = 0; in < inner_size; ++in) {
                size_t base_idx = ((o * seq_len + pos) * inner_size + in) * d;
                for (size_t j = 0; j < half_d; ++j) {
                    double x1 = src[base_idx + 2 * j];
                    double x2 = src[base_idx + 2 * j + 1];
                    double c = cos_table[pos * half_d + j];
                    double s = sin_table[pos * half_d + j];
                    out_data[base_idx + 2 * j]     = x1 * c - x2 * s;
                    out_data[base_idx + 2 * j + 1] = x1 * s + x2 * c;
                }
            }
        }
    }

    return std::make_shared<Tensor>(x->shape(), std::move(out_data), x->requires_grad());
}

AttentionOutput scaled_dot_product_attention(
    const std::shared_ptr<Tensor>& q,
    const std::shared_ptr<Tensor>& k,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& mask,
    std::optional<double> scale)
{
    size_t d_k = q->shape().back();
    double s = scale.value_or(1.0 / std::sqrt(static_cast<double>(d_k)));

    // S = (Q @ K^T) * scale
    auto k_t = k->transpose();
    auto scores = (q->matmul(k_t)) * s;

    if (mask) {
        scores = scores + mask;
    }

    // A = Softmax(S, axis=-1)
    auto attn_weights = scores->softmax(-1);

    // O = A @ V
    auto output = attn_weights->matmul(v);

    return {output, attn_weights};
}

MultiHeadAttention::MultiHeadAttention(
    size_t d_model,
    size_t num_heads,
    bool bias,
    double dropout,
    uint64_t seed)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_k_(d_model / num_heads),
      scale_(1.0 / std::sqrt(static_cast<double>(d_model / num_heads))),
      dropout_(dropout)
{
    if (d_model % num_heads != 0) {
        throw std::invalid_argument(std::format(
            "d_model ({}) must be divisible by num_heads ({})", d_model, num_heads));
    }

    q_proj_ = std::make_shared<nn::Linear>(d_model, d_model, bias, seed);
    k_proj_ = std::make_shared<nn::Linear>(d_model, d_model, bias, seed + 1);
    v_proj_ = std::make_shared<nn::Linear>(d_model, d_model, bias, seed + 2);
    out_proj_ = std::make_shared<nn::Linear>(d_model, d_model, bias, seed + 3);

    if (dropout > 0.0) {
        dropout_layer_ = std::make_shared<nn::Dropout>(dropout, seed + 4);
        register_module("dropout", dropout_layer_);
    }

    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(const std::shared_ptr<Tensor>& input) {
    return forward_with_attention(input, input, input, nullptr, false).output;
}

AttentionOutput MultiHeadAttention::forward_with_attention(
    const std::shared_ptr<Tensor>& q,
    const std::shared_ptr<Tensor>& k_in,
    const std::shared_ptr<Tensor>& v_in,
    const std::shared_ptr<Tensor>& mask,
    bool is_causal)
{
    const auto& k = k_in ? k_in : q;
    const auto& v = v_in ? v_in : k;

    size_t B = q->shape()[0];
    size_t T_q = q->shape()[1];
    size_t T_k = k->shape()[1];

    // 1. Linear projections
    auto q_proj = q_proj_->forward(q);
    auto k_proj = k_proj_->forward(k);
    auto v_proj = v_proj_->forward(v);

    // 2. Reshape and swapaxes to (B, num_heads, T, d_k)
    auto q_heads = q_proj->reshape({B, T_q, num_heads_, d_k_})->swapaxes(1, 2);
    auto k_heads = k_proj->reshape({B, T_k, num_heads_, d_k_})->swapaxes(1, 2);
    auto v_heads = v_proj->reshape({B, T_k, num_heads_, d_k_})->swapaxes(1, 2);

    // 3. Causal mask
    std::shared_ptr<Tensor> attn_mask = mask;
    if (is_causal) {
        auto causal_m = create_causal_mask(T_q);
        attn_mask = mask ? mask + causal_m : causal_m;
    }

    // 4. Scaled dot-product attention
    auto [out_heads, attn_weights] = scaled_dot_product_attention(
        q_heads, k_heads, v_heads, attn_mask, scale_);

    // 5. Concatenate heads and out-project: (B, num_heads, T_q, d_k) -> (B, T_q, d_model)
    auto out_concat = out_heads->swapaxes(1, 2)->reshape({B, T_q, d_model_});
    auto out = out_proj_->forward(out_concat);

    return {out, attn_weights};
}

} // namespace aurora
