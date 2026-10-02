#include "aurora/transformer.hpp"

#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>

namespace aurora {

TransformerBlock::TransformerBlock(
    size_t d_model,
    size_t num_heads,
    std::optional<size_t> d_ff,
    const std::string& norm_type,
    const std::string& activation,
    double dropout,
    bool bias,
    uint64_t seed)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_ff_(d_ff.value_or(4 * d_model)),
      norm_type_(norm_type),
      activation_name_(activation),
      dropout_(dropout)
{
    if (norm_type_ == "layernorm") {
        norm1_ = std::make_shared<nn::LayerNorm>(d_model);
        norm2_ = std::make_shared<nn::LayerNorm>(d_model);
    } else if (norm_type_ == "rmsnorm") {
        norm1_ = std::make_shared<nn::RMSNorm>(d_model);
        norm2_ = std::make_shared<nn::RMSNorm>(d_model);
    } else {
        throw std::invalid_argument(std::format("Unknown norm_type: {}", norm_type_));
    }

    attn_ = std::make_shared<MultiHeadAttention>(d_model, num_heads, bias, dropout, seed + 1);

    if (dropout > 0.0) {
        dropout1_ = std::make_shared<nn::Dropout>(dropout, seed + 2);
        dropout2_ = std::make_shared<nn::Dropout>(dropout, seed + 3);
        register_module("dropout1", dropout1_);
        register_module("dropout2", dropout2_);
    }

    linear1_ = std::make_shared<nn::Linear>(d_model, d_ff_, bias, seed + 4);

    if (activation_name_ == "gelu") {
        act_ = std::make_shared<nn::GELU>();
    } else if (activation_name_ == "relu") {
        act_ = std::make_shared<nn::ReLU>();
    } else if (activation_name_ == "silu") {
        act_ = std::make_shared<nn::SiLU>();
    } else {
        throw std::invalid_argument(std::format("Unknown activation: {}", activation_name_));
    }

    linear2_ = std::make_shared<nn::Linear>(d_ff_, d_model, bias, seed + 5);

    register_module("norm1", norm1_);
    register_module("attn", attn_);
    register_module("norm2", norm2_);
    register_module("linear1", linear1_);
    register_module("linear2", linear2_);
}

std::shared_ptr<Tensor> TransformerBlock::forward(const std::shared_ptr<Tensor>& input) {
    return forward_with_attention(input, nullptr, false).first;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> TransformerBlock::forward_with_attention(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& mask,
    bool is_causal)
{
    // 1. Pre-norm 1 + Attention + Residual
    auto normed1 = norm1_->forward(x);
    auto attn_res = attn_->forward_with_attention(normed1, nullptr, nullptr, mask, is_causal);
    auto attn_out = attn_res.output;
    if (dropout1_) {
        attn_out = dropout1_->forward(attn_out);
    }
    auto x_next = x + attn_out;

    // 2. Pre-norm 2 + FFN + Residual
    auto normed2 = norm2_->forward(x_next);
    auto ffn_h = act_->forward(linear1_->forward(normed2));
    auto ffn_out = linear2_->forward(ffn_h);
    if (dropout2_) {
        ffn_out = dropout2_->forward(ffn_out);
    }
    auto out = x_next + ffn_out;

    return {out, attn_res.attention_weights};
}

TransformerDecoder::TransformerDecoder(
    size_t d_model,
    size_t num_layers,
    size_t num_heads,
    std::optional<size_t> vocab_size,
    std::optional<size_t> in_dim,
    std::optional<size_t> out_dim,
    size_t max_seq_len,
    std::optional<size_t> d_ff,
    const std::string& norm_type,
    const std::string& activation,
    double dropout,
    bool bias,
    uint64_t seed)
    : d_model_(d_model),
      num_layers_(num_layers),
      num_heads_(num_heads),
      vocab_size_(vocab_size),
      in_dim_(in_dim),
      max_seq_len_(max_seq_len)
{
    if (!vocab_size.has_value() && !in_dim.has_value()) {
        throw std::invalid_argument("Either vocab_size or in_dim must be provided");
    }

    if (vocab_size.has_value()) {
        token_emb_ = std::make_shared<nn::Embedding>(*vocab_size, d_model, seed);
        register_module("token_emb", token_emb_);
    } else {
        input_proj_ = std::make_shared<nn::Linear>(*in_dim, d_model, bias, seed);
        register_module("input_proj", input_proj_);
    }

    pos_emb_ = std::make_shared<nn::Embedding>(max_seq_len, d_model, seed + 1);
    register_module("pos_emb", pos_emb_);

    if (dropout > 0.0) {
        drop_ = std::make_shared<nn::Dropout>(dropout, seed + 2);
        register_module("drop", drop_);
    }

    for (size_t i = 0; i < num_layers; ++i) {
        auto block = std::make_shared<TransformerBlock>(
            d_model, num_heads, d_ff, norm_type, activation, dropout, bias, seed + 10 * (i + 1));
        blocks_.push_back(block);
        register_module(std::format("blocks.{}", i), block);
    }

    if (norm_type == "layernorm") {
        final_norm_ = std::make_shared<nn::LayerNorm>(d_model);
    } else if (norm_type == "rmsnorm") {
        final_norm_ = std::make_shared<nn::RMSNorm>(d_model);
    } else {
        throw std::invalid_argument(std::format("Unknown norm_type: {}", norm_type));
    }
    register_module("final_norm", final_norm_);

    size_t target_out = out_dim.value_or(vocab_size.value_or(in_dim.value_or(d_model)));
    head_ = std::make_shared<nn::Linear>(d_model, target_out, bias, seed + 999);
    register_module("head", head_);
}

std::shared_ptr<Tensor> TransformerDecoder::forward_tokens(
    const std::vector<size_t>& tokens,
    const std::vector<size_t>& shape,
    const std::shared_ptr<Tensor>& mask,
    bool is_causal)
{
    if (!token_emb_) {
        throw std::runtime_error("forward_tokens called on continuous decoder model");
    }

    auto h = token_emb_->forward(tokens, shape);
    size_t T = shape[1];
    if (T > max_seq_len_) {
        throw std::out_of_range(std::format(
            "Sequence length {} exceeds max_seq_len {}", T, max_seq_len_));
    }

    std::vector<size_t> pos_indices(T);
    std::iota(pos_indices.begin(), pos_indices.end(), 0);
    auto pos_enc = pos_emb_->forward(pos_indices, {T});
    h = h + pos_enc;

    if (drop_) {
        h = drop_->forward(h);
    }

    for (const auto& block : blocks_) {
        h = block->forward_with_attention(h, mask, is_causal).first;
    }

    h = final_norm_->forward(h);
    return head_->forward(h);
}

std::shared_ptr<Tensor> TransformerDecoder::forward_continuous(
    const std::shared_ptr<Tensor>& states,
    const std::shared_ptr<Tensor>& mask,
    bool is_causal)
{
    if (!input_proj_) {
        throw std::runtime_error("forward_continuous called on discrete token decoder model");
    }

    auto h = input_proj_->forward(states);
    size_t T = states->shape()[1];
    if (T > max_seq_len_) {
        throw std::out_of_range(std::format(
            "Sequence length {} exceeds max_seq_len {}", T, max_seq_len_));
    }

    std::vector<size_t> pos_indices(T);
    std::iota(pos_indices.begin(), pos_indices.end(), 0);
    auto pos_enc = pos_emb_->forward(pos_indices, {T});
    h = h + pos_enc;

    if (drop_) {
        h = drop_->forward(h);
    }

    for (const auto& block : blocks_) {
        h = block->forward_with_attention(h, mask, is_causal).first;
    }

    h = final_norm_->forward(h);
    return head_->forward(h);
}

std::shared_ptr<Tensor> TransformerDecoder::forward(const std::shared_ptr<Tensor>& input) {
    if (input_proj_) {
        return forward_continuous(input, nullptr, true);
    } else {
        auto vec = input->to_vector();
        std::vector<size_t> tokens(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            tokens[i] = static_cast<size_t>(vec[i]);
        }
        return forward_tokens(tokens, input->shape(), nullptr, true);
    }
}

} // namespace aurora
