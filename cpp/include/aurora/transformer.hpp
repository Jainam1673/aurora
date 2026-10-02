#pragma once

#include "aurora/attention.hpp"
#include "aurora/nn.hpp"
#include "aurora/tensor.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aurora {

class TransformerBlock : public nn::Module {
public:
    TransformerBlock(
        size_t d_model,
        size_t num_heads,
        std::optional<size_t> d_ff = std::nullopt,
        const std::string& norm_type = "layernorm",
        const std::string& activation = "gelu",
        double dropout = 0.0,
        bool bias = true,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward_with_attention(
        const std::shared_ptr<Tensor>& x,
        const std::shared_ptr<Tensor>& mask = nullptr,
        bool is_causal = false);

    [[nodiscard]] size_t d_model() const noexcept { return d_model_; }
    [[nodiscard]] size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] size_t d_ff() const noexcept { return d_ff_; }
    [[nodiscard]] double dropout() const noexcept { return dropout_; }

    [[nodiscard]] std::shared_ptr<nn::Module> norm1() const noexcept { return norm1_; }
    [[nodiscard]] std::shared_ptr<MultiHeadAttention> attn() const noexcept { return attn_; }
    [[nodiscard]] std::shared_ptr<nn::Module> norm2() const noexcept { return norm2_; }
    [[nodiscard]] std::shared_ptr<nn::Linear> linear1() const noexcept { return linear1_; }
    [[nodiscard]] std::shared_ptr<nn::Linear> linear2() const noexcept { return linear2_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t d_ff_;
    std::string norm_type_;
    std::string activation_name_;
    double dropout_;

    std::shared_ptr<nn::Module> norm1_;
    std::shared_ptr<MultiHeadAttention> attn_;
    std::shared_ptr<nn::Dropout> dropout1_{nullptr};

    std::shared_ptr<nn::Module> norm2_;
    std::shared_ptr<nn::Linear> linear1_;
    std::shared_ptr<nn::Module> act_;
    std::shared_ptr<nn::Linear> linear2_;
    std::shared_ptr<nn::Dropout> dropout2_{nullptr};
};

class TransformerDecoder : public nn::Module {
public:
    TransformerDecoder(
        size_t d_model,
        size_t num_layers,
        size_t num_heads,
        std::optional<size_t> vocab_size = std::nullopt,
        std::optional<size_t> in_dim = std::nullopt,
        std::optional<size_t> out_dim = std::nullopt,
        size_t max_seq_len = 1024,
        std::optional<size_t> d_ff = std::nullopt,
        const std::string& norm_type = "layernorm",
        const std::string& activation = "gelu",
        double dropout = 0.0,
        bool bias = true,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::shared_ptr<Tensor> forward_tokens(
        const std::vector<size_t>& tokens,
        const std::vector<size_t>& shape,
        const std::shared_ptr<Tensor>& mask = nullptr,
        bool is_causal = true);

    std::shared_ptr<Tensor> forward_continuous(
        const std::shared_ptr<Tensor>& states,
        const std::shared_ptr<Tensor>& mask = nullptr,
        bool is_causal = true);

    [[nodiscard]] size_t d_model() const noexcept { return d_model_; }
    [[nodiscard]] size_t num_layers() const noexcept { return num_layers_; }
    [[nodiscard]] size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] size_t max_seq_len() const noexcept { return max_seq_len_; }

    [[nodiscard]] const std::vector<std::shared_ptr<TransformerBlock>>& blocks() const noexcept {
        return blocks_;
    }

private:
    size_t d_model_;
    size_t num_layers_;
    size_t num_heads_;
    std::optional<size_t> vocab_size_;
    std::optional<size_t> in_dim_;
    size_t max_seq_len_;

    std::shared_ptr<nn::Embedding> token_emb_{nullptr};
    std::shared_ptr<nn::Linear> input_proj_{nullptr};
    std::shared_ptr<nn::Embedding> pos_emb_;
    std::shared_ptr<nn::Dropout> drop_{nullptr};

    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    std::shared_ptr<nn::Module> final_norm_;
    std::shared_ptr<nn::Linear> head_;
};

} // namespace aurora
