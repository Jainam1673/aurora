#pragma once

#include "aurora/nn.hpp"
#include "aurora/tensor.hpp"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace aurora {

std::shared_ptr<Tensor> create_causal_mask(size_t seq_len);

std::shared_ptr<Tensor> apply_rotary_pos_emb(const std::shared_ptr<Tensor>& x, int seq_dim = -2);

struct AttentionOutput {
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> attention_weights;
};

AttentionOutput scaled_dot_product_attention(
    const std::shared_ptr<Tensor>& q,
    const std::shared_ptr<Tensor>& k,
    const std::shared_ptr<Tensor>& v,
    const std::shared_ptr<Tensor>& mask = nullptr,
    std::optional<double> scale = std::nullopt);

class MultiHeadAttention : public nn::Module {
public:
    MultiHeadAttention(
        size_t d_model,
        size_t num_heads,
        bool bias = true,
        double dropout = 0.0,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    AttentionOutput forward_with_attention(
        const std::shared_ptr<Tensor>& q,
        const std::shared_ptr<Tensor>& k = nullptr,
        const std::shared_ptr<Tensor>& v = nullptr,
        const std::shared_ptr<Tensor>& mask = nullptr,
        bool is_causal = false);

    [[nodiscard]] size_t d_model() const noexcept { return d_model_; }
    [[nodiscard]] size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] size_t d_k() const noexcept { return d_k_; }
    [[nodiscard]] double scale() const noexcept { return scale_; }
    [[nodiscard]] double dropout() const noexcept { return dropout_; }

    [[nodiscard]] std::shared_ptr<nn::Linear> q_proj() const noexcept { return q_proj_; }
    [[nodiscard]] std::shared_ptr<nn::Linear> k_proj() const noexcept { return k_proj_; }
    [[nodiscard]] std::shared_ptr<nn::Linear> v_proj() const noexcept { return v_proj_; }
    [[nodiscard]] std::shared_ptr<nn::Linear> out_proj() const noexcept { return out_proj_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t d_k_;
    double scale_;
    double dropout_;

    std::shared_ptr<nn::Linear> q_proj_;
    std::shared_ptr<nn::Linear> k_proj_;
    std::shared_ptr<nn::Linear> v_proj_;
    std::shared_ptr<nn::Linear> out_proj_;
    std::shared_ptr<nn::Dropout> dropout_layer_{nullptr};
};

} // namespace aurora
