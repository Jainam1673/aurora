#include "aurora/attention.hpp"
#include "aurora/optim.hpp"
#include "aurora/transformer.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

using namespace aurora;

TEST(TransformerTest, CausalMaskGeneration) {
    auto mask = create_causal_mask(4);
    ASSERT_EQ(mask->shape(), (std::vector<size_t>{4, 4}));

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double val = mask->at({i, j});
            if (j <= i) {
                EXPECT_DOUBLE_EQ(val, 0.0);
            } else {
                EXPECT_DOUBLE_EQ(val, -1e9);
            }
        }
    }
}

TEST(TransformerTest, RotaryPositionalEmbeddingNormPreservation) {
    size_t seq_len = 6;
    size_t d_k = 8;
    std::vector<double> data(2 * seq_len * d_k);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i % 7) - 3.0;
    }
    auto x = std::make_shared<Tensor>(std::vector<size_t>{2, seq_len, d_k}, data, false);
    auto x_rot = apply_rotary_pos_emb(x, 1);

    ASSERT_EQ(x_rot->shape(), x->shape());

    // Check that each (b, pos) vector retains the exact Euclidean norm
    for (size_t b = 0; b < 2; ++b) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            double norm_orig_sq = 0.0;
            double norm_rot_sq = 0.0;
            for (size_t d = 0; d < d_k; ++d) {
                double v1 = x->at({b, pos, d});
                double v2 = x_rot->at({b, pos, d});
                norm_orig_sq += v1 * v1;
                norm_rot_sq += v2 * v2;
            }
            EXPECT_NEAR(std::sqrt(norm_rot_sq), std::sqrt(norm_orig_sq), 1e-9);
        }
    }
}

TEST(TransformerTest, ScaledDotProductAttentionForwardAndBackward) {
    size_t B = 2, H = 2, T = 3, d_k = 4;
    auto q = Tensor::randn({B, H, T, d_k}, /*requires_grad=*/true, 42);
    auto k = Tensor::randn({B, H, T, d_k}, /*requires_grad=*/true, 43);
    auto v = Tensor::randn({B, H, T, d_k}, /*requires_grad=*/true, 44);

    auto mask = create_causal_mask(T);
    auto [out, weights] = scaled_dot_product_attention(q, k, v, mask);

    ASSERT_EQ(out->shape(), (std::vector<size_t>{B, H, T, d_k}));
    ASSERT_EQ(weights->shape(), (std::vector<size_t>{B, H, T, T}));

    // Softmax probabilities must sum to 1 along last axis
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < T; ++i) {
                double sum_p = 0.0;
                for (size_t j = 0; j < T; ++j) {
                    double p = weights->at({b, h, i, j});
                    sum_p += p;
                    if (j > i) {
                        EXPECT_NEAR(p, 0.0, 1e-5);
                    }
                }
                EXPECT_NEAR(sum_p, 1.0, 1e-6);
            }
        }
    }

    // Backward pass
    auto loss = out->sum();
    loss->backward();

    ASSERT_NE(q->grad(), nullptr);
    ASSERT_NE(k->grad(), nullptr);
    ASSERT_NE(v->grad(), nullptr);

    for (double g : q->grad()->to_vector()) {
        EXPECT_FALSE(std::isnan(g));
    }
    for (double g : k->grad()->to_vector()) {
        EXPECT_FALSE(std::isnan(g));
    }
    for (double g : v->grad()->to_vector()) {
        EXPECT_FALSE(std::isnan(g));
    }
}

TEST(TransformerTest, MultiHeadAttentionForwardAndBackward) {
    size_t B = 2, T = 4, d_model = 16, num_heads = 4;
    auto mha = std::make_shared<MultiHeadAttention>(d_model, num_heads, /*bias=*/true, /*dropout=*/0.0, 123);

    auto x = Tensor::randn({B, T, d_model}, /*requires_grad=*/true, 456);
    auto res = mha->forward_with_attention(x, nullptr, nullptr, nullptr, /*is_causal=*/true);

    ASSERT_EQ(res.output->shape(), (std::vector<size_t>{B, T, d_model}));
    ASSERT_EQ(res.attention_weights->shape(), (std::vector<size_t>{B, num_heads, T, T}));

    auto loss = res.output->sum();
    loss->backward();

    ASSERT_NE(x->grad(), nullptr);
    for (const auto& p : mha->parameters()) {
        ASSERT_NE(p->grad(), nullptr);
        for (double g : p->grad()->to_vector()) {
            EXPECT_FALSE(std::isnan(g));
        }
    }
}

TEST(TransformerTest, TransformerBlockForwardAndBackward) {
    size_t B = 2, T = 3, d_model = 12, num_heads = 3;
    auto block = std::make_shared<TransformerBlock>(
        d_model, num_heads, /*d_ff=*/24, "layernorm", "gelu", 0.0, true, 789);

    auto x = Tensor::randn({B, T, d_model}, /*requires_grad=*/true, 101);
    auto [out, attn] = block->forward_with_attention(x, nullptr, /*is_causal=*/true);

    ASSERT_EQ(out->shape(), (std::vector<size_t>{B, T, d_model}));
    ASSERT_EQ(attn->shape(), (std::vector<size_t>{B, num_heads, T, T}));

    auto loss = out->sum();
    loss->backward();

    ASSERT_NE(x->grad(), nullptr);
    for (const auto& p : block->parameters()) {
        ASSERT_NE(p->grad(), nullptr);
        for (double g : p->grad()->to_vector()) {
            EXPECT_FALSE(std::isnan(g));
        }
    }
}

TEST(TransformerTest, TransformerBlockRMSNormAndSiLU) {
    size_t B = 2, T = 3, d_model = 8, num_heads = 2;
    auto block = std::make_shared<TransformerBlock>(
        d_model, num_heads, /*d_ff=*/16, "rmsnorm", "silu", 0.0, true, 202);

    auto x = Tensor::randn({B, T, d_model}, /*requires_grad=*/false, 303);
    auto out = block->forward(x);
    ASSERT_EQ(out->shape(), (std::vector<size_t>{B, T, d_model}));
}

TEST(TransformerTest, TransformerDecoderDiscreteAndContinuous) {
    // 1. Discrete token decoder
    size_t vocab_size = 20;
    size_t d_model = 16;
    auto decoder_discrete = std::make_shared<TransformerDecoder>(
        d_model, /*num_layers=*/2, /*num_heads=*/2, vocab_size,
        /*in_dim=*/std::nullopt, /*out_dim=*/std::nullopt,
        /*max_seq_len=*/32, /*d_ff=*/32, "layernorm", "gelu", 0.0, true, 404);

    std::vector<size_t> tokens = {1, 4, 9, 2, 0, 3, 5, 8};
    auto logits = decoder_discrete->forward_tokens(tokens, {2, 4});
    ASSERT_EQ(logits->shape(), (std::vector<size_t>{2, 4, vocab_size}));

    // 2. Continuous input decoder
    size_t in_dim = 6;
    size_t out_dim = 6;
    auto decoder_continuous = std::make_shared<TransformerDecoder>(
        d_model, /*num_layers=*/2, /*num_heads=*/2, /*vocab_size=*/std::nullopt,
        in_dim, out_dim, /*max_seq_len=*/32, /*d_ff=*/32, "layernorm", "gelu", 0.0, true, 505);

    auto states = Tensor::randn({2, 5, in_dim}, /*requires_grad=*/false, 606);
    auto pred = decoder_continuous->forward_continuous(states);
    ASSERT_EQ(pred->shape(), (std::vector<size_t>{2, 5, out_dim}));
}

TEST(TransformerTest, TransformerDecoderOptimizationStep) {
    size_t vocab_size = 8;
    size_t d_model = 16;
    auto decoder = std::make_shared<TransformerDecoder>(
        d_model, /*num_layers=*/2, /*num_heads=*/2, vocab_size,
        /*in_dim=*/std::nullopt, /*out_dim=*/std::nullopt,
        /*max_seq_len=*/16, /*d_ff=*/32, "layernorm", "gelu", 0.0, true, 707);

    optim::AdamW opt(decoder->parameters(), 1e-3);

    std::vector<size_t> tokens = {1, 2, 3, 4};
    std::vector<size_t> target = {2, 3, 4, 5};

    // Forward
    auto logits = decoder->forward_tokens(tokens, {1, 4});
    std::vector<double> target_one_hot(1 * 4 * vocab_size, 0.0);
    for (size_t t = 0; t < 4; ++t) {
        target_one_hot[t * vocab_size + target[t]] = 1.0;
    }
    auto target_t = std::make_shared<Tensor>(
        std::vector<size_t>{1, 4, vocab_size}, std::move(target_one_hot), false);

    auto diff = logits - target_t;
    auto loss = (diff * diff)->mean();
    double initial_loss = loss->item();

    opt.zero_grad();
    loss->backward();
    opt.step();

    auto logits2 = decoder->forward_tokens(tokens, {1, 4});
    auto diff2 = logits2 - target_t;
    auto loss2 = (diff2 * diff2)->mean();

    EXPECT_LT(loss2->item(), initial_loss);
}
