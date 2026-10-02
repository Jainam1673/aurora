#include "aurora/nn.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace aurora;
using namespace aurora::nn;

TEST(NNTest, LinearForwardAndBackward) {
    auto lin = std::make_shared<Linear>(4, 2, true);
    auto x = Tensor::randn({3, 4}, 123, true);

    auto y = lin->forward(x);
    EXPECT_EQ(y->shape(), (std::vector<size_t>{3, 2}));

    auto loss = y->sum();
    loss->backward();

    ASSERT_NE(lin->weight()->grad(), nullptr);
    ASSERT_NE(lin->bias()->grad(), nullptr);
    ASSERT_NE(x->grad(), nullptr);

    EXPECT_EQ(lin->weight()->grad()->shape(), (std::vector<size_t>{2, 4}));
    EXPECT_EQ(lin->bias()->grad()->shape(), (std::vector<size_t>{2}));
    EXPECT_EQ(x->grad()->shape(), (std::vector<size_t>{3, 4}));
}

TEST(NNTest, EmbeddingForwardAndBackward) {
    auto emb = std::make_shared<Embedding>(6, 4, 42);
    // Index 1 appears twice: index 0 and index 2
    std::vector<size_t> indices = {1, 3, 1};
    auto out = emb->forward(indices, {3});

    EXPECT_EQ(out->shape(), (std::vector<size_t>{3, 4}));

    // Verify row equality
    auto w_data = emb->weight()->to_vector();
    auto out_data = out->to_vector();
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_DOUBLE_EQ(out_data[0 * 4 + d], w_data[1 * 4 + d]);
        EXPECT_DOUBLE_EQ(out_data[1 * 4 + d], w_data[3 * 4 + d]);
        EXPECT_DOUBLE_EQ(out_data[2 * 4 + d], w_data[1 * 4 + d]);
    }

    auto loss = out->sum();
    loss->backward();

    ASSERT_NE(emb->weight()->grad(), nullptr);
    auto gw_data = emb->weight()->grad()->to_vector();
    // Index 1 was looked up twice -> grad is 2.0
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_DOUBLE_EQ(gw_data[1 * 4 + d], 2.0);
        EXPECT_DOUBLE_EQ(gw_data[3 * 4 + d], 1.0);
        EXPECT_DOUBLE_EQ(gw_data[0 * 4 + d], 0.0);
    }
}

TEST(NNTest, LayerNormBehavior) {
    auto ln = std::make_shared<LayerNorm>(4);
    auto x = Tensor::create({2, 4}, {1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0}, true);
    auto y = ln->forward(x);

    EXPECT_EQ(y->shape(), (std::vector<size_t>{2, 4}));
    auto mean_y = y->mean(-1);
    for (double val : mean_y->to_vector()) {
        EXPECT_NEAR(val, 0.0, 1e-5);
    }
}

TEST(NNTest, RMSNormBehavior) {
    auto rms = std::make_shared<RMSNorm>(4);
    auto x = Tensor::create({2, 4}, {2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0}, true);
    auto y = rms->forward(x);

    EXPECT_EQ(y->shape(), (std::vector<size_t>{2, 4}));
    // For vector of identical values c, RMS = c, so y = c / c * 1.0 = 1.0
    for (double val : y->to_vector()) {
        EXPECT_NEAR(val, 1.0, 1e-4);
    }
}

TEST(NNTest, DropoutModes) {
    auto drop = std::make_shared<Dropout>(0.5, 999);
    auto x = Tensor::ones({10, 10});

    // Eval mode: must be exact identity
    drop->eval();
    auto y_eval = drop->forward(x);
    for (double v : y_eval->to_vector()) {
        EXPECT_DOUBLE_EQ(v, 1.0);
    }

    // Train mode: elements are either 0 or 1 / (1 - 0.5) = 2.0
    drop->train(true);
    auto y_train = drop->forward(x);
    bool has_zero = false;
    bool has_scale = false;
    for (double v : y_train->to_vector()) {
        if (v == 0.0) has_zero = true;
        if (std::abs(v - 2.0) < 1e-5) has_scale = true;
    }
    EXPECT_TRUE(has_zero);
    EXPECT_TRUE(has_scale);
}

TEST(NNTest, SequentialAndMLP) {
    auto mlp = std::make_shared<MLP>(4, std::vector<size_t>{8, 8}, 2, "gelu", 0.0);
    auto x = Tensor::randn({5, 4}, 777, true);
    auto y = mlp->forward(x);

    EXPECT_EQ(y->shape(), (std::vector<size_t>{5, 2}));
    auto loss = y->sum();
    loss->backward();

    EXPECT_NE(x->grad(), nullptr);
    for (const auto& p : mlp->parameters()) {
        EXPECT_NE(p->grad(), nullptr);
    }
}

TEST(NNTest, ResidualBlock) {
    auto lin = std::make_shared<Linear>(4, 4);
    auto block = std::make_shared<ResidualBlock>(lin);

    auto x = Tensor::ones({2, 4});
    auto y = block->forward(x);
    EXPECT_EQ(y->shape(), (std::vector<size_t>{2, 4}));
}

TEST(NNTest, StateDictRoundtrip) {
    auto lin1 = std::make_shared<Linear>(3, 2, true, 42);
    auto dict = lin1->state_dict();

    EXPECT_TRUE(dict.contains("weight"));
    EXPECT_TRUE(dict.contains("bias"));

    auto lin2 = std::make_shared<Linear>(3, 2, true, 100);
    // Lin2 has different weights initially
    EXPECT_NE(lin1->weight()->to_vector(), lin2->weight()->to_vector());

    lin2->load_state_dict(dict);
    EXPECT_EQ(lin1->weight()->to_vector(), lin2->weight()->to_vector());
    EXPECT_EQ(lin1->bias()->to_vector(), lin2->bias()->to_vector());
}
