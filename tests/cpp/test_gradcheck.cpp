#include "aurora/autograd.hpp"
#include "aurora/gradcheck.hpp"
#include "aurora/tensor.hpp"

#include <gtest/gtest.h>

TEST(GradcheckTest, AddAndSub) {
    auto a = aurora::Tensor::randn({2, 3}, 101, true);
    auto b = aurora::Tensor::randn({2, 3}, 102, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->add(inps[1])->sum(); },
        {a, b}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->sub(inps[1])->sum(); },
        {a, b}
    ));
}

TEST(GradcheckTest, BroadcastingAddAndMul) {
    auto a = aurora::Tensor::randn({2, 3}, 201, true);
    auto b = aurora::Tensor::randn({3}, 202, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->add(inps[1])->sum(); },
        {a, b}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->mul(inps[1])->sum(); },
        {a, b}
    ));
}

TEST(GradcheckTest, Div) {
    auto a = aurora::Tensor::create({2, 2}, std::vector<double>{2.0, 3.0, 4.0, 5.0}, true);
    auto b = aurora::Tensor::create({2, 2}, std::vector<double>{1.5, 2.5, 3.5, 4.5}, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->div(inps[1])->sum(); },
        {a, b}
    ));
}

TEST(GradcheckTest, Matmul) {
    auto a = aurora::Tensor::randn({3, 4}, 301, true);
    auto b = aurora::Tensor::randn({4, 2}, 302, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->matmul(inps[1])->sum(); },
        {a, b}
    ));
}

TEST(GradcheckTest, SumAndMean) {
    auto a = aurora::Tensor::randn({3, 4}, 401, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->sum(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->sum(0, true)->sum(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->mean(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->mean(1, false)->sum(); },
        {a}
    ));
}

TEST(GradcheckTest, ReshapeAndTranspose) {
    auto a = aurora::Tensor::randn({2, 3, 4}, 501, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->reshape({6, 4})->sum(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->transpose({1, 0, 2})->sum(); },
        {a}
    ));
}

TEST(GradcheckTest, UnaryMath) {
    auto a = aurora::Tensor::create({4}, std::vector<double>{1.2, 2.5, 3.1, 0.8}, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->exp()->sum(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->log()->sum(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->sqrt()->sum(); },
        {a}
    ));
}

TEST(GradcheckTest, Activations) {
    auto x_relu = aurora::Tensor::create({5}, std::vector<double>{-2.5, -1.0, 0.5, 1.8, 3.2}, true);
    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->relu()->sum(); },
        {x_relu}
    ));

    auto x_gelu = aurora::Tensor::create({5}, std::vector<double>{-1.5, -0.5, 0.0, 0.8, 2.1}, true);
    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->gelu()->sum(); },
        {x_gelu}
    ));

    auto x_silu = aurora::Tensor::create({5}, std::vector<double>{-2.0, -0.5, 0.0, 1.0, 2.5}, true);
    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->silu()->sum(); },
        {x_silu}
    ));
}

TEST(GradcheckTest, SoftmaxAndLogSoftmax) {
    auto a = aurora::Tensor::randn({2, 4}, 601, true);
    auto w = aurora::Tensor::randn({2, 4}, 602, false);

    EXPECT_TRUE(aurora::gradcheck(
        [&w](const auto& inps) { return inps[0]->softmax(-1)->mul(w)->sum(); },
        {a}
    ));

    EXPECT_TRUE(aurora::gradcheck(
        [&w](const auto& inps) { return inps[0]->log_softmax(-1)->mul(w)->sum(); },
        {a}
    ));
}

TEST(GradcheckTest, LayerNorm) {
    auto x = aurora::Tensor::randn({2, 4}, 701, true);
    auto gamma = aurora::Tensor::randn({4}, 702, true);
    auto beta = aurora::Tensor::randn({4}, 703, true);

    EXPECT_TRUE(aurora::gradcheck(
        [](const auto& inps) { return inps[0]->layer_norm(inps[1], inps[2])->sum(); },
        {x, gamma, beta}
    ));
}
