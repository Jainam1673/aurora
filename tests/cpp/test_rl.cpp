#include "aurora/buffer.hpp"
#include "aurora/distribution.hpp"
#include "aurora/environment.hpp"
#include "aurora/tensor.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <vector>

using namespace aurora;

TEST(RLTest, CategoricalDistribution) {
    // 2 batch items, 3 classes each
    // Row 0: uniform logits [0, 0, 0] -> probs [1/3, 1/3, 1/3]
    // Row 1: logits [1, 2, 3]
    auto logits = Tensor::create({2, 3}, std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 3.0}, true);
    Categorical dist(logits);

    auto probs = dist.probs();
    ASSERT_EQ(probs->shape(), (std::vector<size_t>{2, 3}));
    EXPECT_NEAR(probs->at({0, 0}), 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(probs->at({0, 1}), 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(probs->at({0, 2}), 1.0 / 3.0, 1e-6);

    // Entropy of uniform row should be ln(3)
    auto ent = dist.entropy();
    ASSERT_EQ(ent->shape(), (std::vector<size_t>{2}));
    EXPECT_NEAR(ent->at({0}), std::log(3.0), 1e-6);

    // Log prob of class 0 in row 0
    auto actions = Tensor::create({2}, std::vector<double>{0.0, 2.0}, false);
    auto lp = dist.log_prob(actions);
    ASSERT_EQ(lp->shape(), (std::vector<size_t>{2}));
    EXPECT_NEAR(lp->at({0}), std::log(1.0 / 3.0), 1e-6);

    // Sample
    auto sample = dist.sample(42);
    ASSERT_EQ(sample->shape(), (std::vector<size_t>{2}));
    double s0 = sample->at({0});
    EXPECT_TRUE(s0 >= 0.0 && s0 < 3.0);
}

TEST(RLTest, NormalDistribution) {
    auto loc = Tensor::create({2, 2}, std::vector<double>{0.0, 1.0, 2.0, -1.0}, true);
    auto scale = Tensor::create({2, 2}, std::vector<double>{1.0, 2.0, 0.5, 1.5}, true);
    Normal dist(loc, scale);

    // Log prob evaluated at the mean
    auto lp = dist.log_prob(loc);
    ASSERT_EQ(lp->shape(), (std::vector<size_t>{2}));

    // Analytical: -0.5 * sum_i(ln(2 * pi * sigma_i^2))
    double expected_lp_0 = -0.5 * (std::log(2.0 * std::numbers::pi * 1.0 * 1.0) +
                                   std::log(2.0 * std::numbers::pi * 2.0 * 2.0));
    EXPECT_NEAR(lp->at({0}), expected_lp_0, 1e-6);

    // Entropy: 0.5 * sum_i(1 + ln(2 * pi * sigma_i^2))
    auto ent = dist.entropy();
    ASSERT_EQ(ent->shape(), (std::vector<size_t>{2}));
    double expected_ent_0 = 0.5 * (1.0 + std::log(2.0 * std::numbers::pi * 1.0 * 1.0) +
                                   1.0 + std::log(2.0 * std::numbers::pi * 2.0 * 2.0));
    EXPECT_NEAR(ent->at({0}), expected_ent_0, 1e-6);

    // Reparameterized sampling & gradient flow
    auto sample = dist.rsample(12345);
    auto loss = sample->sum();
    loss->backward();
    ASSERT_TRUE(loc->grad() != nullptr);
    ASSERT_TRUE(scale->grad() != nullptr);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(loc->grad()->at({i / 2, i % 2}), 1.0, 1e-6);
    }
}

TEST(RLTest, TanhNormalDistribution) {
    auto loc = Tensor::create({1, 2}, std::vector<double>{0.0, 0.5}, true);
    auto scale = Tensor::create({1, 2}, std::vector<double>{1.0, 1.0}, true);
    TanhNormal dist(loc, scale);

    auto [action, pre_tanh] = dist.rsample_with_pre_tanh(42);
    ASSERT_EQ(action->shape(), (std::vector<size_t>{1, 2}));
    ASSERT_EQ(pre_tanh->shape(), (std::vector<size_t>{1, 2}));

    // Action must be bounded in (-1, 1)
    EXPECT_TRUE(std::abs(action->at({0, 0})) < 1.0);
    EXPECT_TRUE(std::abs(action->at({0, 1})) < 1.0);

    // Log prob with and without pre_tanh should be identical
    auto lp1 = dist.log_prob(action, pre_tanh);
    auto lp2 = dist.log_prob(action);
    EXPECT_NEAR(lp1->at({0}), lp2->at({0}), 1e-5);

    // Reparameterization gradient flow
    auto loss = action->sum();
    loss->backward();
    ASSERT_TRUE(loc->grad() != nullptr);
    ASSERT_TRUE(scale->grad() != nullptr);
}

TEST(RLTest, SpacesAndEnvironments) {
    // DiscreteSpace
    DiscreteSpace d_space(5);
    EXPECT_EQ(d_space.n(), 5);
    for (size_t i = 0; i < 20; ++i) {
        size_t s = d_space.sample(i + 1);
        EXPECT_LT(s, 5);
        EXPECT_TRUE(d_space.contains({static_cast<double>(s)}));
    }
    EXPECT_FALSE(d_space.contains({-1.0}));
    EXPECT_FALSE(d_space.contains({5.0}));

    // BoxSpace
    BoxSpace b_space({-2.0, -1.0}, {2.0, 1.0}, {2});
    auto b_sample = b_space.sample(42);
    EXPECT_EQ(b_sample.size(), 2);
    EXPECT_TRUE(b_space.contains(b_sample));
    EXPECT_FALSE(b_space.contains({3.0, 0.0}));

    // CartPole
    CartPole cp;
    auto cp_obs = cp.reset(42);
    EXPECT_EQ(cp_obs.size(), 4);
    EXPECT_TRUE(cp.observation_space().contains(cp_obs));

    auto cp_step = cp.step({1.0});
    EXPECT_EQ(cp_step.next_obs.size(), 4);
    EXPECT_DOUBLE_EQ(cp_step.reward, 1.0);

    // Pendulum
    Pendulum pend;
    auto pend_obs = pend.reset(123);
    EXPECT_EQ(pend_obs.size(), 3);
    EXPECT_TRUE(pend.observation_space().contains(pend_obs));

    auto pend_step = pend.step({-0.5});
    EXPECT_EQ(pend_step.next_obs.size(), 3);
    EXPECT_FALSE(pend_step.terminated);
}

TEST(RLTest, RolloutBufferGAE) {
    RolloutBuffer buf(3, {2}, {1}, 1);

    buf.add({1.0, 2.0}, {0.0}, 1.0, false, 0.5, -0.2);
    buf.add({2.0, 3.0}, {1.0}, 2.0, false, 1.0, -0.3);
    buf.add({3.0, 4.0}, {0.0}, 3.0, true, 1.5, -0.4);

    EXPECT_EQ(buf.size(), 3);

    buf.compute_returns_and_advantages(0.0, true, 0.99, 0.95);

    // Step 2 (t=2, done=true):
    // delta = r + gamma * next_val * 0 - V = 3.0 - 1.5 = 1.5
    // adv = 1.5, ret = 1.5 + 1.5 = 3.0
    EXPECT_NEAR(buf.advantages()[2], 1.5, 1e-5);
    EXPECT_NEAR(buf.returns()[2], 3.0, 1e-5);
}

TEST(RLTest, ReplayBufferSampling) {
    ReplayBuffer buf(10, {2}, {1});
    EXPECT_EQ(buf.size(), 0);

    for (size_t i = 0; i < 5; ++i) {
        double v = static_cast<double>(i);
        buf.add({v, v}, {v}, v, {v + 1.0, v + 1.0}, false);
    }
    EXPECT_EQ(buf.size(), 5);

    auto batch = buf.sample(4, 42);
    ASSERT_EQ(batch.obs->shape(), (std::vector<size_t>{4, 2}));
    ASSERT_EQ(batch.actions->shape(), (std::vector<size_t>{4, 1}));
    ASSERT_EQ(batch.rewards->shape(), (std::vector<size_t>{4}));
    ASSERT_EQ(batch.next_obs->shape(), (std::vector<size_t>{4, 2}));
    ASSERT_EQ(batch.dones->shape(), (std::vector<size_t>{4}));
}

TEST(RLTest, TensorConcatAndTanhAutograd) {
    auto a = Tensor::create({2, 3}, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    auto b = Tensor::create({2, 2}, std::vector<double>{7.0, 8.0, 9.0, 10.0}, true);

    auto ab = Tensor::concat({a, b}, -1);
    ASSERT_EQ(ab->shape(), (std::vector<size_t>{2, 5}));
    EXPECT_DOUBLE_EQ(ab->at({0, 2}), 3.0);
    EXPECT_DOUBLE_EQ(ab->at({0, 3}), 7.0);

    auto squashed = ab->tanh();
    auto loss = squashed->sum();
    loss->backward();

    ASSERT_TRUE(a->grad() != nullptr);
    ASSERT_TRUE(b->grad() != nullptr);

    // d/dx tanh(x) = 1 - tanh^2(x)
    double val_a0 = a->at({0, 0});
    double expected_grad_a0 = 1.0 - std::pow(std::tanh(val_a0), 2.0);
    EXPECT_NEAR(a->grad()->at({0, 0}), expected_grad_a0, 1e-6);

    double val_b0 = b->at({0, 0});
    double expected_grad_b0 = 1.0 - std::pow(std::tanh(val_b0), 2.0);
    EXPECT_NEAR(b->grad()->at({0, 0}), expected_grad_b0, 1e-6);
}
