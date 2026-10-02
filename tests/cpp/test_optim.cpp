#include "aurora/checkpoint.hpp"
#include "aurora/nn.hpp"
#include "aurora/optim.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>

using namespace aurora;
using namespace aurora::nn;
using namespace aurora::optim;

TEST(OptimTest, SGDQuadraticConvergence) {
    auto w = Parameter::create({1}, 0.0);
    auto target = Tensor::create({1}, 3.0);
    SGD opt({w}, 0.05, 0.8);

    double init_loss = std::pow(w->to_vector()[0] - 3.0, 2);
    for (int step = 0; step < 70; ++step) {
        opt.zero_grad();
        auto diff = w - target;
        auto loss = diff * diff;
        loss->backward();
        opt.step();
    }

    double final_loss = std::pow(w->to_vector()[0] - 3.0, 2);
    EXPECT_LT(final_loss, init_loss * 1e-3);
    EXPECT_NEAR(w->to_vector()[0], 3.0, 2e-2);
}

TEST(OptimTest, AdamWLinearRegression) {
    auto lin = std::make_shared<Linear>(1, 1, true, 42);
    AdamW opt(lin->parameters(), 0.05, 0.9, 0.999, 1e-8, 0.0);

    // Target: y = 2 * x + 1
    std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_true = {3.0, 5.0, 7.0, 9.0};

    auto x = Tensor::create({4, 1}, x_data);
    auto target = Tensor::create({4, 1}, y_true);

    double initial_loss = 0.0;
    double final_loss = 0.0;

    for (int step = 0; step < 120; ++step) {
        opt.zero_grad();
        auto pred = lin->forward(x);
        auto diff = pred - target;
        auto loss = (diff * diff)->mean();

        if (step == 0) initial_loss = loss->item();
        final_loss = loss->item();

        loss->backward();
        opt.step();
    }

    EXPECT_LT(final_loss, initial_loss * 0.01);
}

TEST(OptimTest, GradientClipping) {
    auto p1 = Parameter::create({2}, {1.0, 2.0});
    auto p2 = Parameter::create({2}, {3.0, 4.0});
    p1->set_grad(Tensor::create({2}, {100.0, 200.0}));
    p2->set_grad(Tensor::create({2}, {-100.0, -200.0}));

    double orig_norm = clip_grad_norm({p1, p2}, 1.0);
    EXPECT_GT(orig_norm, 300.0);

    // After clipping with max_norm=1.0, total norm must be <= 1.0 + eps
    double clipped_norm = clip_grad_norm({p1, p2}, 1.0);
    EXPECT_NEAR(clipped_norm, 1.0, 1e-4);

    // Clip grad value
    p1->set_grad(Tensor::create({2}, {10.0, -20.0}));
    clip_grad_value({p1}, 0.5);
    auto g1 = p1->grad()->to_vector();
    EXPECT_DOUBLE_EQ(g1[0], 0.5);
    EXPECT_DOUBLE_EQ(g1[1], -0.5);
}

TEST(OptimTest, Schedulers) {
    auto w = Parameter::create({1}, 1.0);
    AdamW opt({w}, 0.1);

    LinearWarmupDecayLR sched(opt, 10, 100, 0.1, 0.01);
    EXPECT_DOUBLE_EQ(sched.get_lr(), 0.0); // step 0

    for (int i = 0; i < 10; ++i) sched.step();
    EXPECT_NEAR(sched.get_lr(), 0.1, 1e-5); // step 10 (peak)

    for (int i = 0; i < 90; ++i) sched.step();
    EXPECT_NEAR(sched.get_lr(), 0.01, 1e-5); // step 100 (min_lr)
}

TEST(OptimTest, CheckpointSaveAndLoadRoundtrip) {
    auto lin1 = std::make_shared<Linear>(3, 2, true, 777);
    AdamW opt1(lin1->parameters(), 0.01);

    // Take a step to populate optimizer state
    auto x = Tensor::randn({2, 3}, 111, true);
    auto y = lin1->forward(x);
    y->sum()->backward();
    opt1.step();

    std::string test_ckpt = "/tmp/aurora_test_checkpoint.json";
    checkpoint::save_checkpoint(test_ckpt, *lin1, &opt1, {{"step", "1"}});
    EXPECT_TRUE(std::filesystem::exists(test_ckpt));

    auto lin2 = std::make_shared<Linear>(3, 2, true, 999);
    AdamW opt2(lin2->parameters(), 0.05);

    auto loaded = checkpoint::load_checkpoint(test_ckpt, lin2.get(), &opt2);

    EXPECT_TRUE(loaded.has_optimizer);
    EXPECT_EQ(loaded.metadata["step"], "1");
    EXPECT_EQ(opt2.step_count(), 1);

    // Weights must match lin1 exactly
    EXPECT_EQ(lin1->weight()->to_vector(), lin2->weight()->to_vector());
    EXPECT_EQ(lin1->bias()->to_vector(), lin2->bias()->to_vector());

    std::filesystem::remove(test_ckpt);
}
