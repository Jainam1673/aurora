#include <gtest/gtest.h>
#include "aurora/reproductions.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace aurora;
using namespace aurora::reproductions;

TEST(ReproductionsTest, LambdaReturnsGoldenAnalytical) {
    // 2 steps: H=2, B=1
    // rewards: r0=1.0, r1=2.0
    // discounts: g0=0.9, g1=0.9
    // values: v0=0.5, v1=1.5, v2=2.5
    // lambda = 0.8
    // V_2^lambda = v2 = 2.5
    // V_1^lambda = 2.0 + 0.9 * (0.2 * 2.5 + 0.8 * 2.5) = 4.25
    // V_0^lambda = 1.0 + 0.9 * (0.2 * 1.5 + 0.8 * 4.25) = 4.33
    auto rewards = std::make_shared<Tensor>(std::vector<size_t>{2, 1}, std::vector<double>{1.0, 2.0}, false);
    auto discounts = std::make_shared<Tensor>(std::vector<size_t>{2, 1}, std::vector<double>{0.9, 0.9}, false);
    auto values = std::make_shared<Tensor>(std::vector<size_t>{3, 1}, std::vector<double>{0.5, 1.5, 2.5}, false);

    auto returns = compute_lambda_returns(rewards, discounts, values, 0.8);
    ASSERT_EQ(returns->shape(), (std::vector<size_t>{2, 1}));

    EXPECT_NEAR(returns->data()[0], 4.33, 1e-12);
    EXPECT_NEAR(returns->data()[1], 4.25, 1e-12);
}

TEST(ReproductionsTest, CEMPlannerOptimization) {
    CEMConfig config;
    config.horizon = 3;
    config.num_samples = 20;
    config.num_elites = 5;
    config.iterations = 4;
    config.action_dim = 2;
    config.alpha = 0.6;
    config.seed = 123;

    CEMPlanner planner(config);

    // Latent dynamics: z_{t+1} = z_t + 0.1 * a_t
    auto dynamics = [](const std::vector<double>& z, const std::vector<double>& a) {
        std::vector<double> next_z(z.size());
        for (size_t i = 0; i < z.size(); ++i) {
            next_z[i] = z[i] + 0.1 * a[i];
        }
        return next_z;
    };

    // Reward: quadratic reward penalizing distance from target [1.0, 1.0]
    auto reward = [](const std::vector<double>& z, const std::vector<double>& /*a*/) {
        double dist_sq = 0.0;
        for (double val : z) {
            const double diff = val - 1.0;
            dist_sq += diff * diff;
        }
        return -dist_sq;
    };

    auto terminal_value = [](const std::vector<double>& z) {
        double dist_sq = 0.0;
        for (double val : z) {
            const double diff = val - 1.0;
            dist_sq += diff * diff;
        }
        return -dist_sq;
    };

    std::vector<double> z0 = {0.0, 0.0};
    std::vector<double> action = planner.plan(z0, dynamics, reward, terminal_value);

    ASSERT_EQ(action.size(), 2U);
    // Actions should be positive to move towards 1.0
    EXPECT_GT(action[0], 0.0);
    EXPECT_GT(action[1], 0.0);
    EXPECT_LE(action[0], 1.0);
    EXPECT_LE(action[1], 1.0);
}

TEST(ReproductionsTest, PUCTPlannerSearch) {
    MCTSConfig config;
    config.action_dim = 3;
    config.num_simulations = 20;
    config.discount = 0.95;

    PUCTPlanner planner(config);

    // Mock discrete dynamics
    auto dynamics = [](const std::vector<double>& z, size_t action) {
        std::vector<double> next_z = z;
        next_z[0] += static_cast<double>(action) * 0.1;
        const double r = (action == 1) ? 2.0 : 0.5; // action 1 gives higher reward
        return std::make_pair(next_z, r);
    };

    // Mock prediction function: uniform prior, value 1.0
    auto prediction = [](const std::vector<double>& /*z*/) {
        std::vector<double> priors = {0.33, 0.34, 0.33};
        return std::make_pair(priors, 1.0);
    };

    std::vector<double> root_z = {0.0, 0.0};
    auto [best_action, pi_probs] = planner.search(root_z, dynamics, prediction);

    ASSERT_LT(best_action, 3U);
    ASSERT_EQ(pi_probs.size(), 3U);

    double prob_sum = std::accumulate(pi_probs.begin(), pi_probs.end(), 0.0);
    EXPECT_NEAR(prob_sum, 1.0, 1e-6);

    // Action 1 provides higher reward, should be explored and preferred
    EXPECT_EQ(best_action, 1U);
}

TEST(ReproductionsTest, MBPOBufferManagerHybridSampling) {
    std::vector<size_t> obs_shape = {4};
    std::vector<size_t> action_shape = {2};

    MBPOBufferManager manager(100, 100, obs_shape, action_shape, 0.5);

    // Add 20 env transitions
    for (size_t i = 0; i < 20; ++i) {
        manager.add_env({1.0, 1.0, 1.0, 1.0}, {0.1, 0.2}, 1.0, {1.1, 1.1, 1.1, 1.1}, false);
    }
    // Add 20 model transitions
    for (size_t i = 0; i < 20; ++i) {
        manager.add_model({2.0, 2.0, 2.0, 2.0}, {-0.1, -0.2}, 0.5, {2.1, 2.1, 2.1, 2.1}, false);
    }

    EXPECT_EQ(manager.env_size(), 20U);
    EXPECT_EQ(manager.model_size(), 20U);

    ReplayBatch batch = manager.sample_mixed(10, 42);
    ASSERT_EQ(batch.obs->shape(), (std::vector<size_t>{10, 4}));
    ASSERT_EQ(batch.actions->shape(), (std::vector<size_t>{10, 2}));
    ASSERT_EQ(batch.rewards->shape(), (std::vector<size_t>{10}));
    ASSERT_EQ(batch.next_obs->shape(), (std::vector<size_t>{10, 4}));
    ASSERT_EQ(batch.dones->shape(), (std::vector<size_t>{10}));
}
