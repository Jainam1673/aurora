#include "aurora/world_model.hpp"
#include "aurora/tensor.hpp"
#include "aurora/buffer.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace aurora;
using namespace aurora::world_model;

TEST(WorldModelTest, GaussianNLLLossForwardAndBackward) {
    auto mean = Tensor::create({2, 3}, {0.0, 1.0, -1.0, 2.0, 0.5, -0.5}, true);
    auto log_var = Tensor::create({2, 3}, {0.0, 0.5, -0.5, 0.2, -0.2, 0.1}, true);
    auto target = Tensor::create({2, 3}, {0.2, 0.8, -0.9, 1.8, 0.6, -0.4}, false);

    auto loss = gaussian_nll_loss(mean, log_var, target);
    EXPECT_GT(loss->item(), 0.0);

    loss->backward();
    EXPECT_NE(mean->grad(), nullptr);
    EXPECT_NE(log_var->grad(), nullptr);
    EXPECT_EQ(mean->grad()->shape(), mean->shape());
    EXPECT_EQ(log_var->grad()->shape(), log_var->shape());
}

TEST(WorldModelTest, UncertaintyDecompositionAgreementAndDisagreement) {
    // 1. Perfect agreement
    auto m1 = Tensor::create({1, 2}, {1.0, 2.0}, false);
    auto m2 = Tensor::create({1, 2}, {1.0, 2.0}, false);
    auto v1 = Tensor::create({1, 2}, {0.2, 0.3}, false);
    auto v2 = Tensor::create({1, 2}, {0.4, 0.5}, false);

    auto unc_agree = decompose_uncertainty({m1, m2}, {v1, v2});
    auto ep_agree = unc_agree.epistemic->to_vector();
    EXPECT_NEAR(ep_agree[0], 0.0, 1e-12);
    EXPECT_NEAR(ep_agree[1], 0.0, 1e-12);
    EXPECT_NEAR(unc_agree.disagreement[0], 0.0, 1e-12);

    auto al_agree = unc_agree.aleatoric->to_vector();
    EXPECT_NEAR(al_agree[0], 0.3, 1e-12);
    EXPECT_NEAR(al_agree[1], 0.4, 1e-12);

    auto tot_agree = unc_agree.total->to_vector();
    EXPECT_NEAR(tot_agree[0], al_agree[0], 1e-12);
    EXPECT_NEAR(tot_agree[1], al_agree[1], 1e-12);

    // 2. Disagreement along dimension 0
    auto md1 = Tensor::create({1, 2}, {0.0, 1.0}, false);
    auto md2 = Tensor::create({1, 2}, {2.0, 1.0}, false);
    auto vd = Tensor::create({1, 2}, {0.5, 0.5}, false);

    auto unc_disagree = decompose_uncertainty({md1, md2}, {vd, vd});
    auto ep_disagree = unc_disagree.epistemic->to_vector();
    EXPECT_NEAR(ep_disagree[0], 1.0, 1e-12);
    EXPECT_NEAR(ep_disagree[1], 0.0, 1e-12);
    EXPECT_NEAR(unc_disagree.disagreement[0], 1.0, 1e-12);
}

TEST(WorldModelTest, EnsembleDynamicsForwardAndLoss) {
    size_t obs_dim = 3;
    size_t action_dim = 1;
    size_t ensemble_size = 3;

    EnsembleDynamics dynamics(obs_dim, action_dim, ensemble_size, {16, 16}, "silu", -10.0, 2.0, true, 42);
    EXPECT_EQ(dynamics.members().size(), ensemble_size);

    auto obs = Tensor::randn({4, obs_dim}, 101, false);
    auto act = Tensor::randn({4, action_dim}, 202, false);
    auto next_obs = Tensor::randn({4, obs_dim}, 303, false);
    auto reward = Tensor::randn({4, 1}, 404, false);

    auto [means, log_vars] = dynamics.forward_ensemble(obs, act);
    EXPECT_EQ(means.size(), ensemble_size);
    EXPECT_EQ(log_vars.size(), ensemble_size);

    for (size_t e = 0; e < ensemble_size; ++e) {
        EXPECT_EQ(means[e]->shape(), (std::vector<size_t>{4, obs_dim + 1}));
        EXPECT_EQ(log_vars[e]->shape(), (std::vector<size_t>{4, obs_dim + 1}));
    }

    auto loss = dynamics.compute_loss(obs, act, next_obs, reward);
    EXPECT_GT(loss->item(), 0.0);

    loss->backward();
    for (const auto& member : dynamics.members()) {
        for (const auto& p : member->parameters()) {
            if (p->requires_grad()) {
                EXPECT_NE(p->grad(), nullptr);
            }
        }
    }
}

TEST(WorldModelTest, EnsembleDynamicsPredictAndStep) {
    size_t obs_dim = 2;
    size_t action_dim = 1;
    size_t ensemble_size = 4;

    EnsembleDynamics dynamics(obs_dim, action_dim, ensemble_size, {16}, "silu", -10.0, 2.0, true, 99);

    std::vector<double> obs = {1.0, -0.5};
    std::vector<double> act = {0.5};

    auto [next_states, rewards, variances] = dynamics.predict(obs, act);
    EXPECT_EQ(next_states.size(), ensemble_size);
    EXPECT_EQ(rewards.size(), ensemble_size);
    EXPECT_EQ(variances.size(), ensemble_size);

    for (size_t e = 0; e < ensemble_size; ++e) {
        EXPECT_EQ(next_states[e].size(), obs_dim);
        EXPECT_EQ(rewards[e].size(), 1);
        EXPECT_EQ(variances[e].size(), obs_dim + 1);
    }

    auto [ns, r] = dynamics.step(obs, act, 0);
    EXPECT_EQ(ns.size(), obs_dim);
    EXPECT_NEAR(ns[0], next_states[0][0], 1e-12);
    EXPECT_NEAR(r, rewards[0][0], 1e-12);
}

TEST(WorldModelTest, ImaginationEngineRolloutAndTruncation) {
    size_t obs_dim = 2;
    size_t action_dim = 1;
    auto dynamics = std::make_shared<EnsembleDynamics>(obs_dim, action_dim, 3, std::vector<size_t>{16}, "silu", -10.0, 2.0, true, 42);

    auto policy = [](const std::vector<double>&) -> std::vector<double> {
        return {0.0};
    };

    // Test with artificially tiny threshold: must truncate early
    ImaginationEngine engine_trunc(dynamics, policy, 20, 1e-6, true, "ts1", 42);
    auto res_trunc = engine_trunc.generate_rollouts({{1.0, 0.0}, {-0.5, 0.5}});

    EXPECT_GT(res_trunc.truncated_trajectories, 0);
    EXPECT_LT(res_trunc.mean_horizon, 20.0);

    // Test with large threshold: runs full horizon
    ImaginationEngine engine_full(dynamics, policy, 5, 100.0, true, "ts1", 42);
    auto res_full = engine_full.generate_rollouts({{1.0, 0.0}, {-0.5, 0.5}});

    EXPECT_EQ(res_full.truncated_trajectories, 0);
    EXPECT_DOUBLE_EQ(res_full.mean_horizon, 5.0);

    // Inject into buffer
    ReplayBuffer buf(100, {obs_dim}, {action_dim});
    auto res_buf = engine_full.inject_into_buffer(buf, {{0.1, 0.2}});
    EXPECT_EQ(res_buf.total_transitions, 5);
    EXPECT_EQ(buf.size(), 5);
}

TEST(WorldModelTest, GRUCellForwardAndBackward) {
    GRUCell cell(4, 8, 42);
    auto x = Tensor::randn({2, 4}, 10, true);
    auto h = Tensor::randn({2, 8}, 20, true);

    auto h_next = cell.forward(x, h);
    EXPECT_EQ(h_next->shape(), (std::vector<size_t>{2, 8}));

    auto loss = h_next->sum();
    loss->backward();
    EXPECT_NE(x->grad(), nullptr);
    EXPECT_NE(h->grad(), nullptr);
}

TEST(WorldModelTest, RSSMStepAndDecode) {
    size_t obs_dim = 4;
    size_t action_dim = 2;
    size_t deter_dim = 16;
    size_t stoch_dim = 8;

    RSSM rssm(obs_dim, action_dim, deter_dim, stoch_dim, 16, 0.8, 1.0, 0.1, 42);

    auto state = rssm.initial_state(2);
    EXPECT_EQ(state.h->shape(), (std::vector<size_t>{2, deter_dim}));
    EXPECT_EQ(state.z->shape(), (std::vector<size_t>{2, stoch_dim}));

    auto action = Tensor::randn({2, action_dim}, 1, false);
    auto obs = Tensor::randn({2, obs_dim}, 2, false);

    auto [prior_state, post_state] = rssm.observe_step(state, action, obs);
    EXPECT_EQ(prior_state.h->shape(), (std::vector<size_t>{2, deter_dim}));
    EXPECT_EQ(post_state.h->shape(), (std::vector<size_t>{2, deter_dim}));
    EXPECT_EQ(prior_state.z->shape(), (std::vector<size_t>{2, stoch_dim}));
    EXPECT_EQ(post_state.z->shape(), (std::vector<size_t>{2, stoch_dim}));

    auto [obs_hat, r_hat, gamma_hat] = rssm.decode(post_state);
    EXPECT_EQ(obs_hat->shape(), (std::vector<size_t>{2, obs_dim}));
    EXPECT_EQ(r_hat->shape(), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(gamma_hat->shape(), (std::vector<size_t>{2, 1}));

    for (double g : gamma_hat->to_vector()) {
        EXPECT_GE(g, 0.0);
        EXPECT_LE(g, 1.0);
    }
}

TEST(WorldModelTest, RSSMKLDivergenceZeroWhenIdentical) {
    auto mu = Tensor::create({1, 3}, {1.0, -2.0, 0.5}, false);
    auto log_std = Tensor::create({1, 3}, {0.0, 0.2, -0.5}, false);

    auto kl = RSSM::kl_divergence(mu, log_std, mu, log_std);
    EXPECT_NEAR(kl->item(), 0.0, 1e-12);
}

TEST(WorldModelTest, RSSMLossAndBackward) {
    RSSM rssm(3, 1, 8, 4, 8, 0.8, 1.0, 0.1, 42);
    auto state0 = rssm.initial_state(2);

    auto a = Tensor::randn({2, 1}, 10, false);
    auto o = Tensor::randn({2, 3}, 20, false);
    auto r = Tensor::randn({2, 1}, 30, false);
    auto d = Tensor::zeros({2, 1}, false);

    auto [prior_s, post_s] = rssm.observe_step(state0, a, o);
    auto [loss, metrics] = rssm.compute_loss(prior_s, post_s, o, r, d);

    EXPECT_GT(metrics["total_loss"], 0.0);
    EXPECT_TRUE(metrics.contains("kl_loss"));

    loss->backward();
    for (const auto& p : rssm.parameters()) {
        if (p->requires_grad() && p->grad()) {
            for (double g : p->grad()->to_vector()) {
                EXPECT_FALSE(std::isnan(g));
            }
        }
    }
}
