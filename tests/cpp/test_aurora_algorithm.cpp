#include <gtest/gtest.h>
#include "aurora/aurora_algorithm.hpp"

#include <cmath>
#include <vector>

using namespace aurora;
using namespace aurora::algorithm;

TEST(AURORAAlgorithmTest, ThresholdUpdateDecay) {
    AdaptiveHorizonConfig config;
    config.tau_base = 0.8;
    config.kappa = 2.0;

    AdaptiveHorizonScheduler scheduler(config);
    EXPECT_DOUBLE_EQ(scheduler.current_tau(), 0.8);

    // Zero validation loss => threshold remains tau_base
    double tau0 = scheduler.update_threshold(0.0);
    EXPECT_NEAR(tau0, 0.8, 1e-12);

    // Validation loss = 0.5 => tau_base * exp(-2.0 * 0.5) = 0.8 * exp(-1.0)
    double tau_half = scheduler.update_threshold(0.5);
    EXPECT_NEAR(tau_half, 0.8 * std::exp(-1.0), 1e-12);

    // Validation loss >= 1.0 clamped => tau_base * exp(-2.0)
    double tau1 = scheduler.update_threshold(1.5);
    EXPECT_NEAR(tau1, 0.8 * std::exp(-2.0), 1e-12);
}

TEST(AURORAAlgorithmTest, HorizonTruncationConditions) {
    AdaptiveHorizonConfig config;
    config.horizon_min = 2;
    config.horizon_max = 8;
    config.tau_base = 0.5;
    config.kappa = 0.0; // keep tau constant at 0.5
    config.budget_max = 1.0;
    config.gamma = 1.0;

    AdaptiveHorizonScheduler scheduler(config);

    // Step 0: below horizon_min => never truncated even with huge uncertainty
    auto [trunc0, b0] = scheduler.should_truncate(0, 10.0, 0.0);
    EXPECT_FALSE(trunc0);
    EXPECT_NEAR(b0, 10.0, 1e-12);

    // Step 2: above horizon_min, peak uncertainty exceeded (> 0.5)
    auto [trunc2, b2] = scheduler.should_truncate(2, 0.6, 0.2);
    EXPECT_TRUE(trunc2);
    EXPECT_NEAR(b2, 0.2, 1e-12);

    // Step 3: peak uncertainty below tau, but cumulative budget exceeded (> 1.0)
    auto [trunc3, b3] = scheduler.should_truncate(3, 0.4, 0.8);
    EXPECT_TRUE(trunc3);
    EXPECT_NEAR(b3, 1.2, 1e-12);

    // Step 8: reaches horizon_max => truncated
    auto [trunc8, b8] = scheduler.should_truncate(8, 0.1, 0.2);
    EXPECT_TRUE(trunc8);
}

TEST(AURORAAlgorithmTest, ComputeAdaptiveHorizon) {
    AdaptiveHorizonConfig config;
    config.horizon_min = 1;
    config.horizon_max = 10;
    config.tau_base = 0.5;
    config.budget_max = 1.5;
    config.gamma = 0.9;

    AdaptiveHorizonScheduler scheduler(config);

    // Case 1: Low uncertainty throughout -> runs to horizon_max
    std::vector<double> low_u(10, 0.05);
    size_t h_low = scheduler.compute_adaptive_horizon(low_u);
    EXPECT_EQ(h_low, 10);

    // Case 2: High peak uncertainty at step 3 -> truncates at step 3
    std::vector<double> spike_u = {0.1, 0.1, 0.1, 0.9, 0.1, 0.1};
    size_t h_spike = scheduler.compute_adaptive_horizon(spike_u);
    EXPECT_EQ(h_spike, 3);
}

TEST(AURORAAlgorithmTest, DynamicBlendingRatio) {
    DynamicBlendingConfig config;
    config.eta_max = 0.8;
    config.eta_min = 0.1;
    config.u_target = 0.4;
    config.momentum = 0.5;

    DynamicBlendingController controller(config);
    EXPECT_DOUBLE_EQ(controller.current_eta(), 0.1);

    // Near zero uncertainty -> raw eta = 0.8
    // With momentum 0.5: eta_1 = 0.5 * 0.1 + 0.5 * 0.8 = 0.45
    double eta1 = controller.compute_ratio(0.0);
    EXPECT_NEAR(eta1, 0.45, 1e-12);

    // Another zero uncertainty: eta_2 = 0.5 * 0.45 + 0.5 * 0.8 = 0.625
    double eta2 = controller.compute_ratio(0.0);
    EXPECT_NEAR(eta2, 0.625, 1e-12);

    // Extreme uncertainty (1.0 > u_target): raw eta = eta_min = 0.1
    // eta_3 = 0.5 * 0.625 + 0.5 * 0.1 = 0.3625
    double eta3 = controller.compute_ratio(1.0);
    EXPECT_NEAR(eta3, 0.3625, 1e-12);
}

TEST(AURORAAlgorithmTest, PessimisticValuePenalty) {
    auto q1 = std::make_shared<Tensor>(std::vector<size_t>{3, 1}, std::vector<double>{10.0, 5.0, 8.0}, false);
    auto q2 = std::make_shared<Tensor>(std::vector<size_t>{3, 1}, std::vector<double>{7.0, 9.0, 6.0}, false);
    auto u_epi = std::make_shared<Tensor>(std::vector<size_t>{3, 1}, std::vector<double>{1.0, 2.0, 0.5}, false);

    // min(q1, q2) = [7.0, 5.0, 6.0]
    // beta_pess = 0.5
    // q_pess = [7.0 - 0.5, 5.0 - 1.0, 6.0 - 0.25] = [6.5, 4.0, 5.75]
    auto q_pess = compute_pessimistic_value(q1, q2, u_epi, 0.5);
    ASSERT_EQ(q_pess->shape(), (std::vector<size_t>{3, 1}));
    EXPECT_NEAR(q_pess->data()[0], 6.5, 1e-12);
    EXPECT_NEAR(q_pess->data()[1], 4.0, 1e-12);
    EXPECT_NEAR(q_pess->data()[2], 5.75, 1e-12);
}

TEST(AURORAAlgorithmTest, ActiveExplorationTrigger) {
    EXPECT_FALSE(should_trigger_active_exploration(0.4, 0.5));
    EXPECT_TRUE(should_trigger_active_exploration(0.6, 0.5));
}
