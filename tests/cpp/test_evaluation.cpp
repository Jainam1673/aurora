#include <gtest/gtest.h>
#include "aurora/statistical_evaluation.hpp"

#include <vector>

using namespace aurora::evaluation;

TEST(StatisticalEvaluationTest, ComputeIQMTrimsOutliers) {
    // 8 elements: [-1000.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 10000.0]
    // N=8, k1 = 8/4 = 2, k2 = 2, m = 4.
    // Trimmed elements: 11.0, 12.0, 13.0, 14.0
    // Mean = (11 + 12 + 13 + 14) / 4 = 12.5
    std::vector<double> scores = {-1000.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 10000.0};
    double iqm = compute_iqm(scores);
    EXPECT_NEAR(iqm, 12.5, 1e-12);

    // Empty vector
    EXPECT_DOUBLE_EQ(compute_iqm({}), 0.0);

    // Small vector (< 4 elements)
    std::vector<double> small = {2.0, 4.0, 6.0};
    EXPECT_NEAR(compute_iqm(small), 4.0, 1e-12);
}

TEST(StatisticalEvaluationTest, ProbabilityOfImprovement) {
    // Strictly superior: X = [10, 20], Y = [1, 2] -> P(X > Y) = 1.0
    EXPECT_DOUBLE_EQ(probability_of_improvement({10.0, 20.0}, {1.0, 2.0}), 1.0);

    // Identical: X = [1, 2], Y = [1, 2] -> P(X > Y) = 0.5
    EXPECT_DOUBLE_EQ(probability_of_improvement({1.0, 2.0}, {1.0, 2.0}), 0.5);

    // Empty inputs
    EXPECT_DOUBLE_EQ(probability_of_improvement({}, {1.0}), 0.5);

    // Known partial overlap:
    // X = [2, 4], Y = [1, 3]
    // Pairwise: (2>1: 1), (2>3: 0), (4>1: 1), (4>3: 1) -> sum = 3 / 4 = 0.75
    EXPECT_DOUBLE_EQ(probability_of_improvement({2.0, 4.0}, {1.0, 3.0}), 0.75);
}

TEST(StatisticalEvaluationTest, PerformanceProfile) {
    std::vector<double> scores = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> thresholds = {5.0, 15.0, 25.0, 35.0, 45.0};

    auto profile = performance_profile(scores, thresholds);
    ASSERT_EQ(profile.size(), 5);
    EXPECT_DOUBLE_EQ(profile[0], 1.0);   // all >= 5.0
    EXPECT_DOUBLE_EQ(profile[1], 0.75);  // 20, 30, 40 >= 15.0
    EXPECT_DOUBLE_EQ(profile[2], 0.5);   // 30, 40 >= 25.0
    EXPECT_DOUBLE_EQ(profile[3], 0.25);  // 40 >= 35.0
    EXPECT_DOUBLE_EQ(profile[4], 0.0);   // none >= 45.0
}

TEST(StatisticalEvaluationTest, BootstrapConfidenceInterval) {
    std::vector<double> scores = {10.0, 12.0, 11.0, 13.0, 14.0, 12.5, 11.5, 13.5};
    auto [ci_low, ci_high] = bootstrap_ci(scores, 500, 0.95, 42);

    double point_iqm = compute_iqm(scores);
    EXPECT_LE(ci_low, point_iqm);
    EXPECT_GE(ci_high, point_iqm);
}
