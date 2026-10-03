#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace aurora::evaluation {

// Compute Interquartile Mean (IQM) trimming top and bottom 25%
[[nodiscard]] double compute_iqm(const std::vector<double>& scores);

// Compute non-parametric percentile bootstrap confidence interval for IQM
[[nodiscard]] std::pair<double, double> bootstrap_ci(
    const std::vector<double>& scores,
    size_t num_bootstraps = 2000,
    double confidence_level = 0.95,
    uint64_t seed = 42);

// Probability of improvement P(X > Y)
[[nodiscard]] double probability_of_improvement(
    const std::vector<double>& x,
    const std::vector<double>& y);

// Empirical performance profile: fraction of runs achieving score >= threshold
[[nodiscard]] std::vector<double> performance_profile(
    const std::vector<double>& scores,
    const std::vector<double>& thresholds);

} // namespace aurora::evaluation
