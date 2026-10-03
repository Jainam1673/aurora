#include "aurora/statistical_evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace aurora::evaluation {

double compute_iqm(const std::vector<double>& scores) {
    if (scores.empty()) {
        return 0.0;
    }
    const size_t n = scores.size();
    if (n < 4) {
        double sum = 0.0;
        for (double v : scores) {
            sum += v;
        }
        return sum / static_cast<double>(n);
    }

    std::vector<double> sorted = scores;
    std::sort(sorted.begin(), sorted.end());

    const size_t k1 = n / 4;
    const size_t k2 = n / 4;
    const size_t m = n - k1 - k2;

    double trimmed_sum = 0.0;
    for (size_t i = k1; i < n - k2; ++i) {
        trimmed_sum += sorted[i];
    }

    return trimmed_sum / static_cast<double>(m);
}

std::pair<double, double> bootstrap_ci(
    const std::vector<double>& scores,
    size_t num_bootstraps,
    double confidence_level,
    uint64_t seed) {
    if (scores.empty()) {
        return {0.0, 0.0};
    }
    if (scores.size() == 1) {
        return {scores[0], scores[0]};
    }

    const size_t n = scores.size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    std::vector<double> estimates(num_bootstraps, 0.0);
    std::vector<double> resample(n);

    for (size_t b = 0; b < num_bootstraps; ++b) {
        for (size_t i = 0; i < n; ++i) {
            resample[i] = scores[dist(rng)];
        }
        estimates[b] = compute_iqm(resample);
    }

    std::sort(estimates.begin(), estimates.end());

    const double alpha = 1.0 - confidence_level;
    const double lower_f = std::floor((alpha / 2.0) * static_cast<double>(num_bootstraps));
    const double upper_f = std::ceil((1.0 - alpha / 2.0) * static_cast<double>(num_bootstraps)) - 1.0;

    size_t lower_idx = static_cast<size_t>(std::max(0.0, lower_f));
    size_t upper_idx = static_cast<size_t>(std::max(0.0, upper_f));

    if (lower_idx >= num_bootstraps) lower_idx = num_bootstraps - 1;
    if (upper_idx >= num_bootstraps) upper_idx = num_bootstraps - 1;

    return {estimates[lower_idx], estimates[upper_idx]};
}

double probability_of_improvement(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    if (x.empty() || y.empty()) {
        return 0.5;
    }

    double sum = 0.0;
    for (double x_val : x) {
        for (double y_val : y) {
            if (x_val > y_val) {
                sum += 1.0;
            } else if (std::abs(x_val - y_val) < 1e-15) {
                sum += 0.5;
            }
        }
    }

    const double total_pairs = static_cast<double>(x.size() * y.size());
    return sum / total_pairs;
}

std::vector<double> performance_profile(
    const std::vector<double>& scores,
    const std::vector<double>& thresholds) {
    std::vector<double> profile;
    profile.reserve(thresholds.size());

    if (scores.empty()) {
        profile.resize(thresholds.size(), 0.0);
        return profile;
    }

    const double n = static_cast<double>(scores.size());
    for (double tau : thresholds) {
        size_t count = 0;
        for (double s : scores) {
            if (s >= tau) {
                ++count;
            }
        }
        profile.push_back(static_cast<double>(count) / n);
    }

    return profile;
}

} // namespace aurora::evaluation
