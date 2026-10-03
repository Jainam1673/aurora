#pragma once

#include "aurora/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace aurora::algorithm {

struct AdaptiveHorizonConfig {
    size_t horizon_max{15};
    size_t horizon_min{1};
    double tau_base{0.5};
    double kappa{1.0};
    double budget_max{2.0};
    double gamma{0.99};
};

class AdaptiveHorizonScheduler {
public:
    explicit AdaptiveHorizonScheduler(AdaptiveHorizonConfig config = {})
        : config_(config), current_tau_(config.tau_base) {}

    double update_threshold(double validation_loss) noexcept {
        const double clamped_loss = std::min(1.0, std::max(0.0, validation_loss));
        current_tau_ = config_.tau_base * std::exp(-config_.kappa * clamped_loss);
        return current_tau_;
    }

    [[nodiscard]] std::pair<bool, double> should_truncate(
        size_t step,
        double epistemic_uncertainty,
        double cumulative_budget) const noexcept {
        if (step < config_.horizon_min) {
            const double new_budget = cumulative_budget + std::pow(config_.gamma, static_cast<double>(step)) * epistemic_uncertainty;
            return {false, new_budget};
        }

        if (step >= config_.horizon_max) {
            return {true, cumulative_budget};
        }

        // Peak uncertainty threshold check
        if (epistemic_uncertainty > current_tau_) {
            return {true, cumulative_budget};
        }

        // Cumulative discounted uncertainty budget check
        const double new_budget = cumulative_budget + std::pow(config_.gamma, static_cast<double>(step)) * epistemic_uncertainty;
        if (new_budget > config_.budget_max) {
            return {true, new_budget};
        }

        return {false, new_budget};
    }

    [[nodiscard]] size_t compute_adaptive_horizon(const std::vector<double>& epistemic_trajectory) const {
        double cum_budget = 0.0;
        for (size_t h = 0; h < epistemic_trajectory.size(); ++h) {
            auto [trunc, new_b] = should_truncate(h, epistemic_trajectory[h], cum_budget);
            cum_budget = new_b;
            if (trunc) {
                return std::max(config_.horizon_min, h);
            }
        }
        return std::min(config_.horizon_max, epistemic_trajectory.size());
    }

    [[nodiscard]] size_t compute_adaptive_horizon(const std::shared_ptr<Tensor>& epistemic_trajectory) const {
        return compute_adaptive_horizon(epistemic_trajectory->to_vector());
    }

    [[nodiscard]] double current_tau() const noexcept { return current_tau_; }
    [[nodiscard]] const AdaptiveHorizonConfig& config() const noexcept { return config_; }

private:
    AdaptiveHorizonConfig config_;
    double current_tau_;
};

struct DynamicBlendingConfig {
    double eta_max{0.9};
    double eta_min{0.0};
    double u_target{0.5};
    double momentum{0.8};
};

class DynamicBlendingController {
public:
    explicit DynamicBlendingController(DynamicBlendingConfig config = {})
        : config_(config), current_eta_(config.eta_min) {}

    double compute_ratio(double mean_epistemic_uncertainty) noexcept {
        const double u = std::max(0.0, mean_epistemic_uncertainty);
        const double scaled_u = std::min(1.0, u / std::max(1e-8, config_.u_target));
        double raw_eta = config_.eta_max * (1.0 - scaled_u);
        raw_eta = std::max(config_.eta_min, std::min(config_.eta_max, raw_eta));

        current_eta_ = config_.momentum * current_eta_ + (1.0 - config_.momentum) * raw_eta;
        return current_eta_;
    }

    [[nodiscard]] double current_eta() const noexcept { return current_eta_; }
    [[nodiscard]] const DynamicBlendingConfig& config() const noexcept { return config_; }

private:
    DynamicBlendingConfig config_;
    double current_eta_;
};

// Pessimistic Q-value computation: Q_tilde = min(Q1, Q2) - beta_pess * u_epi
[[nodiscard]] std::shared_ptr<Tensor> compute_pessimistic_value(
    const std::shared_ptr<Tensor>& q1,
    const std::shared_ptr<Tensor>& q2,
    const std::shared_ptr<Tensor>& epistemic_uncertainty,
    double beta_pess);

// Active exploration trigger: returns true if mean epistemic uncertainty exceeds tau_active
[[nodiscard]] bool should_trigger_active_exploration(
    double mean_epistemic_uncertainty,
    double tau_active) noexcept;

} // namespace aurora::algorithm
