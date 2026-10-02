#pragma once

#include "aurora/tensor.hpp"
#include "aurora/nn.hpp"
#include "aurora/buffer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace aurora {
namespace world_model {

// --- Heteroscedastic Gaussian NLL Loss ---

std::shared_ptr<Tensor> gaussian_nll_loss(
    const std::shared_ptr<Tensor>& mean,
    const std::shared_ptr<Tensor>& log_var,
    const std::shared_ptr<Tensor>& target);

// --- Uncertainty Decomposition ---

struct UncertaintyMetrics {
    std::shared_ptr<Tensor> mean;
    std::shared_ptr<Tensor> aleatoric;
    std::shared_ptr<Tensor> epistemic;
    std::shared_ptr<Tensor> total;
    std::vector<double> disagreement; // Max epistemic variance per sample
};

UncertaintyMetrics decompose_uncertainty(
    const std::vector<std::shared_ptr<Tensor>>& means,
    const std::vector<std::shared_ptr<Tensor>>& vars);

// --- Ensemble Dynamics Member ---

class EnsembleMember : public nn::Module {
public:
    EnsembleMember(
        size_t in_features,
        size_t out_features,
        const std::vector<size_t>& hidden_dims,
        const std::string& activation = "silu",
        double log_var_min = -10.0,
        double log_var_max = 2.0,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward_member(
        const std::shared_ptr<Tensor>& x);

    [[nodiscard]] size_t in_features() const noexcept { return in_features_; }
    [[nodiscard]] size_t out_features() const noexcept { return out_features_; }

private:
    size_t in_features_;
    size_t out_features_;
    double log_var_min_;
    double log_var_max_;

    std::shared_ptr<nn::MLP> backbone_;
    std::shared_ptr<nn::Linear> mu_head_;
    std::shared_ptr<nn::Linear> log_var_head_;
};

// --- Ensemble Dynamics Model ---

class EnsembleDynamics : public nn::Module {
public:
    EnsembleDynamics(
        size_t obs_dim,
        size_t action_dim,
        size_t ensemble_size = 5,
        const std::vector<size_t>& hidden_dims = {200, 200, 200},
        const std::string& activation = "silu",
        double log_var_min = -10.0,
        double log_var_max = 2.0,
        bool predict_reward = true,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::pair<std::vector<std::shared_ptr<Tensor>>, std::vector<std::shared_ptr<Tensor>>> forward_ensemble(
        const std::shared_ptr<Tensor>& obs,
        const std::shared_ptr<Tensor>& action);

    std::shared_ptr<Tensor> compute_loss(
        const std::shared_ptr<Tensor>& obs,
        const std::shared_ptr<Tensor>& action,
        const std::shared_ptr<Tensor>& next_obs,
        const std::shared_ptr<Tensor>& reward = nullptr);

    // CPU fast predictions for stepping and imagination
    // Returns: next_states (E, obs_dim), rewards (E, 1), variances (E, out_dim)
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    predict(const std::vector<double>& obs, const std::vector<double>& action);

    std::pair<std::vector<double>, double> step(
        const std::vector<double>& obs,
        const std::vector<double>& action,
        std::optional<size_t> member_idx = std::nullopt);

    [[nodiscard]] size_t obs_dim() const noexcept { return obs_dim_; }
    [[nodiscard]] size_t action_dim() const noexcept { return action_dim_; }
    [[nodiscard]] size_t ensemble_size() const noexcept { return ensemble_size_; }
    [[nodiscard]] size_t out_dim() const noexcept { return out_dim_; }
    [[nodiscard]] bool predict_reward() const noexcept { return predict_reward_; }
    [[nodiscard]] double log_var_min() const noexcept { return log_var_min_; }
    [[nodiscard]] double log_var_max() const noexcept { return log_var_max_; }

    [[nodiscard]] const std::vector<std::shared_ptr<EnsembleMember>>& members() const noexcept {
        return members_;
    }

private:
    size_t obs_dim_;
    size_t action_dim_;
    size_t ensemble_size_;
    size_t out_dim_;
    double log_var_min_;
    double log_var_max_;
    bool predict_reward_;

    std::vector<std::shared_ptr<EnsembleMember>> members_;
    mutable std::mt19937_64 rng_;
};

// --- Imagination Rollout Engine ---

struct ImaginationResult {
    size_t total_transitions{0};
    size_t truncated_trajectories{0};
    size_t completed_trajectories{0};
    double mean_horizon{0.0};
    double mean_reward{0.0};
    double mean_epistemic_uncertainty{0.0};
    std::vector<std::tuple<std::vector<double>, std::vector<double>, double, std::vector<double>, bool>> transitions;
};

class ImaginationEngine {
public:
    ImaginationEngine(
        std::shared_ptr<EnsembleDynamics> dynamics,
        std::function<std::vector<double>(const std::vector<double>&)> policy,
        size_t max_horizon = 15,
        double uncertainty_threshold = 0.5,
        bool adaptive_truncation = true,
        std::string sampling_mode = "ts1",
        uint64_t seed = 42);

    std::tuple<std::vector<std::tuple<std::vector<double>, std::vector<double>, double, std::vector<double>, bool>>, bool, std::vector<double>>
    rollout_single(const std::vector<double>& initial_state);

    ImaginationResult generate_rollouts(const std::vector<std::vector<double>>& initial_states);

    ImaginationResult inject_into_buffer(ReplayBuffer& replay_buffer, const std::vector<std::vector<double>>& initial_states);

private:
    std::shared_ptr<EnsembleDynamics> dynamics_;
    std::function<std::vector<double>(const std::vector<double>&)> policy_;
    size_t max_horizon_;
    double uncertainty_threshold_;
    bool adaptive_truncation_;
    std::string sampling_mode_;
    mutable std::mt19937_64 rng_;
};

// --- GRUCell ---

class GRUCell : public nn::Module {
public:
    GRUCell(size_t input_dim, size_t hidden_dim, uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& h);

    [[nodiscard]] size_t input_dim() const noexcept { return input_dim_; }
    [[nodiscard]] size_t hidden_dim() const noexcept { return hidden_dim_; }

private:
    size_t input_dim_;
    size_t hidden_dim_;

    std::shared_ptr<nn::Linear> w_ir_;
    std::shared_ptr<nn::Linear> w_hr_;
    std::shared_ptr<nn::Linear> w_iz_;
    std::shared_ptr<nn::Linear> w_hz_;
    std::shared_ptr<nn::Linear> w_in_;
    std::shared_ptr<nn::Linear> w_hn_;
};

// --- RSSM (Recurrent State-Space Model) ---

struct RSSMState {
    std::shared_ptr<Tensor> h;
    std::shared_ptr<Tensor> z;
    std::shared_ptr<Tensor> mu;
    std::shared_ptr<Tensor> log_std;
};

class RSSM : public nn::Module {
public:
    RSSM(
        size_t obs_dim,
        size_t action_dim,
        size_t deter_dim = 200,
        size_t stoch_dim = 30,
        size_t hidden_dim = 200,
        double kl_alpha = 0.8,
        double kl_scale = 1.0,
        double min_std = 0.1,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    RSSMState initial_state(size_t batch_size = 1) const;

    std::pair<RSSMState, RSSMState> observe_step(
        const RSSMState& prev_state,
        const std::shared_ptr<Tensor>& action,
        const std::shared_ptr<Tensor>& obs);

    RSSMState imagine_step(
        const RSSMState& prev_state,
        const std::shared_ptr<Tensor>& action);

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
    decode(const RSSMState& state);

    static std::shared_ptr<Tensor> kl_divergence(
        const std::shared_ptr<Tensor>& mu_q,
        const std::shared_ptr<Tensor>& log_std_q,
        const std::shared_ptr<Tensor>& mu_p,
        const std::shared_ptr<Tensor>& log_std_p);

    std::pair<std::shared_ptr<Tensor>, std::map<std::string, double>> compute_loss(
        const RSSMState& prior_state,
        const RSSMState& post_state,
        const std::shared_ptr<Tensor>& target_obs,
        const std::shared_ptr<Tensor>& target_reward,
        const std::shared_ptr<Tensor>& target_done = nullptr);

    [[nodiscard]] size_t obs_dim() const noexcept { return obs_dim_; }
    [[nodiscard]] size_t action_dim() const noexcept { return action_dim_; }
    [[nodiscard]] size_t deter_dim() const noexcept { return deter_dim_; }
    [[nodiscard]] size_t stoch_dim() const noexcept { return stoch_dim_; }

private:
    size_t obs_dim_;
    size_t action_dim_;
    size_t deter_dim_;
    size_t stoch_dim_;
    double kl_alpha_;
    double kl_scale_;
    double min_std_;

    std::shared_ptr<GRUCell> cell_;
    std::shared_ptr<nn::MLP> prior_mlp_;
    std::shared_ptr<nn::Linear> prior_mu_;
    std::shared_ptr<nn::Linear> prior_log_std_;

    std::shared_ptr<nn::MLP> post_mlp_;
    std::shared_ptr<nn::Linear> post_mu_;
    std::shared_ptr<nn::Linear> post_log_std_;

    std::shared_ptr<nn::MLP> obs_decoder_;
    std::shared_ptr<nn::MLP> reward_decoder_;
    std::shared_ptr<nn::MLP> cont_decoder_;

    mutable std::mt19937_64 rng_;

    std::shared_ptr<Tensor> sample_gaussian(
        const std::shared_ptr<Tensor>& mu,
        const std::shared_ptr<Tensor>& log_std) const;
};

} // namespace world_model
} // namespace aurora
