#pragma once

#include "aurora/buffer.hpp"
#include "aurora/nn.hpp"
#include "aurora/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aurora {
namespace reproductions {

// --- Dreamer: Generalized Lambda-Returns ---

std::shared_ptr<Tensor> compute_lambda_returns(
    const std::shared_ptr<Tensor>& rewards,
    const std::shared_ptr<Tensor>& discounts,
    const std::shared_ptr<Tensor>& values,
    double lambda = 0.95);

// --- TD-MPC: Cross-Entropy Method (CEM) Trajectory Planner ---

struct CEMConfig {
    size_t horizon{5};
    size_t num_samples{50};
    size_t num_elites{10};
    size_t iterations{5};
    size_t action_dim{1};
    double alpha{0.5};
    double min_std{0.05};
    double clip_min{-1.0};
    double clip_max{1.0};
    uint64_t seed{42};
};

class CEMPlanner {
public:
    explicit CEMPlanner(CEMConfig config);

    std::vector<double> plan(
        const std::vector<double>& initial_latent,
        const std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>& dynamics,
        const std::function<double(const std::vector<double>&, const std::vector<double>&)>& reward,
        const std::function<double(const std::vector<double>&)>& terminal_value,
        double gamma = 0.99);

    [[nodiscard]] const CEMConfig& config() const noexcept { return config_; }

private:
    CEMConfig config_;
};

// --- MuZero: PUCT Latent Search Tree ---

struct MCTSConfig {
    size_t action_dim{2};
    size_t num_simulations{50};
    double discount{0.99};
    double c1{1.25};
    double c2{19652.0};
};

struct PUCTNode {
    double prior{0.0};
    size_t visit_count{0};
    double value_sum{0.0};
    double reward{0.0};
    std::vector<double> latent_state{};
    std::vector<std::unique_ptr<PUCTNode>> children{};

    [[nodiscard]] double value() const noexcept {
        return visit_count > 0 ? value_sum / static_cast<double>(visit_count) : 0.0;
    }

    [[nodiscard]] bool expanded() const noexcept {
        return !children.empty();
    }
};

class PUCTPlanner {
public:
    explicit PUCTPlanner(MCTSConfig config);

    std::pair<size_t, std::vector<double>> search(
        const std::vector<double>& root_latent,
        const std::function<std::pair<std::vector<double>, double>(const std::vector<double>&, size_t)>& dynamics_step,
        const std::function<std::pair<std::vector<double>, double>(const std::vector<double>&)>& prediction_step);

    [[nodiscard]] const MCTSConfig& config() const noexcept { return config_; }

private:
    MCTSConfig config_;
};

// --- MBPO: Hybrid Environment & Model Replay Sampler ---

class MBPOBufferManager {
public:
    MBPOBufferManager(
        size_t env_capacity,
        size_t model_capacity,
        std::vector<size_t> obs_shape,
        std::vector<size_t> action_shape,
        double model_ratio = 0.5);

    void add_env(
        const std::vector<double>& obs,
        const std::vector<double>& action,
        double reward,
        const std::vector<double>& next_obs,
        bool done);

    void add_model(
        const std::vector<double>& obs,
        const std::vector<double>& action,
        double reward,
        const std::vector<double>& next_obs,
        bool done);

    [[nodiscard]] ReplayBatch sample_mixed(size_t batch_size, uint64_t seed = 0);

    [[nodiscard]] size_t env_size() const noexcept { return env_buffer_.size(); }
    [[nodiscard]] size_t model_size() const noexcept { return model_buffer_.size(); }
    [[nodiscard]] double model_ratio() const noexcept { return model_ratio_; }

private:
    ReplayBuffer env_buffer_;
    ReplayBuffer model_buffer_;
    double model_ratio_;
};

} // namespace reproductions
} // namespace aurora
