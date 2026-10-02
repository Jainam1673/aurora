#pragma once

#include "aurora/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aurora {

class RolloutBuffer {
public:
    RolloutBuffer(size_t buffer_size,
                  std::vector<size_t> obs_shape,
                  std::vector<size_t> action_shape,
                  size_t batch_size = 1);

    void reset();
    void add(const std::vector<double>& obs,
             const std::vector<double>& action,
             double reward,
             bool done,
             double value,
             double log_prob);

    void compute_returns_and_advantages(double last_value,
                                        bool done,
                                        double gamma = 0.99,
                                        double gae_lambda = 0.95);

    [[nodiscard]] size_t size() const noexcept { return ptr_; }
    [[nodiscard]] size_t buffer_size() const noexcept { return buffer_size_; }
    [[nodiscard]] const std::vector<double>& advantages() const noexcept { return advantages_; }
    [[nodiscard]] const std::vector<double>& returns() const noexcept { return returns_; }
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }
    [[nodiscard]] const std::vector<double>& rewards() const noexcept { return rewards_; }
    [[nodiscard]] const std::vector<double>& log_probs() const noexcept { return log_probs_; }

private:
    size_t buffer_size_;
    std::vector<size_t> obs_shape_;
    std::vector<size_t> action_shape_;
    size_t batch_size_;

    size_t obs_dim_;
    size_t action_dim_;

    std::vector<double> obs_;
    std::vector<double> actions_;
    std::vector<double> rewards_;
    std::vector<double> dones_;
    std::vector<double> values_;
    std::vector<double> log_probs_;
    std::vector<double> advantages_;
    std::vector<double> returns_;

    size_t ptr_{0};
};

struct ReplayBatch {
    std::shared_ptr<Tensor> obs;
    std::shared_ptr<Tensor> actions;
    std::shared_ptr<Tensor> rewards;
    std::shared_ptr<Tensor> next_obs;
    std::shared_ptr<Tensor> dones;
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity,
                 std::vector<size_t> obs_shape,
                 std::vector<size_t> action_shape);

    void add(const std::vector<double>& obs,
             const std::vector<double>& action,
             double reward,
             const std::vector<double>& next_obs,
             bool done);

    [[nodiscard]] ReplayBatch sample(size_t batch_size, uint64_t seed = 0) const;
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

private:
    size_t capacity_;
    std::vector<size_t> obs_shape_;
    std::vector<size_t> action_shape_;

    size_t obs_dim_;
    size_t action_dim_;

    std::vector<double> obs_;
    std::vector<double> actions_;
    std::vector<double> rewards_;
    std::vector<double> next_obs_;
    std::vector<double> dones_;

    size_t ptr_{0};
    size_t size_{0};
};

} // namespace aurora
