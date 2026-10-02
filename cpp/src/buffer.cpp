#include "aurora/buffer.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace aurora {

// --- RolloutBuffer ---

RolloutBuffer::RolloutBuffer(size_t buffer_size,
                             std::vector<size_t> obs_shape,
                             std::vector<size_t> action_shape,
                             size_t batch_size)
    : buffer_size_(buffer_size),
      obs_shape_(std::move(obs_shape)),
      action_shape_(std::move(action_shape)),
      batch_size_(batch_size) {
    obs_dim_ = obs_shape_.empty() ? 1 : std::accumulate(obs_shape_.begin(), obs_shape_.end(), size_t{1}, std::multiplies<size_t>());
    action_dim_ = action_shape_.empty() ? 1 : std::accumulate(action_shape_.begin(), action_shape_.end(), size_t{1}, std::multiplies<size_t>());

    reset();
}

void RolloutBuffer::reset() {
    size_t total_steps = buffer_size_ * batch_size_;
    obs_.assign(total_steps * obs_dim_, 0.0);
    actions_.assign(total_steps * action_dim_, 0.0);
    rewards_.assign(total_steps, 0.0);
    dones_.assign(total_steps, 0.0);
    values_.assign(total_steps, 0.0);
    log_probs_.assign(total_steps, 0.0);
    advantages_.assign(total_steps, 0.0);
    returns_.assign(total_steps, 0.0);
    ptr_ = 0;
}

void RolloutBuffer::add(const std::vector<double>& obs,
                        const std::vector<double>& action,
                        double reward,
                        bool done,
                        double value,
                        double log_prob) {
    if (ptr_ >= buffer_size_) {
        throw std::runtime_error("RolloutBuffer is full");
    }

    for (size_t i = 0; i < obs_dim_; ++i) {
        obs_[ptr_ * obs_dim_ + i] = (i < obs.size()) ? obs[i] : 0.0;
    }
    for (size_t i = 0; i < action_dim_; ++i) {
        actions_[ptr_ * action_dim_ + i] = (i < action.size()) ? action[i] : 0.0;
    }
    rewards_[ptr_] = reward;
    dones_[ptr_] = done ? 1.0 : 0.0;
    values_[ptr_] = value;
    log_probs_[ptr_] = log_prob;

    ptr_++;
}

void RolloutBuffer::compute_returns_and_advantages(double last_value,
                                                   bool done,
                                                   double gamma,
                                                   double gae_lambda) {
    double last_done = done ? 1.0 : 0.0;
    double last_gae = 0.0;

    for (int t = static_cast<int>(buffer_size_) - 1; t >= 0; --t) {
        auto ut = static_cast<size_t>(t);
        double next_non_terminal;
        double next_val;
        if (ut == buffer_size_ - 1) {
            next_non_terminal = 1.0 - last_done;
            next_val = last_value;
        } else {
            next_non_terminal = 1.0 - dones_[ut + 1];
            next_val = values_[ut + 1];
        }

        double delta = rewards_[ut] + gamma * next_val * next_non_terminal - values_[ut];
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
        advantages_[ut] = last_gae;
    }

    for (size_t t = 0; t < buffer_size_; ++t) {
        returns_[t] = advantages_[t] + values_[t];
    }
}

// --- ReplayBuffer ---

ReplayBuffer::ReplayBuffer(size_t capacity,
                           std::vector<size_t> obs_shape,
                           std::vector<size_t> action_shape)
    : capacity_(capacity),
      obs_shape_(std::move(obs_shape)),
      action_shape_(std::move(action_shape)) {
    obs_dim_ = obs_shape_.empty() ? 1 : std::accumulate(obs_shape_.begin(), obs_shape_.end(), size_t{1}, std::multiplies<size_t>());
    action_dim_ = action_shape_.empty() ? 1 : std::accumulate(action_shape_.begin(), action_shape_.end(), size_t{1}, std::multiplies<size_t>());

    obs_.assign(capacity_ * obs_dim_, 0.0);
    actions_.assign(capacity_ * action_dim_, 0.0);
    rewards_.assign(capacity_, 0.0);
    next_obs_.assign(capacity_ * obs_dim_, 0.0);
    dones_.assign(capacity_, 0.0);
}

void ReplayBuffer::add(const std::vector<double>& obs,
                       const std::vector<double>& action,
                       double reward,
                       const std::vector<double>& next_obs,
                       bool done) {
    for (size_t i = 0; i < obs_dim_; ++i) {
        obs_[ptr_ * obs_dim_ + i] = (i < obs.size()) ? obs[i] : 0.0;
        next_obs_[ptr_ * obs_dim_ + i] = (i < next_obs.size()) ? next_obs[i] : 0.0;
    }
    for (size_t i = 0; i < action_dim_; ++i) {
        actions_[ptr_ * action_dim_ + i] = (i < action.size()) ? action[i] : 0.0;
    }
    rewards_[ptr_] = reward;
    dones_[ptr_] = done ? 1.0 : 0.0;

    ptr_ = (ptr_ + 1) % capacity_;
    size_ = std::min(size_ + 1, capacity_);
}

ReplayBatch ReplayBuffer::sample(size_t batch_size, uint64_t seed) const {
    if (size_ < batch_size) {
        throw std::runtime_error("ReplayBuffer has fewer elements than batch_size");
    }

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<size_t> dist(0, size_ - 1);

    std::vector<double> b_obs(batch_size * obs_dim_);
    std::vector<double> b_actions(batch_size * action_dim_);
    std::vector<double> b_rewards(batch_size);
    std::vector<double> b_next_obs(batch_size * obs_dim_);
    std::vector<double> b_dones(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = dist(rng);
        for (size_t d = 0; d < obs_dim_; ++d) {
            b_obs[i * obs_dim_ + d] = obs_[idx * obs_dim_ + d];
            b_next_obs[i * obs_dim_ + d] = next_obs_[idx * obs_dim_ + d];
        }
        for (size_t d = 0; d < action_dim_; ++d) {
            b_actions[i * action_dim_ + d] = actions_[idx * action_dim_ + d];
        }
        b_rewards[i] = rewards_[idx];
        b_dones[i] = dones_[idx];
    }

    std::vector<size_t> obs_s = {batch_size};
    obs_s.insert(obs_s.end(), obs_shape_.begin(), obs_shape_.end());

    std::vector<size_t> act_s = {batch_size};
    act_s.insert(act_s.end(), action_shape_.begin(), action_shape_.end());

    return ReplayBatch{
        .obs = Tensor::create(obs_s, std::move(b_obs), false),
        .actions = Tensor::create(act_s, std::move(b_actions), false),
        .rewards = Tensor::create({batch_size}, std::move(b_rewards), false),
        .next_obs = Tensor::create(obs_s, std::move(b_next_obs), false),
        .dones = Tensor::create({batch_size}, std::move(b_dones), false),
    };
}

} // namespace aurora
