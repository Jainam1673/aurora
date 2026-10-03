#include "aurora/reproductions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace aurora {
namespace reproductions {

// --- Dreamer: Generalized Lambda-Returns ---

std::shared_ptr<Tensor> compute_lambda_returns(
    const std::shared_ptr<Tensor>& rewards,
    const std::shared_ptr<Tensor>& discounts,
    const std::shared_ptr<Tensor>& values,
    double lambda) {
    if (!rewards || !discounts || !values) {
        throw std::invalid_argument("Null tensor passed to compute_lambda_returns");
    }

    const auto& r_shape = rewards->shape();
    const auto& d_shape = discounts->shape();
    const auto& v_shape = values->shape();

    if (r_shape.empty() || v_shape.empty()) {
        throw std::invalid_argument("Tensors must have at least 1 dimension");
    }

    const size_t H = r_shape[0];
    const size_t B = (r_shape.size() > 1) ? r_shape[1] : 1;

    if (d_shape[0] != H || v_shape[0] != H + 1) {
        throw std::invalid_argument("Horizon dimension mismatch: rewards H, values H+1");
    }

    std::vector<double> returns_data(H * B, 0.0);
    const auto& r_data = rewards->data();
    const auto& d_data = discounts->data();
    const auto& v_data = values->data();

    // Next step return accumulator initialized to terminal value v_H
    std::vector<double> next_return(B, 0.0);
    for (size_t b = 0; b < B; ++b) {
        next_return[b] = v_data[H * B + b];
    }

    // Backwards recursion from H-1 down to 0
    for (size_t step = 0; step < H; ++step) {
        const size_t t = H - 1 - step;
        for (size_t b = 0; b < B; ++b) {
            const size_t idx = t * B + b;
            const size_t next_v_idx = (t + 1) * B + b;
            const double r = r_data[idx];
            const double g = d_data[idx];
            const double v_next = v_data[next_v_idx];

            const double ret = r + g * ((1.0 - lambda) * v_next + lambda * next_return[b]);
            returns_data[idx] = ret;
            next_return[b] = ret;
        }
    }

    return std::make_shared<Tensor>(r_shape, returns_data, false);
}

// --- TD-MPC: Cross-Entropy Method (CEM) Trajectory Planner ---

CEMPlanner::CEMPlanner(CEMConfig config)
    : config_(std::move(config)) {}

std::vector<double> CEMPlanner::plan(
    const std::vector<double>& initial_latent,
    const std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>& dynamics,
    const std::function<double(const std::vector<double>&, const std::vector<double>&)>& reward,
    const std::function<double(const std::vector<double>&)>& terminal_value,
    double gamma) {

    const size_t H = config_.horizon;
    const size_t D = config_.action_dim;
    const size_t N = config_.num_samples;
    const size_t M = config_.num_elites;

    // Proposal parameters: mean and std over (H, D)
    std::vector<double> mean(H * D, 0.0);
    std::vector<double> std_dev(H * D, 1.0);

    std::mt19937_64 rng(config_.seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    std::vector<std::vector<double>> candidates(N, std::vector<double>(H * D, 0.0));
    std::vector<double> returns(N, 0.0);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t iter = 0; iter < config_.iterations; ++iter) {
        // 1. Sample candidate action sequences
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < H * D; ++j) {
                const double val = mean[j] + std_dev[j] * normal_dist(rng);
                candidates[i][j] = std::clamp(val, config_.clip_min, config_.clip_max);
            }
        }

        // 2. Evaluate latent trajectories
        for (size_t i = 0; i < N; ++i) {
            std::vector<double> z = initial_latent;
            double cum_return = 0.0;
            double discount = 1.0;

            for (size_t h = 0; h < H; ++h) {
                std::vector<double> a(D);
                for (size_t d = 0; d < D; ++d) {
                    a[d] = candidates[i][h * D + d];
                }
                const double r = reward(z, a);
                cum_return += discount * r;
                discount *= gamma;
                z = dynamics(z, a);
            }

            cum_return += discount * terminal_value(z);
            returns[i] = cum_return;
        }

        // 3. Select top M elites
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return returns[a] > returns[b];
        });

        // 4. Fit proposal distribution with momentum
        std::vector<double> elite_mean(H * D, 0.0);
        for (size_t m = 0; m < M; ++m) {
            const size_t e_idx = indices[m];
            for (size_t j = 0; j < H * D; ++j) {
                elite_mean[j] += candidates[e_idx][j];
            }
        }
        for (size_t j = 0; j < H * D; ++j) {
            elite_mean[j] /= static_cast<double>(M);
        }

        std::vector<double> elite_std(H * D, 0.0);
        for (size_t m = 0; m < M; ++m) {
            const size_t e_idx = indices[m];
            for (size_t j = 0; j < H * D; ++j) {
                const double diff = candidates[e_idx][j] - elite_mean[j];
                elite_std[j] += diff * diff;
            }
        }
        for (size_t j = 0; j < H * D; ++j) {
            const double var = elite_std[j] / static_cast<double>(M);
            elite_std[j] = std::max(std::sqrt(var + 1e-8), config_.min_std);
        }

        // Momentum update
        for (size_t j = 0; j < H * D; ++j) {
            mean[j] = config_.alpha * elite_mean[j] + (1.0 - config_.alpha) * mean[j];
            std_dev[j] = config_.alpha * elite_std[j] + (1.0 - config_.alpha) * std_dev[j];
        }
    }

    // Return the first planned action
    std::vector<double> best_action(D);
    for (size_t d = 0; d < D; ++d) {
        best_action[d] = mean[d];
    }
    return best_action;
}

// --- MuZero: PUCT Latent Search Tree ---

PUCTPlanner::PUCTPlanner(MCTSConfig config)
    : config_(std::move(config)) {}

std::pair<size_t, std::vector<double>> PUCTPlanner::search(
    const std::vector<double>& root_latent,
    const std::function<std::pair<std::vector<double>, double>(const std::vector<double>&, size_t)>& dynamics_step,
    const std::function<std::pair<std::vector<double>, double>(const std::vector<double>&)>& prediction_step) {

    const size_t A = config_.action_dim;
    auto root = std::make_unique<PUCTNode>();
    root->latent_state = root_latent;

    // Expand root node
    const auto [root_priors, root_val] = prediction_step(root_latent);
    for (size_t a = 0; a < A; ++a) {
        auto child = std::make_unique<PUCTNode>();
        child->prior = root_priors[a];
        root->children.push_back(std::move(child));
    }

    double min_q = root_val;
    double max_q = root_val;

    for (size_t sim = 0; sim < config_.num_simulations; ++sim) {
        PUCTNode* node = root.get();
        std::vector<std::pair<PUCTNode*, size_t>> search_path;

        // 1. Selection
        while (node->expanded()) {
            size_t total_n = 0;
            for (const auto& c : node->children) {
                total_n += c->visit_count;
            }

            double best_score = -1e9;
            size_t best_action = 0;

            for (size_t a = 0; a < A; ++a) {
                const auto& c = node->children[a];
                double q = 0.0;
                if (c->visit_count > 0) {
                    q = c->value();
                    if (max_q > min_q) {
                        q = (q - min_q) / (max_q - min_q);
                    }
                }

                const double pb_c = (std::log((static_cast<double>(total_n) + config_.c2 + 1.0) / config_.c2) + config_.c1) *
                                    (std::sqrt(static_cast<double>(total_n)) / (1.0 + static_cast<double>(c->visit_count)));
                const double score = q + c->prior * pb_c;

                if (score > best_score) {
                    best_score = score;
                    best_action = a;
                }
            }

            search_path.emplace_back(node, best_action);
            node = node->children[best_action].get();
        }

        // 2. Dynamics step (Expansion)
        const auto& [parent_node, act] = search_path.back();
        const auto [next_latent, r] = dynamics_step(parent_node->latent_state, act);
        node->latent_state = next_latent;
        node->reward = r;

        // 3. Prediction
        const auto [priors, leaf_v] = prediction_step(next_latent);
        min_q = std::min(min_q, leaf_v);
        max_q = std::max(max_q, leaf_v);

        for (size_t a = 0; a < A; ++a) {
            auto child = std::make_unique<PUCTNode>();
            child->prior = priors[a];
            node->children.push_back(std::move(child));
        }

        // 4. Backup
        double g = leaf_v;
        for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
            auto* p = it->first;
            const size_t a = it->second;
            auto* c = p->children[a].get();
            g = c->reward + config_.discount * g;
            c->value_sum += g;
            c->visit_count += 1;
            min_q = std::min(min_q, c->value());
            max_q = std::max(max_q, c->value());
        }
    }

    // Policy target from visit counts
    std::vector<double> pi_probs(A, 0.0);
    size_t total_visits = 0;
    size_t best_action = 0;
    size_t max_visits = 0;

    for (size_t a = 0; a < A; ++a) {
        const size_t n = root->children[a]->visit_count;
        pi_probs[a] = static_cast<double>(n);
        total_visits += n;
        if (n > max_visits) {
            max_visits = n;
            best_action = a;
        }
    }

    if (total_visits > 0) {
        for (size_t a = 0; a < A; ++a) {
            pi_probs[a] /= static_cast<double>(total_visits);
        }
    } else {
        std::fill(pi_probs.begin(), pi_probs.end(), 1.0 / static_cast<double>(A));
    }

    return {best_action, pi_probs};
}

// --- MBPO: Hybrid Environment & Model Replay Sampler ---

MBPOBufferManager::MBPOBufferManager(
    size_t env_capacity,
    size_t model_capacity,
    std::vector<size_t> obs_shape,
    std::vector<size_t> action_shape,
    double model_ratio)
    : env_buffer_(env_capacity, obs_shape, action_shape),
      model_buffer_(model_capacity, obs_shape, action_shape),
      model_ratio_(model_ratio) {}

void MBPOBufferManager::add_env(
    const std::vector<double>& obs,
    const std::vector<double>& action,
    double reward,
    const std::vector<double>& next_obs,
    bool done) {
    env_buffer_.add(obs, action, reward, next_obs, done);
}

void MBPOBufferManager::add_model(
    const std::vector<double>& obs,
    const std::vector<double>& action,
    double reward,
    const std::vector<double>& next_obs,
    bool done) {
    model_buffer_.add(obs, action, reward, next_obs, done);
}

ReplayBatch MBPOBufferManager::sample_mixed(size_t batch_size, uint64_t seed) {
    const size_t model_size = static_cast<size_t>(static_cast<double>(batch_size) * model_ratio_);
    const size_t env_size = batch_size - model_size;

    if (model_buffer_.size() < model_size || model_size == 0) {
        return env_buffer_.sample(batch_size, seed);
    }
    if (env_buffer_.size() < env_size || env_size == 0) {
        return model_buffer_.sample(batch_size, seed);
    }

    const auto env_batch = env_buffer_.sample(env_size, seed);
    const auto model_batch = model_buffer_.sample(model_size, seed + 1);

    return ReplayBatch{
        Tensor::concat({env_batch.obs, model_batch.obs}, 0),
        Tensor::concat({env_batch.actions, model_batch.actions}, 0),
        Tensor::concat({env_batch.rewards, model_batch.rewards}, 0),
        Tensor::concat({env_batch.next_obs, model_batch.next_obs}, 0),
        Tensor::concat({env_batch.dones, model_batch.dones}, 0),
    };
}

} // namespace reproductions
} // namespace aurora
