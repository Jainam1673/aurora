#include "aurora/world_model.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace aurora {
namespace world_model {

// --- Heteroscedastic Gaussian NLL Loss ---

std::shared_ptr<Tensor> gaussian_nll_loss(
    const std::shared_ptr<Tensor>& mean,
    const std::shared_ptr<Tensor>& log_var,
    const std::shared_ptr<Tensor>& target) {
    auto diff = target - mean;
    auto inv_var = (-log_var)->exp();
    auto diff_sq_inv_var = diff * diff * inv_var;
    auto elem_nll = (diff_sq_inv_var + log_var + std::log(2.0 * M_PI)) * 0.5;
    size_t batch_size = (mean->ndim() > 1) ? mean->shape()[0] : 1;
    return elem_nll->sum() / static_cast<double>(batch_size);
}

// --- Uncertainty Decomposition ---

UncertaintyMetrics decompose_uncertainty(
    const std::vector<std::shared_ptr<Tensor>>& means,
    const std::vector<std::shared_ptr<Tensor>>& vars) {
    if (means.empty() || vars.empty()) {
        throw std::invalid_argument("decompose_uncertainty requires at least one ensemble member");
    }
    double num_models = static_cast<double>(means.size());
    double inv_e = 1.0 / num_models;

    // 1. Ensemble Mean
    auto sum_m = means[0];
    for (size_t i = 1; i < means.size(); ++i) {
        sum_m = sum_m + means[i];
    }
    auto ens_mean = sum_m * inv_e;

    // 2. Aleatoric Uncertainty
    auto sum_v = vars[0];
    for (size_t i = 1; i < vars.size(); ++i) {
        sum_v = sum_v + vars[i];
    }
    auto aleatoric = sum_v * inv_e;

    // 3. Epistemic Uncertainty
    auto d0 = means[0] - ens_mean;
    auto sum_sq = d0 * d0;
    for (size_t i = 1; i < means.size(); ++i) {
        auto d = means[i] - ens_mean;
        sum_sq = sum_sq + (d * d);
    }
    auto epistemic = sum_sq * inv_e;

    // 4. Total Predictive Variance
    auto total = aleatoric + epistemic;

    // Disagreement metric: max epistemic variance across feature dimensions per sample
    std::vector<double> disagreement;
    auto ep_vec = epistemic->to_vector();
    size_t num_items = (epistemic->ndim() > 1) ? epistemic->shape()[0] : 1;
    size_t feat_dim = epistemic->shape().back();
    disagreement.resize(num_items);

    for (size_t i = 0; i < num_items; ++i) {
        double max_val = 0.0;
        for (size_t j = 0; j < feat_dim; ++j) {
            double val = ep_vec[i * feat_dim + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        disagreement[i] = max_val;
    }

    return UncertaintyMetrics{
        .mean = std::move(ens_mean),
        .aleatoric = std::move(aleatoric),
        .epistemic = std::move(epistemic),
        .total = std::move(total),
        .disagreement = std::move(disagreement),
    };
}

// --- EnsembleMember Implementation ---

EnsembleMember::EnsembleMember(
    size_t in_features,
    size_t out_features,
    const std::vector<size_t>& hidden_dims,
    const std::string& activation,
    double log_var_min,
    double log_var_max,
    uint64_t seed)
    : in_features_(in_features),
      out_features_(out_features),
      log_var_min_(log_var_min),
      log_var_max_(log_var_max) {
    size_t last_hidden = hidden_dims.empty() ? in_features : hidden_dims.back();
    backbone_ = std::make_shared<nn::MLP>(in_features, hidden_dims, last_hidden, activation, 0.0, seed);
    mu_head_ = std::make_shared<nn::Linear>(last_hidden, out_features, true, seed + 101);
    log_var_head_ = std::make_shared<nn::Linear>(last_hidden, out_features, true, seed + 202);

    register_module("backbone", backbone_);
    register_module("mu_head", mu_head_);
    register_module("log_var_head", log_var_head_);
}

std::shared_ptr<Tensor> EnsembleMember::forward(const std::shared_ptr<Tensor>& input) {
    auto [mu, log_var] = forward_member(input);
    return mu;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> EnsembleMember::forward_member(
    const std::shared_ptr<Tensor>& x) {
    auto feats = backbone_->forward(x);
    auto mu = mu_head_->forward(feats);
    auto log_var = log_var_head_->forward(feats)->clamp(log_var_min_, log_var_max_);
    return {mu, log_var};
}

// --- EnsembleDynamics Implementation ---

EnsembleDynamics::EnsembleDynamics(
    size_t obs_dim,
    size_t action_dim,
    size_t ensemble_size,
    const std::vector<size_t>& hidden_dims,
    const std::string& activation,
    double log_var_min,
    double log_var_max,
    bool predict_reward,
    uint64_t seed)
    : obs_dim_(obs_dim),
      action_dim_(action_dim),
      ensemble_size_(ensemble_size),
      out_dim_(obs_dim + (predict_reward ? 1 : 0)),
      log_var_min_(log_var_min),
      log_var_max_(log_var_max),
      predict_reward_(predict_reward),
      rng_(seed) {
    members_.reserve(ensemble_size);
    for (size_t i = 0; i < ensemble_size; ++i) {
        auto member = std::make_shared<EnsembleMember>(
            obs_dim + action_dim,
            out_dim_,
            hidden_dims,
            activation,
            log_var_min,
            log_var_max,
            seed + (i * 1000));
        register_module(std::format("member_{}", i), member);
        members_.push_back(member);
    }
}

std::shared_ptr<Tensor> EnsembleDynamics::forward(const std::shared_ptr<Tensor>& input) {
    return members_[0]->forward(input);
}

std::pair<std::vector<std::shared_ptr<Tensor>>, std::vector<std::shared_ptr<Tensor>>>
EnsembleDynamics::forward_ensemble(
    const std::shared_ptr<Tensor>& obs,
    const std::shared_ptr<Tensor>& action) {
    auto x = Tensor::concat({obs, action}, -1);
    std::vector<std::shared_ptr<Tensor>> means;
    std::vector<std::shared_ptr<Tensor>> log_vars;
    means.reserve(ensemble_size_);
    log_vars.reserve(ensemble_size_);

    for (const auto& member : members_) {
        auto [mu, lv] = member->forward_member(x);
        means.push_back(mu);
        log_vars.push_back(lv);
    }
    return {means, log_vars};
}

std::shared_ptr<Tensor> EnsembleDynamics::compute_loss(
    const std::shared_ptr<Tensor>& obs,
    const std::shared_ptr<Tensor>& action,
    const std::shared_ptr<Tensor>& next_obs,
    const std::shared_ptr<Tensor>& reward) {
    auto delta_s = next_obs - obs;
    std::shared_ptr<Tensor> target;

    if (predict_reward_) {
        if (!reward) {
            throw std::invalid_argument("reward cannot be null when predict_reward is true");
        }
        std::shared_ptr<Tensor> r = reward;
        if (reward->ndim() == 1) {
            r = reward->reshape({reward->shape()[0], 1});
        }
        target = Tensor::concat({delta_s, r}, -1);
    } else {
        target = delta_s;
    }

    auto [means, log_vars] = forward_ensemble(obs, action);
    auto total_loss = gaussian_nll_loss(means[0], log_vars[0], target);

    for (size_t e = 1; e < ensemble_size_; ++e) {
        total_loss = total_loss + gaussian_nll_loss(means[e], log_vars[e], target);
    }
    return total_loss;
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
EnsembleDynamics::predict(const std::vector<double>& obs, const std::vector<double>& action) {
    auto obs_t = Tensor::create({1, obs_dim_}, obs, false);
    auto act_t = Tensor::create({1, action_dim_}, action, false);

    auto [means, log_vars] = forward_ensemble(obs_t, act_t);

    std::vector<std::vector<double>> next_states;
    std::vector<std::vector<double>> rewards;
    std::vector<std::vector<double>> variances;
    next_states.reserve(ensemble_size_);
    rewards.reserve(ensemble_size_);
    variances.reserve(ensemble_size_);

    for (size_t e = 0; e < ensemble_size_; ++e) {
        auto mu_vec = means[e]->to_vector();
        auto lv_vec = log_vars[e]->to_vector();

        std::vector<double> ns(obs_dim_);
        for (size_t j = 0; j < obs_dim_; ++j) {
            ns[j] = obs[j] + mu_vec[j];
        }
        next_states.push_back(std::move(ns));

        if (predict_reward_) {
            rewards.push_back({mu_vec[obs_dim_]});
        } else {
            rewards.push_back({0.0});
        }

        std::vector<double> var(out_dim_);
        for (size_t j = 0; j < out_dim_; ++j) {
            var[j] = std::exp(lv_vec[j]);
        }
        variances.push_back(std::move(var));
    }

    return {next_states, rewards, variances};
}

std::pair<std::vector<double>, double> EnsembleDynamics::step(
    const std::vector<double>& obs,
    const std::vector<double>& action,
    std::optional<size_t> member_idx) {
    size_t idx = member_idx.value_or(std::uniform_int_distribution<size_t>(0, ensemble_size_ - 1)(rng_));
    auto [next_states, rewards, _] = predict(obs, action);
    return {next_states[idx], rewards[idx][0]};
}

// --- ImaginationEngine Implementation ---

ImaginationEngine::ImaginationEngine(
    std::shared_ptr<EnsembleDynamics> dynamics,
    std::function<std::vector<double>(const std::vector<double>&)> policy,
    size_t max_horizon,
    double uncertainty_threshold,
    bool adaptive_truncation,
    std::string sampling_mode,
    uint64_t seed)
    : dynamics_(std::move(dynamics)),
      policy_(std::move(policy)),
      max_horizon_(max_horizon),
      uncertainty_threshold_(uncertainty_threshold),
      adaptive_truncation_(adaptive_truncation),
      sampling_mode_(std::move(sampling_mode)),
      rng_(seed) {}

std::tuple<std::vector<std::tuple<std::vector<double>, std::vector<double>, double, std::vector<double>, bool>>, bool, std::vector<double>>
ImaginationEngine::rollout_single(const std::vector<double>& initial_state) {
    std::vector<double> curr_state = initial_state;
    std::vector<std::tuple<std::vector<double>, std::vector<double>, double, std::vector<double>, bool>> transitions;
    std::vector<double> uncertainties;
    bool truncated = false;

    size_t obs_dim = dynamics_->obs_dim();
    size_t ensemble_size = dynamics_->ensemble_size();

    for (size_t h = 0; h < max_horizon_; ++h) {
        auto action = policy_(curr_state);
        auto [next_states, rewards, variances] = dynamics_->predict(curr_state, action);

        // Decompose uncertainty across predicted next states
        std::vector<std::shared_ptr<Tensor>> mean_tensors;
        std::vector<std::shared_ptr<Tensor>> var_tensors;
        mean_tensors.reserve(ensemble_size);
        var_tensors.reserve(ensemble_size);

        for (size_t e = 0; e < ensemble_size; ++e) {
            mean_tensors.push_back(Tensor::create({1, obs_dim}, next_states[e], false));
            std::vector<double> obs_vars(variances[e].begin(), variances[e].begin() + static_cast<std::ptrdiff_t>(obs_dim));
            var_tensors.push_back(Tensor::create({1, obs_dim}, obs_vars, false));
        }

        auto unc = decompose_uncertainty(mean_tensors, var_tensors);
        double max_epistemic = unc.disagreement[0];
        uncertainties.push_back(max_epistemic);

        if (adaptive_truncation_ && max_epistemic > uncertainty_threshold_) {
            truncated = true;
            break;
        }

        std::vector<double> next_state(obs_dim);
        double reward = 0.0;

        if (sampling_mode_ == "ts1") {
            size_t idx = std::uniform_int_distribution<size_t>(0, ensemble_size - 1)(rng_);
            next_state = next_states[idx];
            reward = rewards[idx][0];
        } else {
            next_state = unc.mean->to_vector();
            double sum_r = 0.0;
            for (size_t e = 0; e < ensemble_size; ++e) {
                sum_r += rewards[e][0];
            }
            reward = sum_r / static_cast<double>(ensemble_size);
        }

        transitions.emplace_back(curr_state, action, reward, next_state, false);
        curr_state = next_state;
    }

    return {transitions, truncated, uncertainties};
}

ImaginationResult ImaginationEngine::generate_rollouts(
    const std::vector<std::vector<double>>& initial_states) {
    ImaginationResult result;
    double total_reward = 0.0;
    double total_epistemic = 0.0;
    size_t epistemic_samples = 0;

    for (const auto& s0 : initial_states) {
        auto [traj, was_truncated, step_unc] = rollout_single(s0);
        if (was_truncated) {
            result.truncated_trajectories++;
        } else {
            result.completed_trajectories++;
        }
        for (const auto& tr : traj) {
            total_reward += std::get<2>(tr);
            result.transitions.push_back(tr);
        }
        for (double u : step_unc) {
            total_epistemic += u;
            epistemic_samples++;
        }
    }

    result.total_transitions = result.transitions.size();
    if (!initial_states.empty()) {
        result.mean_horizon = static_cast<double>(result.total_transitions) / static_cast<double>(initial_states.size());
        result.mean_reward = total_reward / static_cast<double>(initial_states.size());
    }
    if (epistemic_samples > 0) {
        result.mean_epistemic_uncertainty = total_epistemic / static_cast<double>(epistemic_samples);
    }
    return result;
}

ImaginationResult ImaginationEngine::inject_into_buffer(
    ReplayBuffer& replay_buffer,
    const std::vector<std::vector<double>>& initial_states) {
    auto result = generate_rollouts(initial_states);
    for (const auto& [s, a, r, next_s, done] : result.transitions) {
        replay_buffer.add(s, a, r, next_s, done);
    }
    return result;
}

// --- GRUCell Implementation ---

GRUCell::GRUCell(size_t input_dim, size_t hidden_dim, uint64_t seed)
    : input_dim_(input_dim), hidden_dim_(hidden_dim) {
    w_ir_ = std::make_shared<nn::Linear>(input_dim, hidden_dim, true, seed + 1);
    w_hr_ = std::make_shared<nn::Linear>(hidden_dim, hidden_dim, true, seed + 2);
    w_iz_ = std::make_shared<nn::Linear>(input_dim, hidden_dim, true, seed + 3);
    w_hz_ = std::make_shared<nn::Linear>(hidden_dim, hidden_dim, true, seed + 4);
    w_in_ = std::make_shared<nn::Linear>(input_dim, hidden_dim, true, seed + 5);
    w_hn_ = std::make_shared<nn::Linear>(hidden_dim, hidden_dim, true, seed + 6);

    register_module("w_ir", w_ir_);
    register_module("w_hr", w_hr_);
    register_module("w_iz", w_iz_);
    register_module("w_hz", w_hz_);
    register_module("w_in", w_in_);
    register_module("w_hn", w_hn_);
}

std::shared_ptr<Tensor> GRUCell::forward(const std::shared_ptr<Tensor>& input) {
    auto h0 = Tensor::zeros({input->shape()[0], hidden_dim_}, false);
    return forward(input, h0);
}

std::shared_ptr<Tensor> GRUCell::forward(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& h) {
    auto r = (w_ir_->forward(x) + w_hr_->forward(h))->sigmoid();
    auto z = (w_iz_->forward(x) + w_hz_->forward(h))->sigmoid();
    auto cand = (w_in_->forward(x) + r * w_hn_->forward(h))->tanh();
    auto one_minus_z = 1.0 - z;
    return one_minus_z * cand + z * h;
}

// --- RSSM Implementation ---

RSSM::RSSM(
    size_t obs_dim,
    size_t action_dim,
    size_t deter_dim,
    size_t stoch_dim,
    size_t hidden_dim,
    double kl_alpha,
    double kl_scale,
    double min_std,
    uint64_t seed)
    : obs_dim_(obs_dim),
      action_dim_(action_dim),
      deter_dim_(deter_dim),
      stoch_dim_(stoch_dim),
      kl_alpha_(kl_alpha),
      kl_scale_(kl_scale),
      min_std_(min_std),
      rng_(seed) {
    cell_ = std::make_shared<GRUCell>(stoch_dim + action_dim, deter_dim, seed + 10);

    prior_mlp_ = std::make_shared<nn::MLP>(deter_dim, std::vector<size_t>{hidden_dim}, hidden_dim, "silu", 0.0, seed + 20);
    prior_mu_ = std::make_shared<nn::Linear>(hidden_dim, stoch_dim, true, seed + 30);
    prior_log_std_ = std::make_shared<nn::Linear>(hidden_dim, stoch_dim, true, seed + 40);

    post_mlp_ = std::make_shared<nn::MLP>(deter_dim + obs_dim, std::vector<size_t>{hidden_dim}, hidden_dim, "silu", 0.0, seed + 50);
    post_mu_ = std::make_shared<nn::Linear>(hidden_dim, stoch_dim, true, seed + 60);
    post_log_std_ = std::make_shared<nn::Linear>(hidden_dim, stoch_dim, true, seed + 70);

    obs_decoder_ = std::make_shared<nn::MLP>(deter_dim + stoch_dim, std::vector<size_t>{hidden_dim}, obs_dim, "silu", 0.0, seed + 80);
    reward_decoder_ = std::make_shared<nn::MLP>(deter_dim + stoch_dim, std::vector<size_t>{hidden_dim}, 1, "silu", 0.0, seed + 90);
    cont_decoder_ = std::make_shared<nn::MLP>(deter_dim + stoch_dim, std::vector<size_t>{hidden_dim}, 1, "silu", 0.0, seed + 100);

    register_module("cell", cell_);
    register_module("prior_mlp", prior_mlp_);
    register_module("prior_mu", prior_mu_);
    register_module("prior_log_std", prior_log_std_);
    register_module("post_mlp", post_mlp_);
    register_module("post_mu", post_mu_);
    register_module("post_log_std", post_log_std_);
    register_module("obs_decoder", obs_decoder_);
    register_module("reward_decoder", reward_decoder_);
    register_module("cont_decoder", cont_decoder_);
}

std::shared_ptr<Tensor> RSSM::forward(const std::shared_ptr<Tensor>& input) {
    return input;
}

RSSMState RSSM::initial_state(size_t batch_size) const {
    return RSSMState{
        .h = Tensor::zeros({batch_size, deter_dim_}, false),
        .z = Tensor::zeros({batch_size, stoch_dim_}, false),
        .mu = Tensor::zeros({batch_size, stoch_dim_}, false),
        .log_std = Tensor::zeros({batch_size, stoch_dim_}, false),
    };
}

std::shared_ptr<Tensor> RSSM::sample_gaussian(
    const std::shared_ptr<Tensor>& mu,
    const std::shared_ptr<Tensor>& log_std) const {
    auto std_t = log_std->exp()->clamp(min_std_, 10.0);
    auto eps = Tensor::randn(mu->shape(), rng_(), false);
    return mu + std_t * eps;
}

std::pair<RSSMState, RSSMState> RSSM::observe_step(
    const RSSMState& prev_state,
    const std::shared_ptr<Tensor>& action,
    const std::shared_ptr<Tensor>& obs) {
    // 1. Recurrent Deterministic update
    auto cell_in = Tensor::concat({prev_state.z, action}, -1);
    auto h = cell_->forward(cell_in, prev_state.h);

    // 2. Prior distribution
    auto prior_feats = prior_mlp_->forward(h);
    auto prior_mu = prior_mu_->forward(prior_feats);
    auto prior_log_std = prior_log_std_->forward(prior_feats)->clamp(-5.0, 2.0);
    auto prior_z = sample_gaussian(prior_mu, prior_log_std);
    RSSMState prior_state{
        .h = h,
        .z = prior_z,
        .mu = prior_mu,
        .log_std = prior_log_std,
    };

    // 3. Posterior distribution
    auto post_in = Tensor::concat({h, obs}, -1);
    auto post_feats = post_mlp_->forward(post_in);
    auto post_mu = post_mu_->forward(post_feats);
    auto post_log_std = post_log_std_->forward(post_feats)->clamp(-5.0, 2.0);
    auto post_z = sample_gaussian(post_mu, post_log_std);
    RSSMState post_state{
        .h = h,
        .z = post_z,
        .mu = post_mu,
        .log_std = post_log_std,
    };

    return {prior_state, post_state};
}

RSSMState RSSM::imagine_step(
    const RSSMState& prev_state,
    const std::shared_ptr<Tensor>& action) {
    auto cell_in = Tensor::concat({prev_state.z, action}, -1);
    auto h = cell_->forward(cell_in, prev_state.h);

    auto prior_feats = prior_mlp_->forward(h);
    auto prior_mu = prior_mu_->forward(prior_feats);
    auto prior_log_std = prior_log_std_->forward(prior_feats)->clamp(-5.0, 2.0);
    auto prior_z = sample_gaussian(prior_mu, prior_log_std);

    return RSSMState{
        .h = h,
        .z = prior_z,
        .mu = prior_mu,
        .log_std = prior_log_std,
    };
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RSSM::decode(const RSSMState& state) {
    auto feat = Tensor::concat({state.h, state.z}, -1);
    auto obs_hat = obs_decoder_->forward(feat);
    auto r_hat = reward_decoder_->forward(feat);
    auto gamma_hat = cont_decoder_->forward(feat)->sigmoid();
    return {obs_hat, r_hat, gamma_hat};
}

std::shared_ptr<Tensor> RSSM::kl_divergence(
    const std::shared_ptr<Tensor>& mu_q,
    const std::shared_ptr<Tensor>& log_std_q,
    const std::shared_ptr<Tensor>& mu_p,
    const std::shared_ptr<Tensor>& log_std_p) {
    auto var_q = (log_std_q * 2.0)->exp();
    auto inv_var_p = (-log_std_p * 2.0)->exp();

    auto diff = mu_q - mu_p;
    auto diff_sq = diff * diff;

    auto term1 = log_std_p - log_std_q;
    auto term2 = (var_q + diff_sq) * inv_var_p * 0.5;
    auto kl_elem = term1 + term2 - 0.5;
    return kl_elem->sum(-1);
}

std::pair<std::shared_ptr<Tensor>, std::map<std::string, double>> RSSM::compute_loss(
    const RSSMState& prior_state,
    const RSSMState& post_state,
    const std::shared_ptr<Tensor>& target_obs,
    const std::shared_ptr<Tensor>& target_reward,
    const std::shared_ptr<Tensor>& target_done) {
    auto [obs_hat, r_hat, gamma_hat] = decode(post_state);

    // 1. Observation MSE
    auto obs_diff = target_obs - obs_hat;
    auto obs_loss = (obs_diff * obs_diff * 0.5)->sum(-1)->mean();

    // 2. Reward MSE
    auto r_target = target_reward;
    if (target_reward->ndim() == 1) {
        r_target = target_reward->reshape({target_reward->shape()[0], 1});
    }
    auto r_diff = r_target - r_hat;
    auto reward_loss = (r_diff * r_diff * 0.5)->sum(-1)->mean();

    // 3. Continuation BCE
    std::shared_ptr<Tensor> cont_loss;
    if (target_done) {
        auto done = target_done;
        if (target_done->ndim() == 1) {
            done = target_done->reshape({target_done->shape()[0], 1});
        }
        auto cont_target = 1.0 - done;
        double eps = 1e-7;
        auto pos_term = cont_target * (gamma_hat + eps)->log();
        auto neg_term = (1.0 - cont_target) * (1.0 - gamma_hat + eps)->log();
        auto bce = -(pos_term + neg_term);
        cont_loss = bce->sum(-1)->mean();
    } else {
        cont_loss = Tensor::zeros({1}, false);
    }

    // 4. KL Balancing with stop-gradient
    auto prior_mu_detached = Tensor::create(prior_state.mu->shape(), prior_state.mu->to_vector(), false);
    auto prior_log_std_detached = Tensor::create(prior_state.log_std->shape(), prior_state.log_std->to_vector(), false);
    auto kl_lhs = kl_divergence(post_state.mu, post_state.log_std, prior_mu_detached, prior_log_std_detached)->mean();

    auto post_mu_detached = Tensor::create(post_state.mu->shape(), post_state.mu->to_vector(), false);
    auto post_log_std_detached = Tensor::create(post_state.log_std->shape(), post_state.log_std->to_vector(), false);
    auto kl_rhs = kl_divergence(post_mu_detached, post_log_std_detached, prior_state.mu, prior_state.log_std)->mean();

    auto kl_loss = (kl_lhs * kl_alpha_) + (kl_rhs * (1.0 - kl_alpha_));

    auto total_loss = obs_loss + reward_loss + cont_loss + (kl_loss * kl_scale_);

    std::map<std::string, double> metrics;
    metrics["obs_loss"] = obs_loss->item();
    metrics["reward_loss"] = reward_loss->item();
    metrics["cont_loss"] = cont_loss->item();
    metrics["kl_loss"] = kl_loss->item();
    metrics["total_loss"] = total_loss->item();

    return {total_loss, metrics};
}

} // namespace world_model
} // namespace aurora
