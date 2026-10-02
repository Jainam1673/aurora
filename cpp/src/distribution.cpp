#include "aurora/distribution.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

namespace aurora {

// --- Categorical ---

Categorical::Categorical(std::shared_ptr<Tensor> logits)
    : logits_(std::move(logits)),
      probs_(logits_->softmax(-1)),
      num_classes_(logits_->shape().back()) {}

std::shared_ptr<Tensor> Categorical::sample(uint64_t seed) {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    auto p_vec = probs_->to_vector();
    size_t batch_size = probs_->numel() / num_classes_;

    std::vector<double> samples(batch_size);
    std::vector<double> row_probs(num_classes_);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_classes_; ++c) {
            row_probs[c] = p_vec[b * num_classes_ + c];
        }
        std::discrete_distribution<size_t> dist(row_probs.begin(), row_probs.end());
        samples[b] = static_cast<double>(dist(rng));
    }

    std::vector<size_t> out_shape = probs_->shape();
    out_shape.pop_back();
    if (out_shape.empty()) {
        out_shape = {1};
    }
    return Tensor::create(out_shape, std::move(samples), /*requires_grad=*/false);
}

std::shared_ptr<Tensor> Categorical::log_prob(const std::shared_ptr<Tensor>& value) {
    auto log_p = logits_->log_softmax(-1);
    auto val_vec = value->to_vector();
    size_t batch_size = value->numel();

    if (log_p->numel() != batch_size * num_classes_) {
        throw std::invalid_argument("Value batch size does not match categorical distribution logits");
    }

    // Build one-hot indicator tensor for differentiable indexing
    std::vector<double> one_hot_data(batch_size * num_classes_, 0.0);
    for (size_t b = 0; b < batch_size; ++b) {
        auto cls_idx = static_cast<size_t>(val_vec[b]);
        if (cls_idx < num_classes_) {
            one_hot_data[b * num_classes_ + cls_idx] = 1.0;
        }
    }

    auto one_hot = Tensor::create(log_p->shape(), std::move(one_hot_data), false);
    return (one_hot * log_p)->sum(-1);
}

std::shared_ptr<Tensor> Categorical::entropy() {
    auto log_p = logits_->log_softmax(-1);
    return -(probs_ * log_p)->sum(-1);
}

// --- Normal ---

Normal::Normal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale)
    : loc_(std::move(loc)), scale_(std::move(scale)) {
    if (loc_->shape() != scale_->shape()) {
        throw std::invalid_argument("Normal distribution loc and scale must have matching shapes");
    }
}

std::shared_ptr<Tensor> Normal::sample(uint64_t seed) {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<double> std_norm(0.0, 1.0);

    size_t count = loc_->numel();
    auto loc_vec = loc_->to_vector();
    auto scale_vec = scale_->to_vector();
    std::vector<double> data(count);

    for (size_t i = 0; i < count; ++i) {
        data[i] = loc_vec[i] + scale_vec[i] * std_norm(rng);
    }
    return Tensor::create(loc_->shape(), std::move(data), false);
}

std::shared_ptr<Tensor> Normal::rsample(uint64_t seed) {
    auto eps = Tensor::randn(loc_->shape(), seed == 0 ? std::random_device{}() : seed, false);
    return loc_ + scale_ * eps;
}

std::shared_ptr<Tensor> Normal::log_prob(const std::shared_ptr<Tensor>& value) {
    return log_prob(value, true);
}

std::shared_ptr<Tensor> Normal::log_prob(const std::shared_ptr<Tensor>& value, bool sum_features) {
    // -0.5 * ((x - loc)^2 / scale^2 + 2 * log(scale) + log(2*pi))
    const double log_2pi = std::log(2.0 * std::numbers::pi);
    auto var = scale_ * scale_;
    auto diff = value - loc_;
    auto log_scale = scale_->log();

    auto term1 = (diff * diff) / var;
    auto term2 = log_scale * 2.0;
    auto elem_log_p = -0.5 * (term1 + term2 + log_2pi);

    if (sum_features) {
        return elem_log_p->sum(-1);
    }
    return elem_log_p;
}

std::shared_ptr<Tensor> Normal::entropy() {
    return entropy(true);
}

std::shared_ptr<Tensor> Normal::entropy(bool sum_features) {
    // 0.5 * (1 + log(2*pi) + 2 * log(scale)) = log(scale) + 0.5 * (1 + log(2*pi))
    const double c = 0.5 * (1.0 + std::log(2.0 * std::numbers::pi));
    auto elem_ent = scale_->log() + c;
    if (sum_features) {
        return elem_ent->sum(-1);
    }
    return elem_ent;
}

// --- TanhNormal ---

TanhNormal::TanhNormal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale, double eps)
    : loc_(loc), scale_(scale), normal_(loc, scale), eps_(eps) {}

std::shared_ptr<Tensor> TanhNormal::sample(uint64_t seed) {
    auto u = normal_.sample(seed);
    return u->tanh();
}

std::shared_ptr<Tensor> TanhNormal::rsample(uint64_t seed) {
    auto [a, _] = rsample_with_pre_tanh(seed);
    return a;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> TanhNormal::rsample_with_pre_tanh(uint64_t seed) {
    auto u = normal_.rsample(seed);
    auto a = u->tanh();
    return {a, u};
}

std::shared_ptr<Tensor> TanhNormal::log_prob(const std::shared_ptr<Tensor>& value) {
    return log_prob(value, nullptr);
}

std::shared_ptr<Tensor> TanhNormal::log_prob(const std::shared_ptr<Tensor>& value,
                                             const std::shared_ptr<Tensor>& pre_tanh_value) {
    std::shared_ptr<Tensor> u;
    if (pre_tanh_value != nullptr) {
        u = pre_tanh_value;
    } else {
        // Invert tanh: u = atanh(clip(value, -1 + eps, 1 - eps))
        auto val_vec = value->to_vector();
        std::vector<double> u_data(val_vec.size());
        for (size_t i = 0; i < val_vec.size(); ++i) {
            double clipped = std::clamp(val_vec[i], -1.0 + eps_, 1.0 - eps_);
            u_data[i] = std::atanh(clipped);
        }
        u = Tensor::create(value->shape(), std::move(u_data), value->requires_grad());
    }

    auto log_p_gaussian = normal_.log_prob(u, /*sum_features=*/false);

    // Numerically stable softplus identity:
    // log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
    const double log_2 = std::log(2.0);
    auto u_vec = u->to_vector();
    std::vector<double> log_det_vec(u_vec.size());

    for (size_t i = 0; i < u_vec.size(); ++i) {
        double ui = u_vec[i];
        double neg2u = -2.0 * ui;
        // softplus(neg2u) = max(0, neg2u) + log1p(exp(-|neg2u|))
        double softplus_val = std::max(0.0, neg2u) + std::log1p(std::exp(-std::abs(neg2u)));
        log_det_vec[i] = 2.0 * (log_2 - ui - softplus_val);
    }

    auto log_det = Tensor::create(u->shape(), std::move(log_det_vec), u->requires_grad());
    auto log_p = log_p_gaussian - log_det;
    return log_p->sum(-1);
}

std::shared_ptr<Tensor> TanhNormal::entropy() {
    throw std::runtime_error("Analytic entropy for TanhNormal is intractable; use sample estimation");
}

} // namespace aurora
