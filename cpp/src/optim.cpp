#include "aurora/optim.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace aurora {
namespace optim {

double clip_grad_norm(const std::vector<std::shared_ptr<Tensor>>& params,
                      double max_norm,
                      double norm_type) {
    std::vector<std::shared_ptr<Tensor>> grads;
    for (const auto& p : params) {
        if (p && p->grad()) {
            grads.push_back(p->grad());
        }
    }
    if (grads.empty()) {
        return 0.0;
    }

    double total_norm = 0.0;
    if (std::isinf(norm_type)) {
        for (const auto& g : grads) {
            auto vec = g->to_vector();
            for (double val : vec) {
                total_norm = std::max(total_norm, std::abs(val));
            }
        }
    } else {
        double sum = 0.0;
        for (const auto& g : grads) {
            auto vec = g->to_vector();
            for (double val : vec) {
                sum += std::pow(std::abs(val), norm_type);
            }
        }
        total_norm = std::pow(sum, 1.0 / norm_type);
    }

    double clip_coef = max_norm / (total_norm + 1e-6);
    if (clip_coef < 1.0) {
        for (auto& g : grads) {
            double* data = g->data();
            size_t n = g->numel();
            for (size_t i = 0; i < n; ++i) {
                data[i] *= clip_coef;
            }
        }
    }

    return total_norm;
}

void clip_grad_value(const std::vector<std::shared_ptr<Tensor>>& params,
                     double clip_value) {
    for (const auto& p : params) {
        if (p && p->grad()) {
            double* data = p->grad()->data();
            size_t n = p->grad()->numel();
            for (size_t i = 0; i < n; ++i) {
                data[i] = std::clamp(data[i], -clip_value, clip_value);
            }
        }
    }
}

// --- Optimizer Base ---

Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor>> params)
    : params_(std::move(params)) {}

void Optimizer::zero_grad() noexcept {
    for (auto& p : params_) {
        if (p) {
            p->zero_grad();
        }
    }
}

OptimizerStateDict Optimizer::state_dict() const {
    OptimizerStateDict s;
    s.step_count = step_count_;
    s.defaults = defaults();
    s.state = state_;
    return s;
}

void Optimizer::load_state_dict(const OptimizerStateDict& s) {
    step_count_ = s.step_count;
    state_ = s.state;
}

// --- SGD ---

SGD::SGD(std::vector<std::shared_ptr<Tensor>> params,
         double lr,
         double momentum,
         double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr),
      momentum_(momentum),
      weight_decay_(weight_decay) {}

std::map<std::string, double> SGD::defaults() const {
    return {
        {"lr", lr_},
        {"momentum", momentum_},
        {"weight_decay", weight_decay_},
    };
}

void SGD::step() {
    step_count_++;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p || !p->grad()) continue;

        auto g_vec = p->grad()->to_vector();
        double* p_data = p->data();
        size_t n = p->numel();

        if (momentum_ > 0.0) {
            auto& p_state = state_[i].tensors;
            if (!p_state.contains("v")) {
                p_state["v"] = Tensor::zeros(p->shape());
            }
            double* v_data = p_state["v"]->data();

            for (size_t k = 0; k < n; ++k) {
                double g = g_vec[k];
                if (weight_decay_ > 0.0) {
                    g += weight_decay_ * p_data[k];
                }
                v_data[k] = momentum_ * v_data[k] + g;
                p_data[k] -= lr_ * v_data[k];
            }
        } else {
            for (size_t k = 0; k < n; ++k) {
                double g = g_vec[k];
                if (weight_decay_ > 0.0) {
                    g += weight_decay_ * p_data[k];
                }
                p_data[k] -= lr_ * g;
            }
        }
    }
}

// --- Adam ---

Adam::Adam(std::vector<std::shared_ptr<Tensor>> params,
           double lr,
           double beta1,
           double beta2,
           double eps,
           double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {}

std::map<std::string, double> Adam::defaults() const {
    return {
        {"lr", lr_},
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"eps", eps_},
        {"weight_decay", weight_decay_},
    };
}

void Adam::step() {
    step_count_++;
    double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
    double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p || !p->grad()) continue;

        auto g_vec = p->grad()->to_vector();
        double* p_data = p->data();
        size_t n = p->numel();

        auto& p_state = state_[i].tensors;
        if (!p_state.contains("exp_avg")) {
            p_state["exp_avg"] = Tensor::zeros(p->shape());
        }
        if (!p_state.contains("exp_avg_sq")) {
            p_state["exp_avg_sq"] = Tensor::zeros(p->shape());
        }
        double* m_data = p_state["exp_avg"]->data();
        double* v_data = p_state["exp_avg_sq"]->data();

        for (size_t k = 0; k < n; ++k) {
            double g = g_vec[k];
            if (weight_decay_ > 0.0) {
                g += weight_decay_ * p_data[k];
            }
            m_data[k] = beta1_ * m_data[k] + (1.0 - beta1_) * g;
            v_data[k] = beta2_ * v_data[k] + (1.0 - beta2_) * (g * g);

            double m_hat = m_data[k] / bias_correction1;
            double v_hat = v_data[k] / bias_correction2;
            p_data[k] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

// --- AdamW ---

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> params,
             double lr,
             double beta1,
             double beta2,
             double eps,
             double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {}

std::map<std::string, double> AdamW::defaults() const {
    return {
        {"lr", lr_},
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"eps", eps_},
        {"weight_decay", weight_decay_},
    };
}

void AdamW::step() {
    step_count_++;
    double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
    double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p || !p->grad()) continue;

        auto g_vec = p->grad()->to_vector();
        double* p_data = p->data();
        size_t n = p->numel();

        auto& p_state = state_[i].tensors;
        if (!p_state.contains("exp_avg")) {
            p_state["exp_avg"] = Tensor::zeros(p->shape());
        }
        if (!p_state.contains("exp_avg_sq")) {
            p_state["exp_avg_sq"] = Tensor::zeros(p->shape());
        }
        double* m_data = p_state["exp_avg"]->data();
        double* v_data = p_state["exp_avg_sq"]->data();

        for (size_t k = 0; k < n; ++k) {
            // Decoupled weight decay
            if (weight_decay_ > 0.0) {
                p_data[k] -= lr_ * weight_decay_ * p_data[k];
            }

            double g = g_vec[k];
            m_data[k] = beta1_ * m_data[k] + (1.0 - beta1_) * g;
            v_data[k] = beta2_ * v_data[k] + (1.0 - beta2_) * (g * g);

            double m_hat = m_data[k] / bias_correction1;
            double v_hat = v_data[k] / bias_correction2;
            p_data[k] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

// --- Schedulers ---

void LRScheduler::step() {
    step_count_++;
    optimizer_.set_lr(get_lr());
}

ConstantLR::ConstantLR(Optimizer& optimizer, double lr)
    : LRScheduler(optimizer), lr_(lr) {
    optimizer_.set_lr(lr_);
}

LinearWarmupDecayLR::LinearWarmupDecayLR(Optimizer& optimizer,
                                         size_t warmup_steps,
                                         size_t total_steps,
                                         double base_lr,
                                         double min_lr)
    : LRScheduler(optimizer),
      warmup_steps_(warmup_steps),
      total_steps_(total_steps),
      base_lr_(base_lr),
      min_lr_(min_lr) {
    optimizer_.set_lr(get_lr());
}

double LinearWarmupDecayLR::get_lr() const {
    if (step_count_ < warmup_steps_) {
        if (warmup_steps_ == 0) return base_lr_;
        return base_lr_ * (static_cast<double>(step_count_) / static_cast<double>(warmup_steps_));
    }
    if (step_count_ >= total_steps_) {
        return min_lr_;
    }
    double progress = static_cast<double>(total_steps_ - step_count_) /
                      static_cast<double>(total_steps_ - warmup_steps_);
    return min_lr_ + (base_lr_ - min_lr_) * std::max(0.0, progress);
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer,
                                     size_t warmup_steps,
                                     size_t total_steps,
                                     double base_lr,
                                     double min_lr)
    : LRScheduler(optimizer),
      warmup_steps_(warmup_steps),
      total_steps_(total_steps),
      base_lr_(base_lr),
      min_lr_(min_lr) {
    optimizer_.set_lr(get_lr());
}

double CosineAnnealingLR::get_lr() const {
    if (step_count_ < warmup_steps_) {
        if (warmup_steps_ == 0) return base_lr_;
        return base_lr_ * (static_cast<double>(step_count_) / static_cast<double>(warmup_steps_));
    }
    if (step_count_ >= total_steps_) {
        return min_lr_;
    }
    double progress = static_cast<double>(step_count_ - warmup_steps_) /
                      static_cast<double>(total_steps_ - warmup_steps_);
    return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + std::cos(std::numbers::pi * progress));
}

} // namespace optim
} // namespace aurora
