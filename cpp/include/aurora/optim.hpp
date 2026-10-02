#pragma once

#include "aurora/tensor.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aurora {
namespace optim {

double clip_grad_norm(const std::vector<std::shared_ptr<Tensor>>& params,
                      double max_norm,
                      double norm_type = 2.0);

void clip_grad_value(const std::vector<std::shared_ptr<Tensor>>& params,
                     double clip_value);

struct ParamState {
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
};

struct OptimizerStateDict {
    size_t step_count{0};
    std::map<std::string, double> defaults;
    std::map<size_t, ParamState> state;
};

class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<Tensor>> params);
    virtual ~Optimizer() = default;

    void zero_grad() noexcept;
    virtual void step() = 0;

    [[nodiscard]] size_t step_count() const noexcept { return step_count_; }
    [[nodiscard]] const std::vector<std::shared_ptr<Tensor>>& params() const noexcept { return params_; }

    virtual void set_lr(double lr) = 0;
    [[nodiscard]] virtual double get_lr() const = 0;
    [[nodiscard]] virtual std::map<std::string, double> defaults() const = 0;

    virtual OptimizerStateDict state_dict() const;
    virtual void load_state_dict(const OptimizerStateDict& s);

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
    size_t step_count_{0};
    std::map<size_t, ParamState> state_;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<std::shared_ptr<Tensor>> params,
        double lr,
        double momentum = 0.0,
        double weight_decay = 0.0);

    void step() override;
    void set_lr(double lr) override { lr_ = lr; }
    [[nodiscard]] double get_lr() const override { return lr_; }
    [[nodiscard]] std::map<std::string, double> defaults() const override;

private:
    double lr_;
    double momentum_;
    double weight_decay_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<std::shared_ptr<Tensor>> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0);

    void step() override;
    void set_lr(double lr) override { lr_ = lr; }
    [[nodiscard]] double get_lr() const override { return lr_; }
    [[nodiscard]] std::map<std::string, double> defaults() const override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
};

class AdamW : public Optimizer {
public:
    AdamW(std::vector<std::shared_ptr<Tensor>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 1e-2);

    void step() override;
    void set_lr(double lr) override { lr_ = lr; }
    [[nodiscard]] double get_lr() const override { return lr_; }
    [[nodiscard]] std::map<std::string, double> defaults() const override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
};

// --- Schedulers ---

class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer) : optimizer_(optimizer) {}
    virtual ~LRScheduler() = default;

    virtual void step();
    [[nodiscard]] virtual double get_lr() const = 0;
    [[nodiscard]] size_t step_count() const noexcept { return step_count_; }

protected:
    Optimizer& optimizer_;
    size_t step_count_{0};
};

class ConstantLR : public LRScheduler {
public:
    ConstantLR(Optimizer& optimizer, double lr);
    [[nodiscard]] double get_lr() const override { return lr_; }

private:
    double lr_;
};

class LinearWarmupDecayLR : public LRScheduler {
public:
    LinearWarmupDecayLR(Optimizer& optimizer,
                        size_t warmup_steps,
                        size_t total_steps,
                        double base_lr,
                        double min_lr = 0.0);

    [[nodiscard]] double get_lr() const override;

private:
    size_t warmup_steps_;
    size_t total_steps_;
    double base_lr_;
    double min_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer,
                      size_t warmup_steps,
                      size_t total_steps,
                      double base_lr,
                      double min_lr = 0.0);

    [[nodiscard]] double get_lr() const override;

private:
    size_t warmup_steps_;
    size_t total_steps_;
    double base_lr_;
    double min_lr_;
};

} // namespace optim
} // namespace aurora
