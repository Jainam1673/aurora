#pragma once

#include "aurora/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace aurora {

class Distribution {
public:
    virtual ~Distribution() = default;

    virtual std::shared_ptr<Tensor> sample(uint64_t seed = 0) = 0;
    virtual std::shared_ptr<Tensor> rsample(uint64_t seed = 0) { return sample(seed); }
    virtual std::shared_ptr<Tensor> log_prob(const std::shared_ptr<Tensor>& value) = 0;
    virtual std::shared_ptr<Tensor> entropy() = 0;
    virtual std::shared_ptr<Tensor> mean() const = 0;
};

class Categorical : public Distribution {
public:
    explicit Categorical(std::shared_ptr<Tensor> logits);

    std::shared_ptr<Tensor> sample(uint64_t seed = 0) override;
    std::shared_ptr<Tensor> log_prob(const std::shared_ptr<Tensor>& value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() const override { return probs_; }

    [[nodiscard]] const std::shared_ptr<Tensor>& logits() const noexcept { return logits_; }
    [[nodiscard]] const std::shared_ptr<Tensor>& probs() const noexcept { return probs_; }

private:
    std::shared_ptr<Tensor> logits_;
    std::shared_ptr<Tensor> probs_;
    size_t num_classes_;
};

class Normal : public Distribution {
public:
    Normal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale);

    std::shared_ptr<Tensor> sample(uint64_t seed = 0) override;
    std::shared_ptr<Tensor> rsample(uint64_t seed = 0) override;
    std::shared_ptr<Tensor> log_prob(const std::shared_ptr<Tensor>& value) override;
    std::shared_ptr<Tensor> log_prob(const std::shared_ptr<Tensor>& value, bool sum_features);
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> entropy(bool sum_features);
    std::shared_ptr<Tensor> mean() const override { return loc_; }
    std::shared_ptr<Tensor> variance() const { return scale_->mul(scale_); }

    [[nodiscard]] const std::shared_ptr<Tensor>& loc() const noexcept { return loc_; }
    [[nodiscard]] const std::shared_ptr<Tensor>& scale() const noexcept { return scale_; }

private:
    std::shared_ptr<Tensor> loc_;
    std::shared_ptr<Tensor> scale_;
};

class TanhNormal : public Distribution {
public:
    TanhNormal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale, double eps = 1e-6);

    std::shared_ptr<Tensor> sample(uint64_t seed = 0) override;
    std::shared_ptr<Tensor> rsample(uint64_t seed = 0) override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> rsample_with_pre_tanh(uint64_t seed = 0);
    std::shared_ptr<Tensor> log_prob(const std::shared_ptr<Tensor>& value) override;
    std::shared_ptr<Tensor> log_prob(const std::shared_ptr<Tensor>& value, const std::shared_ptr<Tensor>& pre_tanh_value);
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() const override { return loc_->tanh(); }

    [[nodiscard]] const std::shared_ptr<Tensor>& loc() const noexcept { return loc_; }
    [[nodiscard]] const std::shared_ptr<Tensor>& scale() const noexcept { return scale_; }

private:
    std::shared_ptr<Tensor> loc_;
    std::shared_ptr<Tensor> scale_;
    Normal normal_;
    double eps_;
};

} // namespace aurora
