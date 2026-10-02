#pragma once

#include "aurora/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aurora {
namespace nn {

class Parameter : public Tensor {
public:
    using Tensor::Tensor;
    explicit Parameter(std::vector<size_t> shape, double init_val = 0.0)
        : Tensor(std::move(shape), init_val, /*requires_grad=*/true) {}
    Parameter(std::vector<size_t> shape, std::vector<double> data)
        : Tensor(std::move(shape), std::move(data), /*requires_grad=*/true) {}
    explicit Parameter(const Tensor& t)
        : Tensor(t.shape(), t.to_vector(), /*requires_grad=*/true) {}

    static std::shared_ptr<Parameter> create(std::vector<size_t> shape, double init_val = 0.0) {
        return std::make_shared<Parameter>(std::move(shape), init_val);
    }
    static std::shared_ptr<Parameter> create(std::vector<size_t> shape, std::vector<double> data) {
        return std::make_shared<Parameter>(std::move(shape), std::move(data));
    }
};

class Module {
public:
    virtual ~Module() = default;

    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) = 0;

    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& input) {
        return forward(input);
    }

    void train(bool mode = true) noexcept;
    void eval() noexcept { train(false); }
    [[nodiscard]] bool is_training() const noexcept { return training_; }

    virtual std::vector<std::shared_ptr<Tensor>> parameters(bool recurse = true) const;
    virtual std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_parameters(
        const std::string& prefix = "", bool recurse = true) const;

    virtual void zero_grad() noexcept;

    using StateDict = std::map<std::string, std::shared_ptr<Tensor>>;
    virtual StateDict state_dict(const std::string& prefix = "") const;
    virtual void load_state_dict(const StateDict& dict);

    void register_parameter(const std::string& name, std::shared_ptr<Tensor> param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

protected:
    bool training_{true};
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
};

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true, uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> weight() const noexcept { return weight_; }
    [[nodiscard]] std::shared_ptr<Tensor> bias() const noexcept { return bias_; }
    [[nodiscard]] size_t in_features() const noexcept { return in_features_; }
    [[nodiscard]] size_t out_features() const noexcept { return out_features_; }

private:
    size_t in_features_;
    size_t out_features_;
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_{nullptr};
};

class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim, uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& indices_tensor) override;
    std::shared_ptr<Tensor> forward(const std::vector<size_t>& indices, const std::vector<size_t>& indices_shape);

    [[nodiscard]] std::shared_ptr<Tensor> weight() const noexcept { return weight_; }
    [[nodiscard]] size_t num_embeddings() const noexcept { return num_embeddings_; }
    [[nodiscard]] size_t embedding_dim() const noexcept { return embedding_dim_; }

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
    std::shared_ptr<Tensor> weight_;
};

class LayerNorm : public Module {
public:
    explicit LayerNorm(size_t normalized_shape, double eps = 1e-5, bool elementwise_affine = true);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> weight() const noexcept { return weight_; }
    [[nodiscard]] std::shared_ptr<Tensor> bias() const noexcept { return bias_; }
    [[nodiscard]] size_t normalized_shape() const noexcept { return normalized_shape_; }
    [[nodiscard]] bool elementwise_affine() const noexcept { return elementwise_affine_; }

private:
    size_t normalized_shape_;
    double eps_;
    bool elementwise_affine_;
    std::shared_ptr<Tensor> weight_{nullptr};
    std::shared_ptr<Tensor> bias_{nullptr};
};

class RMSNorm : public Module {
public:
    explicit RMSNorm(size_t dim, double eps = 1e-6);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    [[nodiscard]] std::shared_ptr<Tensor> weight() const noexcept { return weight_; }
    [[nodiscard]] size_t dim() const noexcept { return dim_; }

private:
    size_t dim_;
    double eps_;
    std::shared_ptr<Tensor> weight_;
};

class Dropout : public Module {
public:
    explicit Dropout(double p = 0.5, uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    [[nodiscard]] double p() const noexcept { return p_; }

private:
    double p_;
    uint64_t seed_;
};

class ReLU : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        return input->relu();
    }
};

class GELU : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        return input->gelu();
    }
};

class SiLU : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        return input->silu();
    }
};

class Tanh : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        return input->tanh();
    }
};

class Softmax : public Module {
public:
    explicit Softmax(int axis = -1) : axis_(axis) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        return input->softmax(axis_);
    }

private:
    int axis_;
};

class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int axis = -1) : axis_(axis) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        return input->log_softmax(axis_);
    }

private:
    int axis_;
};

class Sequential : public Module {
public:
    Sequential() = default;
    explicit Sequential(std::vector<std::shared_ptr<Module>> modules);

    void add(const std::string& name, std::shared_ptr<Module> module);
    void add(std::shared_ptr<Module> module);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

    [[nodiscard]] const std::vector<std::shared_ptr<Module>>& modules() const noexcept { return ordered_modules_; }

private:
    std::vector<std::shared_ptr<Module>> ordered_modules_;
};

class MLP : public Module {
public:
    MLP(size_t in_features,
        const std::vector<size_t>& hidden_dims,
        size_t out_features,
        const std::string& activation = "relu",
        double dropout = 0.0,
        uint64_t seed = 42);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

private:
    std::shared_ptr<Sequential> net_;
};

class ResidualBlock : public Module {
public:
    ResidualBlock(std::shared_ptr<Module> block, std::shared_ptr<Module> shortcut = nullptr);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override;

private:
    std::shared_ptr<Module> block_;
    std::shared_ptr<Module> shortcut_{nullptr};
};

} // namespace nn
} // namespace aurora
