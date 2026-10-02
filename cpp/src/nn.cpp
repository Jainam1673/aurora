#include "aurora/nn.hpp"
#include "aurora/autograd.hpp"

#include <cmath>
#include <format>
#include <random>
#include <stdexcept>

namespace aurora {
namespace nn {

// --- Module Base ---

void Module::train(bool mode) noexcept {
    training_ = mode;
    for (auto& [_, submod] : submodules_) {
        if (submod) {
            submod->train(mode);
        }
    }
}

std::vector<std::shared_ptr<Tensor>> Module::parameters(bool recurse) const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& [_, p] : parameters_) {
        params.push_back(p);
    }
    if (recurse) {
        for (const auto& [_, submod] : submodules_) {
            if (submod) {
                auto sub_params = submod->parameters(true);
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }
        }
    }
    return params;
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::named_parameters(
    const std::string& prefix, bool recurse) const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> result;
    for (const auto& [name, p] : parameters_) {
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        result.emplace_back(full_name, p);
    }
    if (recurse) {
        for (const auto& [mod_name, submod] : submodules_) {
            if (submod) {
                std::string sub_prefix = prefix.empty() ? mod_name : prefix + "." + mod_name;
                auto sub_named = submod->named_parameters(sub_prefix, true);
                result.insert(result.end(), sub_named.begin(), sub_named.end());
            }
        }
    }
    return result;
}

void Module::zero_grad() noexcept {
    for (auto& p : parameters(true)) {
        p->zero_grad();
    }
}

Module::StateDict Module::state_dict(const std::string& prefix) const {
    StateDict dict;
    for (const auto& [name, p] : named_parameters(prefix, true)) {
        dict[name] = p;
    }
    return dict;
}

void Module::load_state_dict(const StateDict& dict) {
    auto current_params = named_parameters("", true);
    std::map<std::string, std::shared_ptr<Tensor>> param_map;
    for (const auto& [name, p] : current_params) {
        param_map[name] = p;
    }

    for (const auto& [name, src_tensor] : dict) {
        auto it = param_map.find(name);
        if (it == param_map.end()) {
            throw std::invalid_argument(std::format("Unexpected key '{}' in state_dict", name));
        }
        auto dest_tensor = it->second;
        if (dest_tensor->shape() != src_tensor->shape()) {
            throw std::invalid_argument(std::format("Shape mismatch for parameter '{}'", name));
        }
        auto src_vec = src_tensor->to_vector();
        double* dest_data = dest_tensor->data();
        for (size_t i = 0; i < src_vec.size(); ++i) {
            dest_data[i] = src_vec[i];
        }
    }
}

void Module::register_parameter(const std::string& name, std::shared_ptr<Tensor> param) {
    for (auto& [n, p] : parameters_) {
        if (n == name) {
            p = std::move(param);
            return;
        }
    }
    parameters_.emplace_back(name, std::move(param));
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    for (auto& [n, m] : submodules_) {
        if (n == name) {
            m = std::move(module);
            return;
        }
    }
    submodules_.emplace_back(name, std::move(module));
}

// --- Linear ---

Linear::Linear(size_t in_features, size_t out_features, bool bias, uint64_t seed)
    : in_features_(in_features), out_features_(out_features) {
    std::mt19937_64 rng(seed);
    double k = 1.0 / std::sqrt(static_cast<double>(in_features));
    std::uniform_real_distribution<double> dist(-k, k);

    std::vector<double> w_data(out_features * in_features);
    for (auto& v : w_data) {
        v = dist(rng);
    }
    weight_ = Parameter::create({out_features, in_features}, std::move(w_data));
    register_parameter("weight", weight_);

    if (bias) {
        std::vector<double> b_data(out_features);
        for (auto& v : b_data) {
            v = dist(rng);
        }
        bias_ = Parameter::create({out_features}, std::move(b_data));
        register_parameter("bias", bias_);
    }
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input) {
    auto out = input->matmul(weight_->transpose());
    if (bias_) {
        out = out + bias_;
    }
    return out;
}

// --- Embedding ---

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim, uint64_t seed)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> w_data(num_embeddings * embedding_dim);
    for (auto& v : w_data) {
        v = dist(rng);
    }
    weight_ = Parameter::create({num_embeddings, embedding_dim}, std::move(w_data));
    register_parameter("weight", weight_);
}

std::shared_ptr<Tensor> Embedding::forward(const std::vector<size_t>& indices, const std::vector<size_t>& indices_shape) {
    size_t V = weight_->shape()[0];
    size_t D = weight_->shape()[1];
    size_t N = indices.size();

    std::vector<size_t> out_shape = indices_shape;
    out_shape.push_back(D);

    std::vector<double> out_data(N * D);
    const double* w_data = weight_->data();

    for (size_t i = 0; i < N; ++i) {
        size_t idx = indices[i];
        if (idx >= V) {
            throw std::out_of_range(std::format("Embedding index {} out of range [0, {})", idx, V));
        }
        for (size_t d = 0; d < D; ++d) {
            out_data[i * D + d] = w_data[idx * D + d];
        }
    }

    auto out = std::make_shared<Tensor>(out_shape, std::move(out_data), weight_->requires_grad());
    if (weight_->requires_grad()) {
        out->set_creator(std::make_shared<EmbeddingNode>(weight_, indices, indices_shape));
    }
    return out;
}

std::shared_ptr<Tensor> Embedding::forward(const std::shared_ptr<Tensor>& indices_tensor) {
    auto vec = indices_tensor->to_vector();
    std::vector<size_t> indices(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indices[i] = static_cast<size_t>(vec[i]);
    }
    return forward(indices, indices_tensor->shape());
}

// --- LayerNorm ---

LayerNorm::LayerNorm(size_t normalized_shape, double eps, bool elementwise_affine)
    : normalized_shape_(normalized_shape), eps_(eps), elementwise_affine_(elementwise_affine) {
    if (elementwise_affine) {
        weight_ = Parameter::create({normalized_shape}, 1.0);
        bias_ = Parameter::create({normalized_shape}, 0.0);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
}

std::shared_ptr<Tensor> LayerNorm::forward(const std::shared_ptr<Tensor>& input) {
    return input->layer_norm(weight_, bias_, eps_, -1);
}

// --- RMSNorm ---

RMSNorm::RMSNorm(size_t dim, double eps) : dim_(dim), eps_(eps) {
    weight_ = Parameter::create({dim}, 1.0);
    register_parameter("weight", weight_);
}

std::shared_ptr<Tensor> RMSNorm::forward(const std::shared_ptr<Tensor>& input) {
    // RMS(x) = sqrt(mean(x^2) + eps)
    auto x2 = input * input;
    auto mean_x2 = x2->mean(-1, /*keepdims=*/true);
    auto rms = (mean_x2 + eps_)->sqrt();
    auto x_norm = input / rms;
    return x_norm * weight_;
}

// --- Dropout ---

Dropout::Dropout(double p, uint64_t seed) : p_(p), seed_(seed) {
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument("Dropout probability must be in [0, 1)");
    }
}

std::shared_ptr<Tensor> Dropout::forward(const std::shared_ptr<Tensor>& input) {
    if (!training_ || p_ == 0.0) {
        return input;
    }
    std::mt19937_64 rng(seed_);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double q = 1.0 - p_;
    double scale = 1.0 / q;
    size_t count = input->numel();
    std::vector<double> mask_data(count);

    for (size_t i = 0; i < count; ++i) {
        mask_data[i] = (dist(rng) < q) ? scale : 0.0;
    }

    auto mask_tensor = std::make_shared<Tensor>(input->shape(), std::move(mask_data), /*requires_grad=*/false);
    return input * mask_tensor;
}

// --- Sequential ---

Sequential::Sequential(std::vector<std::shared_ptr<Module>> modules) {
    for (size_t i = 0; i < modules.size(); ++i) {
        add(std::format("layer_{}", i), modules[i]);
    }
}

void Sequential::add(const std::string& name, std::shared_ptr<Module> module) {
    ordered_modules_.push_back(module);
    register_module(name, std::move(module));
}

void Sequential::add(std::shared_ptr<Module> module) {
    add(std::format("layer_{}", ordered_modules_.size()), std::move(module));
}

std::shared_ptr<Tensor> Sequential::forward(const std::shared_ptr<Tensor>& input) {
    auto current = input;
    for (const auto& mod : ordered_modules_) {
        current = mod->forward(current);
    }
    return current;
}

// --- MLP ---

MLP::MLP(size_t in_features,
         const std::vector<size_t>& hidden_dims,
         size_t out_features,
         const std::string& activation,
         double dropout,
         uint64_t seed) {
    net_ = std::make_shared<Sequential>();
    size_t curr_in = in_features;

    auto make_activation = [](const std::string& act) -> std::shared_ptr<Module> {
        if (act == "relu") return std::make_shared<ReLU>();
        if (act == "gelu") return std::make_shared<GELU>();
        if (act == "silu") return std::make_shared<SiLU>();
        if (act == "tanh") return std::make_shared<Tanh>();
        throw std::invalid_argument(std::format("Unknown activation '{}'", act));
    };

    uint64_t cur_seed = seed;
    for (size_t h_dim : hidden_dims) {
        net_->add(std::make_shared<Linear>(curr_in, h_dim, true, cur_seed++));
        net_->add(make_activation(activation));
        if (dropout > 0.0) {
            net_->add(std::make_shared<Dropout>(dropout, cur_seed++));
        }
        curr_in = h_dim;
    }

    net_->add(std::make_shared<Linear>(curr_in, out_features, true, cur_seed++));
    register_module("net", net_);
}

std::shared_ptr<Tensor> MLP::forward(const std::shared_ptr<Tensor>& input) {
    return net_->forward(input);
}

// --- ResidualBlock ---

ResidualBlock::ResidualBlock(std::shared_ptr<Module> block, std::shared_ptr<Module> shortcut)
    : block_(std::move(block)), shortcut_(std::move(shortcut)) {
    register_module("block", block_);
    if (shortcut_) {
        register_module("shortcut", shortcut_);
    }
}

std::shared_ptr<Tensor> ResidualBlock::forward(const std::shared_ptr<Tensor>& input) {
    auto res = block_->forward(input);
    auto sc = shortcut_ ? shortcut_->forward(input) : input;
    return sc + res;
}

} // namespace nn
} // namespace aurora
