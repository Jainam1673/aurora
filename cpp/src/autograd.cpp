#include "aurora/autograd.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace aurora {

std::shared_ptr<Tensor> unbroadcast(const std::shared_ptr<Tensor>& grad, const std::vector<size_t>& target_shape) {
    if (grad->shape() == target_shape) {
        return grad;
    }

    auto current = grad;

    // Handle prepended dimensions (rank difference)
    while (current->ndim() > target_shape.size()) {
        current = current->sum(0, false);
    }

    // Handle dimensions where target was 1
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && current->shape()[i] > 1) {
            current = current->sum(static_cast<int>(i), true);
        }
    }

    if (current->shape() != target_shape) {
        current = current->reshape(target_shape);
    }
    return current;
}

std::vector<std::shared_ptr<Tensor>> AddNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    const auto& y = inputs_[1];

    std::shared_ptr<Tensor> gx = x->requires_grad() ? unbroadcast(grad_output, x->shape()) : nullptr;
    std::shared_ptr<Tensor> gy = y->requires_grad() ? unbroadcast(grad_output, y->shape()) : nullptr;
    return {gx, gy};
}

std::vector<std::shared_ptr<Tensor>> SubNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    const auto& y = inputs_[1];

    std::shared_ptr<Tensor> gx = x->requires_grad() ? unbroadcast(grad_output, x->shape()) : nullptr;
    std::shared_ptr<Tensor> gy = y->requires_grad() ? unbroadcast(grad_output->neg(), y->shape()) : nullptr;
    return {gx, gy};
}

std::vector<std::shared_ptr<Tensor>> MulNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    const auto& y = inputs_[1];

    std::shared_ptr<Tensor> gx = x->requires_grad() ? unbroadcast(grad_output->mul(y), x->shape()) : nullptr;
    std::shared_ptr<Tensor> gy = y->requires_grad() ? unbroadcast(grad_output->mul(x), y->shape()) : nullptr;
    return {gx, gy};
}

std::vector<std::shared_ptr<Tensor>> DivNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    const auto& y = inputs_[1];

    std::shared_ptr<Tensor> gx = x->requires_grad() ? unbroadcast(grad_output->div(y), x->shape()) : nullptr;
    std::shared_ptr<Tensor> gy = nullptr;
    if (y->requires_grad()) {
        auto y2 = y->mul(y);
        auto num = grad_output->neg()->mul(x);
        gy = unbroadcast(num->div(y2), y->shape());
    }
    return {gx, gy};
}

std::vector<std::shared_ptr<Tensor>> MatmulNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& a = inputs_[0];
    const auto& b = inputs_[1];

    std::shared_ptr<Tensor> ga = nullptr;
    std::shared_ptr<Tensor> gb = nullptr;

    if (a->requires_grad()) {
        // dL/dA = grad @ B^T
        auto b_t = b->transpose();
        ga = unbroadcast(grad_output->matmul(b_t), a->shape());
    }
    if (b->requires_grad()) {
        // dL/dB = A^T @ grad
        auto a_t = a->transpose();
        gb = unbroadcast(a_t->matmul(grad_output), b->shape());
    }
    return {ga, gb};
}

std::vector<std::shared_ptr<Tensor>> ReshapeNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) {
        return {nullptr};
    }
    return {grad_output->reshape(orig_shape_)};
}

std::vector<std::shared_ptr<Tensor>> TransposeNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) {
        return {nullptr};
    }

    std::vector<size_t> inv_axes(axes_.size());
    for (size_t i = 0; i < axes_.size(); ++i) {
        inv_axes[axes_[i]] = i;
    }
    return {grad_output->transpose(inv_axes)};
}

std::vector<std::shared_ptr<Tensor>> SumNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) {
        return {nullptr};
    }

    // Broadcast grad_output back to original shape
    std::vector<double> out_data(x->numel());

    if (!axis_.has_value()) {
        double val = grad_output->item();
        std::fill(out_data.begin(), out_data.end(), val);
    } else {
        auto g_reshaped = keepdims_ ? grad_output : [&]() {
            int ax = (*axis_ < 0) ? *axis_ + static_cast<int>(orig_shape_.size()) : *axis_;
            std::vector<size_t> expanded_shape = grad_output->shape();
            expanded_shape.insert(expanded_shape.begin() + ax, 1);
            return grad_output->reshape(expanded_shape);
        }();

        // Broadcast g_reshaped to orig_shape_
        auto dummy_orig = Tensor::zeros(orig_shape_);
        return {dummy_orig->add(g_reshaped)};
    }

    return {std::make_shared<Tensor>(orig_shape_, std::move(out_data), false)};
}

std::vector<std::shared_ptr<Tensor>> MeanNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) {
        return {nullptr};
    }

    size_t reduced_count = axis_.has_value() ?
        orig_shape_[static_cast<size_t>(*axis_ < 0 ? *axis_ + static_cast<int>(orig_shape_.size()) : *axis_)] : x->numel();
    double scale = 1.0 / static_cast<double>(reduced_count);

    auto scaled_grad = grad_output->mul(Tensor::create(grad_output->shape(), scale));

    if (!axis_.has_value()) {
        std::vector<double> out_data(x->numel(), scaled_grad->item());
        return {std::make_shared<Tensor>(orig_shape_, std::move(out_data), false)};
    }

    auto g_reshaped = keepdims_ ? scaled_grad : [&]() {
        int ax = (*axis_ < 0) ? *axis_ + static_cast<int>(orig_shape_.size()) : *axis_;
        std::vector<size_t> expanded_shape = scaled_grad->shape();
        expanded_shape.insert(expanded_shape.begin() + ax, 1);
        return scaled_grad->reshape(expanded_shape);
    }();

    auto dummy_orig = Tensor::zeros(orig_shape_);
    return {dummy_orig->add(g_reshaped)};
}

std::vector<std::shared_ptr<Tensor>> ExpNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};
    return {grad_output->mul(out_)};
}

std::vector<std::shared_ptr<Tensor>> LogNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};
    return {grad_output->div(x)};
}

std::vector<std::shared_ptr<Tensor>> SqrtNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};
    auto two_out = out_->mul(Tensor::create(out_->shape(), 2.0));
    return {grad_output->div(two_out)};
}

std::vector<std::shared_ptr<Tensor>> ReLUNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};

    std::vector<double> mask(x->numel());
    for (size_t i = 0; i < x->numel(); ++i) {
        mask[i] = ((*x)[i] > 0.0) ? 1.0 : 0.0;
    }
    auto mask_t = std::make_shared<Tensor>(x->shape(), std::move(mask), false);
    return {grad_output->mul(mask_t)};
}

std::vector<std::shared_ptr<Tensor>> GELUNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);

    std::vector<double> d_gelu(x->numel());
    for (size_t i = 0; i < x->numel(); ++i) {
        double val = (*x)[i];
        double cdf = 0.5 * (1.0 + std::erf(val * inv_sqrt2));
        double pdf = inv_sqrt_2pi * std::exp(-0.5 * val * val);
        d_gelu[i] = cdf + val * pdf;
    }

    auto d_t = std::make_shared<Tensor>(x->shape(), std::move(d_gelu), false);
    return {grad_output->mul(d_t)};
}

std::vector<std::shared_ptr<Tensor>> SiLUNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};

    std::vector<double> d_silu(x->numel());
    for (size_t i = 0; i < x->numel(); ++i) {
        double val = (*x)[i];
        double sig = 1.0 / (1.0 + std::exp(-val));
        d_silu[i] = sig * (1.0 + val * (1.0 - sig));
    }

    auto d_t = std::make_shared<Tensor>(x->shape(), std::move(d_silu), false);
    return {grad_output->mul(d_t)};
}

std::vector<std::shared_ptr<Tensor>> SoftmaxNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};

    // dL/dx_i = s_i * (dL/ds_i - sum_k(dL/ds_k * s_k))
    auto dot = grad_output->mul(out_)->sum(axis_, true);
    auto sub = grad_output->sub(dot);
    return {out_->mul(sub)};
}

std::vector<std::shared_ptr<Tensor>> LogSoftmaxNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    if (!x->requires_grad()) return {nullptr};

    // dL/dx_i = dL/dy_i - s_i * sum_k(dL/dy_k)
    auto sum_grad = grad_output->sum(axis_, true);
    auto sub = softmax_out_->mul(sum_grad);
    return {grad_output->sub(sub)};
}

std::vector<std::shared_ptr<Tensor>> LayerNormNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& x = inputs_[0];
    const auto& gamma = inputs_[1];
    const auto& beta = inputs_[2];

    size_t uax = (axis_ < 0) ? static_cast<size_t>(axis_ + static_cast<int>(x->ndim())) : static_cast<size_t>(axis_);
    double d_val = static_cast<double>(x->shape()[uax]);

    auto dy_gamma = gamma ? grad_output->mul(gamma) : grad_output;
    auto sum_dy_gamma = dy_gamma->sum(axis_, true);
    auto sum_dy_gamma_xhat = dy_gamma->mul(x_hat_)->sum(axis_, true);

    std::shared_ptr<Tensor> gx = nullptr;
    if (x->requires_grad()) {
        auto d_dy_gamma = dy_gamma->mul(Tensor::create(dy_gamma->shape(), d_val));
        auto term = d_dy_gamma->sub(sum_dy_gamma)->sub(x_hat_->mul(sum_dy_gamma_xhat));
        auto scale = std_inv_->div(Tensor::create(std_inv_->shape(), d_val));
        gx = scale->mul(term);
    }

    std::shared_ptr<Tensor> ggamma = nullptr;
    if (gamma && gamma->requires_grad()) {
        ggamma = unbroadcast(grad_output->mul(x_hat_)->sum(0, true), gamma->shape());
    }

    std::shared_ptr<Tensor> gbeta = nullptr;
    if (beta && beta->requires_grad()) {
        gbeta = unbroadcast(grad_output->sum(0, true), beta->shape());
    }

    return {gx, ggamma, gbeta};
}

std::vector<std::shared_ptr<Tensor>> EmbeddingNode::backward(const std::shared_ptr<Tensor>& grad_output) {
    const auto& w = inputs_[0];
    if (!w->requires_grad()) {
        return {nullptr};
    }
    auto grad_weight = Tensor::zeros(w->shape());
    size_t embed_dim = w->shape()[1];
    auto grad_out_vec = grad_output->to_vector();
    double* grad_w_data = grad_weight->data();

    for (size_t i = 0; i < indices_.size(); ++i) {
        size_t idx = indices_[i];
        if (idx >= w->shape()[0]) {
            throw std::runtime_error("Index out of bounds in embedding backward");
        }
        for (size_t d = 0; d < embed_dim; ++d) {
            grad_w_data[idx * embed_dim + d] += grad_out_vec[i * embed_dim + d];
        }
    }
    return {grad_weight};
}

} // namespace aurora
