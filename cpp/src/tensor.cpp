#include "aurora/tensor.hpp"
#include "aurora/autograd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace aurora {

std::vector<size_t> Tensor::compute_c_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

Tensor::Tensor() : shape_{}, strides_{}, offset_{0}, storage_{std::make_shared<std::vector<double>>(1, 0.0)} {}

Tensor::Tensor(std::vector<size_t> shape, double init_val, bool requires_grad)
    : shape_(std::move(shape)),
      strides_(compute_c_strides(shape_)),
      offset_(0),
      requires_grad_(requires_grad)
{
    size_t count = numel();
    storage_ = std::make_shared<std::vector<double>>(count, init_val);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data, bool requires_grad)
    : shape_(std::move(shape)),
      strides_(compute_c_strides(shape_)),
      offset_(0),
      requires_grad_(requires_grad)
{
    size_t count = numel();
    if (data.size() != count) {
        throw std::invalid_argument(std::format("Data size {} does not match shape count {}", data.size(), count));
    }
    storage_ = std::make_shared<std::vector<double>>(std::move(data));
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset,
               std::shared_ptr<std::vector<double>> storage, bool requires_grad)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      storage_(std::move(storage)),
      requires_grad_(requires_grad) {}

size_t Tensor::numel() const noexcept {
    if (shape_.empty()) {
        return 1;
    }
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
}

bool Tensor::is_contiguous() const noexcept {
    if (shape_.empty()) {
        return true;
    }
    auto expected = compute_c_strides(shape_);
    return strides_ == expected && offset_ == 0;
}

std::vector<double> Tensor::to_vector() const {
    if (is_contiguous()) {
        return std::vector<double>(storage_->begin() + static_cast<std::ptrdiff_t>(offset_),
                                   storage_->begin() + static_cast<std::ptrdiff_t>(offset_ + numel()));
    }
    std::vector<double> out(numel());
    size_t n = numel();
    for (size_t i = 0; i < n; ++i) {
        out[i] = (*this)[i];
    }
    return out;
}

double Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() can only be called on tensors with 1 element");
    }
    return (*this)[0];
}

double& Tensor::operator[](size_t linear_idx) {
    if (is_contiguous()) {
        return (*storage_)[offset_ + linear_idx];
    }
    // Decompose linear index to multi-index
    size_t rem = linear_idx;
    size_t actual_offset = offset_;
    for (size_t d = 0; d < shape_.size(); ++d) {
        size_t c_stride = (d + 1 < shape_.size()) ? compute_c_strides(shape_)[d] : 1;
        size_t idx = rem / c_stride;
        rem %= c_stride;
        actual_offset += idx * strides_[d];
    }
    return (*storage_)[actual_offset];
}

const double& Tensor::operator[](size_t linear_idx) const {
    if (is_contiguous()) {
        return (*storage_)[offset_ + linear_idx];
    }
    size_t rem = linear_idx;
    size_t actual_offset = offset_;
    for (size_t d = 0; d < shape_.size(); ++d) {
        size_t c_stride = (d + 1 < shape_.size()) ? compute_c_strides(shape_)[d] : 1;
        size_t idx = rem / c_stride;
        rem %= c_stride;
        actual_offset += idx * strides_[d];
    }
    return (*storage_)[actual_offset];
}

double& Tensor::at(const std::vector<size_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Rank mismatch in Tensor::at");
    }
    size_t idx = offset_;
    for (size_t d = 0; d < indices.size(); ++d) {
        idx += indices[d] * strides_[d];
    }
    return (*storage_)[idx];
}

const double& Tensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Rank mismatch in Tensor::at");
    }
    size_t idx = offset_;
    for (size_t d = 0; d < indices.size(); ++d) {
        idx += indices[d] * strides_[d];
    }
    return (*storage_)[idx];
}

std::shared_ptr<Tensor> Tensor::contiguous() const {
    if (is_contiguous()) {
        return std::const_pointer_cast<Tensor>(shared_from_this());
    }
    return std::make_shared<Tensor>(shape_, to_vector(), requires_grad_);
}

void Tensor::zero_grad() noexcept {
    grad_ = nullptr;
}

// Broadcasting Helper
static std::pair<std::vector<size_t>, std::pair<std::vector<size_t>, std::vector<size_t>>>
broadcast_shapes_and_strides(const Tensor& a, const Tensor& b) {
    size_t rank_a = a.ndim();
    size_t rank_b = b.ndim();
    size_t max_rank = std::max(rank_a, rank_b);

    std::vector<size_t> out_shape(max_rank, 1);
    std::vector<size_t> strides_a(max_rank, 0);
    std::vector<size_t> strides_b(max_rank, 0);

    for (size_t i = 0; i < max_rank; ++i) {
        size_t dim_a = (i < max_rank - rank_a) ? 1 : a.shape()[i - (max_rank - rank_a)];
        size_t dim_b = (i < max_rank - rank_b) ? 1 : b.shape()[i - (max_rank - rank_b)];

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument(std::format("Incompatible shapes for broadcasting"));
        }

        out_shape[i] = std::max(dim_a, dim_b);

        if (i >= max_rank - rank_a) {
            strides_a[i] = (dim_a == 1) ? 0 : a.strides()[i - (max_rank - rank_a)];
        }
        if (i >= max_rank - rank_b) {
            strides_b[i] = (dim_b == 1) ? 0 : b.strides()[i - (max_rank - rank_b)];
        }
    }

    return {out_shape, {strides_a, strides_b}};
}

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor>& other) const {
    auto [out_shape, strides] = broadcast_shapes_and_strides(*this, *other);
    auto [s_a, s_b] = strides;

    size_t total = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<double> out_data(total);

    if (shape_ == other->shape() && is_contiguous() && other->is_contiguous()) {
        const double* a_ptr = data();
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }
    } else if (is_contiguous() && other->numel() == 1) {
        const double* a_ptr = data();
        double b_val = (*other->storage_)[other->offset_];
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] + b_val;
        }
    } else if (other->is_contiguous() && numel() == 1) {
        double a_val = (*storage_)[offset_];
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_val + b_ptr[idx];
        }
    } else {
        auto c_strides = compute_c_strides(out_shape);
        for (size_t idx = 0; idx < total; ++idx) {
            size_t rem = idx;
            size_t off_a = offset_;
            size_t off_b = other->offset_;

            for (size_t d = 0; d < out_shape.size(); ++d) {
                size_t c_s = (d + 1 < out_shape.size()) ? c_strides[d] : 1;
                size_t coord = rem / c_s;
                rem %= c_s;
                off_a += coord * s_a[d];
                off_b += coord * s_b[d];
            }

            out_data[idx] = (*storage_)[off_a] + (*other->storage_)[off_b];
        }
    }

    bool req = requires_grad_ || other->requires_grad_;
    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), req);
    if (req) {
        result->set_creator(std::make_shared<AddNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this()), other
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor>& other) const {
    auto [out_shape, strides] = broadcast_shapes_and_strides(*this, *other);
    auto [s_a, s_b] = strides;

    size_t total = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<double> out_data(total);

    if (shape_ == other->shape() && is_contiguous() && other->is_contiguous()) {
        const double* a_ptr = data();
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }
    } else if (is_contiguous() && other->numel() == 1) {
        const double* a_ptr = data();
        double b_val = (*other->storage_)[other->offset_];
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] - b_val;
        }
    } else if (other->is_contiguous() && numel() == 1) {
        double a_val = (*storage_)[offset_];
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_val - b_ptr[idx];
        }
    } else {
        auto c_strides = compute_c_strides(out_shape);
        for (size_t idx = 0; idx < total; ++idx) {
            size_t rem = idx;
            size_t off_a = offset_;
            size_t off_b = other->offset_;

            for (size_t d = 0; d < out_shape.size(); ++d) {
                size_t c_s = (d + 1 < out_shape.size()) ? c_strides[d] : 1;
                size_t coord = rem / c_s;
                rem %= c_s;
                off_a += coord * s_a[d];
                off_b += coord * s_b[d];
            }

            out_data[idx] = (*storage_)[off_a] - (*other->storage_)[off_b];
        }
    }

    bool req = requires_grad_ || other->requires_grad_;
    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), req);
    if (req) {
        result->set_creator(std::make_shared<SubNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this()), other
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor>& other) const {
    auto [out_shape, strides] = broadcast_shapes_and_strides(*this, *other);
    auto [s_a, s_b] = strides;

    size_t total = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<double> out_data(total);

    if (shape_ == other->shape() && is_contiguous() && other->is_contiguous()) {
        const double* a_ptr = data();
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }
    } else if (is_contiguous() && other->numel() == 1) {
        const double* a_ptr = data();
        double b_val = (*other->storage_)[other->offset_];
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] * b_val;
        }
    } else if (other->is_contiguous() && numel() == 1) {
        double a_val = (*storage_)[offset_];
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_val * b_ptr[idx];
        }
    } else {
        auto c_strides = compute_c_strides(out_shape);
        for (size_t idx = 0; idx < total; ++idx) {
            size_t rem = idx;
            size_t off_a = offset_;
            size_t off_b = other->offset_;

            for (size_t d = 0; d < out_shape.size(); ++d) {
                size_t c_s = (d + 1 < out_shape.size()) ? c_strides[d] : 1;
                size_t coord = rem / c_s;
                rem %= c_s;
                off_a += coord * s_a[d];
                off_b += coord * s_b[d];
            }

            out_data[idx] = (*storage_)[off_a] * (*other->storage_)[off_b];
        }
    }

    bool req = requires_grad_ || other->requires_grad_;
    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), req);
    if (req) {
        result->set_creator(std::make_shared<MulNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this()), other
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor>& other) const {
    auto [out_shape, strides] = broadcast_shapes_and_strides(*this, *other);
    auto [s_a, s_b] = strides;

    size_t total = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<double> out_data(total);

    if (shape_ == other->shape() && is_contiguous() && other->is_contiguous()) {
        const double* a_ptr = data();
        const double* b_ptr = other->data();
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        }
    } else if (is_contiguous() && other->numel() == 1) {
        const double* a_ptr = data();
        double b_val = (*other->storage_)[other->offset_];
        double* out_ptr = out_data.data();
        for (size_t idx = 0; idx < total; ++idx) {
            out_ptr[idx] = a_ptr[idx] / b_val;
        }
    } else {
        auto c_strides = compute_c_strides(out_shape);
        for (size_t idx = 0; idx < total; ++idx) {
            size_t rem = idx;
            size_t off_a = offset_;
            size_t off_b = other->offset_;

            for (size_t d = 0; d < out_shape.size(); ++d) {
                size_t c_s = (d + 1 < out_shape.size()) ? c_strides[d] : 1;
                size_t coord = rem / c_s;
                rem %= c_s;
                off_a += coord * s_a[d];
                off_b += coord * s_b[d];
            }

            out_data[idx] = (*storage_)[off_a] / (*other->storage_)[off_b];
        }
    }

    bool req = requires_grad_ || other->requires_grad_;
    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), req);
    if (req) {
        result->set_creator(std::make_shared<DivNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this()), other
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::neg() const {
    auto zero = Tensor::zeros(shape_);
    return zero->sub(std::const_pointer_cast<Tensor>(shared_from_this()));
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) const {
    if (ndim() < 2 || other->ndim() < 2) {
        throw std::invalid_argument("matmul requires tensors of rank at least 2");
    }

    size_t M = shape_[ndim() - 2];
    size_t K1 = shape_[ndim() - 1];
    size_t K2 = other->shape()[other->ndim() - 2];
    size_t N = other->shape()[other->ndim() - 1];

    if (K1 != K2) {
        throw std::invalid_argument(std::format("Incompatible matrix dimensions: {} vs {}", K1, K2));
    }

    // Determine batch shape
    std::vector<size_t> batch_a(shape_.begin(), shape_.end() - 2);
    std::vector<size_t> batch_b(other->shape().begin(), other->shape().end() - 2);

    size_t batch_size_a = std::accumulate(batch_a.begin(), batch_a.end(), size_t{1}, std::multiplies<size_t>());
    size_t batch_size_b = std::accumulate(batch_b.begin(), batch_b.end(), size_t{1}, std::multiplies<size_t>());

    std::vector<size_t> out_shape = batch_a.empty() ? batch_b : batch_a;
    out_shape.push_back(M);
    out_shape.push_back(N);

    size_t total_batches = std::max(batch_size_a, batch_size_b);
    std::vector<double> out_data(total_batches * M * N, 0.0);

    auto a_contig = contiguous();
    auto b_contig = other->contiguous();

    for (size_t b_idx = 0; b_idx < total_batches; ++b_idx) {
        size_t a_b = (batch_size_a == 1) ? 0 : b_idx;
        size_t b_b = (batch_size_b == 1) ? 0 : b_idx;

        const double* ptr_a = a_contig->data() + a_b * M * K1;
        const double* ptr_b = b_contig->data() + b_b * K1 * N;
        double* ptr_c = out_data.data() + b_idx * M * N;

        // Cache-friendly i-k-j loop order
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K1; ++k) {
                double a_ik = ptr_a[i * K1 + k];
                for (size_t j = 0; j < N; ++j) {
                    ptr_c[i * N + j] += a_ik * ptr_b[k * N + j];
                }
            }
        }
    }

    bool req = requires_grad_ || other->requires_grad_;
    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), req);
    if (req) {
        result->set_creator(std::make_shared<MatmulNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this()), other
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_count = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if (new_count != numel()) {
        throw std::invalid_argument(std::format("Cannot reshape count {} to {}", numel(), new_count));
    }

    auto contig = contiguous();
    auto result = std::make_shared<Tensor>(new_shape, contig->to_vector(), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<ReshapeNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), shape_
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::transpose(const std::vector<size_t>& axes) const {
    if (axes.size() != ndim()) {
        throw std::invalid_argument("Axes count must match tensor ndim");
    }

    std::vector<size_t> new_shape(ndim());
    std::vector<size_t> new_strides(ndim());

    for (size_t i = 0; i < ndim(); ++i) {
        new_shape[i] = shape_[axes[i]];
        new_strides[i] = strides_[axes[i]];
    }

    auto result = std::make_shared<Tensor>(new_shape, new_strides, offset_, storage_, requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<TransposeNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), axes
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::swapaxes(int axis1, int axis2) const {
    int n = static_cast<int>(ndim());
    int a1 = (axis1 >= 0) ? axis1 : n + axis1;
    int a2 = (axis2 >= 0) ? axis2 : n + axis2;
    if (a1 < 0 || a1 >= n || a2 < 0 || a2 >= n) {
        throw std::out_of_range("swapaxes axis out of range");
    }
    std::vector<size_t> axes(ndim());
    std::iota(axes.begin(), axes.end(), 0);
    std::swap(axes[static_cast<size_t>(a1)], axes[static_cast<size_t>(a2)]);
    return transpose(axes);
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (ndim() < 2) {
        throw std::invalid_argument("transpose() requires rank at least 2");
    }
    return swapaxes(-1, -2);
}

std::shared_ptr<Tensor> Tensor::sum(std::optional<int> axis, bool keepdims) const {
    if (!axis.has_value()) {
        // Full sum to scalar
        double total = 0.0;
        if (is_contiguous()) {
            const double* ptr = data();
            size_t n = numel();
            for (size_t i = 0; i < n; ++i) {
                total += ptr[i];
            }
        } else {
            for (size_t i = 0; i < numel(); ++i) {
                total += (*this)[i];
            }
        }
        std::vector<size_t> out_shape = keepdims ? std::vector<size_t>(ndim(), 1) : std::vector<size_t>{};
        auto result = std::make_shared<Tensor>(out_shape, total, requires_grad_);
        if (requires_grad_) {
            result->set_creator(std::make_shared<SumNode>(
                std::const_pointer_cast<Tensor>(shared_from_this()), axis, keepdims, shape_
            ));
        }
        return result;
    }

    int ax = *axis;
    if (ax < 0) {
        ax += static_cast<int>(ndim());
    }
    size_t uax = static_cast<size_t>(ax);

    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == uax) {
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape_[i]);
        }
    }

    size_t total = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<double> out_data(total, 0.0);

    size_t n = numel();
    auto c_strides = compute_c_strides(shape_);
    auto out_c_strides = compute_c_strides(out_shape);

    for (size_t i = 0; i < n; ++i) {
        // Determine coordinate in output
        size_t rem = i;
        size_t out_idx = 0;
        size_t out_d = 0;

        for (size_t d = 0; d < ndim(); ++d) {
            size_t c_s = (d + 1 < ndim()) ? c_strides[d] : 1;
            size_t coord = rem / c_s;
            rem %= c_s;

            if (d == uax) {
                if (keepdims) {
                    out_d++;
                }
            } else {
                size_t o_s = (out_d + 1 < out_shape.size()) ? out_c_strides[out_d] : 1;
                out_idx += coord * o_s;
                out_d++;
            }
        }

        out_data[out_idx] += (*this)[i];
    }

    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<SumNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), axis, keepdims, shape_
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::mean(std::optional<int> axis, bool keepdims) const {
    size_t reduced_count = axis.has_value() ? shape_[static_cast<size_t>(*axis < 0 ? *axis + static_cast<int>(ndim()) : *axis)] : numel();
    auto s = sum(axis, keepdims);
    auto scale = Tensor::create(s->shape(), 1.0 / static_cast<double>(reduced_count));
    auto res = s->mul(scale);
    if (requires_grad_) {
        res->set_creator(std::make_shared<MeanNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), axis, keepdims, shape_
        ));
    }
    return res;
}

std::shared_ptr<Tensor> Tensor::exp() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = std::exp(in_ptr[i]);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::exp((*this)[i]);
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<ExpNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), result
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::log() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = std::log(in_ptr[i]);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::log((*this)[i]);
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<LogNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this())
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::sqrt() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = std::sqrt(in_ptr[i]);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::sqrt((*this)[i]);
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<SqrtNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), result
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::relu() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            double v = in_ptr[i];
            out_ptr[i] = (v > 0.0) ? v : 0.0;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double v = (*this)[i];
            out[i] = (v > 0.0) ? v : 0.0;
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<ReLUNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this())
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::gelu() const {
    size_t n = numel();
    std::vector<double> out(n);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            double x = in_ptr[i];
            double cdf = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
            out_ptr[i] = x * cdf;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double x = (*this)[i];
            double cdf = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
            out[i] = x * cdf;
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<GELUNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this())
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::silu() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            double x = in_ptr[i];
            double sig = 1.0 / (1.0 + std::exp(-x));
            out_ptr[i] = x * sig;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double x = (*this)[i];
            double sig = 1.0 / (1.0 + std::exp(-x));
            out[i] = x * sig;
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<SiLUNode>(std::vector{
            std::const_pointer_cast<Tensor>(shared_from_this())
        }));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::tanh() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = std::tanh(in_ptr[i]);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::tanh((*this)[i]);
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<TanhNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), result
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = 1.0 / (1.0 + std::exp(-in_ptr[i]));
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = 1.0 / (1.0 + std::exp(-(*this)[i]));
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<SigmoidNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), result
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::clamp(double min_val, double max_val) const {
    size_t n = numel();
    std::vector<double> out(n);
    if (is_contiguous()) {
        const double* in_ptr = data();
        double* out_ptr = out.data();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = std::clamp(in_ptr[i], min_val, max_val);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::clamp((*this)[i], min_val, max_val);
        }
    }
    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<ClampNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), min_val, max_val
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::softmax(int axis) const {
    int ax = (axis < 0) ? axis + static_cast<int>(ndim()) : axis;
    size_t uax = static_cast<size_t>(ax);
    size_t dim_size = shape_[uax];

    auto contig = contiguous();
    std::vector<double> out(numel());

    size_t outer_size = 1;
    for (size_t i = 0; i < uax; ++i) outer_size *= shape_[i];
    size_t inner_size = 1;
    for (size_t i = uax + 1; i < ndim(); ++i) inner_size *= shape_[i];

    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            // Find max for numerical stability
            double max_val = -1e30;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                max_val = std::max(max_val, contig->data()[idx]);
            }

            // Exponentiate and sum
            double sum_exp = 0.0;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                out[idx] = std::exp(contig->data()[idx] - max_val);
                sum_exp += out[idx];
            }

            // Normalize
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                out[idx] /= sum_exp;
            }
        }
    }

    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        result->set_creator(std::make_shared<SoftmaxNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), result, ax
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::log_softmax(int axis) const {
    int ax = (axis < 0) ? axis + static_cast<int>(ndim()) : axis;
    size_t uax = static_cast<size_t>(ax);
    size_t dim_size = shape_[uax];

    auto contig = contiguous();
    std::vector<double> out(numel());
    std::vector<double> s_out(numel());

    size_t outer_size = 1;
    for (size_t i = 0; i < uax; ++i) outer_size *= shape_[i];
    size_t inner_size = 1;
    for (size_t i = uax + 1; i < ndim(); ++i) inner_size *= shape_[i];

    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            double max_val = -1e30;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                max_val = std::max(max_val, contig->data()[idx]);
            }

            double sum_exp = 0.0;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                s_out[idx] = std::exp(contig->data()[idx] - max_val);
                sum_exp += s_out[idx];
            }

            double log_sum_exp = std::log(sum_exp);
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                s_out[idx] /= sum_exp;
                out[idx] = contig->data()[idx] - max_val - log_sum_exp;
            }
        }
    }

    auto result = std::make_shared<Tensor>(shape_, std::move(out), requires_grad_);
    if (requires_grad_) {
        auto softmax_res = std::make_shared<Tensor>(shape_, std::move(s_out), false);
        result->set_creator(std::make_shared<LogSoftmaxNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), softmax_res, ax
        ));
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::layer_norm(
    const std::shared_ptr<Tensor>& gamma,
    const std::shared_ptr<Tensor>& beta,
    double eps,
    int axis) const
{
    int ax = (axis < 0) ? axis + static_cast<int>(ndim()) : axis;
    size_t uax = static_cast<size_t>(ax);
    size_t d_size = shape_[uax];

    auto contig = contiguous();
    std::vector<double> out(numel());
    std::vector<double> x_hat(numel());
    std::vector<double> std_inv(numel());

    size_t outer = 1;
    for (size_t i = 0; i < uax; ++i) outer *= shape_[i];
    size_t inner = 1;
    for (size_t i = uax + 1; i < ndim(); ++i) inner *= shape_[i];

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            double mean_val = 0.0;
            for (size_t d = 0; d < d_size; ++d) {
                size_t idx = o * d_size * inner + d * inner + in;
                mean_val += contig->data()[idx];
            }
            mean_val /= static_cast<double>(d_size);

            double var_val = 0.0;
            for (size_t d = 0; d < d_size; ++d) {
                size_t idx = o * d_size * inner + d * inner + in;
                double diff = contig->data()[idx] - mean_val;
                var_val += diff * diff;
            }
            var_val /= static_cast<double>(d_size);
            double inv = 1.0 / std::sqrt(var_val + eps);

            for (size_t d = 0; d < d_size; ++d) {
                size_t idx = o * d_size * inner + d * inner + in;
                double norm = (contig->data()[idx] - mean_val) * inv;
                x_hat[idx] = norm;
                std_inv[idx] = inv;

                double val = norm;
                if (gamma) {
                    val *= (*gamma)[d];
                }
                if (beta) {
                    val += (*beta)[d];
                }
                out[idx] = val;
            }
        }
    }

    bool req = requires_grad_ || (gamma && gamma->requires_grad()) || (beta && beta->requires_grad());
    auto result = std::make_shared<Tensor>(shape_, std::move(out), req);
    if (req) {
        auto x_hat_t = std::make_shared<Tensor>(shape_, std::move(x_hat), false);
        auto std_inv_t = std::make_shared<Tensor>(shape_, std::move(std_inv), false);
        result->set_creator(std::make_shared<LayerNormNode>(
            std::const_pointer_cast<Tensor>(shared_from_this()), gamma, beta, x_hat_t, std_inv_t, ax
        ));
    }
    return result;
}

// Backpropagation Tape
void Tensor::backward(std::shared_ptr<Tensor> grad_output) {
    if (!requires_grad_) {
        throw std::runtime_error("backward() called on Tensor with requires_grad = false");
    }

    if (!grad_output) {
        if (numel() != 1) {
            throw std::runtime_error("grad can only be implicitly created for scalar outputs");
        }
        grad_output = Tensor::ones(shape_);
    }

    // Topological sort via DFS
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;

    auto dfs = [&](auto& self, Tensor* node) -> void {
        if (visited.contains(node)) return;
        visited.insert(node);
        if (node->creator()) {
            for (const auto& parent : node->creator()->inputs()) {
                if (parent) {
                    self(self, parent.get());
                }
            }
        }
        topo.push_back(node);
    };

    dfs(dfs, this);

    // Gradient accumulation map
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> grads;
    grads[this] = grad_output;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* node = *it;
        auto it_grad = grads.find(node);
        if (it_grad == grads.end()) continue;

        auto g = it_grad->second;

        // Assign to node's grad
        if (!node->grad()) {
            node->set_grad(g);
        } else {
            node->set_grad(node->grad()->add(g));
        }

        if (node->creator()) {
            auto parent_grads = node->creator()->backward(g);
            const auto& parents = node->creator()->inputs();

            for (size_t p = 0; p < parents.size(); ++p) {
                if (parents[p] && parents[p]->requires_grad() && p < parent_grads.size() && parent_grads[p]) {
                    Tensor* p_ptr = parents[p].get();
                    if (!grads.contains(p_ptr)) {
                        grads[p_ptr] = parent_grads[p];
                    } else {
                        grads[p_ptr] = grads[p_ptr]->add(parent_grads[p]);
                    }
                }
            }
        }
    }
}

// Factory functions
std::shared_ptr<Tensor> Tensor::create(std::vector<size_t> shape, double init_val, bool requires_grad) {
    return std::make_shared<Tensor>(std::move(shape), init_val, requires_grad);
}

std::shared_ptr<Tensor> Tensor::create(std::vector<size_t> shape, std::vector<double> data, bool requires_grad) {
    return std::make_shared<Tensor>(std::move(shape), std::move(data), requires_grad);
}

std::shared_ptr<Tensor> Tensor::zeros(std::vector<size_t> shape, bool requires_grad) {
    return std::make_shared<Tensor>(std::move(shape), 0.0, requires_grad);
}

std::shared_ptr<Tensor> Tensor::ones(std::vector<size_t> shape, bool requires_grad) {
    return std::make_shared<Tensor>(std::move(shape), 1.0, requires_grad);
}

std::shared_ptr<Tensor> Tensor::randn(std::vector<size_t> shape, uint64_t seed, bool requires_grad) {
    size_t count = shape.empty() ? 1 : std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return std::make_shared<Tensor>(std::move(shape), std::move(data), requires_grad);
}

std::shared_ptr<Tensor> Tensor::concat(const std::vector<std::shared_ptr<Tensor>>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::invalid_argument("concat requires at least one tensor");
    }
    if (tensors.size() == 1) {
        return tensors[0];
    }
    int ndim_int = static_cast<int>(tensors[0]->ndim());
    int norm_axis = (axis < 0) ? (axis + ndim_int) : axis;
    if (norm_axis < 0 || norm_axis >= ndim_int) {
        throw std::invalid_argument(std::format("concat axis {} out of bounds for ndim {}", axis, ndim_int));
    }
    size_t uaxis = static_cast<size_t>(norm_axis);

    const auto& base_shape = tensors[0]->shape();
    size_t total_axis_size = 0;
    bool any_req_grad = false;
    std::vector<size_t> split_sizes;
    split_sizes.reserve(tensors.size());

    for (const auto& t : tensors) {
        if (static_cast<int>(t->ndim()) != ndim_int) {
            throw std::invalid_argument("All tensors must have the same number of dimensions for concat");
        }
        for (size_t d = 0; d < static_cast<size_t>(ndim_int); ++d) {
            if (d != uaxis && t->shape()[d] != base_shape[d]) {
                throw std::invalid_argument("Tensor dimensions must match except along concat axis");
            }
        }
        split_sizes.push_back(t->shape()[uaxis]);
        total_axis_size += t->shape()[uaxis];
        if (t->requires_grad()) {
            any_req_grad = true;
        }
    }

    std::vector<size_t> out_shape = base_shape;
    out_shape[uaxis] = total_axis_size;

    size_t n_outer = 1;
    for (size_t d = 0; d < uaxis; ++d) {
        n_outer *= base_shape[d];
    }
    size_t n_inner = 1;
    for (size_t d = uaxis + 1; d < static_cast<size_t>(ndim_int); ++d) {
        n_inner *= base_shape[d];
    }

    size_t total_numel = n_outer * total_axis_size * n_inner;
    std::vector<double> out_data(total_numel);

    size_t offset_k = 0;
    for (size_t k = 0; k < tensors.size(); ++k) {
        const auto& t = tensors[k];
        size_t s_k = split_sizes[k];
        for (size_t o = 0; o < n_outer; ++o) {
            for (size_t m = 0; m < s_k; ++m) {
                for (size_t i = 0; i < n_inner; ++i) {
                    size_t src_idx = (o * s_k + m) * n_inner + i;
                    size_t dst_idx = (o * total_axis_size + (offset_k + m)) * n_inner + i;
                    out_data[dst_idx] = (*t)[src_idx];
                }
            }
        }
        offset_k += s_k;
    }

    auto result = std::make_shared<Tensor>(out_shape, std::move(out_data), any_req_grad);
    if (any_req_grad) {
        result->set_creator(std::make_shared<ConcatNode>(tensors, norm_axis, std::move(split_sizes)));
    }
    return result;
}

// Operators
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->add(b);
}

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, double scalar) {
    return a->add(Tensor::create({1}, scalar, false));
}

std::shared_ptr<Tensor> operator+(double scalar, const std::shared_ptr<Tensor>& a) {
    return Tensor::create({1}, scalar, false)->add(a);
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->sub(b);
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, double scalar) {
    return a->sub(Tensor::create({1}, scalar, false));
}

std::shared_ptr<Tensor> operator-(double scalar, const std::shared_ptr<Tensor>& a) {
    return Tensor::create({1}, scalar, false)->sub(a);
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->mul(b);
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, double scalar) {
    return a->mul(Tensor::create({1}, scalar, false));
}

std::shared_ptr<Tensor> operator*(double scalar, const std::shared_ptr<Tensor>& a) {
    return Tensor::create({1}, scalar, false)->mul(a);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->div(b);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, double scalar) {
    return a->div(Tensor::create({1}, scalar, false));
}

std::shared_ptr<Tensor> operator/(double scalar, const std::shared_ptr<Tensor>& a) {
    return Tensor::create({1}, scalar, false)->div(a);
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a) {
    return a->neg();
}

} // namespace aurora
