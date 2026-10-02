#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace aurora {

class Node;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // Constructors
    Tensor();
    explicit Tensor(std::vector<size_t> shape, double init_val = 0.0, bool requires_grad = false);
    Tensor(std::vector<size_t> shape, std::vector<double> data, bool requires_grad = false);
    Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset,
           std::shared_ptr<std::vector<double>> storage, bool requires_grad = false);

    // Metadata
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] size_t offset() const noexcept { return offset_; }
    [[nodiscard]] size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] size_t numel() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;
    [[nodiscard]] bool requires_grad() const noexcept { return requires_grad_; }
    void set_requires_grad(bool req) noexcept { requires_grad_ = req; }

    // Memory access
    [[nodiscard]] double* data() noexcept { return storage_->data() + offset_; }
    [[nodiscard]] const double* data() const noexcept { return storage_->data() + offset_; }
    [[nodiscard]] std::shared_ptr<std::vector<double>> storage() const noexcept { return storage_; }
    [[nodiscard]] std::vector<double> to_vector() const;
    [[nodiscard]] double item() const;

    // Indexing
    [[nodiscard]] double& operator[](size_t linear_idx);
    [[nodiscard]] const double& operator[](size_t linear_idx) const;
    [[nodiscard]] double& at(const std::vector<size_t>& indices);
    [[nodiscard]] const double& at(const std::vector<size_t>& indices) const;

    // Autograd
    [[nodiscard]] std::shared_ptr<Tensor> grad() const noexcept { return grad_; }
    void set_grad(std::shared_ptr<Tensor> g) { grad_ = std::move(g); }
    [[nodiscard]] std::shared_ptr<Node> creator() const noexcept { return creator_; }
    void set_creator(std::shared_ptr<Node> c) { creator_ = std::move(c); }
    void zero_grad() noexcept;
    void backward(std::shared_ptr<Tensor> grad_output = nullptr);

    // Contiguity
    [[nodiscard]] std::shared_ptr<Tensor> contiguous() const;

    // Elementwise Arithmetic (with broadcasting)
    [[nodiscard]] std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other) const;
    [[nodiscard]] std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& other) const;
    [[nodiscard]] std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& other) const;
    [[nodiscard]] std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& other) const;
    [[nodiscard]] std::shared_ptr<Tensor> neg() const;

    // Matrix Multiplication (2D and batched)
    [[nodiscard]] std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other) const;

    // Shape Operations
    [[nodiscard]] std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape) const;
    [[nodiscard]] std::shared_ptr<Tensor> transpose(const std::vector<size_t>& axes) const;
    [[nodiscard]] std::shared_ptr<Tensor> transpose() const; // Default 2D matrix transpose (swaps last two axes)
    [[nodiscard]] std::shared_ptr<Tensor> swapaxes(int axis1, int axis2) const;

    // Reductions
    [[nodiscard]] std::shared_ptr<Tensor> sum(std::optional<int> axis = std::nullopt, bool keepdims = false) const;
    [[nodiscard]] std::shared_ptr<Tensor> mean(std::optional<int> axis = std::nullopt, bool keepdims = false) const;

    // Nonlinearities
    [[nodiscard]] std::shared_ptr<Tensor> exp() const;
    [[nodiscard]] std::shared_ptr<Tensor> log() const;
    [[nodiscard]] std::shared_ptr<Tensor> sqrt() const;
    [[nodiscard]] std::shared_ptr<Tensor> relu() const;
    [[nodiscard]] std::shared_ptr<Tensor> gelu() const;
    [[nodiscard]] std::shared_ptr<Tensor> silu() const;
    [[nodiscard]] std::shared_ptr<Tensor> tanh() const;
    [[nodiscard]] std::shared_ptr<Tensor> sigmoid() const;
    [[nodiscard]] std::shared_ptr<Tensor> clamp(double min_val, double max_val) const;
    [[nodiscard]] std::shared_ptr<Tensor> softmax(int axis = -1) const;
    [[nodiscard]] std::shared_ptr<Tensor> log_softmax(int axis = -1) const;
    [[nodiscard]] std::shared_ptr<Tensor> layer_norm(
        const std::shared_ptr<Tensor>& gamma = nullptr,
        const std::shared_ptr<Tensor>& beta = nullptr,
        double eps = 1e-5,
        int axis = -1) const;

    // Static Factories and Combiners
    static std::shared_ptr<Tensor> create(std::vector<size_t> shape, double init_val = 0.0, bool requires_grad = false);
    static std::shared_ptr<Tensor> create(std::vector<size_t> shape, std::vector<double> data, bool requires_grad = false);
    static std::shared_ptr<Tensor> zeros(std::vector<size_t> shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> ones(std::vector<size_t> shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> randn(std::vector<size_t> shape, uint64_t seed = 42, bool requires_grad = false);
    static std::shared_ptr<Tensor> concat(const std::vector<std::shared_ptr<Tensor>>& tensors, int axis = 0);

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_{0};
    std::shared_ptr<std::vector<double>> storage_;
    bool requires_grad_{false};
    std::shared_ptr<Tensor> grad_{nullptr};
    std::shared_ptr<Node> creator_{nullptr};

    static std::vector<size_t> compute_c_strides(const std::vector<size_t>& shape);
};

// Operator Overloads
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, double scalar);
std::shared_ptr<Tensor> operator+(double scalar, const std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, double scalar);
std::shared_ptr<Tensor> operator-(double scalar, const std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, double scalar);
std::shared_ptr<Tensor> operator*(double scalar, const std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, double scalar);
std::shared_ptr<Tensor> operator/(double scalar, const std::shared_ptr<Tensor>& a);

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a);

} // namespace aurora
