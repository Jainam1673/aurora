#pragma once

#include "aurora/tensor.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace aurora {

std::shared_ptr<Tensor> unbroadcast(const std::shared_ptr<Tensor>& grad, const std::vector<size_t>& target_shape);

class Node {
public:
    explicit Node(std::vector<std::shared_ptr<Tensor>> inputs) : inputs_(std::move(inputs)) {}
    virtual ~Node() = default;

    [[nodiscard]] const std::vector<std::shared_ptr<Tensor>>& inputs() const noexcept { return inputs_; }
    [[nodiscard]] virtual std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) = 0;

protected:
    std::vector<std::shared_ptr<Tensor>> inputs_;
};

class AddNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class SubNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class MulNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class DivNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class MatmulNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class ReshapeNode : public Node {
public:
    ReshapeNode(std::shared_ptr<Tensor> input, std::vector<size_t> orig_shape)
        : Node({std::move(input)}), orig_shape_(std::move(orig_shape)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::vector<size_t> orig_shape_;
};

class TransposeNode : public Node {
public:
    TransposeNode(std::shared_ptr<Tensor> input, std::vector<size_t> axes)
        : Node({std::move(input)}), axes_(std::move(axes)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::vector<size_t> axes_;
};

class SumNode : public Node {
public:
    SumNode(std::shared_ptr<Tensor> input, std::optional<int> axis, bool keepdims, std::vector<size_t> orig_shape)
        : Node({std::move(input)}), axis_(axis), keepdims_(keepdims), orig_shape_(std::move(orig_shape)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::optional<int> axis_;
    bool keepdims_;
    std::vector<size_t> orig_shape_;
};

class MeanNode : public Node {
public:
    MeanNode(std::shared_ptr<Tensor> input, std::optional<int> axis, bool keepdims, std::vector<size_t> orig_shape)
        : Node({std::move(input)}), axis_(axis), keepdims_(keepdims), orig_shape_(std::move(orig_shape)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::optional<int> axis_;
    bool keepdims_;
    std::vector<size_t> orig_shape_;
};

class ExpNode : public Node {
public:
    ExpNode(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> out)
        : Node({std::move(input)}), out_(std::move(out)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::shared_ptr<Tensor> out_;
};

class LogNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class SqrtNode : public Node {
public:
    SqrtNode(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> out)
        : Node({std::move(input)}), out_(std::move(out)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::shared_ptr<Tensor> out_;
};

class ReLUNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class GELUNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class SiLUNode : public Node {
public:
    using Node::Node;
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;
};

class SoftmaxNode : public Node {
public:
    SoftmaxNode(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> out, int axis)
        : Node({std::move(input)}), out_(std::move(out)), axis_(axis) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::shared_ptr<Tensor> out_;
    int axis_;
};

class LogSoftmaxNode : public Node {
public:
    LogSoftmaxNode(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> softmax_out, int axis)
        : Node({std::move(input)}), softmax_out_(std::move(softmax_out)), axis_(axis) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::shared_ptr<Tensor> softmax_out_;
    int axis_;
};

class LayerNormNode : public Node {
public:
    LayerNormNode(std::shared_ptr<Tensor> x,
                  std::shared_ptr<Tensor> gamma,
                  std::shared_ptr<Tensor> beta,
                  std::shared_ptr<Tensor> x_hat,
                  std::shared_ptr<Tensor> std_inv,
                  int axis)
        : Node({std::move(x), std::move(gamma), std::move(beta)}),
          x_hat_(std::move(x_hat)),
          std_inv_(std::move(std_inv)),
          axis_(axis) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::shared_ptr<Tensor> x_hat_;
    std::shared_ptr<Tensor> std_inv_;
    int axis_;
};

class EmbeddingNode : public Node {
public:
    EmbeddingNode(std::shared_ptr<Tensor> weight, std::vector<size_t> indices, std::vector<size_t> indices_shape)
        : Node({std::move(weight)}), indices_(std::move(indices)), indices_shape_(std::move(indices_shape)) {}
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override;

private:
    std::vector<size_t> indices_;
    std::vector<size_t> indices_shape_;
};

} // namespace aurora
