#include "aurora/tensor.hpp"

#include <gtest/gtest.h>

TEST(TensorCoreTest, CreationAndMetadata) {
    auto t = aurora::Tensor::create({2, 3}, 5.0);
    EXPECT_EQ(t->ndim(), 2);
    EXPECT_EQ(t->numel(), 6);
    EXPECT_TRUE(t->is_contiguous());
    EXPECT_EQ(t->shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(t->strides(), (std::vector<size_t>{3, 1}));
    EXPECT_DOUBLE_EQ(t->at({0, 1}), 5.0);
}

TEST(TensorCoreTest, ReshapeAndTranspose) {
    auto t = aurora::Tensor::create({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    auto r = t->reshape({3, 2});
    EXPECT_EQ(r->shape(), (std::vector<size_t>{3, 2}));
    EXPECT_DOUBLE_EQ(r->at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(r->at({2, 1}), 6.0);

    auto trans = t->transpose();
    EXPECT_EQ(trans->shape(), (std::vector<size_t>{3, 2}));
    EXPECT_DOUBLE_EQ(trans->at({1, 0}), 2.0);
    EXPECT_DOUBLE_EQ(trans->at({0, 1}), 4.0);
}

TEST(TensorCoreTest, ElementwiseArithmetic) {
    auto a = aurora::Tensor::create({2, 2}, std::vector<double>{1, 2, 3, 4});
    auto b = aurora::Tensor::create({2, 2}, std::vector<double>{5, 6, 7, 8});

    auto c_add = a + b;
    EXPECT_DOUBLE_EQ(c_add->at({0, 0}), 6.0);
    EXPECT_DOUBLE_EQ(c_add->at({1, 1}), 12.0);

    auto c_sub = b - a;
    EXPECT_DOUBLE_EQ(c_sub->at({0, 0}), 4.0);
    EXPECT_DOUBLE_EQ(c_sub->at({1, 1}), 4.0);

    auto c_mul = a * b;
    EXPECT_DOUBLE_EQ(c_mul->at({0, 0}), 5.0);
    EXPECT_DOUBLE_EQ(c_mul->at({1, 1}), 32.0);

    auto c_div = b / a;
    EXPECT_DOUBLE_EQ(c_div->at({0, 0}), 5.0);
    EXPECT_DOUBLE_EQ(c_div->at({1, 1}), 2.0);
}

TEST(TensorCoreTest, BroadcastingArithmetic) {
    auto a = aurora::Tensor::create({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    auto b = aurora::Tensor::create({3}, std::vector<double>{10, 20, 30});

    auto c = a + b;
    EXPECT_EQ(c->shape(), (std::vector<size_t>{2, 3}));
    EXPECT_DOUBLE_EQ(c->at({0, 0}), 11.0);
    EXPECT_DOUBLE_EQ(c->at({0, 2}), 33.0);
    EXPECT_DOUBLE_EQ(c->at({1, 0}), 14.0);
    EXPECT_DOUBLE_EQ(c->at({1, 2}), 36.0);
}

TEST(TensorCoreTest, MatrixMultiplication) {
    auto a = aurora::Tensor::create({2, 2}, std::vector<double>{1, 2, 3, 4});
    auto b = aurora::Tensor::create({2, 2}, std::vector<double>{5, 6, 7, 8});

    auto c = a->matmul(b);
    EXPECT_DOUBLE_EQ(c->at({0, 0}), 19.0);
    EXPECT_DOUBLE_EQ(c->at({0, 1}), 22.0);
    EXPECT_DOUBLE_EQ(c->at({1, 0}), 43.0);
    EXPECT_DOUBLE_EQ(c->at({1, 1}), 50.0);
}

TEST(TensorCoreTest, Reductions) {
    auto a = aurora::Tensor::create({2, 2}, std::vector<double>{1, 2, 3, 4});
    EXPECT_DOUBLE_EQ(a->sum()->item(), 10.0);
    EXPECT_DOUBLE_EQ(a->mean()->item(), 2.5);

    auto sum_ax0 = a->sum(0);
    EXPECT_EQ(sum_ax0->shape(), (std::vector<size_t>{2}));
    EXPECT_DOUBLE_EQ(sum_ax0->at({0}), 4.0);
    EXPECT_DOUBLE_EQ(sum_ax0->at({1}), 6.0);
}
