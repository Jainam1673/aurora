#pragma once

#include "aurora/tensor.hpp"

#include <cmath>
#include <format>
#include <functional>
#include <stdexcept>
#include <vector>

namespace aurora {

inline bool gradcheck(
    const std::function<std::shared_ptr<Tensor>(const std::vector<std::shared_ptr<Tensor>>&)>& func,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    double eps = 1e-6,
    double atol = 1e-5,
    double rtol = 1e-4)
{
    // 1. Analytical backward pass
    for (const auto& inp : inputs) {
        inp->zero_grad();
    }

    auto out = func(inputs);
    if (out->numel() != 1) {
        throw std::invalid_argument("gradcheck function must produce a scalar output");
    }

    out->backward();

    // 2. Numerical gradient via central differences
    for (size_t inp_idx = 0; inp_idx < inputs.size(); ++inp_idx) {
        const auto& inp = inputs[inp_idx];
        if (!inp->requires_grad()) {
            continue;
        }

        auto grad = inp->grad();
        if (!grad) {
            throw std::runtime_error(std::format("Input {} requires grad but has null grad", inp_idx));
        }

        std::vector<double> analytical = grad->to_vector();
        std::vector<double> numerical(inp->numel(), 0.0);

        auto orig_storage = *inp->storage();

        for (size_t elem_idx = 0; elem_idx < inp->numel(); ++elem_idx) {
            // f(x + eps)
            (*inp->storage())[inp->offset() + elem_idx] = orig_storage[inp->offset() + elem_idx] + eps;
            double out_pos = func(inputs)->item();

            // f(x - eps)
            (*inp->storage())[inp->offset() + elem_idx] = orig_storage[inp->offset() + elem_idx] - eps;
            double out_neg = func(inputs)->item();

            numerical[elem_idx] = (out_pos - out_neg) / (2.0 * eps);

            // Restore original value
            (*inp->storage())[inp->offset() + elem_idx] = orig_storage[inp->offset() + elem_idx];
        }

        // Compare gradients
        for (size_t i = 0; i < analytical.size(); ++i) {
            double diff = std::abs(analytical[i] - numerical[i]);
            double denom = std::max(std::abs(analytical[i]), std::abs(numerical[i])) + 1e-8;
            double rel_diff = diff / denom;

            if (diff > atol && rel_diff > rtol) {
                throw std::runtime_error(std::format(
                    "Gradient check failed for input {} at element {}:\n"
                    "  Analytical: {:.8e}\n"
                    "  Numerical:  {:.8e}\n"
                    "  Abs Error:  {:.8e} (atol: {:.8e})\n"
                    "  Rel Error:  {:.8e} (rtol: {:.8e})",
                    inp_idx, i, analytical[i], numerical[i], diff, atol, rel_diff, rtol
                ));
            }
        }
    }

    return true;
}

} // namespace aurora
