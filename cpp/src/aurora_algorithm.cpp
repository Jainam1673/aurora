#include "aurora/aurora_algorithm.hpp"

#include <algorithm>
#include <stdexcept>

namespace aurora::algorithm {

std::shared_ptr<Tensor> compute_pessimistic_value(
    const std::shared_ptr<Tensor>& q1,
    const std::shared_ptr<Tensor>& q2,
    const std::shared_ptr<Tensor>& epistemic_uncertainty,
    double beta_pess) {
    if (!q1 || !q2 || !epistemic_uncertainty) {
        throw std::invalid_argument("Null tensor passed to compute_pessimistic_value");
    }

    if (q1->numel() != q2->numel()) {
        throw std::invalid_argument("Q1 and Q2 must have the same number of elements");
    }

    const size_t n = q1->numel();
    const auto* d1 = q1->data();
    const auto* d2 = q2->data();
    const auto* du = epistemic_uncertainty->data();
    const size_t nu = epistemic_uncertainty->numel();

    std::vector<double> out_data(n);
    for (size_t i = 0; i < n; ++i) {
        const double q_min = std::min(d1[i], d2[i]);
        const double u = (nu == 1) ? du[0] : (i < nu ? du[i] : 0.0);
        out_data[i] = q_min - beta_pess * u;
    }

    return Tensor::create(q1->shape(), out_data, false);
}

bool should_trigger_active_exploration(
    double mean_epistemic_uncertainty,
    double tau_active) noexcept {
    return mean_epistemic_uncertainty > tau_active;
}

} // namespace aurora::algorithm
