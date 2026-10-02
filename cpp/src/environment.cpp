#include "aurora/environment.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>

namespace aurora {

// --- DiscreteSpace ---

DiscreteSpace::DiscreteSpace(size_t n) : n_(n) {
    if (n_ == 0) {
        throw std::invalid_argument("Discrete space size must be > 0");
    }
}

size_t DiscreteSpace::sample(uint64_t seed) const {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<size_t> dist(0, n_ - 1);
    return dist(rng);
}

bool DiscreteSpace::contains(const std::vector<double>& val) const {
    if (val.size() != 1) return false;
    double v = val[0];
    return v >= 0.0 && v < static_cast<double>(n_) && std::floor(v) == v;
}

// --- BoxSpace ---

BoxSpace::BoxSpace(std::vector<double> low, std::vector<double> high, std::vector<size_t> shape)
    : low_(std::move(low)), high_(std::move(high)), shape_(std::move(shape)) {
    if (low_.size() != high_.size()) {
        throw std::invalid_argument("BoxSpace low and high must have same dimension");
    }
}

std::vector<double> BoxSpace::sample(uint64_t seed) const {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::vector<double> out(low_.size());
    for (size_t i = 0; i < low_.size(); ++i) {
        double lo = std::isinf(low_[i]) ? -1e3 : low_[i];
        double hi = std::isinf(high_[i]) ? 1e3 : high_[i];
        std::uniform_real_distribution<double> dist(lo, hi);
        out[i] = dist(rng);
    }
    return out;
}

bool BoxSpace::contains(const std::vector<double>& val) const {
    if (val.size() != low_.size()) return false;
    for (size_t i = 0; i < val.size(); ++i) {
        if (val[i] < low_[i] || val[i] > high_[i]) return false;
    }
    return true;
}

// --- CartPole ---

CartPole::CartPole() {
    double inf = 1e30;
    obs_space_ = std::make_shared<BoxSpace>(
        std::vector<double>{-x_threshold_ * 2.0, -inf, -theta_threshold_radians_ * 2.0, -inf},
        std::vector<double>{x_threshold_ * 2.0, inf, theta_threshold_radians_ * 2.0, inf},
        std::vector<size_t>{4}
    );
    act_space_ = std::make_shared<DiscreteSpace>(2);
}

std::vector<double> CartPole::reset(uint64_t seed) {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);

    x_ = dist(rng);
    x_dot_ = dist(rng);
    theta_ = dist(rng);
    theta_dot_ = dist(rng);
    steps_ = 0;

    return {x_, x_dot_, theta_, theta_dot_};
}

StepResult CartPole::step(const std::vector<double>& action) {
    double act_val = action.empty() ? 0.0 : action[0];
    double force = (act_val > 0.5) ? force_mag_ : -force_mag_;

    double costheta = std::cos(theta_);
    double sintheta = std::sin(theta_);

    double temp = (force + polemass_length_ * theta_dot_ * theta_dot_ * sintheta) / total_mass_;
    double thetaacc = (gravity_ * sintheta - costheta * temp) /
        (length_ * (4.0 / 3.0 - masspole_ * costheta * costheta / total_mass_));
    double xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

    // Semi-implicit Euler integration
    x_ += tau_ * x_dot_;
    x_dot_ += tau_ * xacc;
    theta_ += tau_ * theta_dot_;
    theta_dot_ += tau_ * thetaacc;

    steps_++;

    bool terminated = (x_ < -x_threshold_ || x_ > x_threshold_ ||
                       theta_ < -theta_threshold_radians_ || theta_ > theta_threshold_radians_);
    bool truncated = (steps_ >= max_steps_);

    return StepResult{
        .next_obs = {x_, x_dot_, theta_, theta_dot_},
        .reward = 1.0,
        .terminated = terminated,
        .truncated = truncated,
    };
}

// --- Pendulum ---

Pendulum::Pendulum() {
    obs_space_ = std::make_shared<BoxSpace>(
        std::vector<double>{-1.0, -1.0, -max_speed_},
        std::vector<double>{1.0, 1.0, max_speed_},
        std::vector<size_t>{3}
    );
    act_space_ = std::make_shared<BoxSpace>(
        std::vector<double>{-max_torque_},
        std::vector<double>{max_torque_},
        std::vector<size_t>{1}
    );
}

std::vector<double> Pendulum::reset(uint64_t seed) {
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist_th(-std::numbers::pi, std::numbers::pi);
    std::uniform_real_distribution<double> dist_thdot(-1.0, 1.0);

    theta_ = dist_th(rng);
    theta_dot_ = dist_thdot(rng);

    return {std::cos(theta_), std::sin(theta_), theta_dot_};
}

StepResult Pendulum::step(const std::vector<double>& action) {
    double act_val = action.empty() ? 0.0 : action[0];
    double u = std::clamp(act_val, -max_torque_, max_torque_);

    // Normalize theta to [-pi, pi]
    double norm_th = std::fmod(theta_ + std::numbers::pi, 2.0 * std::numbers::pi);
    if (norm_th < 0.0) norm_th += 2.0 * std::numbers::pi;
    norm_th -= std::numbers::pi;

    double costs = norm_th * norm_th + 0.1 * theta_dot_ * theta_dot_ + 0.001 * u * u;

    double newthdot = theta_dot_ + (3.0 * g_ / (2.0 * l_) * std::sin(theta_) + 3.0 / (m_ * l_ * l_) * u) * dt_;
    newthdot = std::clamp(newthdot, -max_speed_, max_speed_);
    theta_ += newthdot * dt_;
    theta_dot_ = newthdot;

    return StepResult{
        .next_obs = {std::cos(theta_), std::sin(theta_), theta_dot_},
        .reward = -costs,
        .terminated = false,
        .truncated = false,
    };
}

} // namespace aurora
