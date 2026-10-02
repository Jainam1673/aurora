#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aurora {

class Space {
public:
    virtual ~Space() = default;
    virtual bool contains(const std::vector<double>& val) const = 0;
    virtual std::vector<size_t> shape() const = 0;
};

class DiscreteSpace : public Space {
public:
    explicit DiscreteSpace(size_t n);

    [[nodiscard]] size_t n() const noexcept { return n_; }
    size_t sample(uint64_t seed = 0) const;
    bool contains(const std::vector<double>& val) const override;
    std::vector<size_t> shape() const override { return {}; }

private:
    size_t n_;
};

class BoxSpace : public Space {
public:
    BoxSpace(std::vector<double> low, std::vector<double> high, std::vector<size_t> shape);

    std::vector<double> sample(uint64_t seed = 0) const;
    bool contains(const std::vector<double>& val) const override;
    std::vector<size_t> shape() const override { return shape_; }
    [[nodiscard]] const std::vector<double>& low() const noexcept { return low_; }
    [[nodiscard]] const std::vector<double>& high() const noexcept { return high_; }

private:
    std::vector<double> low_;
    std::vector<double> high_;
    std::vector<size_t> shape_;
};

struct StepResult {
    std::vector<double> next_obs;
    double reward;
    bool terminated;
    bool truncated;
};

class Environment {
public:
    virtual ~Environment() = default;

    virtual std::vector<double> reset(uint64_t seed = 0) = 0;
    virtual StepResult step(const std::vector<double>& action) = 0;
    virtual const Space& observation_space() const = 0;
    virtual const Space& action_space() const = 0;
};

class CartPole : public Environment {
public:
    CartPole();

    std::vector<double> reset(uint64_t seed = 0) override;
    StepResult step(const std::vector<double>& action) override;
    const Space& observation_space() const override { return *obs_space_; }
    const Space& action_space() const override { return *act_space_; }

    [[nodiscard]] double masscart() const noexcept { return masscart_; }
    [[nodiscard]] double masspole() const noexcept { return masspole_; }
    [[nodiscard]] double gravity() const noexcept { return gravity_; }

private:
    double gravity_{9.8};
    double masscart_{1.0};
    double masspole_{0.1};
    double total_mass_{1.1};
    double length_{0.5}; // half-length of pole
    double polemass_length_{0.05};
    double force_mag_{10.0};
    double tau_{0.02}; // seconds between state updates
    double theta_threshold_radians_{12.0 * 2.0 * 3.14159265358979323846 / 360.0};
    double x_threshold_{2.4};

    double x_{0.0};
    double x_dot_{0.0};
    double theta_{0.0};
    double theta_dot_{0.0};
    int steps_{0};
    int max_steps_{500};

    std::shared_ptr<BoxSpace> obs_space_;
    std::shared_ptr<DiscreteSpace> act_space_;
};

class Pendulum : public Environment {
public:
    Pendulum();

    std::vector<double> reset(uint64_t seed = 0) override;
    StepResult step(const std::vector<double>& action) override;
    const Space& observation_space() const override { return *obs_space_; }
    const Space& action_space() const override { return *act_space_; }

private:
    double max_speed_{8.0};
    double max_torque_{2.0};
    double dt_{0.05};
    double g_{10.0};
    double m_{1.0};
    double l_{1.0};

    double theta_{0.0};
    double theta_dot_{0.0};

    std::shared_ptr<BoxSpace> obs_space_;
    std::shared_ptr<BoxSpace> act_space_;
};

} // namespace aurora
