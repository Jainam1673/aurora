/**
 * AURORA C++23 Native Systems Performance & Throughput Benchmark Suite.
 *
 * Measures:
 * 1. Low-level tensor memory operations & SIMD contiguous fast-paths.
 * 2. GEMM (Matrix Multiplication) compute throughput and GFLOP/s.
 * 3. Elementwise activation throughput (ReLU, GELU, SiLU).
 * 4. Deep dynamics ensemble forward prediction & uncertainty evaluation throughput.
 * 5. Adaptive imagination engine rollout throughput across varying horizons.
 * 6. Scientific statistical evaluation (IQM and bootstrap confidence intervals).
 */

#include "aurora/tensor.hpp"
#include "aurora/nn.hpp"
#include "aurora/world_model.hpp"
#include "aurora/statistical_evaluation.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <format>
#include <functional>

namespace aurora::benchmarks {

template <typename T>
inline void do_not_optimize(T&& val) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+m,r"(val) : : "memory");
#else
    (void)val;
#endif
}

struct BenchmarkMetric {
    std::string category;
    std::string name;
    size_t iterations{0};
    double mean_us{0.0};
    double median_us{0.0};
    double p90_us{0.0};
    double p99_us{0.0};
    double stddev_us{0.0};
    double throughput{0.0};
    std::string throughput_unit;
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(size_t warmup = 5, size_t default_trials = 30)
        : warmup_count_(warmup), default_trials_(default_trials) {}

    BenchmarkMetric run(
        const std::string& category,
        const std::string& name,
        size_t items_per_iter,
        const std::string& unit,
        const std::function<void()>& fn,
        size_t trials = 0
    ) {
        size_t n_trials = (trials > 0) ? trials : default_trials_;

        // Warmup phase
        for (size_t w = 0; w < warmup_count_; ++w) {
            fn();
        }

        // Measurement phase
        std::vector<double> latencies_us;
        latencies_us.reserve(n_trials);

        for (size_t t = 0; t < n_trials; ++t) {
            auto start = std::chrono::high_resolution_clock::now();
            fn();
            auto end = std::chrono::high_resolution_clock::now();
            double dur_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies_us.push_back(dur_us);
        }

        // Statistical summary
        std::sort(latencies_us.begin(), latencies_us.end());
        double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double mean = sum / static_cast<double>(n_trials);

        double sq_sum = 0.0;
        for (double v : latencies_us) {
            sq_sum += (v - mean) * (v - mean);
        }
        double stddev = std::sqrt(sq_sum / static_cast<double>(n_trials));

        double p50 = latencies_us[n_trials / 2];
        double p90 = latencies_us[static_cast<size_t>(0.90 * static_cast<double>(n_trials - 1))];
        double p99 = latencies_us[static_cast<size_t>(0.99 * static_cast<double>(n_trials - 1))];

        double items_per_sec = (static_cast<double>(items_per_iter) / (mean * 1e-6));

        BenchmarkMetric m{
            .category = category,
            .name = name,
            .iterations = n_trials,
            .mean_us = mean,
            .median_us = p50,
            .p90_us = p90,
            .p99_us = p99,
            .stddev_us = stddev,
            .throughput = items_per_sec,
            .throughput_unit = unit
        };

        results_.push_back(m);
        return m;
    }

    const std::vector<BenchmarkMetric>& results() const noexcept {
        return results_;
    }

    void print_table() const {
        std::cout << "\n=========================================================================================================\n";
        std::cout << "                                  AURORA C++23 SYSTEMS BENCHMARK REPORT                                  \n";
        std::cout << "=========================================================================================================\n";
        std::cout << std::left 
                  << std::setw(18) << "Category"
                  << std::setw(32) << "Benchmark"
                  << std::setw(12) << "Mean (us)"
                  << std::setw(12) << "p50 (us)"
                  << std::setw(12) << "p99 (us)"
                  << std::setw(16) << "Throughput"
                  << "Unit\n";
        std::cout << "---------------------------------------------------------------------------------------------------------\n";

        for (const auto& r : results_) {
            std::cout << std::left
                      << std::setw(18) << r.category
                      << std::setw(32) << r.name
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << r.mean_us
                      << std::setw(12) << r.median_us
                      << std::setw(12) << r.p99_us
                      << std::setw(16) << r.throughput
                      << r.throughput_unit << "\n";
        }
        std::cout << "=========================================================================================================\n\n";
    }

    void export_json(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open " << filepath << " for writing.\n";
            return;
        }

        ofs << "{\n  \"benchmarks\": [\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& r = results_[i];
            ofs << "    {\n"
                << "      \"category\": \"" << r.category << "\",\n"
                << "      \"name\": \"" << r.name << "\",\n"
                << "      \"iterations\": " << r.iterations << ",\n"
                << "      \"mean_us\": " << r.mean_us << ",\n"
                << "      \"median_us\": " << r.median_us << ",\n"
                << "      \"p90_us\": " << r.p90_us << ",\n"
                << "      \"p99_us\": " << r.p99_us << ",\n"
                << "      \"stddev_us\": " << r.stddev_us << ",\n"
                << "      \"throughput\": " << r.throughput << ",\n"
                << "      \"throughput_unit\": \"" << r.throughput_unit << "\"\n"
                << "    }" << (i + 1 < results_.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n}\n";
    }

private:
    size_t warmup_count_;
    size_t default_trials_;
    std::vector<BenchmarkMetric> results_;
};

} // namespace aurora::benchmarks

int main(int argc, char* argv[]) {
    using namespace aurora;
    using namespace aurora::benchmarks;

    std::string json_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        }
    }

    std::cout << "AURORA C++23 High-Precision Systems Benchmarking Engine\n";
    std::cout << "Initializing hardware timers and memory caches...\n";

    BenchmarkRunner runner(/*warmup=*/5, /*default_trials=*/30);

    // ==========================================
    // 1. TENSOR MICROBENCHMARKS
    // ==========================================
    {
        const size_t N_ELEM = 500'000;
        runner.run("Tensor", "Contiguous Allocation & Fill", N_ELEM, "elements/s", [&]() {
            auto t = Tensor::zeros({500, 1000});
            do_not_optimize(t->data());
        });

        auto t1 = Tensor::ones({500, 1000});
        auto t2 = Tensor::ones({500, 1000});
        runner.run("Tensor", "Contiguous Elementwise Add", N_ELEM, "elements/s", [&]() {
            auto res = t1->add(t2);
            do_not_optimize(res->data());
        });

        runner.run("Tensor", "Contiguous Elementwise Mul", N_ELEM, "elements/s", [&]() {
            auto res = t1->mul(t2);
            do_not_optimize(res->data());
        });

        runner.run("Tensor", "Contiguous ReLU Activation", N_ELEM, "elements/s", [&]() {
            auto res = t1->relu();
            do_not_optimize(res->data());
        });

        runner.run("Tensor", "Contiguous GELU Activation", N_ELEM, "elements/s", [&]() {
            auto res = t1->gelu();
            do_not_optimize(res->data());
        });

        runner.run("Tensor", "Contiguous SiLU Activation", N_ELEM, "elements/s", [&]() {
            auto res = t1->silu();
            do_not_optimize(res->data());
        });

        runner.run("Tensor", "Full Scalar Sum Reduction", N_ELEM, "elements/s", [&]() {
            auto res = t1->sum();
            do_not_optimize(res->data());
        });
    }

    // ==========================================
    // 2. GEMM (MATRIX MULTIPLICATION) GFLOPS
    // ==========================================
    {
        std::vector<size_t> sizes = {64, 128, 256};
        for (size_t sz : sizes) {
            auto a = Tensor::ones({sz, sz});
            auto b = Tensor::ones({sz, sz});
            size_t flops = 2 * sz * sz * sz;

            runner.run("GEMM (Compute)", std::format("Matmul {}x{}x{}", sz, sz, sz), flops, "FLOPs/s", [&]() {
                auto c = a->matmul(b);
                do_not_optimize(c->data());
            }, /*trials=*/15);
        }
    }

    // ==========================================
    // 3. DYNAMICS ENSEMBLE THROUGHPUT
    // ==========================================
    {
        auto dynamics = std::make_shared<world_model::EnsembleDynamics>(
            /*obs_dim=*/4,
            /*action_dim=*/1,
            /*ensemble_size=*/5,
            /*hidden_dims=*/std::vector<size_t>{64, 64},
            /*activation=*/"silu"
        );

        std::vector<size_t> batch_sizes = {1, 16, 64};
        for (size_t b : batch_sizes) {
            auto obs = Tensor::zeros({b, 4});
            auto act = Tensor::zeros({b, 1});

            runner.run("Dynamics Ensemble", std::format("Forward Ensemble (B={})", b), b * 5, "transitions/s", [&]() {
                auto [means, log_vars] = dynamics->forward_ensemble(obs, act);
                do_not_optimize(means[0]->data());
                do_not_optimize(log_vars[0]->data());
            }, /*trials=*/20);
        }
    }

    // ==========================================
    // 4. ADAPTIVE IMAGINATION ENGINE ROLLOUTS
    // ==========================================
    {
        auto dynamics = std::make_shared<world_model::EnsembleDynamics>(
            /*obs_dim=*/3,
            /*action_dim=*/1,
            /*ensemble_size=*/3,
            /*hidden_dims=*/std::vector<size_t>{48, 48}
        );

        auto policy_fn = [](const std::vector<double>& /*s*/) -> std::vector<double> {
            return {0.0};
        };

        std::vector<size_t> horizons = {1, 4, 8, 12};
        const size_t NUM_SEEDS = 32;

        std::vector<std::vector<double>> initial_states(NUM_SEEDS, std::vector<double>(3, 0.0));

        for (size_t h : horizons) {
            world_model::ImaginationEngine engine(
                dynamics,
                policy_fn,
                /*max_horizon=*/h,
                /*uncertainty_threshold=*/10.0,
                /*adaptive_truncation=*/true
            );

            runner.run("Imagination", std::format("Rollout Trajectories (N={}, H={})", NUM_SEEDS, h), NUM_SEEDS * h, "transitions/s", [&]() {
                auto result = engine.generate_rollouts(initial_states);
                do_not_optimize(result.total_transitions);
            }, /*trials=*/15);
        }
    }

    // ==========================================
    // 5. STATISTICAL EVALUATION THROUGHPUT
    // ==========================================
    {
        std::vector<double> scores_100(100);
        for (size_t i = 0; i < 100; ++i) scores_100[i] = static_cast<double>(i % 30) * 1.5;

        runner.run("Statistics", "IQM (N=100)", 100, "samples/s", [&]() {
            double iqm = evaluation::compute_iqm(scores_100);
            do_not_optimize(iqm);
        }, /*trials=*/100);

        runner.run("Statistics", "Bootstrap CI (N=100, R=1000)", 100 * 1000, "resamples/s", [&]() {
            auto ci = evaluation::bootstrap_ci(scores_100, /*num_bootstraps=*/1000, /*confidence_level=*/0.95);
            do_not_optimize(ci.first);
            do_not_optimize(ci.second);
        }, /*trials=*/20);
    }

    // Print consolidated summary
    runner.print_table();

    if (!json_path.empty()) {
        runner.export_json(json_path);
        std::cout << "Benchmark results exported to: " << json_path << "\n";
    }

    return 0;
}
