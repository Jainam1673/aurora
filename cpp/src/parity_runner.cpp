#include "aurora/tensor.hpp"
#include "aurora/nn.hpp"
#include "aurora/optim.hpp"
#include "aurora/checkpoint.hpp"
#include "aurora/attention.hpp"
#include "aurora/transformer.hpp"
#include "aurora/distribution.hpp"
#include "aurora/environment.hpp"
#include "aurora/buffer.hpp"
#include "aurora/world_model.hpp"
#include "aurora/reproductions.hpp"
#include "aurora/aurora_algorithm.hpp"
#include "aurora/statistical_evaluation.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace aurora {

// Simple lightweight JSON parser and serializer for tensor parity
struct ParityTensorPayload {
    std::vector<size_t> shape;
    std::vector<double> data;
    bool requires_grad{false};
};

struct ParityRequest {
    std::string op;
    std::vector<ParityTensorPayload> inputs;
    std::optional<int> axis;
    bool keepdims{false};
    double eps{1e-5};
};

struct ParityResponse {
    ParityTensorPayload output;
    std::vector<ParityTensorPayload> grads;
};

// Fast parser for numbers from stream
static double parse_double(std::istream& is) {
    double v = 0.0;
    is >> v;
    return v;
}

static size_t parse_size_t(std::istream& is) {
    size_t v = 0;
    is >> v;
    return v;
}

ParityRequest parse_request(std::istream& is) {
    ParityRequest req;
    std::string token;
    while (is >> token) {
        if (token == "OP:") {
            is >> req.op;
        } else if (token == "AXIS:") {
            int ax = 0;
            is >> ax;
            req.axis = ax;
        } else if (token == "KEEPDIMS:") {
            int kd = 0;
            is >> kd;
            req.keepdims = (kd != 0);
        } else if (token == "EPS:") {
            is >> req.eps;
        } else if (token == "TENSOR:") {
            ParityTensorPayload tp;
            size_t rank = 0;
            is >> rank;
            tp.shape.resize(rank);
            for (size_t r = 0; r < rank; ++r) {
                tp.shape[r] = parse_size_t(is);
            }
            int req_grad = 0;
            is >> req_grad;
            tp.requires_grad = (req_grad != 0);

            size_t count = 1;
            for (auto s : tp.shape) count *= s;
            tp.data.resize(count);
            for (size_t i = 0; i < count; ++i) {
                tp.data[i] = parse_double(is);
            }
            req.inputs.push_back(std::move(tp));
        } else if (token == "END") {
            break;
        }
    }
    return req;
}

void write_response(std::ostream& os, const ParityResponse& resp) {
    os << std::setprecision(17);
    os << "STATUS: OK\n";
    os << "OUT_SHAPE: " << resp.output.shape.size();
    for (auto s : resp.output.shape) os << " " << s;
    os << "\nOUT_DATA: " << resp.output.data.size();
    for (auto v : resp.output.data) os << " " << v;
    os << "\nGRADS_COUNT: " << resp.grads.size() << "\n";
    for (const auto& g : resp.grads) {
        os << "GRAD_SHAPE: " << g.shape.size();
        for (auto s : g.shape) os << " " << s;
        os << "\nGRAD_DATA: " << g.data.size();
        for (auto v : g.data) os << " " << v;
        os << "\n";
    }
}

} // namespace aurora

int main(int argc, char** argv) {
    using namespace aurora;

    if (argc >= 5 && std::string(argv[1]) == "--checkpoint-step") {
        std::string in_ckpt = argv[2];
        std::string x_ckpt = argv[3];
        std::string out_ckpt = argv[4];

        auto mlp = std::make_shared<nn::MLP>(4, std::vector<size_t>{8}, 2, "relu", 0.0);
        auto opt = std::make_shared<optim::AdamW>(mlp->parameters(), 0.01, 0.9, 0.999, 1e-8, 0.01);

        checkpoint::load_checkpoint(in_ckpt, mlp.get(), opt.get());

        auto x_data = checkpoint::load_checkpoint(x_ckpt);
        auto x = x_data.model_state_dict.at("x");

        opt->zero_grad();
        auto pred = mlp->forward(x);
        auto loss = pred->sum();
        loss->backward();
        opt->step();

        checkpoint::save_checkpoint(out_ckpt, *mlp, opt.get(), {{"step", std::to_string(opt->step_count())}});
        std::cout << "STATUS: OK\n";
        return 0;
    }

    if (argc >= 5 && std::string(argv[1]) == "--transformer-step") {
        std::string in_ckpt = argv[2];
        std::string x_ckpt = argv[3];
        std::string out_ckpt = argv[4];

        size_t d_model = 16;
        size_t num_heads = 4;
        size_t d_ff = 32;

        auto block = std::make_shared<TransformerBlock>(
            d_model, num_heads, d_ff, "layernorm", "gelu", 0.0, true);
        auto opt = std::make_shared<optim::AdamW>(block->parameters(), 0.01, 0.9, 0.999, 1e-8, 0.01);

        checkpoint::load_checkpoint(in_ckpt, block.get(), opt.get());

        auto x_data = checkpoint::load_checkpoint(x_ckpt);
        auto x = x_data.model_state_dict.at("x");

        opt->zero_grad();
        auto [out, attn] = block->forward_with_attention(x, nullptr, true);
        auto loss = out->sum();
        loss->backward();
        opt->step();

        checkpoint::save_checkpoint(out_ckpt, *block, opt.get(), {{"step", std::to_string(opt->step_count())}});
        std::cout << "STATUS: OK\n";
        return 0;
    }

    auto req = parse_request(std::cin);

    std::vector<std::shared_ptr<Tensor>> inps;
    for (const auto& p : req.inputs) {
        inps.push_back(Tensor::create(p.shape, p.data, p.requires_grad));
    }

    std::shared_ptr<Tensor> out = nullptr;
    if (req.op == "add") {
        out = inps[0]->add(inps[1]);
    } else if (req.op == "sub") {
        out = inps[0]->sub(inps[1]);
    } else if (req.op == "mul") {
        out = inps[0]->mul(inps[1]);
    } else if (req.op == "div") {
        out = inps[0]->div(inps[1]);
    } else if (req.op == "matmul") {
        out = inps[0]->matmul(inps[1]);
    } else if (req.op == "sum") {
        out = inps[0]->sum(req.axis, req.keepdims);
    } else if (req.op == "mean") {
        out = inps[0]->mean(req.axis, req.keepdims);
    } else if (req.op == "exp") {
        out = inps[0]->exp();
    } else if (req.op == "log") {
        out = inps[0]->log();
    } else if (req.op == "sqrt") {
        out = inps[0]->sqrt();
    } else if (req.op == "relu") {
        out = inps[0]->relu();
    } else if (req.op == "gelu") {
        out = inps[0]->gelu();
    } else if (req.op == "silu") {
        out = inps[0]->silu();
    } else if (req.op == "softmax") {
        out = inps[0]->softmax(req.axis.value_or(-1));
    } else if (req.op == "log_softmax") {
        out = inps[0]->log_softmax(req.axis.value_or(-1));
    } else if (req.op == "layer_norm") {
        auto gamma = (inps.size() > 1) ? inps[1] : nullptr;
        auto beta = (inps.size() > 2) ? inps[2] : nullptr;
        out = inps[0]->layer_norm(gamma, beta, req.eps, req.axis.value_or(-1));
    } else if (req.op == "attention") {
        auto q = inps[0];
        auto k = inps[1];
        auto v = inps[2];
        std::shared_ptr<Tensor> mask = (inps.size() > 3) ? inps[3] : nullptr;
        auto attn_res = scaled_dot_product_attention(q, k, v, mask);
        out = attn_res.output;
    } else if (req.op == "tanh") {
        out = inps[0]->tanh();
    } else if (req.op == "clamp") {
        out = inps[0]->clamp(-1.0, 1.0);
    } else if (req.op == "concat") {
        out = Tensor::concat(inps, req.axis.value_or(0));
    } else if (req.op == "categorical_log_prob") {
        Categorical dist(inps[0]);
        out = dist.log_prob(inps[1]);
    } else if (req.op == "categorical_entropy") {
        Categorical dist(inps[0]);
        out = dist.entropy();
    } else if (req.op == "normal_log_prob") {
        Normal dist(inps[0], inps[1]);
        out = dist.log_prob(inps[2]);
    } else if (req.op == "normal_entropy") {
        Normal dist(inps[0], inps[1]);
        out = dist.entropy();
    } else if (req.op == "tanh_normal_log_prob") {
        TanhNormal dist(inps[0], inps[1]);
        out = dist.log_prob(inps[2]);
    } else if (req.op == "gae") {
        auto rewards = inps[0]->to_vector();
        auto values = inps[1]->to_vector();
        auto dones = inps[2]->to_vector();
        double last_v = inps[3]->item();
        bool last_d = inps[4]->item() > 0.5;
        double gamma = (inps.size() > 5) ? inps[5]->item() : 0.99;
        double lambda = (inps.size() > 6) ? inps[6]->item() : 0.95;

        RolloutBuffer buf(rewards.size(), {1}, {1}, 1);
        for (size_t t = 0; t < rewards.size(); ++t) {
            buf.add({0.0}, {0.0}, rewards[t], dones[t] > 0.5, values[t], 0.0);
        }
        buf.compute_returns_and_advantages(last_v, last_d, gamma, lambda);
        out = Tensor::create({rewards.size()}, buf.advantages(), false);
    } else if (req.op == "sigmoid") {
        out = inps[0]->sigmoid();
    } else if (req.op == "gaussian_nll_loss") {
        out = world_model::gaussian_nll_loss(inps[0], inps[1], inps[2]);
    } else if (req.op == "uncertainty_decompose") {
        size_t half = inps.size() / 2;
        std::vector<std::shared_ptr<Tensor>> means(inps.begin(), inps.begin() + static_cast<std::ptrdiff_t>(half));
        std::vector<std::shared_ptr<Tensor>> vars(inps.begin() + static_cast<std::ptrdiff_t>(half), inps.end());
        auto unc = world_model::decompose_uncertainty(means, vars);
        int choice = req.axis.value_or(2);
        if (choice == 0) out = unc.mean;
        else if (choice == 1) out = unc.aleatoric;
        else if (choice == 2) out = unc.epistemic;
        else out = unc.total;
    } else if (req.op == "rssm_kl_divergence") {
        out = world_model::RSSM::kl_divergence(inps[0], inps[1], inps[2], inps[3]);
    } else if (req.op == "lambda_returns") {
        double lambda = (inps.size() > 3) ? inps[3]->item() : 0.95;
        out = reproductions::compute_lambda_returns(inps[0], inps[1], inps[2], lambda);
    } else if (req.op == "cem_planning") {
        reproductions::CEMConfig config;
        config.horizon = (inps.size() > 1) ? static_cast<size_t>(inps[1]->item()) : 3;
        config.num_samples = 30;
        config.num_elites = 6;
        config.iterations = 4;
        config.action_dim = (inps.size() > 2) ? static_cast<size_t>(inps[2]->item()) : 2;
        config.alpha = 0.5;
        config.seed = 42;

        reproductions::CEMPlanner planner(config);
        auto dynamics = [](const std::vector<double>& z, const std::vector<double>& a) {
            std::vector<double> next_z(z.size());
            for (size_t i = 0; i < z.size(); ++i) {
                next_z[i] = z[i] + 0.1 * a[i % a.size()];
            }
            return next_z;
        };
        auto reward = [](const std::vector<double>& z, const std::vector<double>& /*a*/) {
            double s = 0.0;
            for (double val : z) {
                s += val;
            }
            return s;
        };
        auto terminal_value = [](const std::vector<double>& z) {
            double s = 0.0;
            for (double val : z) {
                s += 2.0 * val;
            }
            return s;
        };

        auto z0 = inps[0]->to_vector();
        auto best_act = planner.plan(z0, dynamics, reward, terminal_value);
        out = Tensor::create({best_act.size()}, best_act, false);
    } else if (req.op == "aurora_adaptive_horizon") {
        algorithm::AdaptiveHorizonConfig cfg;
        if (inps.size() > 1) {
            auto cfg_data = inps[1]->to_vector();
            if (cfg_data.size() >= 6) {
                cfg.horizon_max = static_cast<size_t>(cfg_data[0]);
                cfg.horizon_min = static_cast<size_t>(cfg_data[1]);
                cfg.tau_base = cfg_data[2];
                cfg.kappa = cfg_data[3];
                cfg.budget_max = cfg_data[4];
                cfg.gamma = cfg_data[5];
            }
        }
        algorithm::AdaptiveHorizonScheduler scheduler(cfg);
        if (inps.size() > 2) {
            scheduler.update_threshold(inps[2]->item());
        }
        size_t h_star = scheduler.compute_adaptive_horizon(inps[0]);
        out = Tensor::create({1}, static_cast<double>(h_star), false);
    } else if (req.op == "aurora_blending_ratio") {
        algorithm::DynamicBlendingConfig cfg;
        if (inps.size() > 1) {
            auto cfg_data = inps[1]->to_vector();
            if (cfg_data.size() >= 4) {
                cfg.eta_max = cfg_data[0];
                cfg.eta_min = cfg_data[1];
                cfg.u_target = cfg_data[2];
                cfg.momentum = cfg_data[3];
            }
        }
        algorithm::DynamicBlendingController controller(cfg);
        auto u_seq = inps[0]->to_vector();
        std::vector<double> etas;
        etas.reserve(u_seq.size());
        for (double u : u_seq) {
            etas.push_back(controller.compute_ratio(u));
        }
        out = Tensor::create({etas.size()}, etas, false);
    } else if (req.op == "aurora_pessimistic_value") {
        double beta_pess = (inps.size() > 3) ? inps[3]->item() : 0.5;
        out = algorithm::compute_pessimistic_value(inps[0], inps[1], inps[2], beta_pess);
    } else if (req.op == "eval_iqm") {
        double val = evaluation::compute_iqm(inps[0]->to_vector());
        out = Tensor::create({1}, val, false);
    } else if (req.op == "eval_probability_of_improvement") {
        double val = evaluation::probability_of_improvement(inps[0]->to_vector(), inps[1]->to_vector());
        out = Tensor::create({1}, val, false);
    } else if (req.op == "eval_performance_profile") {
        auto prof = evaluation::performance_profile(inps[0]->to_vector(), inps[1]->to_vector());
        out = Tensor::create({prof.size()}, prof, false);
    } else {
        std::cerr << "Unknown op: " << req.op << "\n";
        return 1;
    }

    // Run backward if any input requires grad
    bool has_req_grad = false;
    for (const auto& inp : inps) {
        if (inp->requires_grad()) has_req_grad = true;
    }

    if (has_req_grad) {
        // Reduce out to scalar sum to backward
        auto scalar_loss = out->sum();
        scalar_loss->backward();
    }

    ParityResponse resp;
    resp.output.shape = out->shape();
    resp.output.data = out->to_vector();

    for (const auto& inp : inps) {
        ParityTensorPayload gp;
        if (inp->grad()) {
            gp.shape = inp->grad()->shape();
            gp.data = inp->grad()->to_vector();
        } else {
            gp.shape = inp->shape();
            gp.data.resize(inp->numel(), 0.0);
        }
        resp.grads.push_back(std::move(gp));
    }

    write_response(std::cout, resp);
    return 0;
}
