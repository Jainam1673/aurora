#include "aurora/tensor.hpp"

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

int main() {
    using namespace aurora;
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
