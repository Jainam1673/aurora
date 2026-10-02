#pragma once

#include "aurora/nn.hpp"
#include "aurora/optim.hpp"

#include <map>
#include <memory>
#include <string>

namespace aurora {
namespace checkpoint {

struct CheckpointData {
    std::map<std::string, std::string> metadata;
    nn::Module::StateDict model_state_dict;
    optim::OptimizerStateDict optimizer_state_dict;
    bool has_optimizer{false};
};

void save_checkpoint(const std::string& filepath,
                     const nn::Module& model,
                     const optim::Optimizer* optimizer = nullptr,
                     const std::map<std::string, std::string>& metadata = {});

CheckpointData load_checkpoint(const std::string& filepath,
                               nn::Module* model = nullptr,
                               optim::Optimizer* optimizer = nullptr);

} // namespace checkpoint
} // namespace aurora
