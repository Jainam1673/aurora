#pragma once

#include "aurora/version.hpp"

#include <string>
#include <string_view>

namespace aurora {

struct SystemInfo {
    std::string version;
    std::string cpp_standard;
    std::string compiler;
    bool cuda_enabled{false};
};

[[nodiscard]] std::string_view description() noexcept;

[[nodiscard]] SystemInfo get_system_info();

} // namespace aurora
