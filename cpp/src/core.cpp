#include "aurora/aurora.hpp"

#include <format>
#include <string>
#include <string_view>

namespace aurora {

std::string_view description() noexcept {
    return "AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents";
}

SystemInfo get_system_info() {
    SystemInfo info;
    info.version = std::string(kVersion);

#if defined(__cplusplus)
    info.cpp_standard = std::format("{}", __cplusplus);
#else
    info.cpp_standard = "unknown";
#endif

#if defined(__clang__)
    info.compiler = std::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    info.compiler = std::format("GCC {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    info.compiler = std::format("MSVC {}", _MSC_VER);
#else
    info.compiler = "Unknown";
#endif

#if defined(AURORA_HAS_CUDA)
    info.cuda_enabled = true;
#else
    info.cuda_enabled = false;
#endif

    return info;
}

} // namespace aurora
