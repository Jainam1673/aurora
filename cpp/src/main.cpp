#include "aurora/aurora.hpp"

#include <print>

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    auto info = aurora::get_system_info();
    std::println("==================================================");
    std::println("{}", aurora::description());
    std::println("AURORA Native C++23 Engine v{}", info.version);
    std::println("C++ Standard: {}", info.cpp_standard);
    std::println("Compiler: {}", info.compiler);
    std::println("CUDA Enabled: {}", info.cuda_enabled ? "Yes" : "No");
    std::println("==================================================");

    return 0;
}
