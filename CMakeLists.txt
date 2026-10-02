cmake_minimum_required(VERSION 3.28)

project(AURORA
    VERSION 0.1.0
    DESCRIPTION "Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents"
    HOMEPAGE_URL "https://github.com/aurora-rl/aurora"
    LANGUAGES C CXX
)

# Enforce strict C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(AURORA_BUILD_TESTS "Build AURORA C++ unit test suite" ON)
option(AURORA_BUILD_BENCHMARKS "Build AURORA C++ benchmarks" OFF)
option(AURORA_ENABLE_CUDA "Enable modular CUDA acceleration" OFF)
option(AURORA_STRICT_WARNINGS "Treat compiler warnings as errors" OFF)

# CUDA Support (Modular and optional)
if(AURORA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_compile_definitions(AURORA_HAS_CUDA=1)
        message(STATUS "AURORA: CUDA support enabled with compiler: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "AURORA: CUDA requested but nvcc compiler not found. Falling back to CPU only.")
    endif()
endif()

# Compiler warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wconversion
    )
    if(AURORA_STRICT_WARNINGS)
        add_compile_options(-Werror)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
    if(AURORA_STRICT_WARNINGS)
        add_compile_options(/WX)
    endif()
endif()

# Core Library
add_library(aurora_core
    cpp/src/core.cpp
    cpp/src/tensor.cpp
    cpp/src/autograd.cpp
)
add_library(aurora::core ALIAS aurora_core)

target_include_directories(aurora_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(aurora_core PUBLIC cxx_std_23)

# CLI Binary
add_executable(aurora_cli
    cpp/src/main.cpp
)
target_link_libraries(aurora_cli PRIVATE aurora::core)

# Parity Runner Binary
add_executable(aurora_parity_runner
    cpp/src/parity_runner.cpp
)
target_link_libraries(aurora_parity_runner PRIVATE aurora::core)

# Testing
if(AURORA_BUILD_TESTS)
    enable_testing()

    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GoogleTest not found in system; fetching via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    include(GoogleTest)

    # Smoke Test
    add_executable(aurora_cpp_smoke_test
        tests/cpp/test_smoke.cpp
    )
    target_link_libraries(aurora_cpp_smoke_test
        PRIVATE
            aurora::core
            GTest::gtest_main
    )
    gtest_discover_tests(aurora_cpp_smoke_test)

    # Tensor Core Test
    add_executable(aurora_cpp_tensor_test
        tests/cpp/test_tensor.cpp
    )
    target_link_libraries(aurora_cpp_tensor_test
        PRIVATE
            aurora::core
            GTest::gtest_main
    )
    gtest_discover_tests(aurora_cpp_tensor_test)

    # Gradcheck Test
    add_executable(aurora_cpp_gradcheck_test
        tests/cpp/test_gradcheck.cpp
    )
    target_link_libraries(aurora_cpp_gradcheck_test
        PRIVATE
            aurora::core
            GTest::gtest_main
    )
    gtest_discover_tests(aurora_cpp_gradcheck_test)
endif()
