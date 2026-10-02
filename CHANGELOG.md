# Changelog

All notable changes to the AURORA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-03

### Added
- **M0 Repository Bootstrap**:
  - Python 3.14 package configuration with `uv` (`pyproject.toml`, `uv.lock`).
  - C++23 native build system using CMake and Ninja (`CMakeLists.txt`, `CMakePresets.json`).
  - Architectural blueprints: `ARCHITECTURE.md`, `ROADMAP.md`, `STATUS.md`, `DECISIONS.md`, `TODO.md`.
  - Python smoke test infrastructure (`tests/python/test_smoke.py`).
  - C++ smoke test infrastructure with GoogleTest (`tests/cpp/test_smoke.cpp`).
  - Continuous Integration configuration (`.github/workflows/ci.yml`).
  - Core directory structure separating research, reproductions, native systems, and paper artifacts.
