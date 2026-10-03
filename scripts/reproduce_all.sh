#!/usr/bin/env bash
# ==============================================================================
# AURORA One-Click Scientific Reproducibility Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Starting AURORA One-Click Reproducibility Suite..."
uv run python scripts/reproduce_all.py "$@"
