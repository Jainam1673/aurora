"""Immutable experiment manifest generator for scientific reproducibility."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_info() -> dict[str, Any]:
    """Retrieve current git commit, branch, and status."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        status_out = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        dirty = len(status_out) > 0
    except Exception:
        commit = "unknown"
        branch = "unknown"
        dirty = True

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def create_experiment_manifest(
    experiment_id: str,
    environment_name: str,
    algorithm_name: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Create an immutable, audit-compliant experiment manifest dictionary."""
    git_info = get_git_info()

    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "algorithm": algorithm_name,
        "environment": environment_name,
        "seed": seed,
        "git": git_info,
        "system": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hyperparameters": hyperparameters,
        "metrics": metrics or {},
    }
    return manifest


def save_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Save experiment manifest to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
