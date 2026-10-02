"""Cross-language checkpoint serialization and exchange for AURORA."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aurora.nn.module import Module
from aurora.optim.optimizer import Optimizer
from aurora.version import __version__


def save_checkpoint(
    filepath: str | Path,
    model: Module,
    optimizer: Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model and optimizer state to JSON checkpoint file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_metadata: dict[str, Any] = {
        "aurora_version": __version__,
        "timestamp": int(time.time()),
    }
    if metadata:
        ckpt_metadata.update(metadata)

    payload: dict[str, Any] = {
        "metadata": ckpt_metadata,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_checkpoint(
    filepath: str | Path,
    model: Module | None = None,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Load model and optimizer state from JSON checkpoint file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    with open(path, encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    if model is not None and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
