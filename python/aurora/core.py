"""Core definitions and environment telemetry for AURORA."""

import platform
import sys
from dataclasses import dataclass
from typing import Final

from aurora.version import __version__

PROJECT_NAME: Final[str] = "AURORA"
PROJECT_DESCRIPTION: Final[str] = (
    "Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents"
)


@dataclass(frozen=True, slots=True)
class AuroraInfo:
    """Metadata describing the active AURORA runtime and environment."""

    version: str
    python_version: str
    platform: str
    machine: str
    torch_available: bool = False
    cuda_available: bool = False


def get_system_info() -> AuroraInfo:
    """Collect runtime environment metadata for experimental tracking and reproducibility."""
    torch_avail = False
    cuda_avail = False
    try:
        import torch

        torch_avail = True
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        pass

    return AuroraInfo(
        version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.system(),
        machine=platform.machine(),
        torch_available=torch_avail,
        cuda_available=cuda_avail,
    )
