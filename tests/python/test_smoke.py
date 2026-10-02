"""Smoke tests for AURORA Python package."""

import sys

import aurora
from aurora.core import AuroraInfo, get_system_info
from aurora.version import __version__


def test_version() -> None:
    """Verify package version is consistent."""
    assert __version__ == "0.1.0"
    assert aurora.__version__ == "0.1.0"


def test_system_info() -> None:
    """Verify system telemetry is correctly gathered."""
    info = get_system_info()
    assert isinstance(info, AuroraInfo)
    assert info.version == "0.1.0"
    assert info.python_version == sys.version.split()[0]
    assert info.platform != ""
    assert info.machine != ""


def test_python_version_target() -> None:
    """Verify we are running on Python 3.14+ as specified in project targets."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 14
