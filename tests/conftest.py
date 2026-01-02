"""
Pytest configuration and fixtures.
"""

import shutil
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest import Config, Item


def is_docker_available() -> bool:
    """Check if Docker is available on the system."""
    # Check if docker command exists
    if shutil.which("docker") is None:
        return False

    # Try to run docker info to check if daemon is running
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False


# Register custom markers
def pytest_configure(config: "Config") -> None:
    config.addinivalue_line(
        "markers",
        "requires_docker: mark test as requiring Docker (skipped if Docker unavailable)",
    )


# Skip tests marked with requires_docker if Docker is not available
def pytest_collection_modifyitems(config: "Config", items: list["Item"]) -> None:
    if is_docker_available():
        return

    skip_docker = pytest.mark.skip(reason="Docker is not available")
    for item in items:
        if "requires_docker" in item.keywords:
            item.add_marker(skip_docker)
