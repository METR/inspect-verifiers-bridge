"""
Sandbox bridge: Manage Inspect sandboxes for use in Verifiers environments.

This module provides utilities to create and manage sandbox environments
that can be used during reward computation in RL training.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from inspect_ai.util import ExecResult
from inspect_ai.util._sandbox.context import (
    cleanup_sandbox_environments_sample,
    init_sandbox_environments_sample,
    sandbox_environments_context_var,
)
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv

# Track whether Docker context has been initialized
_docker_context_initialized = False


def _ensure_docker_context() -> None:
    """Initialize Docker-specific context variables if not already done."""
    global _docker_context_initialized
    if _docker_context_initialized:
        return

    try:
        # Import and initialize Docker cleanup context
        from inspect_ai.util._sandbox.docker.cleanup import project_cleanup_startup

        project_cleanup_startup()
        _docker_context_initialized = True
    except ImportError:
        # Docker sandbox not available
        pass


@dataclass
class SandboxConfig:
    """Configuration for sandbox creation."""

    sandbox_type: str = "docker"
    config: str | None = None
    timeout: int = 120


@dataclass
class SandboxInstance:
    """Tracks a sandbox instance with its metadata for cleanup."""

    environments: dict[str, SandboxEnvironment]
    sandbox_type: str
    config: str | None
    task_name: str


async def create_sandbox_for_sample(
    sample_info: dict[str, Any],
    task_name: str,
    sandbox_config: SandboxConfig,
) -> SandboxInstance:
    """
    Create sandbox environment(s) for a sample.

    Args:
        sample_info: The info dict from the converted sample
        task_name: Name of the task
        sandbox_config: Sandbox configuration

    Returns:
        SandboxInstance containing environments and metadata for cleanup
    """
    # Initialize Docker context if using Docker sandbox
    if sandbox_config.sandbox_type == "docker":
        _ensure_docker_context()

    # Get the sandbox environment class
    sandbox_cls = registry_find_sandboxenv(sandbox_config.sandbox_type)

    # Extract files from sample info
    files_raw = sample_info.get("inspect_files") or {}
    files_bytes: dict[str, bytes] = {}
    for name, content in files_raw.items():
        if isinstance(content, bytes):
            files_bytes[name] = content
        elif isinstance(content, str):
            files_bytes[name] = content.encode("utf-8")

    # Extract setup script
    setup = sample_info.get("inspect_setup")
    setup_bytes: bytes | None = None
    if setup:
        setup_bytes = setup.encode("utf-8") if isinstance(setup, str) else setup

    # Get metadata
    metadata_raw = sample_info.get("inspect_metadata") or {}
    metadata: dict[str, Any] = dict(metadata_raw)

    # Initialize sandbox environments
    sandboxes = await init_sandbox_environments_sample(
        sandboxenv_type=sandbox_cls,
        task_name=task_name,
        config=sandbox_config.config,
        files=files_bytes,
        setup=setup_bytes,
        metadata=metadata,
    )

    return SandboxInstance(
        environments=sandboxes,
        sandbox_type=sandbox_config.sandbox_type,
        config=sandbox_config.config,
        task_name=task_name,
    )


async def cleanup_sandbox(instance: SandboxInstance) -> None:
    """Clean up sandbox environment(s)."""
    await cleanup_sandbox_environments_sample(
        type=instance.sandbox_type,
        task_name=instance.task_name,
        config=instance.config,
        environments=instance.environments,
        interrupted=False,
    )


@asynccontextmanager
async def sandbox_context(
    sandboxes: dict[str, SandboxEnvironment],
) -> AsyncIterator[dict[str, SandboxEnvironment]]:
    """
    Context manager that sets up the sandbox context for Inspect scorers.

    This makes sandbox() calls work within the context.

    Args:
        sandboxes: Dictionary of sandbox environments to make available

    Yields:
        The sandboxes dict
    """
    token = sandbox_environments_context_var.set(sandboxes)
    try:
        yield sandboxes
    finally:
        sandbox_environments_context_var.reset(token)


@dataclass
class SandboxManager:
    """
    Manages sandbox lifecycle for a batch of samples.

    This class handles creating, caching, and cleaning up sandboxes
    for reward computation during RL training.
    """

    sandbox_config: SandboxConfig
    task_name: str
    _instances: dict[str | int, SandboxInstance] = field(
        default_factory=dict, init=False
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_sandbox(
        self,
        sample_id: str | int,
        sample_info: dict[str, Any],
    ) -> dict[str, SandboxEnvironment]:
        """
        Get or create sandbox for a sample.

        Args:
            sample_id: Unique identifier for the sample
            sample_info: The info dict from the sample

        Returns:
            Dictionary of sandbox environments
        """
        async with self._lock:
            if sample_id not in self._instances:
                self._instances[sample_id] = await create_sandbox_for_sample(
                    sample_info=sample_info,
                    task_name=self.task_name,
                    sandbox_config=self.sandbox_config,
                )
            return self._instances[sample_id].environments

    async def cleanup_sample(self, sample_id: str | int) -> None:
        """Clean up sandbox for a specific sample."""
        async with self._lock:
            if sample_id in self._instances:
                await cleanup_sandbox(self._instances[sample_id])
                del self._instances[sample_id]

    async def cleanup_all(self) -> None:
        """Clean up all sandboxes."""
        async with self._lock:
            for instance in self._instances.values():
                await cleanup_sandbox(instance)
            self._instances.clear()

    async def __aenter__(self) -> "SandboxManager":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup_all()


async def exec_in_sandbox(
    sandboxes: dict[str, SandboxEnvironment],
    cmd: list[str],
    *,
    sandbox_name: str | None = None,
    timeout: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ExecResult[str]:
    """
    Execute a command in a sandbox.

    Args:
        sandboxes: Dictionary of available sandboxes
        cmd: Command to execute
        sandbox_name: Name of sandbox to use (None for default)
        timeout: Execution timeout in seconds
        cwd: Working directory
        env: Environment variables

    Returns:
        ExecResult with stdout, stderr, and success status
    """
    # Get the appropriate sandbox
    sandbox: SandboxEnvironment
    if sandbox_name and sandbox_name in sandboxes:
        sandbox = sandboxes[sandbox_name]
    elif "default" in sandboxes:
        sandbox = sandboxes["default"]
    elif sandboxes:
        sandbox = next(iter(sandboxes.values()))
    else:
        raise RuntimeError("No sandbox available")

    return await sandbox.exec(
        cmd=cmd,
        timeout=timeout,
        cwd=cwd,
        env=env or {},
    )
