"""
Sandbox bridge: Manage Inspect sandboxes for use in Verifiers environments.

This module provides utilities to create and manage sandbox environments
that can be used during reward computation in RL training.
"""

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from inspect_ai._eval.task.sandbox import read_sandboxenv_file, resolve_sample_files
from inspect_ai.util import ExecResult
from inspect_ai.util._sandbox.context import (
    cleanup_sandbox_environments_sample,
    init_sandbox_environments_sample,
    sandbox_default_context_var,
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
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
    # Check for per-sample sandbox configuration (not yet supported)
    per_sample_sandbox = sample_info.get("inspect_sandbox")
    if per_sample_sandbox is not None:
        raise NotImplementedError(
            f"Per-sample sandbox configuration is not yet supported. "
            f"Sample has sandbox={per_sample_sandbox}, but only task-level sandbox config is used."
        )

    # Initialize Docker context if using Docker sandbox
    if sandbox_config.sandbox_type == "docker":
        _ensure_docker_context()

    # Get the sandbox environment class
    sandbox_cls = registry_find_sandboxenv(sandbox_config.sandbox_type)

    # Resolve files using Inspect's resolution (handles data URIs, HTTP URLs, file paths)
    files_raw = sample_info.get("inspect_files") or {}
    resolved_files = resolve_sample_files(files_raw)
    files_bytes: dict[str, bytes] = {}
    for path, contents in resolved_files.items():
        files_bytes[path] = await read_sandboxenv_file(contents)

    # Resolve setup script using Inspect's resolution
    setup = sample_info.get("inspect_setup")
    setup_bytes: bytes | None = None
    if setup:
        setup_bytes = await read_sandboxenv_file(setup)

    # Get metadata (JSON-serialized in dataset.py for pyarrow compatibility)
    metadata_raw = sample_info.get("inspect_metadata") or {}
    metadata: dict[str, Any] = (
        json.loads(metadata_raw)
        if isinstance(metadata_raw, str)
        else dict(metadata_raw)
    )

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
    Sets all three required ContextVars that Inspect expects:
    - sandbox_environments_context_var: The actual sandbox environments
    - sandbox_default_context_var: Name of the default sandbox
    - sandbox_with_environments_context_var: Cache for sandbox_with lookups

    Args:
        sandboxes: Dictionary of sandbox environments to make available

    Yields:
        The sandboxes dict
    """
    # Determine default sandbox name (first key in dict)
    default_name = next(iter(sandboxes.keys())) if sandboxes else "default"

    # Set all three ContextVars that Inspect expects
    token_envs = sandbox_environments_context_var.set(sandboxes)
    token_default = sandbox_default_context_var.set(default_name)
    token_with = sandbox_with_environments_context_var.set({})
    try:
        yield sandboxes
    finally:
        sandbox_environments_context_var.reset(token_envs)
        sandbox_default_context_var.reset(token_default)
        sandbox_with_environments_context_var.reset(token_with)


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
