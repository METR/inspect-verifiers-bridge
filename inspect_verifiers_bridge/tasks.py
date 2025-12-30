"""
Task introspection and loading utilities for Inspect AI tasks.
"""

from dataclasses import dataclass
from typing import Any, Callable

from inspect_ai import Task
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import Scorer


@dataclass
class InspectTaskInfo:
    """Holds introspected information about an Inspect task."""

    task: Task
    name: str
    dataset: Dataset
    scorers: list[Scorer]
    sandbox_type: str | None
    solver_has_tools: bool
    metadata: dict[str, Any]


def load_inspect_task(
    task_fn: Callable[..., Task],
    **task_kwargs: Any,
) -> InspectTaskInfo:
    """
    Load an Inspect task and extract its components.

    Args:
        task_fn: A callable that returns an Inspect Task (e.g., apps from inspect_evals)
        **task_kwargs: Arguments to pass to the task function

    Returns:
        InspectTaskInfo with extracted task components
    """
    task = task_fn(**task_kwargs)

    # Extract sandbox type
    sandbox_type = None
    if task.sandbox is not None:
        if isinstance(task.sandbox, str):
            sandbox_type = task.sandbox
        elif hasattr(task.sandbox, "type"):
            sandbox_type = task.sandbox.type

    # Extract scorers (normalize to list)
    scorers: list[Scorer] = []
    task_scorer = task.scorer
    if task_scorer is not None:
        if isinstance(task_scorer, list):  # type: ignore[arg-type]
            scorers = task_scorer
        else:
            scorers = [task_scorer]

    # Check if solver uses tools (simple heuristic for now)
    solver_has_tools = _solver_has_tools(task.solver)

    return InspectTaskInfo(
        task=task,
        name=task.name or "unknown",
        dataset=task.dataset,
        scorers=scorers,
        sandbox_type=sandbox_type,
        solver_has_tools=solver_has_tools,
        metadata=task.metadata or {},
    )


def _solver_has_tools(solver: Any) -> bool:
    """Check if a solver chain includes tool usage."""
    # This is a simple heuristic - we look for use_tools or tool-related solvers
    if solver is None:
        return False

    solver_str = str(solver)
    tool_indicators = ["use_tools", "react", "tool", "bash", "python"]
    return any(indicator in solver_str.lower() for indicator in tool_indicators)
