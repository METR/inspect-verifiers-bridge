"""
Task introspection and loading utilities for Inspect AI tasks.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from inspect_ai import Task
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import Scorer

# Built-in Inspect solvers that we know how to handle
KNOWN_SOLVERS = {
    # Prompt/message modifiers we extract content from
    "system_message",
    "prompt_template",
    "chain_of_thought",
    "user_message",
    "multiple_choice",
    # Tool configuration (tracked via solver_has_tools)
    "use_tools",
    # Generation (no input modification)
    "generate",
    # Self-critique (complex, warn if present)
    "self_critique",
}


@dataclass
class InspectTaskInfo:
    """Holds introspected information about an Inspect task."""

    task: Task
    name: str
    dataset: Dataset
    scorers: list[Scorer]
    sandbox_type: str | None
    solver_has_tools: bool
    system_prompt: str | None
    prompt_template: str | None  # Template with {prompt} placeholder
    multiple_choice_template: str | None  # Template for multiple choice formatting
    user_messages: list[str] = field(
        default_factory=list
    )  # Additional user messages to append
    unknown_solvers: list[str] = field(default_factory=list)
    # Ordered list of (transform_type, template) tuples as they appear in solver chain
    prompt_transformations: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverInfo:
    """Class to hold extracted solver information."""

    system_prompt: str | None = None
    prompt_template: str | None = None
    multiple_choice_template: str | None = None
    user_messages: list[str] = field(default_factory=list)
    unknown_solvers: list[str] = field(default_factory=list)
    # Ordered list of (transform_type, template) tuples as they appear in solver chain
    # transform_type is one of: "prompt_template", "multiple_choice"
    prompt_transformations: list[tuple[str, str]] = field(default_factory=list)


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

    # Extract solver info from the chain
    solver_info = _extract_solver_info(task)

    # Emit warnings for unknown solvers
    if solver_info.unknown_solvers:
        warnings.warn(
            f"Task '{task.name or 'unknown'}' uses custom/unknown solvers that may "
            f"modify prompts in ways we cannot replicate: {solver_info.unknown_solvers}. "
            "The dataset prompts may not match what the model sees during inference.",
            UserWarning,
            stacklevel=3,
        )

    return InspectTaskInfo(
        task=task,
        name=task.name or "unknown",
        dataset=task.dataset,
        scorers=scorers,
        sandbox_type=sandbox_type,
        solver_has_tools=solver_has_tools,
        system_prompt=solver_info.system_prompt,
        prompt_template=solver_info.prompt_template,
        multiple_choice_template=solver_info.multiple_choice_template,
        user_messages=solver_info.user_messages,
        unknown_solvers=solver_info.unknown_solvers,
        prompt_transformations=solver_info.prompt_transformations,
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


def _get_solver_name(qualname: str) -> str:
    """Extract the solver name from a qualname like 'system_message.<locals>.solve'.

    Args:
        qualname: The __qualname__ attribute of a solver function

    Returns:
        The extracted solver name

    Raises:
        ValueError: If qualname is empty
    """
    if not qualname:
        raise ValueError("Solver qualname cannot be empty")
    # Pattern: "solver_name.<locals>.solve" or just "solver_name"
    if ".<locals>." in qualname:
        return qualname.split(".<locals>.")[0]
    return qualname


def _is_known_solver(solver_name: str) -> bool:
    """Check if a solver name matches a known built-in solver."""
    return solver_name in KNOWN_SOLVERS


def _extract_solver_info(task: Task) -> SolverInfo:
    """Extract information from task solver chain.

    Recognizes built-in Inspect solvers and extracts relevant content:
    - system_message: system prompt content
    - prompt_template: template with {prompt} placeholder
    - chain_of_thought: treated as prompt_template
    - multiple_choice: template for choice formatting
    - user_messages: additional user messages

    Args:
        task: The Inspect task to introspect

    Returns:
        SolverInfo with extracted content and list of unknown solvers
    """
    solver = task.solver
    info = SolverInfo()

    # If it's a Chain, look at the solver functions
    if hasattr(solver, "_solvers"):
        solvers_list = getattr(solver, "_solvers", [])
        for s in solvers_list:
            qualname = getattr(s, "__qualname__", "")
            closure = getattr(s, "__closure__", None)

            # Skip if no qualname (shouldn't happen for valid solvers)
            if not qualname:
                continue

            solver_name = _get_solver_name(qualname)

            # Check if this is a known solver
            if not _is_known_solver(solver_name):
                info.unknown_solvers.append(solver_name)
                continue

            # Skip solvers that don't need content extraction
            if solver_name in {"generate", "use_tools"}:
                continue

            # Warn about complex solvers we recognize but can't fully handle
            if solver_name == "self_critique":
                warnings.warn(
                    "Task uses self_critique() solver which involves multiple model "
                    "calls. The dataset will contain the original prompts without "
                    "self-critique modifications.",
                    UserWarning,
                    stacklevel=5,
                )
                continue

            # Extract content from closure
            if not closure:
                continue

            for cell in closure:
                content = getattr(cell, "cell_contents", None)
                if not isinstance(content, str):
                    continue

                # system_message: look for substantial string content
                if solver_name == "system_message" and len(content) > 10:
                    info.system_prompt = content
                    break

                # prompt_template: look for {prompt} placeholder
                if solver_name == "prompt_template" and "{prompt}" in content:
                    info.prompt_template = content
                    # Record transformation in order
                    info.prompt_transformations.append(("prompt_template", content))
                    break

                # chain_of_thought: also uses {prompt} placeholder
                if solver_name == "chain_of_thought" and "{prompt}" in content:
                    # chain_of_thought is essentially a prompt_template
                    info.prompt_template = content
                    # Record transformation in order (as prompt_template type)
                    info.prompt_transformations.append(("prompt_template", content))
                    break

                # multiple_choice: look for {question} or {letters} placeholders
                if solver_name == "multiple_choice" and (
                    "{question}" in content or "{letters}" in content
                ):
                    info.multiple_choice_template = content
                    # Record transformation in order
                    info.prompt_transformations.append(("multiple_choice", content))
                    break

                # user_message: look for substantial string content (may have placeholders)
                if solver_name == "user_message" and len(content) > 5:
                    info.user_messages.append(content)
                    break

    return info
