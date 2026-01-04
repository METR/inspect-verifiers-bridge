"""
Scoring bridge: Convert Inspect scorers to Verifiers reward functions.

This module provides the core mechanism to call Inspect scorers within the
Verifiers reward function framework.
"""

import json
import warnings
from functools import partial
from typing import Any, Callable

import verifiers as vf
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score, Scorer, Target, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from inspect_verifiers_bridge.sandbox import SandboxManager, sandbox_context
from inspect_verifiers_bridge.utils import BRIDGE_MODEL_NAME


async def reward_from_inspect_scorer(
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]],
    answer: str | None,
    state: dict[str, Any],
    *,
    scorer: Scorer,
    sandbox_manager: SandboxManager | None = None,
) -> float:
    """
    Verifiers reward function that wraps an Inspect scorer.

    This function reconstructs a minimal TaskState from Verifiers state
    and calls the Inspect scorer to get a reward.

    Args:
        prompt: The prompt messages (from Verifiers)
        completion: The completion messages (from Verifiers)
        answer: The expected answer (from Verifiers dataset)
        state: The Verifiers state dict containing info
        scorer: The Inspect scorer to use
        sandbox_manager: Optional sandbox manager for sandbox-based scorers

    Returns:
        Float reward value (typically 0.0-1.0)
    """
    info = state.get("info", {})

    # Assert expected keys are present in info and have valid values
    assert "inspect_target_raw" in info, "info must contain 'inspect_target_raw'"
    assert "inspect_sample_id" in info, "info must contain 'inspect_sample_id'"
    assert "inspect_metadata" in info, "info must contain 'inspect_metadata'"
    assert "inspect_input_raw" in info, "info must contain 'inspect_input_raw'"

    # Validate sample_id is not None (can happen if Sample.id was None)
    sample_id = info["inspect_sample_id"]
    assert sample_id is not None, "sample_id cannot be None - ensure Sample.id is set"

    # Get the raw target from info, or fall back to answer
    target_raw = info.get("inspect_target_raw", answer)
    if target_raw is None:
        warnings.warn(
            "Target is None - scoring may not work correctly. "
            "Ensure the sample has a valid target.",
            UserWarning,
            stacklevel=2,
        )
    target = Target(target_raw) if target_raw is not None else Target("")

    # Build messages list for TaskState
    messages = _build_inspect_messages(prompt, completion)

    # Build model output from the last assistant message
    model_output = _build_model_output(completion)

    # Get original input from info (pre-solver, matches native Inspect semantics)
    # Convert back to ChatMessage list if it was stored as list of dicts
    input_raw = info["inspect_input_raw"]
    if isinstance(input_raw, str):
        original_input: str | list[Any] = input_raw
    else:
        original_input = _build_inspect_messages(input_raw, [])

    # Build TaskState
    task_state = TaskState(
        model=BRIDGE_MODEL_NAME,
        sample_id=sample_id,
        epoch=0,
        input=original_input,
        messages=messages,
        target=target,
        output=model_output,
        metadata=info["inspect_metadata"],
    )

    # If we have a sandbox manager, set up sandbox context
    score: Score | None
    if sandbox_manager is not None:
        sandboxes = await sandbox_manager.get_sandbox(sample_id, info)
        async with sandbox_context(sandboxes):
            score = await scorer(task_state, target)
    else:
        # Call scorer without sandbox context
        score = await scorer(task_state, target)

    if score is None:
        warnings.warn(
            "Scorer returned None - returning 0.0 as default reward.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0
    return _score_to_float(score)


def _build_inspect_messages(
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]],
) -> list[Any]:
    """Convert Verifiers messages to Inspect ChatMessage objects."""
    messages: list[Any] = []

    for msg in prompt + completion:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            messages.append(ChatMessageSystem(content=content))
        elif role == "user":
            messages.append(ChatMessageUser(content=content))
        elif role == "assistant":
            tool_calls = None
            if "tool_calls" in msg:
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        function=tc["function"]["name"],
                        # Parse arguments from JSON string to dict (OpenAI format stores as string)
                        arguments=json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                        type=tc.get("type", "function"),
                    )
                    for tc in msg["tool_calls"]
                ]
            messages.append(
                ChatMessageAssistant(content=content, tool_calls=tool_calls)
            )
        elif role == "tool":
            messages.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=msg.get("tool_call_id"),
                    function=msg.get("name"),
                )
            )
        else:
            raise ValueError(f"Unknown role: {role}")

    return messages


def _build_model_output(completion: list[dict[str, Any]]) -> ModelOutput:
    """Build ModelOutput from the last assistant message in completion."""
    # Find the last assistant message
    last_assistant_content = ""
    for msg in reversed(completion):
        if msg["role"] == "assistant":
            last_assistant_content = msg["content"]
            break

    return ModelOutput.from_content(
        model="bridge-model",
        content=last_assistant_content,
    )


def _score_to_float(score: Score) -> float:
    """Convert an Inspect Score to a float reward."""
    # Defensive check - some scorers may return None value
    assert score.value is not None
    converter = value_to_float()
    converted_score = converter(score.value)
    return converted_score


def _get_scorer_name(scorer: Scorer) -> str:
    # Add __name__ attribute to partial function for Verifiers compatibility
    # Use __qualname__ to get unique names (e.g., "expression_exact_match.<locals>.score")
    # Extract the parent function name from qualname, or fall back to __name__ or class name
    qualname = getattr(scorer, "__qualname__", "")
    if ".<locals>." in qualname:
        # Extract parent function name: "expression_exact_match.<locals>.score" -> "expression_exact_match"
        scorer_name = qualname.split(".<locals>.")[0]
    else:
        scorer_name = getattr(scorer, "__name__", scorer.__class__.__name__)

    return scorer_name


def build_rubric_from_scorers(
    scorers: list[Scorer],
    weights: list[float] | None = None,
    sandbox_manager: SandboxManager | None = None,
) -> vf.Rubric:
    """
    Build a Verifiers Rubric from a list of Inspect scorers.

    Args:
        scorers: List of Inspect Scorer functions
        weights: Optional weights for each scorer
        sandbox_manager: Optional sandbox manager for sandbox-based scorers

    Returns:
        A Verifiers Rubric that calls the Inspect scorers
    """
    if not scorers:
        raise ValueError("At least one scorer is required")

    # Create reward functions for each scorer
    reward_funcs: list[Callable[..., Any]] = []
    for i, scorer in enumerate(scorers):
        func = partial(
            reward_from_inspect_scorer,
            scorer=scorer,
            sandbox_manager=sandbox_manager,
        )
        scorer_name = _get_scorer_name(scorer)
        # Add index suffix to guarantee uniqueness if there are duplicate names
        func.__name__ = f"inspect_{scorer_name}_{i}"  # type: ignore[attr-defined]
        reward_funcs.append(func)

    return vf.Rubric(funcs=reward_funcs, weights=weights)  # type: ignore[arg-type]
