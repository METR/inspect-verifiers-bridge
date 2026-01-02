"""
Scoring bridge: Convert Inspect scorers to Verifiers reward functions.

This module provides the core mechanism to call Inspect scorers within the
Verifiers reward function framework.
"""

import asyncio
from functools import partial
from typing import Any

import verifiers as vf
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score, Scorer, Target, value_to_float
from inspect_ai.solver import TaskState

from inspect_verifiers_bridge.sandbox import SandboxManager, sandbox_context

# Type alias for model name to avoid strict type checking issues
MODEL_NAME = "bridge-model"


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

    # Get the raw target from info, or fall back to answer
    target_raw = info.get("inspect_target_raw", answer)
    target = Target(target_raw) if target_raw is not None else Target("")

    # Build messages list for TaskState
    messages = _build_inspect_messages(prompt, completion)

    # Build model output from the last assistant message
    model_output = _build_model_output(completion)

    # Reconstruct original input
    original_input = _extract_original_input(prompt)

    # Build TaskState - use a generic model name
    task_state = TaskState(
        model=MODEL_NAME,  # type: ignore[arg-type]
        sample_id=info.get("inspect_sample_id", 0),
        epoch=0,
        input=original_input,
        messages=messages,
        target=target,
        output=model_output,
        metadata=info.get("inspect_metadata", {}),
    )

    # If we have a sandbox manager, set up sandbox context
    score: Score | None
    if sandbox_manager is not None:
        sample_id = info.get("inspect_sample_id", 0)
        sandboxes = await sandbox_manager.get_sandbox(sample_id, info)
        async with sandbox_context(sandboxes):
            score = await scorer(task_state, target)
    else:
        # Call scorer without sandbox context
        score = await scorer(task_state, target)

    if score is None:
        return 0.0
    return _score_to_float(score)


def _build_inspect_messages(
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]],
) -> list[Any]:
    """Convert Verifiers messages to Inspect ChatMessage objects."""
    from inspect_ai.model import ChatMessageSystem

    messages: list[Any] = []

    for msg in prompt:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            messages.append(ChatMessageSystem(content=content))
        elif role == "user":
            messages.append(ChatMessageUser(content=content))
        elif role == "assistant":
            messages.append(ChatMessageAssistant(content=content))

    for msg in completion:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "assistant":
            messages.append(ChatMessageAssistant(content=content))
        elif role == "user":
            messages.append(ChatMessageUser(content=content))

    return messages


def _build_model_output(completion: list[dict[str, Any]]) -> ModelOutput:
    """Build ModelOutput from the last assistant message in completion."""
    # Find the last assistant message
    last_assistant_content = ""
    for msg in reversed(completion):
        if msg.get("role") == "assistant":
            last_assistant_content = msg.get("content", "")
            break

    return ModelOutput.from_content(
        model="bridge-model",
        content=last_assistant_content,
    )


def _extract_original_input(prompt: list[dict[str, Any]]) -> str | list[Any]:
    """Extract the original input from prompt messages."""
    # For single-turn, just get the user message content
    user_messages = [m for m in prompt if m.get("role") == "user"]
    if len(user_messages) == 1:
        return user_messages[0].get("content", "")
    # For multi-turn, return the full message list
    return _build_inspect_messages(prompt, [])


def _score_to_float(score: Score) -> float:
    """Convert an Inspect Score to a float reward."""
    converter = value_to_float()
    return converter(score.value)


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
    reward_funcs = []
    for scorer in scorers:
        func = partial(
            reward_from_inspect_scorer,
            scorer=scorer,
            sandbox_manager=sandbox_manager,
        )
        # Add __name__ attribute to partial function for Verifiers compatibility
        scorer_name = getattr(scorer, "__name__", scorer.__class__.__name__)
        func.__name__ = f"inspect_scorer_{scorer_name}"  # type: ignore[attr-defined]
        reward_funcs.append(func)

    return vf.Rubric(funcs=reward_funcs, weights=weights)  # type: ignore[arg-type]


def sync_reward_wrapper(
    async_reward_func: Any,
) -> Any:
    """
    Wrap an async reward function for synchronous use.

    Some contexts may need synchronous rewards. This wrapper
    handles running the async function in an event loop.
    """

    def sync_wrapper(
        prompt: list[dict[str, Any]],
        completion: list[dict[str, Any]],
        answer: str | None,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    async_reward_func(prompt, completion, answer, state, **kwargs),
                )
                return future.result()
        else:
            return loop.run_until_complete(
                async_reward_func(prompt, completion, answer, state, **kwargs)
            )

    return sync_wrapper
