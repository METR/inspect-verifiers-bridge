"""
Dataset conversion utilities: Inspect Sample -> HuggingFace Dataset.

Uses ground truth solver execution for accurate prompt construction.
"""

import asyncio
import json
import warnings
from typing import Any

from datasets import Dataset as HFDataset
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)

from inspect_verifiers_bridge.ground_truth import get_ground_truth_messages


async def sample_to_row(
    sample: Sample,
    task: Task,
    task_name: str,
    additional_system_content: str | None = None,
) -> dict[str, Any]:
    """
    Convert an Inspect Sample to a Verifiers-compatible dataset row.

    Uses ground truth solver execution to get the actual transformed messages.

    Args:
        sample: An Inspect Sample object
        task: The Inspect Task (contains solver chain)
        task_name: Name of the task (for tracking)
        additional_system_content: Optional content to append to the system message

    Returns:
        Dictionary with prompt, answer, info, and id fields
    """
    # Get ground truth messages from solver pipeline
    messages = await get_ground_truth_messages(task, sample)
    prompt_messages = [_chat_message_to_dict(msg) for msg in messages]

    # Append additional content to system message if provided
    if additional_system_content:
        prompt_messages = _append_to_system_message(
            prompt_messages, additional_system_content
        )

    # Convert target to string answer
    answer = _target_to_text(sample.target)

    # Store original input (pre-solver) for TaskState.input in scoring
    # Convert ChatMessage list to dicts for serialization
    if isinstance(sample.input, str):
        input_raw: str | list[dict[str, Any]] = sample.input
    else:
        input_raw = [_chat_message_to_dict(msg) for msg in sample.input]

    # Store all Inspect-specific data in info for later use
    # Note: inspect_metadata is serialized to JSON string because pyarrow
    # can't handle dicts with varying schemas across samples
    info: dict[str, Any] = {
        "inspect_sample_id": sample.id,
        "inspect_input_raw": input_raw,
        "inspect_target_raw": sample.target,
        "inspect_choices": sample.choices,
        "inspect_metadata": json.dumps(sample.metadata or {}),
        "inspect_sandbox": sample.sandbox,
        "inspect_files": sample.files,
        "inspect_setup": sample.setup,
        "inspect_task_name": task_name,
    }

    return {
        "prompt": prompt_messages,
        "answer": answer,
        "info": info,
        "id": sample.id,
    }


def _extract_content(msg: ChatMessage) -> str:
    """Extract string content from a ChatMessage."""
    content = msg.content
    assert content is not None
    if isinstance(content, str):
        return content
    # Content is a list of content parts - extract text
    # Note: Verifiers expects string content, so we concatenate text parts
    assert isinstance(content, list)
    text_parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
    return "\n".join(text_parts) if text_parts else ""


def _chat_message_to_dict(msg: ChatMessage) -> dict[str, Any]:
    """Convert an Inspect ChatMessage to a Verifiers-compatible dictionary.

    Preserves:
    - role: user, assistant, system, tool
    - content: text content
    - tool_calls: for assistant messages with tool use
    - tool_call_id: for tool response messages
    - name: tool function name for tool responses (OpenAI format)
    """
    match msg:
        case ChatMessageUser() | ChatMessageSystem():
            return {"role": msg.role, "content": _extract_content(msg)}
        case ChatMessageAssistant(tool_calls=tool_calls):
            result: dict[str, Any] = {
                "role": msg.role,
                "content": _extract_content(msg),
            }
            if tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": tc.function,
                            "arguments": tc.arguments
                            if isinstance(tc.arguments, str)
                            else str(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            return result
        case ChatMessageTool(tool_call_id=tool_call_id, function=function):
            result = {"role": msg.role, "content": _extract_content(msg)}
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
            if function:
                result["name"] = function  # OpenAI format uses "name"
            return result


def _append_to_system_message(
    messages: list[dict[str, Any]], content: str
) -> list[dict[str, Any]]:
    """
    Append content to the system message, or create one if it doesn't exist.

    Args:
        messages: List of message dicts
        content: Content to append to system message

    Returns:
        Updated message list with appended system content
    """
    # Find existing system message
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            # Append to existing system message
            existing = msg.get("content", "")
            separator = "\n\n" if existing else ""
            updated_msg = {**msg, "content": f"{existing}{separator}{content}"}
            return messages[:i] + [updated_msg] + messages[i + 1 :]

    # No system message found - create one at the beginning
    system_msg = {"role": "system", "content": content}
    return [system_msg] + messages


def _target_to_text(target: Any) -> str | None:
    """Convert an Inspect target to a text string."""
    if target is None:
        return None
    if isinstance(target, str):
        return target
    if isinstance(target, list):
        warnings.warn(
            f"Converting a list target to a string. {target}",
            UserWarning,
            stacklevel=2,
        )
        # For list targets (like test cases), join them
        if all(isinstance(t, str) for t in target):
            return "\n".join(target)
        return str(target)
    # For other types, try to get text representation
    if hasattr(target, "text"):
        warnings.warn(
            f"Accessing str via .text property. {target}",
            UserWarning,
            stacklevel=2,
        )
        return target.text
    warnings.warn(
        f"Converting a non-string target to a string via str(). {target}",
        UserWarning,
        stacklevel=2,
    )
    return str(target)


def _run_async_in_thread(coro: Any) -> Any:
    """Run an async coroutine in a separate thread to avoid event loop conflicts."""
    import concurrent.futures

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner)
        return future.result()


def inspect_dataset_to_hf(
    task: Task,
    task_name: str,
    max_samples: int | None = None,
    additional_system_content: str | None = None,
) -> HFDataset:
    """
    Convert an Inspect dataset to a HuggingFace Dataset using ground truth.

    Args:
        task: The Inspect Task (contains dataset and solver chain)
        task_name: Name of the task
        max_samples: Optional limit on number of samples to convert
        additional_system_content: Optional content to append to system messages
            (e.g., instructions for using tools)

    Returns:
        A HuggingFace Dataset compatible with Verifiers
    """
    rows = []
    for i, sample in enumerate(task.dataset):
        if max_samples is not None and i >= max_samples:
            break

        # Auto-generate ID if not set (some datasets like GSM8K don't set IDs)
        if sample.id is None:
            sample = Sample(
                input=sample.input,
                target=sample.target,
                id=str(i),
                choices=sample.choices,
                metadata=sample.metadata,
                sandbox=sample.sandbox,
                files=sample.files,
                setup=sample.setup,
            )

        rows.append(
            _run_async_in_thread(
                sample_to_row(sample, task, task_name, additional_system_content)
            )
        )

    return HFDataset.from_list(rows)
