"""
Dataset conversion utilities: Inspect Sample -> HuggingFace Dataset.
"""

from typing import Any

from datasets import Dataset as HFDataset
from inspect_ai.dataset import Dataset as InspectDataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage


def sample_to_row(sample: Sample, task_name: str) -> dict[str, Any]:
    """
    Convert an Inspect Sample to a Verifiers-compatible dataset row.

    Verifiers supports two input formats:
    - "question" column (string): Verifiers wraps with system_prompt + user message
    - "prompt" column (list[ChatMessage]): Used directly as-is

    We use:
    - String inputs → "question" column (simpler, lets Verifiers handle formatting)
    - Chat message inputs → "prompt" column (preserves full conversation history)

    Args:
        sample: An Inspect Sample object
        task_name: Name of the task (for tracking)

    Returns:
        Dictionary with prompt/question, answer, info, and id fields
    """
    sample_input = sample.input

    # Convert target to string answer when possible
    answer = _target_to_text(sample.target)

    # Store all Inspect-specific data in info for later use
    info: dict[str, Any] = {
        "inspect_sample_id": sample.id,
        "inspect_target_raw": sample.target,
        "inspect_choices": sample.choices,
        "inspect_metadata": sample.metadata or {},
        "inspect_sandbox": sample.sandbox,
        "inspect_files": sample.files,
        "inspect_setup": sample.setup,
        "inspect_task_name": task_name,
    }

    # Determine input format
    if isinstance(sample_input, str):
        # String input: use "question" column, let Verifiers add system_prompt
        return {
            "question": sample_input,
            "answer": answer,
            "info": info,
            "id": sample.id,
        }
    elif hasattr(sample_input, "__iter__") and not isinstance(sample_input, str):
        # Chat messages: convert to list[dict] and use "prompt" column
        # This preserves tool calls, assistant messages, multi-turn history
        prompt_messages = [
            _chat_message_to_dict(msg) for msg in sample_input  # type: ignore[union-attr]
        ]
        return {
            "prompt": prompt_messages,
            "answer": answer,
            "info": info,
            "id": sample.id,
        }
    else:
        # Fallback: convert to string
        return {
            "question": str(sample_input),
            "answer": answer,
            "info": info,
            "id": sample.id,
        }


def _chat_message_to_dict(msg: ChatMessage) -> dict[str, Any]:
    """Convert an Inspect ChatMessage to a Verifiers-compatible dictionary.

    Preserves:
    - role: user, assistant, system, tool
    - content: text content
    - tool_calls: for assistant messages with tool use
    - tool_call_id: for tool response messages
    - function: tool function name for tool responses
    """
    result: dict[str, Any] = {"role": msg.role}

    # Get content (prefer .content, fall back to .text)
    if hasattr(msg, "content") and msg.content is not None:
        result["content"] = msg.content
    elif hasattr(msg, "text") and msg.text is not None:
        result["content"] = msg.text
    else:
        result["content"] = ""

    # Preserve tool_calls for assistant messages
    if hasattr(msg, "tool_calls") and msg.tool_calls:
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
            for tc in msg.tool_calls
        ]

    # Preserve tool response metadata
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "function") and msg.function:
        result["name"] = msg.function  # OpenAI format uses "name" for tool responses

    return result


def _target_to_text(target: Any) -> str | None:
    """Convert an Inspect target to a text string."""
    if target is None:
        return None
    if isinstance(target, str):
        return target
    if isinstance(target, list):
        # For list targets (like test cases), join them
        if all(isinstance(t, str) for t in target):
            return "\n".join(target)
        return str(target)
    # For other types, try to get text representation
    if hasattr(target, "text"):
        return target.text
    return str(target)


def inspect_dataset_to_hf(
    dataset: InspectDataset,
    task_name: str,
    max_samples: int | None = None,
) -> HFDataset:
    """
    Convert an Inspect dataset to a HuggingFace Dataset.

    Args:
        dataset: An Inspect Dataset object
        task_name: Name of the task
        max_samples: Optional limit on number of samples to convert

    Returns:
        A HuggingFace Dataset compatible with Verifiers
    """
    rows = []
    for i, sample in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        rows.append(sample_to_row(sample, task_name))

    return HFDataset.from_list(rows)
