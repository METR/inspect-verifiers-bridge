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

    Args:
        sample: An Inspect Sample object
        task_name: Name of the task (for tracking)

    Returns:
        Dictionary with question, answer, info, and id fields
    """
    # Convert input to string question format
    # Verifiers expects a "question" column which it will format with system_prompt
    question: str
    sample_input = sample.input
    if isinstance(sample_input, str):
        question = sample_input
    elif hasattr(sample_input, "__iter__") and not isinstance(sample_input, str):
        # For chat messages, extract just the user content
        # (system messages will be handled via system_prompt parameter)
        user_messages = [
            msg.content if hasattr(msg, "content") else msg.text
            for msg in sample_input  # type: ignore[union-attr]
            if hasattr(msg, "role") and msg.role == "user"
        ]
        question = "\n".join(user_messages) if user_messages else str(sample_input)
    else:
        question = str(sample_input)

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

    return {
        "question": question,
        "answer": answer,
        "info": info,
        "id": sample.id,
    }


def _chat_message_to_dict(msg: ChatMessage) -> dict[str, Any]:
    """Convert an Inspect ChatMessage to a dictionary."""
    result: dict[str, Any] = {"role": msg.role}
    if hasattr(msg, "content"):
        result["content"] = msg.content
    if hasattr(msg, "text"):
        result["content"] = msg.text
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
