"""
Dataset conversion utilities: Inspect Sample -> HuggingFace Dataset.
"""

import logging
from typing import Any

from datasets import Dataset as HFDataset
from inspect_ai.dataset import Dataset as InspectDataset
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
)

logger = logging.getLogger(__name__)


def _substitute_variables(template: str, metadata: dict[str, Any] | None) -> str:
    """Substitute {variable} placeholders in template from metadata.

    Uses str.format_map with a defaulting dict to only substitute
    variables that exist in metadata, leaving others unchanged.
    """
    if not metadata:
        return template

    # Use a custom dict that returns the original placeholder for missing keys
    class DefaultDict(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    # Convert metadata values to strings for safe substitution
    format_dict = DefaultDict({k: str(v) for k, v in metadata.items()})
    return template.format_map(format_dict)


def _format_multiple_choice(
    template: str,
    question: str,
    choices: list[str],
) -> str:
    """Format a multiple choice question using the template.

    Args:
        template: Template with {question}, {letters}, {choices} placeholders
        question: The question text
        choices: List of choice strings

    Returns:
        Formatted multiple choice prompt
    """
    # Generate letters for choices (A, B, C, D, ...)
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    letters_str = ", ".join(letters)

    # Format choices as "A) choice1\nB) choice2\n..."
    formatted_choices = "\n".join(
        f"{letter}) {choice}" for letter, choice in zip(letters, choices)
    )

    # Substitute placeholders
    result = template
    result = result.replace("{question}", question)
    result = result.replace("{letters}", letters_str)
    result = result.replace("{choices}", formatted_choices)

    return result


def sample_to_row(
    sample: Sample,
    task_name: str,
    system_prompt: str | None = None,
    prompt_template: str | None = None,
    multiple_choice_template: str | None = None,
    user_messages: list[str] | None = None,
    prompt_transformations: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Convert an Inspect Sample to a Verifiers-compatible dataset row.

    Always uses the "prompt" column with a list of messages. This format:
    - Preserves full conversation history (multi-turn, tool calls, etc.)
    - Includes the system prompt in the message list
    - Applies prompt_template to format user input
    - Formats multiple choice questions with choices
    - Appends additional user_messages (with variable substitution)
    - Works consistently for both string and chat message inputs

    Args:
        sample: An Inspect Sample object
        task_name: Name of the task (for tracking)
        system_prompt: System prompt to prepend (if not already in messages)
        prompt_template: Template to format user input (e.g., "Question: {prompt}")
        multiple_choice_template: Template for multiple choice formatting
        user_messages: Additional user messages to append (may have {var} placeholders)
        prompt_transformations: Ordered list of (transform_type, template) tuples from solver chain.
            If provided, templates are applied in this order instead of using the individual
            prompt_template/multiple_choice_template parameters.

    Returns:
        Dictionary with prompt, answer, info, and id fields
    """
    sample_input = sample.input
    metadata = sample.metadata
    choices = sample.choices

    # Convert target to string answer when possible
    answer = _target_to_text(sample.target)

    # Store all Inspect-specific data in info for later use
    info: dict[str, Any] = {
        "inspect_sample_id": sample.id,
        "inspect_target_raw": sample.target,
        "inspect_choices": choices,
        "inspect_metadata": metadata or {},
        "inspect_sandbox": sample.sandbox,
        "inspect_files": sample.files,
        "inspect_setup": sample.setup,
        "inspect_task_name": task_name,
    }

    # Build prompt as list of messages
    prompt_messages: list[dict[str, Any]] = []

    if isinstance(sample_input, str):
        # String input: convert to system + user messages
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})

        # Apply templates in order from solver chain
        user_content = sample_input

        if prompt_transformations:
            # Use ordered transformations from solver introspection
            for transform_type, template in prompt_transformations:
                if transform_type == "prompt_template":
                    user_content = template.replace("{prompt}", user_content)
                elif transform_type == "multiple_choice" and choices:
                    user_content = _format_multiple_choice(template, user_content, choices)
        else:
            # Fall back to individual templates (legacy behavior)
            if prompt_template:
                user_content = prompt_template.replace("{prompt}", user_content)

            if multiple_choice_template and choices:
                user_content = _format_multiple_choice(
                    multiple_choice_template, user_content, choices
                )

        prompt_messages.append({"role": "user", "content": user_content})
    elif hasattr(sample_input, "__iter__") and not isinstance(sample_input, str):
        # Chat messages: convert to list[dict]
        prompt_messages = [
            _chat_message_to_dict(msg)
            for msg in sample_input  # type: ignore[union-attr]
        ]
        # Prepend system prompt if not already present and we have one
        if system_prompt and (
            not prompt_messages or prompt_messages[0].get("role") != "system"
        ):
            prompt_messages.insert(0, {"role": "system", "content": system_prompt})
        # Log that templates are not applied to chat messages
        if prompt_template or multiple_choice_template:
            logger.info(
                "Templates are not applied to chat message inputs "
                "(they have their own structure)"
            )
    else:
        # Unexpected input type - this shouldn't happen with valid Inspect samples
        raise ValueError(
            f"Unexpected sample input type: {type(sample_input)}. "
            "Expected str or list of ChatMessage."
        )

    # Append additional user messages (with variable substitution from metadata)
    if user_messages:
        for msg_template in user_messages:
            msg_content = _substitute_variables(msg_template, metadata)
            prompt_messages.append({"role": "user", "content": msg_content})

    return {
        "prompt": prompt_messages,
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

    # Get content - all ChatMessage types have content
    content = msg.content
    if isinstance(content, str):
        result["content"] = content
    elif content:
        # Content is a list of content parts - extract text
        # Note: Verifiers expects string content, so we concatenate text parts
        text_parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))
        result["content"] = "\n".join(text_parts) if text_parts else ""
    else:
        result["content"] = ""

    # Handle type-specific attributes using isinstance checks
    if isinstance(msg, ChatMessageAssistant):
        # Preserve tool_calls for assistant messages
        if msg.tool_calls:
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
    elif isinstance(msg, ChatMessageTool):
        # Preserve tool response metadata
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.function:
            result["name"] = msg.function  # OpenAI format uses "name" for tool responses
    # ChatMessageUser and ChatMessageSystem only have role and content (already handled)

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
    system_prompt: str | None = None,
    prompt_template: str | None = None,
    multiple_choice_template: str | None = None,
    user_messages: list[str] | None = None,
    prompt_transformations: list[tuple[str, str]] | None = None,
    max_samples: int | None = None,
) -> HFDataset:
    """
    Convert an Inspect dataset to a HuggingFace Dataset.

    Args:
        dataset: An Inspect Dataset object
        task_name: Name of the task
        system_prompt: System prompt to include in each sample's prompt list
        prompt_template: Template to format user input (e.g., "Question: {prompt}")
        multiple_choice_template: Template for multiple choice formatting
        user_messages: Additional user messages to append (may have {var} placeholders)
        prompt_transformations: Ordered list of (transform_type, template) tuples from solver chain
        max_samples: Optional limit on number of samples to convert

    Returns:
        A HuggingFace Dataset compatible with Verifiers
    """
    rows = []
    for i, sample in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        rows.append(
            sample_to_row(
                sample,
                task_name,
                system_prompt,
                prompt_template,
                multiple_choice_template,
                user_messages,
                prompt_transformations,
            )
        )

    return HFDataset.from_list(rows)
