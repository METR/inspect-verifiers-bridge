"""
Tests for dataset conversion from Inspect samples to HuggingFace datasets.

These tests verify that:
1. Dataset conversion preserves all information
2. Different input formats work correctly
3. Solver templates are correctly applied
"""

from typing import Any

from inspect_verifiers_bridge.loader import get_inspect_dataset
from inspect_verifiers_bridge.tasks import load_inspect_task

from .fake_tasks import (
    assistant_only_input,
    chat_input,
    code_execution,
    mixed_messages,
    multiple_choice,
    simple_math,
    simple_math_with_template,
    with_metadata,
    with_tool_calls,
)


def _row(item: Any) -> dict[str, Any]:
    """Convert HuggingFace dataset item to dict for type safety."""
    return dict(item)  # type: ignore[arg-type]


class TestDatasetConversion:
    """Test that dataset conversion preserves all information."""

    def test_simple_math_dataset(self) -> None:
        """Test simple math task dataset conversion - no prompt template."""
        task_info = load_inspect_task(simple_math)
        hf_dataset = get_inspect_dataset(simple_math)

        # Check dataset size
        assert len(hf_dataset) == len(list(task_info.dataset))

        # Verify task introspection - system_prompt should be just the system message
        assert task_info.system_prompt == "Answer with just the number, nothing else."
        assert task_info.prompt_template is None

        # Check each sample
        for hf_row, inspect_sample in zip(hf_dataset, task_info.dataset):
            row = _row(hf_row)
            assert row["id"] == inspect_sample.id
            # Always uses "prompt" column with list of messages
            assert "prompt" in row
            assert isinstance(row["prompt"], list)
            # Should have system message + user message (raw input, no template)
            assert len(row["prompt"]) == 2
            assert row["prompt"][0]["role"] == "system"
            assert row["prompt"][0]["content"] == "Answer with just the number, nothing else."
            assert row["prompt"][1]["role"] == "user"
            assert row["prompt"][1]["content"] == inspect_sample.input  # Raw input
            assert row["answer"] == inspect_sample.target
            assert row["info"]["inspect_sample_id"] == inspect_sample.id
            assert row["info"]["inspect_metadata"] == inspect_sample.metadata

    def test_simple_math_with_prompt_template_dataset(self) -> None:
        """Test task with prompt_template - template should format user input."""
        task_info = load_inspect_task(simple_math_with_template)
        hf_dataset = get_inspect_dataset(simple_math_with_template)

        # Check dataset size
        assert len(hf_dataset) == len(list(task_info.dataset))

        # Verify task introspection - system_prompt and prompt_template should be separate
        assert task_info.system_prompt == "Answer with just the number, nothing else."
        assert task_info.prompt_template == "Here is the question: {prompt}"

        # Check each sample
        for hf_row, inspect_sample in zip(hf_dataset, task_info.dataset):
            row = _row(hf_row)
            assert row["id"] == inspect_sample.id
            # Always uses "prompt" column with list of messages
            assert "prompt" in row
            assert isinstance(row["prompt"], list)
            # Should have system message + user message (formatted via template)
            assert len(row["prompt"]) == 2
            assert row["prompt"][0]["role"] == "system"
            assert row["prompt"][0]["content"] == "Answer with just the number, nothing else."
            assert row["prompt"][1]["role"] == "user"
            # User message should be the FORMATTED input using the template
            expected_user_content = f"Here is the question: {inspect_sample.input}"
            assert row["prompt"][1]["content"] == expected_user_content
            assert row["answer"] == inspect_sample.target
            assert row["info"]["inspect_sample_id"] == inspect_sample.id
            assert row["info"]["inspect_metadata"] == inspect_sample.metadata

    def test_multiple_choice_dataset(self) -> None:
        """Test multiple choice task preserves choices."""
        hf_dataset = get_inspect_dataset(multiple_choice)

        for hf_row in hf_dataset:
            row = _row(hf_row)
            assert row["info"]["inspect_choices"] is not None
            assert len(row["info"]["inspect_choices"]) == 4

    def test_chat_input_dataset(self) -> None:
        """Test chat message input format is preserved as prompt list."""
        hf_dataset = get_inspect_dataset(chat_input)

        for hf_row in hf_dataset:
            row = _row(hf_row)
            # Chat input should be converted to list of message dicts
            # using "prompt" column (not "question") to preserve full history
            assert "prompt" in row
            assert isinstance(row["prompt"], list)
            assert len(row["prompt"]) >= 1
            # Check message structure
            for msg in row["prompt"]:
                assert "role" in msg
                assert "content" in msg

    def test_metadata_preserved(self) -> None:
        """Test that metadata is fully preserved."""
        task_info = load_inspect_task(with_metadata)
        hf_dataset = get_inspect_dataset(with_metadata)

        for hf_row, inspect_sample in zip(hf_dataset, task_info.dataset):
            row = _row(hf_row)
            stored_metadata = row["info"]["inspect_metadata"]
            original_metadata = inspect_sample.metadata or {}

            for key, value in original_metadata.items():
                assert key in stored_metadata
                assert stored_metadata[key] == value

    def test_list_target_preserved(self) -> None:
        """Test that list targets (like test cases) are preserved."""
        hf_dataset = get_inspect_dataset(code_execution)

        for hf_row in hf_dataset:
            row = _row(hf_row)
            raw_target = row["info"]["inspect_target_raw"]
            assert isinstance(raw_target, list)
            assert all(isinstance(t, str) for t in raw_target)

    def test_tool_call_messages(self) -> None:
        """Test that samples with tool calls preserve full conversation history."""
        hf_dataset = get_inspect_dataset(with_tool_calls)

        assert len(hf_dataset) == 2

        for hf_row in hf_dataset:
            row = _row(hf_row)
            # Chat input should use "prompt" column with full message list
            assert "prompt" in row
            assert isinstance(row["prompt"], list)
            assert len(row["prompt"]) > 0

            # Metadata should indicate tool calls
            assert row["info"]["inspect_metadata"]["has_tool_calls"] is True

    def test_tool_call_preserves_tool_structure(self) -> None:
        """Test that tool call messages preserve tool_calls and tool_call_id."""
        hf_dataset = get_inspect_dataset(with_tool_calls)

        first_row = _row(hf_dataset[0])

        # Find assistant message with tool calls
        assistant_msgs = [m for m in first_row["prompt"] if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1

        # Check tool_calls are preserved
        tool_call_msg = next((m for m in assistant_msgs if "tool_calls" in m), None)
        assert tool_call_msg is not None
        assert len(tool_call_msg["tool_calls"]) > 0

        # Find tool response message
        tool_msgs = [m for m in first_row["prompt"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "tool_call_id" in tool_msgs[0]

    def test_assistant_only_input(self) -> None:
        """Test that samples with only assistant messages preserve full history."""
        hf_dataset = get_inspect_dataset(assistant_only_input)

        assert len(hf_dataset) == 1
        row = _row(hf_dataset[0])

        # Chat input should use "prompt" column
        assert "prompt" in row
        assert isinstance(row["prompt"], list)

        # Should have system and assistant messages
        roles = [m["role"] for m in row["prompt"]]
        assert "system" in roles
        assert "assistant" in roles

    def test_mixed_message_types(self) -> None:
        """Test that mixed user/assistant conversations preserve full history."""
        hf_dataset = get_inspect_dataset(mixed_messages)

        assert len(hf_dataset) == 2

        # First sample has multiple message types
        first_row = _row(hf_dataset[0])
        assert "prompt" in first_row
        assert isinstance(first_row["prompt"], list)

        # Should have system, user, and assistant messages
        roles = [m["role"] for m in first_row["prompt"]]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

        # Second sample has multiple turns
        second_row = _row(hf_dataset[1])
        assert second_row["info"]["inspect_metadata"]["turn_count"] == 5

        # Should have alternating user/assistant messages
        assert isinstance(second_row["prompt"], list)
        assert len(second_row["prompt"]) == 5  # 5 messages total
