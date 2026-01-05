"""
Regression tests for bug fixes.

These tests ensure that previously fixed bugs don't reoccur.
"""

import asyncio
from typing import Any, cast

import pytest
from inspect_ai.scorer import CORRECT, Score, Scorer, Target, exact, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util._sandbox.environment import (
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from inspect_ai.util._subprocess import ExecResult

from inspect_verifiers_bridge.scoring import (
    _build_inspect_messages,
    build_rubric_from_scorers,
)


class TestToolMessageRoundTrip:
    """
    Regression tests for tool message function name preservation.

    Bug: Tool responses lost their function name during scoring reconstruction
    because dataset stored under "name" (OpenAI format) but scoring code
    might read from wrong key.

    Fix: Both dataset.py and scoring.py consistently use "name" key.
    """

    def test_tool_message_preserves_function_name(self) -> None:
        """Test that tool message function name round-trips correctly."""
        # Simulate a tool message as stored in dataset (OpenAI format)
        tool_msg = {
            "role": "tool",
            "content": "Result: 42",
            "tool_call_id": "call_123",
            "name": "calculator",  # OpenAI format uses "name"
        }

        # Convert through _build_inspect_messages (scoring.py)
        messages = _build_inspect_messages([tool_msg], [])

        assert len(messages) == 1
        from inspect_ai.model import ChatMessageTool

        assert isinstance(messages[0], ChatMessageTool)
        assert messages[0].function == "calculator"
        assert messages[0].tool_call_id == "call_123"
        assert messages[0].content == "Result: 42"

    def test_assistant_tool_calls_preserve_structure(self) -> None:
        """Test that assistant tool calls round-trip correctly."""
        # Simulate an assistant message with tool calls as stored in dataset
        # Note: OpenAI format stores arguments as JSON string
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"x": 1, "y": 2}',
                    },
                }
            ],
        }

        messages = _build_inspect_messages([assistant_msg], [])

        assert len(messages) == 1
        from inspect_ai.model import ChatMessageAssistant

        assert isinstance(messages[0], ChatMessageAssistant)
        assert messages[0].tool_calls is not None
        assert len(messages[0].tool_calls) == 1
        tc = messages[0].tool_calls[0]
        assert tc.id == "call_123"
        assert tc.function == "calculator"
        # ToolCall stores arguments as dict, not string
        assert tc.arguments == {"x": 1, "y": 2}

    def test_full_tool_conversation_round_trip(self) -> None:
        """Test that a full tool conversation round-trips correctly."""
        # Simulate a conversation with tool use
        prompt = [
            {"role": "system", "content": "You are a calculator."},
            {"role": "user", "content": "What is 1 + 2?"},
        ]
        completion = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "3",
                "tool_call_id": "call_abc",
                "name": "add",
            },
            {"role": "assistant", "content": "The answer is 3."},
        ]

        messages = _build_inspect_messages(prompt, completion)

        assert len(messages) == 5

        # Check tool response
        tool_msg = messages[3]
        from inspect_ai.model import ChatMessageTool

        assert isinstance(tool_msg, ChatMessageTool)
        assert tool_msg.function == "add"
        assert tool_msg.tool_call_id == "call_abc"


class TestScorerNaming:
    """
    Regression tests for scorer naming fix.

    Bug: When multiple scorers have the same __name__ (e.g., all inner functions
    named "score"), Verifiers would overwrite results in aggregated_metrics dict,
    causing only the last scorer's results to be visible.

    Fix: Extract unique names from __qualname__ and add index suffix.
    """

    def test_scorers_get_unique_names(self) -> None:
        """Test that scorers with same __name__ get unique function names."""

        # Create multiple scorers that all have __name__ = "score"
        @scorer(metrics=[])
        def scorer_one() -> Scorer:
            async def score(state: TaskState, target: Target) -> Score:
                return Score(value=CORRECT)

            return score

        @scorer(metrics=[])
        def scorer_two() -> Scorer:
            async def score(state: TaskState, target: Target) -> Score:
                return Score(value=CORRECT)

            return score

        @scorer(metrics=[])
        def scorer_three() -> Scorer:
            async def score(state: TaskState, target: Target) -> Score:
                return Score(value=CORRECT)

            return score

        scorers = [scorer_one(), scorer_two(), scorer_three()]

        # All inner functions have __name__ = "score"
        for s in scorers:
            # Scorer is a callable, access __name__ via cast
            assert cast(Any, s).__name__ == "score"

        # Build rubric
        rubric = build_rubric_from_scorers(scorers)

        # Verify unique names
        names = [f.__name__ for f in rubric.funcs]
        assert len(names) == 3
        assert len(set(names)) == 3  # All unique

        # Verify names are unique and contain index
        # Note: scorers defined inside test have qualname like "TestClass.test_method.<locals>.scorer_one"
        # Our extractor gets the parent: "TestClass.test_method"
        assert "_0" in names[0]
        assert "_1" in names[1]
        assert "_2" in names[2]

    def test_scorer_names_include_index(self) -> None:
        """Test that scorer names include index suffix for guaranteed uniqueness."""
        # Use the same scorer twice
        scorers = [exact(), exact()]

        rubric = build_rubric_from_scorers(scorers)
        names = [f.__name__ for f in rubric.funcs]

        # Should have different indices
        assert "_0" in names[0]
        assert "_1" in names[1]
        assert names[0] != names[1]

    def test_qualname_extraction(self) -> None:
        """Test that __qualname__ is properly parsed to extract parent function name."""
        # Use a built-in scorer which has a cleaner qualname
        from inspect_ai.scorer import exact

        scorers = [exact()]
        rubric = build_rubric_from_scorers(scorers)

        # The scorer's qualname should be extracted (e.g., "exact.<locals>.score" -> "exact")
        # The name should contain "exact" and an index
        name = rubric.funcs[0].__name__
        assert "exact" in name.lower() or "score" in name.lower()
        assert "_0" in name

    def test_scorer_name_from_registry_info(self) -> None:
        """
        Regression test for inspect_scout scanner naming.

        Bug: When inspect_scout converts a scanner to a scorer via as_scorer(),
        the scorer's __qualname__ is "as_scorer.<locals>.score" which results in
        names like "inspect_as_scorer_0" instead of the actual scanner name.

        Fix: Check __registry_info__.name first (where the scanner name is stored)
        before falling back to __qualname__ extraction.
        """
        from inspect_ai._util.registry import RegistryInfo

        from inspect_verifiers_bridge.scoring import _get_scorer_name

        # Create a mock scorer that mimics what inspect_scout's as_scorer creates:
        # - __qualname__ is "as_scorer.<locals>.score" (wrong name)
        # - __registry_info__.name contains the actual scanner name
        async def mock_score(state: TaskState, target: Target) -> Score:
            return Score(value=CORRECT)

        # Simulate the qualname that as_scorer creates
        mock_score.__qualname__ = "as_scorer.<locals>.score"

        # Simulate what @scorer(name="ctf_environment") does - stores name in registry info
        mock_score.__registry_info__ = RegistryInfo(  # type: ignore[attr-defined]
            type="scorer",
            name="ctf_environment",
            metadata={},
        )

        # Should extract "ctf_environment" from registry info, not "as_scorer" from qualname
        assert _get_scorer_name(mock_score) == "ctf_environment"

    def test_scorer_name_strips_package_prefix(self) -> None:
        """Test that package prefixes are stripped from registry names."""
        from inspect_ai._util.registry import RegistryInfo

        from inspect_verifiers_bridge.scoring import _get_scorer_name

        async def mock_score(state: TaskState, target: Target) -> Score:
            return Score(value=CORRECT)

        # Registry names can include package prefixes like "my_package/scanner_name"
        mock_score.__registry_info__ = RegistryInfo(  # type: ignore[attr-defined]
            type="scorer",
            name="inspect_scout/ctf_scanner",
            metadata={},
        )

        # Should strip the package prefix and return just "ctf_scanner"
        assert _get_scorer_name(mock_score) == "ctf_scanner"


class MockSandbox(SandboxEnvironment):
    """Mock sandbox for testing."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,  # noqa: A002
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def write_file(self, file: str, contents: str | bytes) -> None:
        pass

    async def read_file(self, file: str, text: bool = True) -> str | bytes:  # type: ignore[override]
        return "" if text else b""

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, "SandboxEnvironment"],
        interrupted: bool,
    ) -> None:
        pass


class TestSandboxContext:
    """
    Regression tests for sandbox context fix.

    Bug: When running multiple rollouts concurrently via asyncio.gather(),
    only the first rollout had the sandbox ContextVars set (via init_sandbox_environments_sample).
    Subsequent rollouts would fail because sandbox_default_context_var wasn't set.

    Fix: Set all three required ContextVars in sandbox_context():
    - sandbox_environments_context_var
    - sandbox_default_context_var
    - sandbox_with_environments_context_var
    """

    @pytest.mark.asyncio
    async def test_sandbox_context_sets_all_contextvars(self) -> None:
        """Test that sandbox_context sets all required ContextVars."""
        from inspect_ai.util._sandbox.context import (
            sandbox_default_context_var,
            sandbox_environments_context_var,
            sandbox_with_environments_context_var,
        )

        from inspect_verifiers_bridge.sandbox import sandbox_context

        mock_sandboxes: dict[str, SandboxEnvironment] = {"default": MockSandbox()}

        async with sandbox_context(mock_sandboxes):
            # All three ContextVars should be set
            envs = sandbox_environments_context_var.get(None)
            default = sandbox_default_context_var.get(None)
            with_envs = sandbox_with_environments_context_var.get(None)

            assert envs is not None
            assert envs == mock_sandboxes
            assert default == "default"
            assert with_envs == {}

        # After context exits, vars should be reset (get returns None or raises)
        # We can't easily test this without knowing initial state

    @pytest.mark.asyncio
    async def test_sandbox_context_concurrent_access(self) -> None:
        """Test that sandbox_context works correctly with concurrent coroutines."""
        from inspect_ai.util._sandbox.context import (
            sandbox_default_context_var,
            sandbox_environments_context_var,
        )

        from inspect_verifiers_bridge.sandbox import sandbox_context

        results: list[dict[str, Any]] = []

        async def check_context(sandbox_name: str) -> None:
            mock_sandboxes: dict[str, SandboxEnvironment] = {
                sandbox_name: MockSandbox(sandbox_name)
            }
            async with sandbox_context(mock_sandboxes):
                # Small delay to simulate work and encourage interleaving
                await asyncio.sleep(0.01)

                envs = sandbox_environments_context_var.get(None)
                default = sandbox_default_context_var.get(None)

                # Each coroutine should see its own context
                results.append(
                    {
                        "expected": sandbox_name,
                        "got_envs": list(envs.keys())[0] if envs else None,
                        "got_default": default,
                    }
                )

        # Run multiple coroutines concurrently
        await asyncio.gather(
            check_context("sandbox_a"),
            check_context("sandbox_b"),
            check_context("sandbox_c"),
        )

        # Each should have seen its own sandbox
        assert len(results) == 3
        for r in results:
            assert r["expected"] == r["got_envs"]
            assert r["expected"] == r["got_default"]

    @pytest.mark.asyncio
    async def test_sandbox_context_default_name_selection(self) -> None:
        """Test that default sandbox name is correctly selected from dict keys."""
        from inspect_ai.util._sandbox.context import sandbox_default_context_var

        from inspect_verifiers_bridge.sandbox import sandbox_context

        # Test with multiple sandboxes - first key should be default
        mock_sandboxes: dict[str, SandboxEnvironment] = {
            "first": MockSandbox("first"),
            "second": MockSandbox("second"),
        }

        async with sandbox_context(mock_sandboxes):
            default = sandbox_default_context_var.get()
            assert default == "first"

    @pytest.mark.asyncio
    async def test_sandbox_context_empty_sandboxes(self) -> None:
        """Test that sandbox_context handles empty sandbox dict."""
        from inspect_ai.util._sandbox.context import sandbox_default_context_var

        from inspect_verifiers_bridge.sandbox import sandbox_context

        # Empty dict should use "default" as fallback
        async with sandbox_context({}):
            default = sandbox_default_context_var.get()
            assert default == "default"


def _row(item: Any) -> dict[str, Any]:
    """Convert HuggingFace dataset item to dict for type safety."""
    return dict(item)  # type: ignore[arg-type]


class TestSandboxScoringConcurrent:
    """
    Test that sandbox-based scoring works with concurrent rollouts.

    This is the actual scenario that was failing: vf-eval runs multiple
    rollouts per example concurrently, and second+ rollouts would fail.

    In the new architecture, each rollout gets its own sandbox via setup_state,
    so this tests that multiple sandboxes can run concurrently.
    """

    @pytest.mark.asyncio
    async def test_multiple_concurrent_scoring_calls(self) -> None:
        """Test that multiple concurrent scoring calls all succeed."""
        from inspect_verifiers_bridge import load_environment
        from inspect_verifiers_bridge.sandbox import (
            SandboxConfig,
            cleanup_sandbox,
            create_sandbox_for_sample,
        )

        from .fake_tasks import code_execution

        env = load_environment(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )

        correct_code = """```python
def add(a, b):
    return a + b
```"""

        dataset = env.dataset
        assert dataset is not None
        sample = _row(dataset[0])

        async def score_once() -> float:
            """Score with a fresh sandbox (mimics InspectSandboxEnv.setup_state)."""
            # Create sandbox for this "rollout"
            sandbox_config = SandboxConfig(sandbox_type="local")
            sandbox_instance = await create_sandbox_for_sample(
                sample_info=sample["info"],
                task_name="test_task",
                sandbox_config=sandbox_config,
            )
            state = {
                "info": sample["info"],
                "_sandbox_envs": sandbox_instance.environments,
            }

            try:
                reward_fn = env.rubric.funcs[0]
                result = reward_fn(
                    prompt=sample["prompt"],
                    completion=[{"role": "assistant", "content": correct_code}],
                    answer=sample["answer"],
                    state=state,
                )
                # result may be a coroutine or a value
                if asyncio.iscoroutine(result):
                    result = await result
                return float(cast(float, result))
            finally:
                await cleanup_sandbox(sandbox_instance)

        # Run multiple scoring calls concurrently (simulating multiple rollouts)
        results = await asyncio.gather(
            score_once(),
            score_once(),
            score_once(),
        )

        # All should succeed (not just the first one)
        assert all(r == 1.0 for r in results), f"Expected all 1.0, got {results}"


class TestScoreDetailsCaching:
    """
    Tests for full Score details caching (value, answer, explanation, metadata).

    This ensures that InspectSandboxEnv captures and exposes all Score information
    for logging and custom reward functions, not just the float value.
    """

    @pytest.mark.asyncio
    async def test_run_inspect_scorer_returns_full_score(self) -> None:
        """Test that run_inspect_scorer returns the full Score object."""
        from inspect_verifiers_bridge.scoring import run_inspect_scorer

        # Create a custom scorer that returns rich Score data
        @scorer(metrics=[])
        def rich_scorer() -> Scorer:
            async def score(state: TaskState, target: Target) -> Score:
                return Score(
                    value=CORRECT,
                    answer="extracted",
                    explanation="because reasons",
                    metadata={"custom": "data"},
                )

            return score

        # Minimal state for scoring
        state: dict[str, Any] = {
            "info": {
                "inspect_target_raw": "expected",
                "inspect_sample_id": "test-1",
                "inspect_metadata": "{}",
                "inspect_input_raw": "What is 1+1?",
            },
        }

        result = await run_inspect_scorer(
            prompt=[{"role": "user", "content": "What is 1+1?"}],
            completion=[{"role": "assistant", "content": "2"}],
            answer="2",
            state=state,
            scorer=rich_scorer(),
        )

        assert result is not None
        assert result.answer == "extracted"
        assert result.explanation == "because reasons"
        assert result.metadata == {"custom": "data"}
