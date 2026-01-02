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

from inspect_verifiers_bridge.scoring import build_rubric_from_scorers


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


class TestTransformationOrdering:
    """
    Regression tests for template transformation ordering.

    Bug: Template transformations (prompt_template, multiple_choice) were applied
    in a hardcoded order regardless of the actual order in the solver chain.

    Fix: Use prompt_transformations list from solver introspection to apply
    templates in the correct order as defined in the task's solver chain.
    """

    def test_prompt_template_then_multiple_choice(self) -> None:
        """Test ordering: prompt_template applied before multiple_choice."""
        from inspect_ai.dataset import Sample

        from inspect_verifiers_bridge.dataset import sample_to_row

        sample = Sample(
            input="What is 2+2?",
            target="B",
            choices=["3", "4", "5"],
        )

        # Order: prompt_template first, then multiple_choice
        transformations: list[tuple[str, str]] = [
            ("prompt_template", "QUESTION: {prompt}"),
            ("multiple_choice", "{question}\n\nChoices:\n{choices}\n\nAnswer with {letters}"),
        ]

        row = sample_to_row(
            sample,
            task_name="test",
            prompt_transformations=transformations,
        )

        user_content = row["prompt"][0]["content"]
        # prompt_template wraps original: "QUESTION: What is 2+2?"
        # multiple_choice then formats that as the question
        assert "QUESTION: What is 2+2?" in user_content
        assert "A) 3" in user_content
        assert "B) 4" in user_content

    def test_multiple_choice_then_prompt_template(self) -> None:
        """Test ordering: multiple_choice applied before prompt_template."""
        from inspect_ai.dataset import Sample

        from inspect_verifiers_bridge.dataset import sample_to_row

        sample = Sample(
            input="What is 2+2?",
            target="B",
            choices=["3", "4", "5"],
        )

        # Order: multiple_choice first, then prompt_template
        transformations: list[tuple[str, str]] = [
            ("multiple_choice", "{question}\n\nChoices:\n{choices}\n\nAnswer with {letters}"),
            ("prompt_template", "SOLVE THIS:\n{prompt}"),
        ]

        row = sample_to_row(
            sample,
            task_name="test",
            prompt_transformations=transformations,
        )

        user_content = row["prompt"][0]["content"]
        # multiple_choice formats the question with choices first
        # prompt_template then wraps the entire thing
        assert user_content.startswith("SOLVE THIS:")
        assert "What is 2+2?" in user_content
        assert "A) 3" in user_content

    def test_different_orders_produce_different_results(self) -> None:
        """Test that different transformation orders produce different outputs."""
        from inspect_ai.dataset import Sample

        from inspect_verifiers_bridge.dataset import sample_to_row

        sample = Sample(
            input="Question",
            target="A",
            choices=["Yes", "No"],
        )

        # Use templates that clearly demonstrate order dependence
        # prompt_template adds text AFTER the prompt
        # multiple_choice adds choices AFTER the question
        order_a: list[tuple[str, str]] = [
            ("prompt_template", "{prompt} [THINK STEP BY STEP]"),
            ("multiple_choice", "{question}\n{choices}"),
        ]

        order_b: list[tuple[str, str]] = [
            ("multiple_choice", "{question}\n{choices}"),
            ("prompt_template", "{prompt} [THINK STEP BY STEP]"),
        ]

        row_a = sample_to_row(sample, task_name="test", prompt_transformations=order_a)
        row_b = sample_to_row(sample, task_name="test", prompt_transformations=order_b)

        content_a = row_a["prompt"][0]["content"]
        content_b = row_b["prompt"][0]["content"]

        # Order A: prompt_template first -> "Question [THINK STEP BY STEP]"
        #          then multiple_choice -> "Question [THINK STEP BY STEP]\nA) Yes\nB) No"
        # Order B: multiple_choice first -> "Question\nA) Yes\nB) No"
        #          then prompt_template -> "Question\nA) Yes\nB) No [THINK STEP BY STEP]"
        assert content_a != content_b

        # In order A, [THINK STEP BY STEP] appears BEFORE choices
        assert "[THINK STEP BY STEP]\nA)" in content_a
        # In order B, [THINK STEP BY STEP] appears AFTER choices (at the very end)
        assert content_b.endswith("[THINK STEP BY STEP]")

    def test_prompt_transformations_from_task_introspection(self) -> None:
        """Test that prompt_transformations from task introspection are used."""
        from inspect_ai import Task
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import exact
        from inspect_ai.solver import chain_of_thought, generate

        from inspect_verifiers_bridge.tasks import load_inspect_task

        def task_with_cot() -> Task:
            return Task(
                dataset=[Sample(input="test", target="answer")],
                solver=[chain_of_thought(), generate()],
                scorer=exact(),
            )

        task_info = load_inspect_task(task_with_cot)

        # chain_of_thought should be recorded as a prompt_template transformation
        assert len(task_info.prompt_transformations) > 0
        transform_types = [t[0] for t in task_info.prompt_transformations]
        assert "prompt_template" in transform_types

    def test_legacy_fallback_when_no_transformations(self) -> None:
        """Test that individual templates work when prompt_transformations is None."""
        from inspect_ai.dataset import Sample

        from inspect_verifiers_bridge.dataset import sample_to_row

        sample = Sample(
            input="What is 2+2?",
            target="4",
        )

        # No prompt_transformations, use legacy parameters
        row = sample_to_row(
            sample,
            task_name="test",
            prompt_template="Q: {prompt}",
            prompt_transformations=None,
        )

        user_content = row["prompt"][0]["content"]
        assert user_content == "Q: What is 2+2?"


class TestSandboxScoringConcurrent:
    """
    Test that sandbox-based scoring works with concurrent rollouts.

    This is the actual scenario that was failing: vf-eval runs multiple
    rollouts per example concurrently, and second+ rollouts would fail.
    """

    @pytest.mark.asyncio
    async def test_multiple_concurrent_scoring_calls(self) -> None:
        """Test that multiple concurrent scoring calls all succeed."""
        from inspect_verifiers_bridge import load_environment

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
            reward_fn = env.rubric.funcs[0]
            result = reward_fn(
                prompt=sample["prompt"],
                completion=[{"role": "assistant", "content": correct_code}],
                answer=sample["answer"],
                state={"info": sample["info"]},
            )
            # result may be a coroutine or a value
            if asyncio.iscoroutine(result):
                result = await result
            return float(cast(float, result))

        # Run multiple scoring calls concurrently (simulating multiple rollouts)
        results = await asyncio.gather(
            score_once(),
            score_once(),
            score_once(),
        )

        # All should succeed (not just the first one)
        assert all(r == 1.0 for r in results), f"Expected all 1.0, got {results}"
