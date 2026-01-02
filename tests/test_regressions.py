"""
Regression tests for bug fixes.

These tests ensure that previously fixed bugs don't reoccur.
"""

import asyncio
from functools import partial

import pytest
from inspect_ai.scorer import exact, includes, scorer
from inspect_ai.scorer import Score, Scorer, Target, CORRECT, INCORRECT
from inspect_ai.solver import TaskState

from inspect_verifiers_bridge.scoring import build_rubric_from_scorers


class TestScorerNaming:
    """
    Regression tests for scorer naming fix.

    Bug: When multiple scorers have the same __name__ (e.g., all inner functions
    named "score"), Verifiers would overwrite results in aggregated_metrics dict,
    causing only the last scorer's results to be visible.

    Fix: Extract unique names from __qualname__ and add index suffix.
    """

    def test_scorers_get_unique_names(self):
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
            assert s.__name__ == "score"

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

    def test_scorer_names_include_index(self):
        """Test that scorer names include index suffix for guaranteed uniqueness."""
        # Use the same scorer twice
        scorers = [exact(), exact()]

        rubric = build_rubric_from_scorers(scorers)
        names = [f.__name__ for f in rubric.funcs]

        # Should have different indices
        assert "_0" in names[0]
        assert "_1" in names[1]
        assert names[0] != names[1]

    def test_qualname_extraction(self):
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
    async def test_sandbox_context_sets_all_contextvars(self):
        """Test that sandbox_context sets all required ContextVars."""
        from inspect_ai.util._sandbox.context import (
            sandbox_default_context_var,
            sandbox_environments_context_var,
            sandbox_with_environments_context_var,
        )

        from inspect_verifiers_bridge.sandbox import sandbox_context

        # Create a mock sandbox dict
        class MockSandbox:
            pass

        mock_sandboxes = {"default": MockSandbox()}

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
    async def test_sandbox_context_concurrent_access(self):
        """Test that sandbox_context works correctly with concurrent coroutines."""
        from inspect_ai.util._sandbox.context import (
            sandbox_default_context_var,
            sandbox_environments_context_var,
        )

        from inspect_verifiers_bridge.sandbox import sandbox_context

        class MockSandbox:
            def __init__(self, name: str):
                self.name = name

        results = []

        async def check_context(sandbox_name: str):
            mock_sandboxes = {sandbox_name: MockSandbox(sandbox_name)}
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
    async def test_sandbox_context_default_name_selection(self):
        """Test that default sandbox name is correctly selected from dict keys."""
        from inspect_ai.util._sandbox.context import sandbox_default_context_var

        from inspect_verifiers_bridge.sandbox import sandbox_context

        class MockSandbox:
            pass

        # Test with multiple sandboxes - first key should be default
        mock_sandboxes = {
            "first": MockSandbox(),
            "second": MockSandbox(),
        }

        async with sandbox_context(mock_sandboxes):
            default = sandbox_default_context_var.get()
            assert default == "first"

    @pytest.mark.asyncio
    async def test_sandbox_context_empty_sandboxes(self):
        """Test that sandbox_context handles empty sandbox dict."""
        from inspect_ai.util._sandbox.context import sandbox_default_context_var

        from inspect_verifiers_bridge.sandbox import sandbox_context

        # Empty dict should use "default" as fallback
        async with sandbox_context({}):
            default = sandbox_default_context_var.get()
            assert default == "default"


class TestSandboxScoringConcurrent:
    """
    Test that sandbox-based scoring works with concurrent rollouts.

    This is the actual scenario that was failing: vf-eval runs multiple
    rollouts per example concurrently, and second+ rollouts would fail.
    """

    @pytest.mark.asyncio
    async def test_multiple_concurrent_scoring_calls(self):
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

        sample = env.dataset[0]

        async def score_once():
            return await env.rubric.funcs[0](
                prompt=sample["prompt"],
                completion=[{"role": "assistant", "content": correct_code}],
                answer=sample["answer"],
                state={"info": sample["info"]},
            )

        # Run multiple scoring calls concurrently (simulating multiple rollouts)
        results = await asyncio.gather(
            score_once(),
            score_once(),
            score_once(),
        )

        # All should succeed (not just the first one)
        assert all(r == 1.0 for r in results), f"Expected all 1.0, got {results}"
