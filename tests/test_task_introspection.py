"""
Tests for task introspection using real inspect_evals tasks.

These tests verify that the bridge correctly detects:
- Sandbox configurations
- Tool usage in solvers
- Various scorer configurations
"""

from inspect_verifiers_bridge.tasks import (
    InspectTaskInfo,
    _solver_has_tools,
    load_inspect_task,
)


class TestSandboxDetection:
    """Test detection of sandbox configurations from real tasks."""

    def test_humaneval_has_docker_sandbox(self):
        """HumanEval requires Docker sandbox for code execution."""
        from inspect_evals.humaneval import humaneval

        task_info = load_inspect_task(humaneval)

        assert task_info.sandbox_type == "docker"

    def test_apps_has_docker_sandbox(self):
        """APPS requires Docker sandbox for code execution."""
        from inspect_evals.apps import apps

        task_info = load_inspect_task(apps)

        assert task_info.sandbox_type == "docker"

    def test_bigcodebench_has_docker_sandbox(self):
        """BigCodeBench requires Docker sandbox for code execution."""
        from inspect_evals.bigcodebench import bigcodebench

        task_info = load_inspect_task(bigcodebench)

        assert task_info.sandbox_type == "docker"

    def test_gsm8k_has_no_sandbox(self):
        """GSM8K is a math task with no sandbox needed."""
        from inspect_evals.gsm8k import gsm8k

        task_info = load_inspect_task(gsm8k)

        assert task_info.sandbox_type is None

    def test_math_has_no_sandbox(self):
        """MATH is a math task with no sandbox needed."""
        from inspect_evals.math import math

        task_info = load_inspect_task(math)

        assert task_info.sandbox_type is None

    def test_gpqa_has_no_sandbox(self):
        """GPQA is a multiple choice task with no sandbox needed."""
        from inspect_evals.gpqa import gpqa_diamond

        task_info = load_inspect_task(gpqa_diamond)

        assert task_info.sandbox_type is None

    def test_mmlu_has_no_sandbox(self):
        """MMLU is a multiple choice task with no sandbox needed."""
        from inspect_evals.mmlu import mmlu_0_shot

        task_info = load_inspect_task(mmlu_0_shot)

        assert task_info.sandbox_type is None


class TestScorerConfigurations:
    """Test detection of various scorer configurations."""

    def test_single_scorer_exact_match(self):
        """GSM8K uses a single str_match scorer."""
        from inspect_evals.gsm8k import gsm8k

        task_info = load_inspect_task(gsm8k)

        assert len(task_info.scorers) == 1
        scorer_name = getattr(task_info.scorers[0], "__qualname__", "")
        assert "match" in scorer_name.lower() or "score" in scorer_name.lower()

    def test_single_scorer_verify(self):
        """HumanEval uses a single verify scorer for code execution."""
        from inspect_evals.humaneval import humaneval

        task_info = load_inspect_task(humaneval)

        assert len(task_info.scorers) == 1
        scorer_name = getattr(task_info.scorers[0], "__qualname__", "")
        assert "verify" in scorer_name.lower() or "score" in scorer_name.lower()

    def test_multiple_scorers_math(self):
        """MATH uses 3 different scorers for expression equivalence."""
        from inspect_evals.math import math

        task_info = load_inspect_task(math)

        assert len(task_info.scorers) == 3

        # Get scorer names
        scorer_names = [
            getattr(s, "__qualname__", "").lower() for s in task_info.scorers
        ]

        # Should have different types of equivalence checks
        assert any(
            "equivalance" in name or "equivalence" in name for name in scorer_names
        )
        assert any("exact" in name for name in scorer_names)
        assert any("sympy" in name for name in scorer_names)

    def test_choice_scorer(self):
        """GPQA uses a choice scorer for multiple choice."""
        from inspect_evals.gpqa import gpqa_diamond

        task_info = load_inspect_task(gpqa_diamond)

        assert len(task_info.scorers) == 1
        scorer_name = getattr(task_info.scorers[0], "__qualname__", "")
        assert "choice" in scorer_name.lower() or "score" in scorer_name.lower()


class TestToolDetectionHeuristic:
    """Test the heuristic for detecting tool usage in solvers."""

    def test_simple_generate_no_tools(self):
        """Simple generate() solver has no tools."""
        from inspect_evals.gsm8k import gsm8k

        task_info = load_inspect_task(gsm8k)

        # GSM8K uses system_message + generate chain, no tools
        assert task_info.solver_has_tools is False

    def test_humaneval_no_tools_in_solver(self):
        """HumanEval solver doesn't use tools (sandbox is separate)."""
        from inspect_evals.humaneval import humaneval

        task_info = load_inspect_task(humaneval)

        # HumanEval uses generate(), sandbox is for scoring not solver
        assert task_info.solver_has_tools is False

    def test_solver_has_tools_heuristic_with_tool_string(self):
        """Test that solver string containing 'tool' is detected."""

        # Create a mock solver that looks like it uses tools
        class MockToolSolver:
            def __str__(self):
                return "use_tools(bash(), python())"

        assert _solver_has_tools(MockToolSolver()) is True

    def test_solver_has_tools_heuristic_with_bash(self):
        """Test that solver string containing 'bash' is detected."""

        class MockBashSolver:
            def __str__(self):
                return "basic_agent(tools=[bash()])"

        assert _solver_has_tools(MockBashSolver()) is True

    def test_solver_has_tools_heuristic_with_react(self):
        """Test that solver string containing 'react' is detected."""

        class MockReactSolver:
            def __str__(self):
                return "react_agent(tools=...)"

        assert _solver_has_tools(MockReactSolver()) is True

    def test_solver_has_tools_returns_false_for_none(self):
        """Test that None solver returns False."""
        assert _solver_has_tools(None) is False

    def test_solver_has_tools_returns_false_for_simple_chain(self):
        """Test that simple solver chain without tools returns False."""

        class MockSimpleSolver:
            def __str__(self):
                return "Chain(system_message, generate)"

        assert _solver_has_tools(MockSimpleSolver()) is False


class TestTaskInfoStructure:
    """Test that InspectTaskInfo contains expected fields."""

    def test_task_info_has_all_fields(self):
        """Test that task info has all required fields."""
        from inspect_evals.gsm8k import gsm8k

        task_info = load_inspect_task(gsm8k)

        assert isinstance(task_info, InspectTaskInfo)
        assert task_info.task is not None
        assert task_info.name is not None and len(task_info.name) > 0
        assert task_info.dataset is not None
        assert isinstance(task_info.scorers, list)
        assert isinstance(task_info.solver_has_tools, bool)
        assert task_info.system_prompt is None or isinstance(task_info.system_prompt, str)
        assert task_info.prompt_template is None or isinstance(task_info.prompt_template, str)
        assert (
            task_info.multiple_choice_template is None
            or isinstance(task_info.multiple_choice_template, str)
        )
        assert isinstance(task_info.user_messages, list)
        assert isinstance(task_info.unknown_solvers, list)
        assert isinstance(task_info.metadata, dict)

    def test_task_name_extraction(self):
        """Test that task name is correctly extracted."""
        from inspect_evals.humaneval import humaneval

        task_info = load_inspect_task(humaneval)

        # HumanEval should have a recognizable name
        assert "humaneval" in task_info.name.lower() or task_info.name != "unknown"

    def test_dataset_is_iterable(self):
        """Test that dataset can be iterated."""
        from inspect_evals.gsm8k import gsm8k

        task_info = load_inspect_task(gsm8k)

        # Should be able to iterate
        samples = list(task_info.dataset)
        assert len(samples) > 0

    def test_metadata_preserved(self):
        """Test that task metadata is preserved."""
        from inspect_evals.math import math

        task_info = load_inspect_task(math)

        # Metadata should be a dict (may be empty)
        assert isinstance(task_info.metadata, dict)


class TestSandboxConfigFormats:
    """Test that different sandbox config formats are handled correctly."""

    def test_string_sandbox_type(self):
        """Test tasks with string sandbox type."""
        from inspect_evals.humaneval import humaneval

        task = humaneval()

        # HumanEval specifies sandbox as string or tuple
        assert task.sandbox is not None

        # Our loader should extract the type
        task_info = load_inspect_task(humaneval)
        assert task_info.sandbox_type == "docker"

    def test_tuple_sandbox_with_config(self):
        """Test tasks with tuple sandbox (type, config)."""
        from inspect_evals.bigcodebench import bigcodebench

        task = bigcodebench()

        # BigCodeBench has sandbox with compose file
        assert task.sandbox is not None

        task_info = load_inspect_task(bigcodebench)
        assert task_info.sandbox_type == "docker"


class TestRealWorldScenarios:
    """Integration tests using real task configurations."""

    def test_code_execution_task_detection(self):
        """Test that code execution tasks are properly identified."""
        from inspect_evals.apps import apps
        from inspect_evals.humaneval import humaneval

        for task_fn in [humaneval, apps]:
            task_info = load_inspect_task(task_fn)

            # Should have sandbox
            assert task_info.sandbox_type is not None, (
                f"{task_fn.__name__} should have sandbox"
            )

            # Should have at least one scorer
            assert len(task_info.scorers) >= 1, (
                f"{task_fn.__name__} should have scorers"
            )

    def test_math_reasoning_task_detection(self):
        """Test that math reasoning tasks are properly identified."""
        from inspect_evals.gsm8k import gsm8k
        from inspect_evals.math import math

        for task_fn in [gsm8k, math]:
            task_info = load_inspect_task(task_fn)

            # Should NOT have sandbox
            assert task_info.sandbox_type is None, (
                f"{task_fn.__name__} should not have sandbox"
            )

            # Should have at least one scorer
            assert len(task_info.scorers) >= 1, (
                f"{task_fn.__name__} should have scorers"
            )

    def test_multiple_choice_task_detection(self):
        """Test that multiple choice tasks are properly identified."""
        from inspect_evals.gpqa import gpqa_diamond
        from inspect_evals.mmlu import mmlu_0_shot

        for task_fn in [gpqa_diamond, mmlu_0_shot]:
            task_info = load_inspect_task(task_fn)

            # Should NOT have sandbox
            assert task_info.sandbox_type is None, (
                f"{task_fn.__name__} should not have sandbox"
            )

            # Should have exactly one choice scorer
            assert len(task_info.scorers) == 1, (
                f"{task_fn.__name__} should have one scorer"
            )
