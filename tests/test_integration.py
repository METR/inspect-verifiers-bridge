"""
Integration tests comparing bridge output with native Inspect.

These tests verify that:
1. Scoring produces the same results as native Inspect
2. Environment creation works correctly
3. Different scoring configurations work correctly
"""

import asyncio
from typing import Any, Callable, Coroutine, TypeVar, cast

import pytest
from datasets import Dataset
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Target, value_to_float
from inspect_ai.solver import TaskState

from inspect_verifiers_bridge import load_environment
from inspect_verifiers_bridge.tasks import InspectTaskInfo, load_inspect_task

from .fake_tasks import (
    chat_input,
    code_execution,
    multiple_choice,
    simple_math,
    trivia_includes,
)

T = TypeVar("T")


def _row(item: Any) -> dict[str, Any]:
    """Convert HuggingFace dataset item to dict for type safety."""
    return dict(item)  # type: ignore[arg-type]


def _get_dataset(env: Any) -> Dataset:
    """Get dataset from environment with type assertion."""
    dataset = env.dataset
    assert dataset is not None
    return cast(Dataset, dataset)


async def _call_reward(reward_fn: Any, **kwargs: Any) -> float:
    """Call reward function and handle async/sync variants."""
    result = reward_fn(**kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    return float(cast(float, result))


class TestScoringComparison:
    """Test that bridge scoring matches native Inspect scoring."""

    def _create_task_state(
        self,
        task_info: InspectTaskInfo,
        sample: Sample,
        completion: str,
    ) -> TaskState:
        """Create a TaskState from sample and completion for native Inspect scoring."""
        target = Target(sample.target)

        # Build messages
        messages: list[Any] = []
        if isinstance(sample.input, str):
            from inspect_ai.model import ChatMessageUser

            messages.append(ChatMessageUser(content=sample.input))
        else:
            messages.extend(sample.input)

        messages.append(ChatMessageAssistant(content=completion))

        return TaskState(
            model="test-model",  # type: ignore[arg-type]
            sample_id=sample.id or 0,
            epoch=0,
            input=sample.input,
            messages=messages,
            target=target,
            output=ModelOutput.from_content(model="test-model", content=completion),
            metadata=sample.metadata or {},
        )

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run an async coroutine."""
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _score_with_bridge(
        self, task_fn: Callable[..., Task], sample_idx: int, completion: str
    ) -> float:
        """Score using the bridge."""
        env = load_environment(task_fn, scoring_mode="live", sandbox_type="local")

        dataset = _get_dataset(env)
        sample = _row(dataset[sample_idx])
        # prompt is now always a list of messages
        prompt_messages = sample["prompt"]
        completion_messages = [{"role": "assistant", "content": completion}]
        state = {"info": sample["info"]}

        return await _call_reward(
            env.rubric.funcs[0],
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

    async def _score_with_inspect(
        self, task_fn: Callable[..., Task], sample_idx: int, completion: str
    ) -> float:
        """Score using native Inspect."""
        task_info = load_inspect_task(task_fn)
        samples = list(task_info.dataset)
        sample = samples[sample_idx]

        task_state = self._create_task_state(task_info, sample, completion)
        target = Target(sample.target)

        scorer = task_info.scorers[0]
        score = await scorer(task_state, target)
        assert score is not None

        converter = value_to_float()
        score_value = score.value
        assert score_value is not None
        return converter(score_value)

    @pytest.mark.asyncio
    async def test_exact_match_correct(self) -> None:
        """Test exact match scoring with correct answer."""
        completion = "4"

        bridge_score = await self._score_with_bridge(simple_math, 0, completion)
        inspect_score = await self._score_with_inspect(simple_math, 0, completion)

        assert bridge_score == inspect_score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_incorrect(self) -> None:
        """Test exact match scoring with incorrect answer."""
        completion = "5"  # Wrong answer for "2 + 2"

        bridge_score = await self._score_with_bridge(simple_math, 0, completion)
        inspect_score = await self._score_with_inspect(simple_math, 0, completion)

        assert bridge_score == inspect_score == 0.0

    @pytest.mark.asyncio
    async def test_includes_correct(self) -> None:
        """Test includes scoring with correct answer."""
        completion = "The capital of France is Paris."

        bridge_score = await self._score_with_bridge(trivia_includes, 0, completion)
        inspect_score = await self._score_with_inspect(trivia_includes, 0, completion)

        assert bridge_score == inspect_score == 1.0

    @pytest.mark.asyncio
    async def test_includes_incorrect(self) -> None:
        """Test includes scoring with incorrect answer."""
        completion = "The capital of France is London."

        bridge_score = await self._score_with_bridge(trivia_includes, 0, completion)
        inspect_score = await self._score_with_inspect(trivia_includes, 0, completion)

        assert bridge_score == inspect_score == 0.0

    @pytest.mark.asyncio
    async def test_match_correct(self) -> None:
        """Test match scoring for multiple choice."""
        completion = "B"  # Correct answer for sky color

        bridge_score = await self._score_with_bridge(multiple_choice, 0, completion)
        inspect_score = await self._score_with_inspect(multiple_choice, 0, completion)

        assert bridge_score == inspect_score == 1.0

    @pytest.mark.asyncio
    async def test_match_incorrect(self) -> None:
        """Test match scoring for multiple choice with wrong answer."""
        completion = "A"  # Wrong answer

        bridge_score = await self._score_with_bridge(multiple_choice, 0, completion)
        inspect_score = await self._score_with_inspect(multiple_choice, 0, completion)

        assert bridge_score == inspect_score == 0.0


class TestEnvironmentCreation:
    """Test environment creation and configuration."""

    def test_single_turn_env_creation(self) -> None:
        """Test creating a SingleTurnEnv."""
        env = load_environment(
            simple_math,
            scoring_mode="live",
            env_type="single_turn",
            sandbox_type="local",
        )

        assert env is not None
        dataset = _get_dataset(env)
        assert len(dataset) == 3
        assert env.rubric is not None
        assert len(env.rubric.funcs) == 1

    def test_system_prompt_in_prompts(self) -> None:
        """Test that system prompt is included in each sample's prompt list."""
        env = load_environment(simple_math, scoring_mode="live", sandbox_type="local")
        dataset = _get_dataset(env)

        # System prompt should be in the first message of each sample
        for hf_sample in dataset:
            sample = _row(hf_sample)
            prompt = sample["prompt"]
            assert isinstance(prompt, list)
            assert len(prompt) >= 1
            # First message should be system with the task's system prompt
            assert prompt[0]["role"] == "system"
            assert "number" in prompt[0]["content"].lower()

    def test_custom_reward_function(self) -> None:
        """Test using a custom reward function."""

        def custom_reward(
            prompt: Any, completion: Any, answer: Any, state: Any, **kwargs: Any
        ) -> float:
            # Simple reward: 1.0 if completion contains the answer
            comp_text = str(completion)
            return 1.0 if answer and answer in comp_text else 0.0

        env = load_environment(
            simple_math,
            scoring_mode="custom",
            custom_reward_fn=custom_reward,
        )

        assert env is not None
        assert len(env.rubric.funcs) == 1

    def test_sandbox_type_override(self) -> None:
        """Test overriding sandbox type."""
        # code_execution task defaults to docker
        task_info = load_inspect_task(code_execution)
        assert task_info.sandbox_type == "docker"

        # But we can override to local
        env = load_environment(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )
        assert env is not None


class TestSandboxScoring:
    """Test sandbox-based scoring."""

    @pytest.mark.asyncio
    async def test_code_execution_correct(self) -> None:
        """Test code execution scoring with correct code."""
        correct_code = """```python
def add(a, b):
    return a + b
```"""

        env = load_environment(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )

        dataset = _get_dataset(env)
        sample = _row(dataset[0])
        prompt_messages = sample["prompt"]
        completion_messages = [{"role": "assistant", "content": correct_code}]
        state = {"info": sample["info"]}

        reward = await _call_reward(
            env.rubric.funcs[0],
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_code_execution_incorrect(self) -> None:
        """Test code execution scoring with incorrect code."""
        wrong_code = """```python
def add(a, b):
    return a - b  # Wrong!
```"""

        env = load_environment(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )

        dataset = _get_dataset(env)
        sample = _row(dataset[0])
        prompt_messages = sample["prompt"]
        completion_messages = [{"role": "assistant", "content": wrong_code}]
        state = {"info": sample["info"]}

        reward = await _call_reward(
            env.rubric.funcs[0],
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

        assert reward == 0.0

    @pytest.mark.asyncio
    @pytest.mark.requires_docker
    async def test_docker_sandbox_scoring(self) -> None:
        """Test scoring with Docker sandbox."""
        correct_code = """```python
def double(x):
    return x * 2
```"""

        env = load_environment(
            code_execution,
            scoring_mode="live",
            sandbox_type="docker",
        )

        dataset = _get_dataset(env)
        sample = _row(dataset[1])  # double function
        prompt_messages = sample["prompt"]
        completion_messages = [{"role": "assistant", "content": correct_code}]
        state = {"info": sample["info"]}

        reward = await _call_reward(
            env.rubric.funcs[0],
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

        assert reward == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_completion(self) -> None:
        """Test scoring with empty completion."""

        async def test() -> float:
            env = load_environment(
                simple_math, scoring_mode="live", sandbox_type="local"
            )
            dataset = _get_dataset(env)
            sample = _row(dataset[0])

            return await _call_reward(
                env.rubric.funcs[0],
                prompt=sample["prompt"],
                completion=[{"role": "assistant", "content": ""}],
                answer=sample["answer"],
                state={"info": sample["info"]},
            )

        reward = asyncio.get_event_loop().run_until_complete(test())
        assert reward == 0.0

    def test_max_samples_limit(self) -> None:
        """Test limiting number of samples."""
        env = load_environment(
            simple_math,
            scoring_mode="live",
            sandbox_type="local",
            max_samples=2,
        )

        dataset = env.dataset
        assert dataset is not None
        assert len(dataset) == 2

    def test_task_with_no_system_prompt(self) -> None:
        """Test task without system message in solver."""
        # chat_input task doesn't have system_message in solver
        env = load_environment(chat_input, scoring_mode="live", sandbox_type="local")
        # Should still work, system_prompt might be None
        assert env is not None
