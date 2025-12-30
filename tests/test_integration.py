"""
Integration tests comparing bridge output with native Inspect.

These tests verify that:
1. Dataset conversion preserves all information
2. Scoring produces the same results
3. Different input formats work correctly
"""

import asyncio

import pytest
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Target, value_to_float
from inspect_ai.solver import TaskState

from inspect_verifiers_bridge import load_inspect_as_env
from inspect_verifiers_bridge.loader import get_inspect_dataset
from inspect_verifiers_bridge.tasks import load_inspect_task

from .fake_tasks import (
    chat_input,
    code_execution,
    multiple_choice,
    simple_math,
    trivia_includes,
    with_metadata,
)


class TestDatasetConversion:
    """Test that dataset conversion preserves all information."""

    def test_simple_math_dataset(self):
        """Test simple math task dataset conversion."""
        task_info = load_inspect_task(simple_math)
        hf_dataset = get_inspect_dataset(simple_math)

        # Check dataset size
        assert len(hf_dataset) == len(list(task_info.dataset))

        # Check each sample
        for i, (hf_row, inspect_sample) in enumerate(
            zip(hf_dataset, task_info.dataset)
        ):
            assert hf_row["id"] == inspect_sample.id
            assert hf_row["prompt"] == inspect_sample.input
            assert hf_row["answer"] == inspect_sample.target
            assert hf_row["info"]["inspect_sample_id"] == inspect_sample.id
            assert hf_row["info"]["inspect_metadata"] == inspect_sample.metadata

    def test_multiple_choice_dataset(self):
        """Test multiple choice task preserves choices."""
        hf_dataset = get_inspect_dataset(multiple_choice)

        for row in hf_dataset:
            assert row["info"]["inspect_choices"] is not None
            assert len(row["info"]["inspect_choices"]) == 4

    def test_chat_input_dataset(self):
        """Test chat message input format is preserved."""
        hf_dataset = get_inspect_dataset(chat_input)

        for row in hf_dataset:
            # Chat input should be converted to list of dicts
            assert isinstance(row["prompt"], list)
            assert len(row["prompt"]) >= 1
            # Check message structure
            for msg in row["prompt"]:
                assert "role" in msg
                assert "content" in msg

    def test_metadata_preserved(self):
        """Test that metadata is fully preserved."""
        task_info = load_inspect_task(with_metadata)
        hf_dataset = get_inspect_dataset(with_metadata)

        for hf_row, inspect_sample in zip(hf_dataset, task_info.dataset):
            stored_metadata = hf_row["info"]["inspect_metadata"]
            original_metadata = inspect_sample.metadata or {}

            for key, value in original_metadata.items():
                assert key in stored_metadata
                assert stored_metadata[key] == value

    def test_list_target_preserved(self):
        """Test that list targets (like test cases) are preserved."""
        hf_dataset = get_inspect_dataset(code_execution)

        for row in hf_dataset:
            raw_target = row["info"]["inspect_target_raw"]
            assert isinstance(raw_target, list)
            assert all(isinstance(t, str) for t in raw_target)


class TestScoringComparison:
    """Test that bridge scoring matches native Inspect scoring."""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def _create_task_state(
        self,
        task_info,
        sample,
        completion: str,
    ) -> TaskState:
        """Create a TaskState from sample and completion for native Inspect scoring."""
        target = Target(sample.target)

        # Build messages
        messages = []
        if isinstance(sample.input, str):
            from inspect_ai.model import ChatMessageUser

            messages.append(ChatMessageUser(content=sample.input))
        else:
            messages.extend(sample.input)

        messages.append(ChatMessageAssistant(content=completion))

        return TaskState(
            model="test-model",
            sample_id=sample.id or 0,
            epoch=0,
            input=sample.input,
            messages=messages,
            target=target,
            output=ModelOutput.from_content(model="test-model", content=completion),
            metadata=sample.metadata or {},
        )

    def _run_async(self, coro):
        """Run an async coroutine."""
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _score_with_bridge(
        self, task_fn, sample_idx: int, completion: str
    ) -> float:
        """Score using the bridge."""
        env = load_inspect_as_env(task_fn, scoring_mode="live", sandbox_type="local")

        sample = env.dataset[sample_idx]
        prompt_messages = [{"role": "user", "content": sample["prompt"]}]
        completion_messages = [{"role": "assistant", "content": completion}]
        state = {"info": sample["info"]}

        reward_func = env.rubric.funcs[0]
        return await reward_func(
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

    async def _score_with_inspect(
        self, task_fn, sample_idx: int, completion: str
    ) -> float:
        """Score using native Inspect."""
        task_info = load_inspect_task(task_fn)
        samples = list(task_info.dataset)
        sample = samples[sample_idx]

        task_state = self._create_task_state(task_info, sample, completion)
        target = Target(sample.target)

        scorer = task_info.scorers[0]
        score = await scorer(task_state, target)

        converter = value_to_float()
        return converter(score.value)

    @pytest.mark.asyncio
    async def test_exact_match_correct(self):
        """Test exact match scoring with correct answer."""
        completion = "4"

        bridge_score = await self._score_with_bridge(simple_math, 0, completion)
        inspect_score = await self._score_with_inspect(simple_math, 0, completion)

        assert bridge_score == inspect_score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_incorrect(self):
        """Test exact match scoring with incorrect answer."""
        completion = "5"  # Wrong answer for "2 + 2"

        bridge_score = await self._score_with_bridge(simple_math, 0, completion)
        inspect_score = await self._score_with_inspect(simple_math, 0, completion)

        assert bridge_score == inspect_score == 0.0

    @pytest.mark.asyncio
    async def test_includes_correct(self):
        """Test includes scoring with correct answer."""
        completion = "The capital of France is Paris."

        bridge_score = await self._score_with_bridge(trivia_includes, 0, completion)
        inspect_score = await self._score_with_inspect(trivia_includes, 0, completion)

        assert bridge_score == inspect_score == 1.0

    @pytest.mark.asyncio
    async def test_includes_incorrect(self):
        """Test includes scoring with incorrect answer."""
        completion = "The capital of France is London."

        bridge_score = await self._score_with_bridge(trivia_includes, 0, completion)
        inspect_score = await self._score_with_inspect(trivia_includes, 0, completion)

        assert bridge_score == inspect_score == 0.0

    @pytest.mark.asyncio
    async def test_match_correct(self):
        """Test match scoring for multiple choice."""
        completion = "B"  # Correct answer for sky color

        bridge_score = await self._score_with_bridge(multiple_choice, 0, completion)
        inspect_score = await self._score_with_inspect(multiple_choice, 0, completion)

        assert bridge_score == inspect_score == 1.0

    @pytest.mark.asyncio
    async def test_match_incorrect(self):
        """Test match scoring for multiple choice with wrong answer."""
        completion = "A"  # Wrong answer

        bridge_score = await self._score_with_bridge(multiple_choice, 0, completion)
        inspect_score = await self._score_with_inspect(multiple_choice, 0, completion)

        assert bridge_score == inspect_score == 0.0


class TestEnvironmentCreation:
    """Test environment creation and configuration."""

    def test_single_turn_env_creation(self):
        """Test creating a SingleTurnEnv."""
        env = load_inspect_as_env(
            simple_math,
            scoring_mode="live",
            env_type="single_turn",
            sandbox_type="local",
        )

        assert env is not None
        assert len(env.dataset) == 3
        assert env.rubric is not None
        assert len(env.rubric.funcs) == 1

    def test_system_prompt_extraction(self):
        """Test that system prompt is extracted from task."""
        env = load_inspect_as_env(
            simple_math, scoring_mode="live", sandbox_type="local"
        )

        assert env.system_prompt is not None
        assert "number" in env.system_prompt.lower()

    def test_custom_reward_function(self):
        """Test using a custom reward function."""

        def custom_reward(prompt, completion, answer, state, **kwargs):
            # Simple reward: 1.0 if completion contains the answer
            comp_text = str(completion)
            return 1.0 if answer and answer in comp_text else 0.0

        env = load_inspect_as_env(
            simple_math,
            scoring_mode="custom",
            custom_reward_fn=custom_reward,
        )

        assert env is not None
        assert len(env.rubric.funcs) == 1

    def test_sandbox_type_override(self):
        """Test overriding sandbox type."""
        # code_execution task defaults to docker
        task_info = load_inspect_task(code_execution)
        assert task_info.sandbox_type == "docker"

        # But we can override to local
        env = load_inspect_as_env(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )
        assert env is not None


class TestSandboxScoring:
    """Test sandbox-based scoring."""

    @pytest.mark.asyncio
    async def test_code_execution_correct(self):
        """Test code execution scoring with correct code."""
        correct_code = """```python
def add(a, b):
    return a + b
```"""

        env = load_inspect_as_env(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )

        sample = env.dataset[0]
        prompt_messages = [{"role": "user", "content": sample["prompt"]}]
        completion_messages = [{"role": "assistant", "content": correct_code}]
        state = {"info": sample["info"]}

        reward = await env.rubric.funcs[0](
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_code_execution_incorrect(self):
        """Test code execution scoring with incorrect code."""
        wrong_code = """```python
def add(a, b):
    return a - b  # Wrong!
```"""

        env = load_inspect_as_env(
            code_execution,
            scoring_mode="live",
            sandbox_type="local",
        )

        sample = env.dataset[0]
        prompt_messages = [{"role": "user", "content": sample["prompt"]}]
        completion_messages = [{"role": "assistant", "content": wrong_code}]
        state = {"info": sample["info"]}

        reward = await env.rubric.funcs[0](
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_docker_sandbox_scoring(self):
        """Test scoring with Docker sandbox."""
        correct_code = """```python
def double(x):
    return x * 2
```"""

        env = load_inspect_as_env(
            code_execution,
            scoring_mode="live",
            sandbox_type="docker",
        )

        sample = env.dataset[1]  # double function
        prompt_messages = [{"role": "user", "content": sample["prompt"]}]
        completion_messages = [{"role": "assistant", "content": correct_code}]
        state = {"info": sample["info"]}

        reward = await env.rubric.funcs[0](
            prompt=prompt_messages,
            completion=completion_messages,
            answer=sample["answer"],
            state=state,
        )

        assert reward == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_completion(self):
        """Test scoring with empty completion."""

        async def test():
            env = load_inspect_as_env(
                simple_math, scoring_mode="live", sandbox_type="local"
            )
            sample = env.dataset[0]

            reward = await env.rubric.funcs[0](
                prompt=[{"role": "user", "content": sample["prompt"]}],
                completion=[{"role": "assistant", "content": ""}],
                answer=sample["answer"],
                state={"info": sample["info"]},
            )
            return reward

        reward = asyncio.get_event_loop().run_until_complete(test())
        assert reward == 0.0

    def test_max_samples_limit(self):
        """Test limiting number of samples."""
        env = load_inspect_as_env(
            simple_math,
            scoring_mode="live",
            sandbox_type="local",
            max_samples=2,
        )

        assert len(env.dataset) == 2

    def test_task_with_no_system_prompt(self):
        """Test task without system message in solver."""
        # chat_input task doesn't have system_message in solver
        env = load_inspect_as_env(chat_input, scoring_mode="live", sandbox_type="local")
        # Should still work, system_prompt might be None
        assert env is not None
