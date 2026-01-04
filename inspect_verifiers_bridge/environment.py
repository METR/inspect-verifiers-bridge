"""
Custom environment that integrates Inspect sandboxes with Verifiers lifecycle.

Creates a fresh sandbox per rollout, destroyed after scoring completes.
Supports configurable tools (bash, submit) for interactive tasks.

IMPORTANT: Scoring happens AFTER cleanup in verifiers' lifecycle. We use
post_rollout() to run Inspect scorers and cache results BEFORE the sandbox
is destroyed, following verifiers' recommended pattern.
"""

from typing import Any, Callable

import verifiers as vf
from datasets import Dataset as HFDataset
from inspect_ai.scorer import Scorer

from inspect_verifiers_bridge.sandbox import (
    SandboxConfig,
    SandboxInstance,
    cleanup_sandbox,
    create_sandbox_for_sample,
)


class InspectSandboxEnv(vf.StatefulToolEnv):
    """
    Verifiers environment with per-rollout Inspect sandbox lifecycle.

    Creates a fresh sandbox for each rollout in setup_state,
    destroys it in @vf.cleanup after scoring completes.

    Supports configurable tools:
    - bash: Execute commands in the sandbox (default: enabled)
    - submit: Submit final answer to end multi-turn rollout (auto-enabled for max_turns > 1)

    The sandbox is made available to scorers via state["_sandbox_envs"]
    which is picked up by reward_from_inspect_scorer.

    IMPORTANT: Verifiers calls cleanup BEFORE scoring. We use post_rollout()
    to run Inspect scorers and cache results in state["_cached_scores"] before
    the sandbox is destroyed. The rubric reward functions then return cached values.
    """

    def __init__(
        self,
        dataset: HFDataset,
        rubric: vf.Rubric,
        sandbox_config: SandboxConfig,
        task_name: str,
        scorers: list[Scorer] | None = None,
        max_turns: int = 1,
        tools: list[Callable[..., Any]] | None = None,
        include_bash: bool = True,
        include_submit: bool | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the environment.

        Args:
            dataset: HuggingFace dataset with prompts
            rubric: Verifiers rubric for scoring
            sandbox_config: Configuration for sandbox creation
            task_name: Name of the Inspect task
            scorers: List of Inspect scorers (used for pre-cleanup scoring)
            max_turns: Maximum conversation turns (1 for single-turn)
            tools: Optional list of additional tool functions
            include_bash: Whether to include bash tool (default: True)
            include_submit: Whether to include submit tool (default: auto, True if max_turns > 1)
            **kwargs: Additional arguments passed to StatefulToolEnv
        """
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            tools=tools or [],
            max_turns=max_turns,
            **kwargs,
        )
        self.sandbox_config = sandbox_config
        self.task_name = task_name
        self.scorers = scorers or []
        self._active_instances: dict[int, SandboxInstance] = {}

        # Add bash tool if requested
        if include_bash:
            self.add_tool(self.bash, args_to_skip=["sandbox"])

        # Add submit tool for multi-turn (auto-enable if max_turns > 1)
        if include_submit or (include_submit is None and max_turns > 1):
            self.add_tool(self.submit, args_to_skip=["state"])

    # === Tools ===
    # Note: sandbox and state parameters use Any type because pydantic schema
    # generation runs before args_to_skip removes them. At runtime, they receive
    # the actual SandboxEnvironment and State dict via update_tool_args.

    async def bash(self, command: str, sandbox: Any = None) -> str:
        """Execute a bash command in the sandbox."""
        if sandbox is None:
            return "Error: No sandbox available"
        result = await sandbox.exec(cmd=["bash", "-c", command], timeout=30)
        output = result.stdout
        if result.stderr:
            output = f"{output}\nstderr: {result.stderr}"
        return output or "(no output)"

    async def submit(self, state: Any = None) -> str:
        """Submit to complete the task. Call this when you are done."""
        if state is not None:
            state["_submitted_answer"] = True
        return "Task submitted. Rollout complete."

    # === StatefulToolEnv overrides ===

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject sandbox/state into tool calls."""
        if tool_name == "bash":
            sandbox_envs = state.get("_sandbox_envs", {})
            sandbox = next(iter(sandbox_envs.values()), None)
            return {**tool_args, "sandbox": sandbox}
        if tool_name == "submit":
            return {**tool_args, "state": state}
        return tool_args

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """
        Generate environment response to model's tool calls.

        Handles two special cases:
        1. Model responds without tools (multi-turn mode) - return empty list
        2. Model calls submit - set final_env_response to prevent extra model call
        """
        assert isinstance(messages, list)
        last_message = messages[-1]

        # If no tool calls, return empty list (no-op turn)
        if "tool_calls" not in last_message or last_message["tool_calls"] is None:
            return []

        # Call parent to execute tools
        response = await super().env_response(messages, state, **kwargs)

        # If submit was called, mark this as final response to prevent wasted model call
        # See: https://docs.primeintellect.ai/verifiers/source/environments#final-environment-responses
        if "_submitted_answer" in state:
            state["final_env_response"] = response

        return response

    # === Stop Conditions ===

    @vf.stop(priority=10)
    async def answer_submitted(self, state: vf.State) -> bool:
        """Stop when model calls submit tool."""
        return "_submitted_answer" in state

    @vf.stop(priority=0)
    async def no_tools_called(self, state: vf.State) -> bool:
        """
        Override ToolEnv's no_tools_called to allow multi-turn without tool use.

        In single-turn mode (max_turns=1), we keep the default behavior: stop if
        the model doesn't call tools.

        In multi-turn mode (max_turns>1), we disable this stop condition so the
        model must either call submit or reach max_turns to end the rollout.
        """
        if self.max_turns > 1:
            # Multi-turn: don't stop just because no tools were called
            return False
        # Single-turn: use default ToolEnv behavior
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        is_assistant = last_message["role"] == "assistant"
        no_tool_calls = (
            "tool_calls" not in last_message or last_message["tool_calls"] is None
        )
        return is_assistant and no_tool_calls

    # === Lifecycle ===

    async def setup_state(self, state: vf.State) -> vf.State:
        """Create per-rollout sandbox."""
        info = state.get("info", {})

        # Create sandbox for this rollout
        instance = await create_sandbox_for_sample(
            sample_info=info,
            task_name=self.task_name,
            sandbox_config=self.sandbox_config,
        )

        # Store in state for scoring and tools
        state["_sandbox_instance"] = instance
        state["_sandbox_envs"] = instance.environments

        # Track for teardown
        self._active_instances[id(instance)] = instance

        return await super().setup_state(state)

    async def post_rollout(self, state: vf.State) -> None:
        """
        Run Inspect scorers and cache results BEFORE sandbox is destroyed.

        Verifiers calls cleanup before scoring, so we must compute scores here
        while the sandbox is still alive. Results are cached in state["_cached_scores"]
        and the rubric reward functions will return these cached values.
        """
        if not self.scorers:
            return

        # Import here to avoid circular dependency
        from inspect_verifiers_bridge.scoring import (
            _get_scorer_name,
            reward_from_inspect_scorer,
        )

        sandbox_envs = state.get("_sandbox_envs")
        if sandbox_envs is None:
            return

        # Prepare arguments for the reward function
        prompt = state.get("prompt", [])
        completion = state.get("completion", [])
        answer = state.get("answer")

        # Cache scores for each scorer
        cached_scores: dict[str, float] = {}
        for i, scorer in enumerate(self.scorers):
            scorer_name = _get_scorer_name(scorer)
            cache_key = f"inspect_{scorer_name}_{i}"
            try:
                score = await reward_from_inspect_scorer(
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    scorer=scorer,
                )
                cached_scores[cache_key] = score
            except Exception as e:
                # Log error but continue with other scorers
                self.logger.error(f"Error in post_rollout scoring for {cache_key}: {e}")
                cached_scores[cache_key] = 0.0

        state["_cached_scores"] = cached_scores

    @vf.cleanup
    async def destroy_sandbox(self, state: vf.State) -> None:
        """Clean up sandbox after rollout. Runs post_rollout first to cache scores."""
        # Run scoring before destroying sandbox
        await self.post_rollout(state)

        instance: SandboxInstance | None = state.get("_sandbox_instance")
        if instance is None:
            return

        instance_id = id(instance)
        await cleanup_sandbox(instance)
        self._active_instances.pop(instance_id, None)

    @vf.teardown
    async def teardown_all_sandboxes(self) -> None:
        """Clean up any remaining sandboxes on process exit."""
        for instance in list(self._active_instances.values()):
            await cleanup_sandbox(instance)
        self._active_instances.clear()
