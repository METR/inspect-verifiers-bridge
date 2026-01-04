"""
Custom environment that integrates Inspect sandboxes with Verifiers lifecycle.

Creates a fresh sandbox per rollout, destroyed after scoring completes.
Supports configurable tools (bash, submit) for interactive tasks.
"""

from typing import Any, Callable

import verifiers as vf
from datasets import Dataset as HFDataset

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
    """

    def __init__(
        self,
        dataset: HFDataset,
        rubric: vf.Rubric,
        sandbox_config: SandboxConfig,
        task_name: str,
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
        self._active_instances: dict[int, SandboxInstance] = {}

        # Add bash tool if requested
        if include_bash:
            self.add_tool(self._bash, args_to_skip=["state"])

        # Add submit tool for multi-turn (auto-enable if max_turns > 1)
        if include_submit or (include_submit is None and max_turns > 1):
            self.add_tool(self._submit, args_to_skip=["state"])

    # === Tools ===
    # Note: state is typed as str but actually receives dict from update_tool_args.
    # This is because args_to_skip removes it from schema, but pydantic still validates
    # the signature. Using dict[str, Any] fails strict JSON schema validation.

    async def _bash(self, command: str, state: str = "") -> str:  # type: ignore[assignment]
        """Execute a bash command in the sandbox."""
        state_dict: dict[str, Any] = state  # type: ignore[assignment]
        sandbox_envs = state_dict.get("_sandbox_envs")
        if not sandbox_envs:
            return "Error: No sandbox available"
        sandbox = next(iter(sandbox_envs.values()))
        result = await sandbox.exec(cmd=["bash", "-c", command], timeout=30)
        output = result.stdout
        if result.stderr:
            output = f"{output}\nstderr: {result.stderr}"
        return output or "(no output)"

    async def _submit(self, answer: str, state: str = "") -> str:  # type: ignore[assignment]
        """Submit your final answer to complete the task."""
        state_dict: dict[str, Any] = state  # type: ignore[assignment]
        state_dict["_submitted_answer"] = answer
        return f"Answer submitted: {answer}"

    # === StatefulToolEnv abstract method ===

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject state into tool calls."""
        if tool_name in ("_bash", "_submit"):
            return {**tool_args, "state": state}
        return tool_args

    # === Stop Conditions ===

    @vf.stop(priority=10)
    async def answer_submitted(self, state: vf.State) -> bool:
        """Stop when model calls submit tool."""
        return "_submitted_answer" in state

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

    @vf.cleanup
    async def destroy_sandbox(self, state: vf.State) -> None:
        """Clean up sandbox after rollout (including scoring)."""
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
