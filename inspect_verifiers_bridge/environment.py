"""
Custom environment that integrates Inspect sandboxes with Verifiers lifecycle.

Creates a fresh sandbox per rollout, destroyed after scoring completes.
"""

from typing import Any

import verifiers as vf
from datasets import Dataset as HFDataset

from inspect_verifiers_bridge.sandbox import (
    SandboxConfig,
    SandboxInstance,
    cleanup_sandbox,
    create_sandbox_for_sample,
)


class InspectSandboxEnv(vf.MultiTurnEnv):
    """
    Verifiers environment with per-rollout Inspect sandbox lifecycle.

    Creates a fresh sandbox for each rollout in setup_state,
    destroys it in @vf.cleanup after scoring completes.

    The sandbox is made available to scorers via state["_sandbox_envs"]
    which is picked up by reward_from_inspect_scorer.
    """

    def __init__(
        self,
        dataset: HFDataset,
        rubric: vf.Rubric,
        sandbox_config: SandboxConfig,
        task_name: str,
        max_turns: int,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
        self.sandbox_config = sandbox_config
        self.task_name = task_name
        self._active_instances: dict[int, SandboxInstance] = {}

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """
        Return environment response after model's turn.

        For basic sandbox environments without tools, we return empty list
        (no environment interaction needed - just scoring).
        """
        return []

    async def setup_state(self, state: vf.State) -> vf.State:
        """Create per-rollout sandbox."""
        info = state.get("info", {})

        # Create sandbox for this rollout
        instance = await create_sandbox_for_sample(
            sample_info=info,
            task_name=self.task_name,
            sandbox_config=self.sandbox_config,
        )

        # Store in state for scoring
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
