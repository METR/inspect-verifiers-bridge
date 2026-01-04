"""
Main loader: Convert Inspect tasks to Verifiers environments.
"""

from typing import Any, Callable, Literal

import verifiers as vf
from inspect_ai import Task

from inspect_verifiers_bridge import dataset as ds
from inspect_verifiers_bridge import scoring, tasks
from inspect_verifiers_bridge.sandbox import SandboxConfig, SandboxManager


def load_environment(
    task: Callable[..., Task],
    *,
    scoring_mode: Literal["live", "custom"] = "live",
    custom_reward_fn: Callable[..., float] | None = None,
    env_type: Literal["single_turn", "multi_turn", "tool"] = "single_turn",
    max_samples: int | None = None,
    max_turns: int = 8,
    sandbox_type: str | None = None,
    sandbox_config: str | None = None,
    **task_kwargs: Any,
) -> vf.Environment:
    """
    Load an Inspect task and convert it to a Verifiers environment.

    Args:
        task: A callable that returns an Inspect Task
        scoring_mode: How to handle scoring:
            - "live": Use Inspect scorers directly (requires sandbox if task uses one)
            - "custom": Use a custom reward function
        custom_reward_fn: Required if scoring_mode="custom". Signature:
            (prompt, completion, answer, state) -> float
        env_type: Type of Verifiers environment to create:
            - "single_turn": SingleTurnEnv for Q&A tasks
            - "multi_turn": MultiTurnEnv for multi-step tasks
            - "tool": ToolEnv for tool-using tasks
        max_samples: Limit number of samples from dataset
        max_turns: Max turns for multi-turn/tool environments
        sandbox_type: Override sandbox type (e.g., "docker", "local")
        sandbox_config: Sandbox configuration file path
        **task_kwargs: Arguments to pass to the Inspect task function

    Returns:
        A Verifiers Environment ready for training
    """
    # Load and introspect the task
    task_info = tasks.load_inspect_task(task, **task_kwargs)

    # Convert dataset using ground truth solver execution
    hf_dataset = ds.inspect_dataset_to_hf(
        task_info.task,
        task_name=task_info.name,
        max_samples=max_samples,
    )

    # Determine if we need a sandbox
    effective_sandbox_type = sandbox_type or task_info.sandbox_type
    sandbox_manager: SandboxManager | None = None

    if scoring_mode == "live" and effective_sandbox_type:
        # Create sandbox manager for sandbox-based scoring
        sandbox_manager = SandboxManager(
            sandbox_config=SandboxConfig(
                sandbox_type=effective_sandbox_type,
                config=sandbox_config,
            ),
            task_name=task_info.name,
        )

    # Build rubric based on scoring mode
    if scoring_mode == "live":
        if not task_info.scorers:
            raise ValueError(
                f"Task {task_info.name} has no scorers. "
                "Use scoring_mode='custom' with a custom_reward_fn."
            )
        rubric = scoring.build_rubric_from_scorers(
            task_info.scorers,
            sandbox_manager=sandbox_manager,
        )
    elif scoring_mode == "custom":
        if custom_reward_fn is None:
            raise ValueError("custom_reward_fn is required when scoring_mode='custom'")
        rubric = vf.Rubric(funcs=[custom_reward_fn])
    else:
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

    # Create the appropriate environment type
    # Note: system_prompt is already included in each sample's prompt list,
    # so we don't pass it separately to the environment
    if env_type == "single_turn":
        return vf.SingleTurnEnv(
            dataset=hf_dataset,
            rubric=rubric,
        )
    elif env_type == "multi_turn":
        # Note: MultiTurnEnv is abstract - for multi-turn we use ToolEnv with no tools
        # or users should create a custom env subclass
        return vf.ToolEnv(
            dataset=hf_dataset,
            rubric=rubric,
            tools=[],
            max_turns=max_turns,
        )
    elif env_type == "tool":
        return vf.ToolEnv(
            dataset=hf_dataset,
            rubric=rubric,
            tools=[],  # TODO: Extract tools from task
            max_turns=max_turns,
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
