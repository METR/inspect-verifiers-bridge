"""
Main loader: Convert Inspect tasks to Verifiers environments.
"""

from typing import Any, Callable, Literal

import verifiers as vf
from inspect_ai import Task

from inspect_verifiers_bridge import dataset as ds
from inspect_verifiers_bridge import scoring, tasks
from inspect_verifiers_bridge.sandbox import SandboxConfig


def load_environment(
    task: Callable[..., Task],
    *,
    scoring_mode: Literal["live", "custom"] = "live",
    custom_reward_fn: Callable[..., float] | None = None,
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
        max_samples: Limit number of samples from dataset
        max_turns: Max turns for sandbox environments
        sandbox_type: Override sandbox type (e.g., "docker", "local")
        sandbox_config: Sandbox configuration file path
        **task_kwargs: Arguments to pass to the Inspect task function

    Returns:
        A Verifiers Environment ready for training.
        - InspectSandboxEnv if task uses sandboxes (per-rollout sandbox lifecycle)
        - SingleTurnEnv for non-sandbox tasks
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

    # Build rubric based on scoring mode
    if scoring_mode == "live":
        if not task_info.scorers:
            raise ValueError(
                f"Task {task_info.name} has no scorers. "
                "Use scoring_mode='custom' with a custom_reward_fn."
            )
        rubric = scoring.build_rubric_from_scorers(task_info.scorers)
    elif scoring_mode == "custom":
        if custom_reward_fn is None:
            raise ValueError("custom_reward_fn is required when scoring_mode='custom'")
        rubric = vf.Rubric(funcs=[custom_reward_fn])
    else:
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

    # Use InspectSandboxEnv for sandbox tasks, SingleTurnEnv otherwise
    if effective_sandbox_type:
        from inspect_verifiers_bridge.environment import InspectSandboxEnv

        return InspectSandboxEnv(
            dataset=hf_dataset,
            rubric=rubric,
            sandbox_config=SandboxConfig(
                sandbox_type=effective_sandbox_type,
                config=sandbox_config,
            ),
            task_name=task_info.name,
            max_turns=max_turns,
        )

    return vf.SingleTurnEnv(dataset=hf_dataset, rubric=rubric)
