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
    scorer_weights: list[float] | None = None,
    env_type: Literal["single_turn", "multi_turn"] = "single_turn",
    max_samples: int | None = None,
    max_turns: int = 10,
    sandbox_type: str | None = None,
    sandbox_config: str | None = None,
    include_bash: bool = True,
    include_submit: bool | None = None,
    submit_instruction: str
    | None = "You must call submit() when you are done to complete the task.",
    **task_kwargs: Any,
) -> vf.Environment:
    """
    Load an Inspect task and convert it to a Verifiers environment.

    Args:
        task: A callable that returns an Inspect Task
        scoring_mode: How to handle scoring:
            - "live": Use Inspect scorers directly (requires sandbox if task uses one)
            - "custom": Use a custom reward function
        scorer_weights: Weights for each Inspect scorer (must match number of scorers).
            If None, all scorers are weighted equally.
        env_type: Environment type:
            - "single_turn": Single response from model
            - "multi_turn": Multi-turn with tools (requires sandbox)
        max_samples: Limit number of samples from dataset
        max_turns: Max turns for multi-turn environments (default: 10)
        sandbox_type: Override sandbox type (e.g., "docker", "local")
        sandbox_config: Sandbox configuration file path
        include_bash: Include bash tool in sandbox environments (default: True)
        include_submit: Include submit tool for multi-turn termination
            (default: auto, True if env_type="multi_turn")
        submit_instruction: Instruction appended to system prompt for multi-turn
            environments explaining how to use the submit tool.
            - str: Use custom instruction
            - None: No instruction added
        **task_kwargs: Arguments to pass to the Inspect task function

    Returns:
        A Verifiers Environment ready for training.

    Environment selection:
        - single_turn + no sandbox → SingleTurnEnv
        - single_turn + sandbox → InspectSandboxEnv(max_turns=1)
        - multi_turn + no sandbox → NotImplementedError
        - multi_turn + sandbox → InspectSandboxEnv(max_turns=N, submit tool)
    """
    # Load and introspect the task
    task_info = tasks.load_inspect_task(task, **task_kwargs)

    # Determine if submit tool will be enabled
    will_include_submit = (
        include_submit if include_submit is not None else (env_type == "multi_turn")
    )

    hf_dataset = ds.inspect_dataset_to_hf(
        task_info.task,
        task_name=task_info.name,
        max_samples=max_samples,
        additional_system_content=submit_instruction
        if will_include_submit and submit_instruction is not None
        else None,
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
        if scorer_weights is not None and len(scorer_weights) != len(task_info.scorers):
            raise ValueError(
                f"scorer_weights has {len(scorer_weights)} elements but task has "
                f"{len(task_info.scorers)} scorers. They must match."
            )
        rubric = scoring.build_rubric_from_scorers(
            task_info.scorers, weights=scorer_weights
        )
    elif scoring_mode == "custom":
        if custom_reward_fn is None:
            raise ValueError("custom_reward_fn is required when scoring_mode='custom'")
        rubric = vf.Rubric(funcs=[custom_reward_fn])
    else:
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

    # Environment selection based on env_type and sandbox
    if env_type == "single_turn":
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
                scorers=task_info.scorers if scoring_mode == "live" else None,
                max_turns=1,
                include_bash=include_bash,
                include_submit=False,  # No submit for single turn
            )
        return vf.SingleTurnEnv(dataset=hf_dataset, rubric=rubric)

    elif env_type == "multi_turn":
        if not effective_sandbox_type:
            raise NotImplementedError(
                "Multi-turn environment requires a sandbox. "
                "Either use a task with sandbox configuration or specify sandbox_type."
            )
        from inspect_verifiers_bridge.environment import InspectSandboxEnv

        return InspectSandboxEnv(
            dataset=hf_dataset,
            rubric=rubric,
            sandbox_config=SandboxConfig(
                sandbox_type=effective_sandbox_type,
                config=sandbox_config,
            ),
            task_name=task_info.name,
            scorers=task_info.scorers if scoring_mode == "live" else None,
            max_turns=max_turns,
            include_bash=include_bash,
            include_submit=include_submit if include_submit is not None else True,
        )

    else:
        raise ValueError(f"Unknown env_type: {env_type}")
