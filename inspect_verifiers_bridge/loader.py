"""
Main loader: Convert Inspect tasks to Verifiers environments.
"""

from typing import Any, Callable, Literal

import verifiers as vf
from datasets import Dataset as HFDataset
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
    system_prompt: str | None = None,
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
        system_prompt: Override the system prompt (if None, extract from task)
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

    # Convert dataset
    hf_dataset = ds.inspect_dataset_to_hf(
        task_info.dataset,
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

    # Extract system prompt from task if not provided
    if system_prompt is None:
        system_prompt = _extract_system_prompt(task_info.task)

    # Create the appropriate environment type
    if env_type == "single_turn":
        return vf.SingleTurnEnv(
            dataset=hf_dataset,
            rubric=rubric,
            system_prompt=system_prompt,
        )
    elif env_type == "multi_turn":
        # Note: MultiTurnEnv is abstract - for multi-turn we use ToolEnv with no tools
        # or users should create a custom env subclass
        return vf.ToolEnv(
            dataset=hf_dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            tools=[],
            max_turns=max_turns,
        )
    elif env_type == "tool":
        return vf.ToolEnv(
            dataset=hf_dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            tools=[],  # TODO: Extract tools from task
            max_turns=max_turns,
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def _extract_system_prompt(task: Task) -> str | None:
    """Extract system prompt from task solver chain if possible.

    Looks for both system_message and prompt_template solvers.
    """
    solver = task.solver
    system_message = None
    prompt_template = None

    # If it's a Chain, look at the solver functions
    if hasattr(solver, "_solvers"):
        solvers_list = getattr(solver, "_solvers", [])
        for s in solvers_list:
            func_name = getattr(s, "__qualname__", "")
            closure = getattr(s, "__closure__", None)

            # Extract system_message
            if "system_message" in func_name and closure:
                for cell in closure:
                    content = getattr(cell, "cell_contents", None)
                    if isinstance(content, str) and len(content) > 10:
                        system_message = content
                        break

            # Extract prompt_template
            elif "prompt_template" in func_name and closure:
                for cell in closure:
                    content = getattr(cell, "cell_contents", None)
                    # prompt_template template is a string with {prompt} placeholder
                    if (
                        isinstance(content, str)
                        and len(content) > 20
                        and "{prompt}" in content
                    ):
                        # Remove the {prompt} placeholder and use as instructions
                        prompt_template = content.replace("{prompt}", "").strip()
                        break

    # Combine system message and prompt template if both exist
    if system_message and prompt_template:
        return f"{system_message}\n\n{prompt_template}"
    elif prompt_template:
        return prompt_template
    elif system_message:
        return system_message

    return None


def get_inspect_dataset(
    task: Callable[..., Task],
    max_samples: int | None = None,
    **task_kwargs: Any,
) -> HFDataset:
    """
    Convenience function to just get the HuggingFace dataset from an Inspect task.

    Useful for inspection or custom processing.

    Args:
        task: A callable that returns an Inspect Task
        max_samples: Limit number of samples
        **task_kwargs: Arguments to pass to the task function

    Returns:
        HuggingFace Dataset
    """
    task_info = tasks.load_inspect_task(task, **task_kwargs)
    return ds.inspect_dataset_to_hf(
        task_info.dataset,
        task_name=task_info.name,
        max_samples=max_samples,
    )
