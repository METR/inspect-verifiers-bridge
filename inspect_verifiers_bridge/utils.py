"""
Utilities for inspect-verifiers-bridge.
"""

from typing import Any, Callable

from datasets import Dataset as HFDataset
from inspect_ai import Task
from inspect_ai.model._model import ModelName

from inspect_verifiers_bridge import dataset as ds
from inspect_verifiers_bridge import tasks

# A proper ModelName for use in TaskState when we don't have a real model.
# Uses "bridge" as the API provider and "bridge-model" as the model name.
# This allows scorers that access task_state.model.api or .name to work.
BRIDGE_MODEL_NAME = ModelName("bridge/bridge-model")


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
        task_info.task,
        task_name=task_info.name,
        max_samples=max_samples,
    )
