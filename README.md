# inspect-bridge

A bridge to convert [Inspect AI](https://inspect.aisi.org.uk/) tasks into [Verifiers](https://verifiers.readthedocs.io/) environments for RL training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Overview

Inspect AI is a framework for evaluating LLMs with a rich ecosystem of evaluation tasks. This bridge allows you to:

- Import existing Inspect tasks and train on them with prime-rl
- Preserve Inspect scoring semantics as Verifiers reward functions
- Support sandbox-based scoring (Docker, local) for code execution tasks
- Convert Inspect datasets to HuggingFace datasets

## Installation

```bash
uv add inspect-bridge
```

Or for development:

```bash
git clone <repo>
cd inspect_bridge
uv sync
```

## Quick Start

```python
from inspect_evals.apps import apps
from inspect_bridge import load_inspect_as_env

# Load an Inspect task as a Verifiers environment
env = load_inspect_as_env(
    apps,
    scoring_mode="live",      # Use Inspect's native scorers
    sandbox_type="docker",    # Use Docker for code execution
    max_samples=100,          # Limit dataset size
)

# The environment is ready for training
print(f"Dataset size: {len(env.dataset)}")
print(f"System prompt: {env.system_prompt[:100]}...")
```

## API Reference

### `load_inspect_as_env`

Main function to convert an Inspect task to a Verifiers environment.

```python
def load_inspect_as_env(
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
    **task_kwargs,
) -> vf.Environment:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `Callable[..., Task]` | required | Inspect task function (e.g., `apps` from inspect_evals) |
| `scoring_mode` | `"live" \| "custom"` | `"live"` | Use Inspect scorers directly or provide custom reward |
| `custom_reward_fn` | `Callable` | `None` | Custom reward function (required if `scoring_mode="custom"`) |
| `env_type` | `str` | `"single_turn"` | Environment type: `single_turn`, `multi_turn`, or `tool` |
| `system_prompt` | `str` | `None` | Override system prompt (auto-extracted if None) |
| `max_samples` | `int` | `None` | Limit number of samples from dataset |
| `max_turns` | `int` | `8` | Max turns for multi-turn/tool environments |
| `sandbox_type` | `str` | `None` | Override sandbox type (`"docker"`, `"local"`) |
| `sandbox_config` | `str` | `None` | Path to sandbox config file |
| `**task_kwargs` | `Any` | - | Arguments passed to the Inspect task function |

### `get_inspect_dataset`

Convenience function to just get the HuggingFace dataset from an Inspect task.

```python
from inspect_bridge.loader import get_inspect_dataset

dataset = get_inspect_dataset(apps, max_samples=50)
print(dataset[0])  # {'prompt': ..., 'answer': ..., 'info': ..., 'id': ...}
```

### `load_inspect_task`

Load and introspect an Inspect task without converting it.

```python
from inspect_bridge.tasks import load_inspect_task

task_info = load_inspect_task(apps)
print(f"Task: {task_info.name}")
print(f"Sandbox: {task_info.sandbox_type}")
print(f"Has tools: {task_info.solver_has_tools}")
```

## Scoring Modes

### Live Scoring

Uses Inspect's native scorers directly. Supports all built-in scorers (`exact`, `includes`, `match`, `model_graded_fact`, etc.) and custom scorers.

```python
env = load_inspect_as_env(
    my_task,
    scoring_mode="live",
    sandbox_type="local",  # or "docker" for isolated execution
)
```

### Custom Scoring

Provide your own reward function:

```python
def my_reward(prompt, completion, answer, state, **kwargs):
    # prompt: list of message dicts
    # completion: list of message dicts (model response)
    # answer: expected answer string
    # state: dict containing 'info' with Inspect metadata
    return 1.0 if answer in str(completion) else 0.0

env = load_inspect_as_env(
    my_task,
    scoring_mode="custom",
    custom_reward_fn=my_reward,
)
```

## Sandbox Support

For tasks that require code execution (like APPS, HumanEval), the bridge supports:

- **Docker sandbox**: Full isolation, recommended for untrusted code
- **Local sandbox**: Faster, runs code directly on host

```python
# Docker sandbox (default for tasks that specify sandbox="docker")
env = load_inspect_as_env(apps, sandbox_type="docker")

# Local sandbox (faster, less isolated)
env = load_inspect_as_env(apps, sandbox_type="local")
```

## Dataset Format

The bridge converts Inspect `Sample` objects to HuggingFace dataset rows:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str \| list[dict]` | Input text or chat messages |
| `answer` | `str \| None` | Target answer (converted to string) |
| `id` | `str \| int` | Sample identifier |
| `info` | `dict` | All Inspect metadata preserved |

The `info` dict contains:
- `inspect_sample_id`: Original sample ID
- `inspect_target_raw`: Original target (may be list, dict, etc.)
- `inspect_choices`: Multiple choice options
- `inspect_metadata`: Sample metadata
- `inspect_sandbox`: Per-sample sandbox config
- `inspect_files`: Files to copy into sandbox
- `inspect_setup`: Setup script
- `inspect_task_name`: Task name

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| String input/output | ✅ | Full support |
| Chat message input | ✅ | Converted to message dicts |
| Multiple choice | ✅ | Choices preserved in info |
| Exact/includes/match scorers | ✅ | Full support |
| Model-graded scorers | ✅ | Requires API access |
| Sandbox scoring | ✅ | Docker and local |
| Custom scorers | ✅ | Full support |
| Tool use | 🚧 | Planned |
| Multi-agent | ❌ | Out of scope |

## Testing

Run the test suite:

```bash
uv run pytest tests/ -v
```

Tests cover:
- Dataset conversion (preserving all fields)
- Scoring comparison (bridge vs native Inspect)
- Environment creation
- Sandbox scoring (local and Docker)
- Edge cases

## Architecture

```
inspect_bridge/
├── __init__.py      # Public API (load_inspect_as_env)
├── loader.py        # Main loader and environment creation
├── tasks.py         # Task introspection utilities
├── dataset.py       # Sample → HuggingFace dataset conversion
├── scoring.py       # Inspect scorer → Verifiers rubric bridge
└── sandbox.py       # Sandbox management for code execution
```

## Development

```bash
# Install dev dependencies
uv sync

# Run linting
uv run ruff check .
uv run basedpyright .

# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format .
```
