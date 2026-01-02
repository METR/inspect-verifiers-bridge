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
cd inspect_verifiers_bridge
uv sync
```

## Quick Start

```python
from inspect_evals.apps import apps
from inspect_verifiers_bridge import load_environment

# Load an Inspect task as a Verifiers environment
env = load_environment(
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

### `load_environment`

Main function to convert an Inspect task to a Verifiers environment.

```python
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
from inspect_verifiers_bridge.loader import get_inspect_dataset

dataset = get_inspect_dataset(apps, max_samples=50)
print(dataset[0])  # {'prompt': ..., 'answer': ..., 'info': ..., 'id': ...}
```

### `load_inspect_task`

Load and introspect an Inspect task without converting it.

```python
from inspect_verifiers_bridge.tasks import load_inspect_task

task_info = load_inspect_task(apps)
print(f"Task: {task_info.name}")
print(f"Sandbox: {task_info.sandbox_type}")
print(f"Has tools: {task_info.solver_has_tools}")
```

## Scoring Modes

### Live Scoring

Uses Inspect's native scorers directly. Supports all built-in scorers (`exact`, `includes`, `match`, `model_graded_fact`, etc.) and custom scorers.

```python
env = load_environment(
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

env = load_environment(
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
env = load_environment(apps, sandbox_type="docker")

# Local sandbox (faster, less isolated)
env = load_environment(apps, sandbox_type="local")
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
inspect_verifiers_bridge/
├── __init__.py      # Public API (load_environment)
├── loader.py        # Main loader and environment creation
├── tasks.py         # Task introspection utilities
├── dataset.py       # Sample → HuggingFace dataset conversion
├── scoring.py       # Inspect scorer → Verifiers rubric bridge
└── sandbox.py       # Sandbox management for code execution
```

## Control Flow: Loading an Inspect Task

This section provides a detailed walkthrough of what happens when you call `load_environment()`.

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          load_environment(task_fn)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. TASK INTROSPECTION                                                      │
│     load_inspect_task(task_fn) → InspectTaskInfo                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DATASET CONVERSION                                                      │
│     inspect_dataset_to_hf(dataset) → HuggingFace Dataset                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. SANDBOX SETUP (if needed)                                               │
│     SandboxManager(config) → manages sandbox lifecycle                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. RUBRIC CREATION                                                         │
│     build_rubric_from_scorers(scorers) → Verifiers Rubric                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. SYSTEM PROMPT EXTRACTION                                                │
│     _extract_system_prompt(task) → str | None                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. ENVIRONMENT CREATION                                                    │
│     vf.SingleTurnEnv | vf.ToolEnv → ready for training                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Task Introspection

**Entry Point:** `loader.py:54`

```python
task_info = tasks.load_inspect_task(task, **task_kwargs)
```

**What happens in `load_inspect_task()`:**

```python
# tasks.py:40 - Invoke the task function to get a Task object
task = task_fn(**task_kwargs)

# tasks.py:43-48 - Extract sandbox type
sandbox_type = None
if task.sandbox is not None:
    if isinstance(task.sandbox, str):
        sandbox_type = task.sandbox           # e.g., "docker"
    elif hasattr(task.sandbox, "type"):
        sandbox_type = task.sandbox.type      # SandboxSpec object

# tasks.py:51-57 - Normalize scorers to a list
scorers: list[Scorer] = []
if task.scorer is not None:
    if isinstance(task.scorer, list):
        scorers = task.scorer                 # Already a list
    else:
        scorers = [task.scorer]               # Single scorer → list

# tasks.py:60 - Check for tool usage (heuristic)
solver_has_tools = _solver_has_tools(task.solver)
```

**Returns:** `InspectTaskInfo` dataclass with:
- `task`: The Inspect Task object
- `name`: Task name (e.g., "humaneval")
- `dataset`: Inspect Dataset
- `scorers`: List of scorer functions
- `sandbox_type`: "docker" | "local" | None
- `solver_has_tools`: bool
- `metadata`: dict

---

### Step 2: Dataset Conversion

**Entry Point:** `loader.py:57-61`

```python
hf_dataset = ds.inspect_dataset_to_hf(
    task_info.dataset,
    task_name=task_info.name,
    max_samples=max_samples,
)
```

**What happens in `inspect_dataset_to_hf()`:**

```python
# dataset.py:106-110 - Iterate over samples
rows = []
for i, sample in enumerate(dataset):
    if max_samples is not None and i >= max_samples:
        break
    rows.append(sample_to_row(sample, task_name))
```

**Sample conversion (`sample_to_row`):**

```python
# dataset.py:26-38 - Convert input to question string
sample_input = sample.input

if isinstance(sample_input, str):
    question = sample_input

elif hasattr(sample_input, "__iter__"):
    # Chat messages: extract only user content
    # System messages → handled via system_prompt parameter
    # Tool calls, assistant messages → filtered out
    user_messages = [
        msg.content
        for msg in sample_input
        if hasattr(msg, "role") and msg.role == "user"
    ]
    question = "\n".join(user_messages)
```

**Output row structure:**

```python
{
    "question": "What is 2+2?",              # User content only
    "answer": "4",                            # String target
    "info": {
        "inspect_sample_id": "math_1",
        "inspect_target_raw": "4",            # Original target (may be list)
        "inspect_choices": None,              # Multiple choice options
        "inspect_metadata": {"difficulty": "easy"},
        "inspect_sandbox": None,              # Per-sample sandbox config
        "inspect_files": {},                  # Files for sandbox
        "inspect_setup": None,                # Setup script
        "inspect_task_name": "simple_math",
    },
    "id": "math_1",
}
```

> **Note:** The column is named `"question"` (not `"prompt"`) because Verifiers only injects `system_prompt` when the `"prompt"` column doesn't exist. This ensures the system prompt is properly prepended.

---

### Step 3: Sandbox Setup

**Entry Point:** `loader.py:64-75`

```python
effective_sandbox_type = sandbox_type or task_info.sandbox_type
sandbox_manager: SandboxManager | None = None

if scoring_mode == "live" and effective_sandbox_type:
    sandbox_manager = SandboxManager(
        sandbox_config=SandboxConfig(
            sandbox_type=effective_sandbox_type,  # "docker" or "local"
            config=sandbox_config,
        ),
        task_name=task_info.name,
    )
```

**Branch conditions:**

| scoring_mode | sandbox_type | sandbox_manager |
|--------------|--------------|-----------------|
| `"live"` | `"docker"` | ✅ Created |
| `"live"` | `"local"` | ✅ Created |
| `"live"` | `None` | ❌ Not needed |
| `"custom"` | any | ❌ Not needed |

**Sandbox lifecycle (during reward computation):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SandboxManager.get_sandbox(sample_id, info)                                │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │ First call for ID   │───▶│ create_sandbox_for  │                         │
│  │ (not cached)        │    │ _sample()           │                         │
│  └─────────────────────┘    └─────────────────────┘                         │
│           │                           │                                     │
│           │                           ▼                                     │
│           │                 ┌─────────────────────┐                         │
│           │                 │ init_sandbox_envs   │  Sets ContextVars       │
│           │                 │ _sample()           │                         │
│           │                 └─────────────────────┘                         │
│           │                           │                                     │
│           ▼                           ▼                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │ Subsequent calls    │───▶│ Return cached       │                         │
│  │ (same sample_id)    │    │ sandbox             │                         │
│  └─────────────────────┘    └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 4: Rubric Creation

**Entry Point:** `loader.py:78-93`

```python
if scoring_mode == "live":
    if not task_info.scorers:
        raise ValueError("Task has no scorers")
    rubric = scoring.build_rubric_from_scorers(
        task_info.scorers,
        sandbox_manager=sandbox_manager,
    )
elif scoring_mode == "custom":
    if custom_reward_fn is None:
        raise ValueError("custom_reward_fn required")
    rubric = vf.Rubric(funcs=[custom_reward_fn])
```

**Building reward functions from scorers:**

```python
# scoring.py:177-195
reward_funcs = []
for i, scorer in enumerate(scorers):
    # Wrap scorer in a partial function
    func = partial(
        reward_from_inspect_scorer,
        scorer=scorer,
        sandbox_manager=sandbox_manager,
    )

    # Extract unique name from __qualname__
    # e.g., "expression_exact_match.<locals>.score" → "expression_exact_match"
    qualname = getattr(scorer, "__qualname__", "")
    if ".<locals>." in qualname:
        scorer_name = qualname.split(".<locals>.")[0]
    else:
        scorer_name = getattr(scorer, "__name__", ...)

    # Add index for uniqueness (prevents metric overwriting)
    func.__name__ = f"inspect_{scorer_name}_{i}"
    reward_funcs.append(func)

return vf.Rubric(funcs=reward_funcs, weights=weights)
```

**Reward function flow (during training):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  reward_from_inspect_scorer(prompt, completion, answer, state)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │ Extract target from   │           │ Build TaskState for   │
        │ info["inspect_target  │           │ Inspect scorer        │
        │ _raw"]                │           │                       │
        └───────────────────────┘           └───────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ sandbox_manager is not None?        │
                    └─────────────────────────────────────┘
                           │                    │
                         Yes                   No
                           ▼                    ▼
            ┌───────────────────────┐  ┌───────────────────────┐
            │ get_sandbox()         │  │ Call scorer directly  │
            │ async with sandbox_   │  │                       │
            │ context(sandboxes):   │  │                       │
            │   score = scorer()    │  │                       │
            └───────────────────────┘  └───────────────────────┘
                           │                    │
                           └────────┬───────────┘
                                    ▼
                    ┌───────────────────────────────────┐
                    │ _score_to_float(score) → 0.0-1.0 │
                    └───────────────────────────────────┘
```

**Critical: ContextVar Setup for Concurrent Rollouts**

When scoring with sandboxes, the `sandbox_context()` must set all three ContextVars that Inspect expects:

```python
# sandbox.py:155-166
async with sandbox_context(sandboxes):
    # Sets these ContextVars:
    token_envs = sandbox_environments_context_var.set(sandboxes)
    token_default = sandbox_default_context_var.set(default_name)
    token_with = sandbox_with_environments_context_var.set({})

    # Now sandbox() calls inside scorer will work
    yield sandboxes

    # Reset all on exit
    sandbox_environments_context_var.reset(token_envs)
    sandbox_default_context_var.reset(token_default)
    sandbox_with_environments_context_var.reset(token_with)
```

> **Why this matters:** Verifiers runs multiple rollouts concurrently via `asyncio.gather()`. Each coroutine has its own ContextVar context. Without setting all three ContextVars per-coroutine, only the first rollout succeeds.

---

### Step 5: System Prompt Extraction

**Entry Point:** `loader.py:96-97`

```python
if system_prompt is None:
    system_prompt = _extract_system_prompt(task_info.task)
```

**Extraction logic:**

```python
# loader.py:128-174
def _extract_system_prompt(task: Task) -> str | None:
    solver = task.solver
    system_message = None
    prompt_template = None

    # Check if solver is a Chain with multiple solvers
    if hasattr(solver, "_solvers"):
        for s in solver._solvers:
            func_name = getattr(s, "__qualname__", "")
            closure = getattr(s, "__closure__", None)

            # Look for system_message solver
            if "system_message" in func_name and closure:
                for cell in closure:
                    content = getattr(cell, "cell_contents", None)
                    if isinstance(content, str) and len(content) > 10:
                        system_message = content

            # Look for prompt_template solver
            elif "prompt_template" in func_name and closure:
                for cell in closure:
                    content = getattr(cell, "cell_contents", None)
                    if isinstance(content, str) and "{prompt}" in content:
                        # Remove placeholder, keep instructions
                        prompt_template = content.replace("{prompt}", "").strip()

    # Combine if both exist
    if system_message and prompt_template:
        return f"{system_message}\n\n{prompt_template}"
    return prompt_template or system_message or None
```

**Example:** For the MATH task:

```python
# Inspect task definition:
solver=[
    prompt_template("Solve this problem:\n{prompt}\n\nAnswer with ANSWER: <answer>"),
    generate(),
]

# Extracted system_prompt:
"Solve this problem:\n\nAnswer with ANSWER: <answer>"
```

---

### Step 6: Environment Creation

**Entry Point:** `loader.py:100-125`

```python
if env_type == "single_turn":
    return vf.SingleTurnEnv(
        dataset=hf_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
    )

elif env_type == "multi_turn":
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
```

**Environment type selection:**

| env_type | Verifiers Class | Use Case |
|----------|-----------------|----------|
| `"single_turn"` | `SingleTurnEnv` | Q&A, math, classification |
| `"multi_turn"` | `ToolEnv` (no tools) | Conversations, reasoning chains |
| `"tool"` | `ToolEnv` | Tasks requiring tool use |

---

### Complete Example

```python
from inspect_evals.humaneval import humaneval
from inspect_verifiers_bridge import load_environment

# This call triggers the entire flow above
env = load_environment(
    humaneval,                    # Step 1: Introspect task
    scoring_mode="live",          # Step 4: Use Inspect scorers
    sandbox_type="local",         # Step 3: Create SandboxManager
    max_samples=10,               # Step 2: Limit dataset
)

# Result:
# - env.dataset: HuggingFace Dataset with 10 samples
# - env.rubric: Verifiers Rubric wrapping humaneval's verify() scorer
# - env.system_prompt: "Write Python code..."
```

---

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
