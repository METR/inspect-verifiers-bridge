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
# System prompt is embedded in each sample's prompt list
print(f"First sample prompt: {env.dataset[0]['prompt'][:2]}...")
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
| `prompt` | `list[dict]` | List of messages (always includes system prompt) |
| `answer` | `str \| None` | Target answer (converted to string) |
| `id` | `str \| int` | Sample identifier |
| `info` | `dict` | All Inspect metadata preserved |

The `prompt` field is always a list of message dicts with `role` and `content` keys. For chat inputs with tool calls, it also preserves `tool_calls` and `tool_call_id`.

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
│     Extracts: system_prompt, prompt_template, multiple_choice_template,     │
│               user_messages, scorers, sandbox_type, unknown_solvers         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DATASET CONVERSION                                                      │
│     inspect_dataset_to_hf(dataset, templates...) → HuggingFace Dataset      │
│     Applies: system_prompt, prompt_template, multiple_choice_template,      │
│              user_messages (with variable substitution from metadata)       │
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
│  5. ENVIRONMENT CREATION                                                    │
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
# tasks.py - Invoke the task function to get a Task object
task = task_fn(**task_kwargs)

# Extract sandbox type
sandbox_type = None
if task.sandbox is not None:
    if isinstance(task.sandbox, str):
        sandbox_type = task.sandbox           # e.g., "docker"
    elif hasattr(task.sandbox, "type"):
        sandbox_type = task.sandbox.type      # SandboxSpec object

# Normalize scorers to a list
scorers: list[Scorer] = []
if task.scorer is not None:
    if isinstance(task.scorer, list):
        scorers = task.scorer
    else:
        scorers = [task.scorer]

# Check for tool usage (heuristic)
solver_has_tools = _solver_has_tools(task.solver)

# Extract solver information (system_prompt, templates, etc.)
solver_info = _extract_solver_info(task)
```

**Solver extraction (`_extract_solver_info`)** inspects the solver chain and extracts content from known built-in solvers:

| Solver | Extracted As | Description |
|--------|--------------|-------------|
| `system_message` | `system_prompt` | System prompt text |
| `prompt_template` | `prompt_template` | Template with `{prompt}` placeholder |
| `chain_of_thought` | `prompt_template` | CoT template (also uses `{prompt}`) |
| `multiple_choice` | `multiple_choice_template` | Template with `{question}`, `{letters}`, `{choices}` |
| `user_message` | `user_messages` | Additional messages (may have `{var}` placeholders) |
| `generate` | - | No extraction needed |
| `use_tools` | - | Tracked via `solver_has_tools` flag |
| `self_critique` | - | Warning emitted (complex multi-model flow) |

Unknown/custom solvers are tracked in `unknown_solvers` and emit a warning.

**Returns:** `InspectTaskInfo` dataclass with:
- `task`: The Inspect Task object
- `name`: Task name (e.g., "humaneval")
- `dataset`: Inspect Dataset
- `scorers`: List of scorer functions
- `sandbox_type`: "docker" | "local" | None
- `solver_has_tools`: bool
- `system_prompt`: str | None
- `prompt_template`: str | None (template with `{prompt}`)
- `multiple_choice_template`: str | None (template with `{question}`, `{letters}`, `{choices}`)
- `user_messages`: list[str] (additional user messages, may have `{var}` placeholders)
- `unknown_solvers`: list[str]
- `metadata`: dict

**Example:** For a chain-of-thought task:

```python
# Inspect task definition:
solver=[
    system_message("You are a math tutor."),
    chain_of_thought(),
    generate(),
]

# Extracted:
# system_prompt: "You are a math tutor."
# prompt_template: "{prompt}\n\nBefore answering, reason step-by-step..."
```

---

### Step 2: Dataset Conversion

**Entry Point:** `loader.py:60-68`

```python
hf_dataset = ds.inspect_dataset_to_hf(
    task_info.dataset,
    task_name=task_info.name,
    system_prompt=effective_system_prompt,
    prompt_template=task_info.prompt_template,
    multiple_choice_template=task_info.multiple_choice_template,
    user_messages=task_info.user_messages or None,
    max_samples=max_samples,
)
```

**What happens in `inspect_dataset_to_hf()`:**

For each sample, `sample_to_row()` builds the prompt as a list of messages, applying templates:

1. **System message**: Added first (if `system_prompt` is provided)
2. **User message**: Formatted based on available templates:
   - If `multiple_choice_template` + choices: Format with `{question}`, `{letters}`, `{choices}`
   - If `prompt_template`: Replace `{prompt}` with sample input
   - Otherwise: Use raw sample input
3. **Additional user messages**: Appended from `user_messages` with `{var}` substitution from metadata

**Template application examples:**

```python
# prompt_template: "Question: {prompt}"
# Input: "What is 2+2?"
# Result: "Question: What is 2+2?"

# multiple_choice_template: "{question}\n\n{choices}\n\nAnswer: {letters}"
# Input: "What color is the sky?"
# Choices: ["Red", "Blue", "Green"]
# Result: "What color is the sky?\n\nA) Red\nB) Blue\nC) Green\n\nAnswer: A, B, C"

# user_message with variable substitution:
# Template: "The text is: {text_to_translate}"
# Metadata: {"text_to_translate": "Hello"}
# Result: "The text is: Hello"
```

**Output row structure:**

```python
{
    "prompt": [                               # Always a list of messages
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?\n\nReason step-by-step..."},
    ],
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

> **Note:** The `"prompt"` column is always a list of message dicts, ready for use. This preserves full conversation history including tool calls, assistant messages, and multi-turn history.

---

### Step 3: Sandbox Setup

**Entry Point:** `loader.py:70-82`

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

**Entry Point:** `loader.py:84-100`

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

### Step 5: Environment Creation

**Entry Point:** `loader.py:102-127`

```python
# System prompt is already embedded in each sample's "prompt" list,
# so we don't pass it separately to the environment
if env_type == "single_turn":
    return vf.SingleTurnEnv(
        dataset=hf_dataset,
        rubric=rubric,
    )

elif env_type == "multi_turn":
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
```

> **Note:** The system prompt is not passed to the environment because it's already included in each sample's `"prompt"` list. Verifiers uses the prompt directly when the `"prompt"` column contains a list of messages.

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
    humaneval,                    # Step 1: Introspect task (extract solvers, scorers)
    scoring_mode="live",          # Step 4: Use Inspect scorers
    sandbox_type="local",         # Step 3: Create SandboxManager
    max_samples=10,               # Step 2: Limit dataset
)

# Result:
# - env.dataset: HuggingFace Dataset with 10 samples
# - env.rubric: Verifiers Rubric wrapping humaneval's verify() scorer
# - Each sample's "prompt" contains [system_message, user_message, ...]
#   with templates applied (prompt_template, multiple_choice, etc.)
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
