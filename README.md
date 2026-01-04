# inspect-verifiers-bridge

A bridge to convert [Inspect AI](https://inspect.aisi.org.uk/) tasks into [Verifiers](https://verifiers.readthedocs.io/) environments for RL training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Overview

Inspect AI is a framework for evaluating LLMs with a rich ecosystem of evaluation tasks. This bridge allows you to:

- Import existing Inspect tasks and train on them with prime-rl
- Preserve Inspect scoring semantics as Verifiers reward functions
- Support sandbox-based scoring (Docker, local) for code execution tasks
- Multi-turn agentic environments with bash and submit tools
- Convert Inspect datasets to HuggingFace datasets

## Installation

```bash
uv add inspect-verifiers-bridge
```

Or for development:

```bash
git clone <repo>
cd inspect-verifiers-bridge
uv sync
```

## Quick Start

```python
from inspect_evals.humaneval import humaneval
from inspect_verifiers_bridge import load_environment

# Load an Inspect task as a Verifiers environment
env = load_environment(
    humaneval,
    env_type="single_turn",   # or "multi_turn" for agentic tasks
    scoring_mode="live",      # Use Inspect's native scorers
    sandbox_type="local",     # Use local sandbox for code execution
    max_samples=100,          # Limit dataset size
)

# The environment is ready for training
print(f"Environment: {type(env).__name__}")
print(f"Dataset size: {len(env.dataset)}")
```

## Examples

The `examples/` directory contains vf-eval compatible scripts for each environment type:

```bash
# Single turn, no sandbox (GSM8K math reasoning)
vf-eval gsm8k_example -p examples/ -m gpt-4o-mini -n 10

# Single turn with sandbox (HumanEval code generation)
vf-eval humaneval_example -p examples/ -m gpt-4o-mini -n 10

# Multi-turn with tools (HumanEval agentic)
vf-eval humaneval_multiturn_example -p examples/ -m gpt-4o-mini -n 5
```

Each example exports a `load_environment()` function that vf-eval can call:

| Example | Environment | Tools | Use Case |
|---------|-------------|-------|----------|
| `gsm8k_example.py` | `SingleTurnEnv` | None | Math reasoning |
| `humaneval_example.py` | `InspectSandboxEnv` | `_bash` | Code generation |
| `humaneval_multiturn_example.py` | `InspectSandboxEnv` | `_bash`, `_submit` | Agentic coding |

## API Reference

### `load_environment`

Main function to convert an Inspect task to a Verifiers environment.

```python
def load_environment(
    task: Callable[..., Task],
    *,
    scoring_mode: Literal["live", "custom"] = "live",
    custom_reward_fn: Callable[..., float] | None = None,
    env_type: Literal["single_turn", "multi_turn"] = "single_turn",
    max_samples: int | None = None,
    max_turns: int = 10,
    sandbox_type: str | None = None,
    sandbox_config: str | None = None,
    include_bash: bool = True,
    include_submit: bool | None = None,
    **task_kwargs,
) -> vf.Environment:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `Callable[..., Task]` | required | Inspect task function (e.g., `humaneval` from inspect_evals) |
| `scoring_mode` | `"live" \| "custom"` | `"live"` | Use Inspect scorers directly or provide custom reward |
| `custom_reward_fn` | `Callable` | `None` | Custom reward function (required if `scoring_mode="custom"`) |
| `env_type` | `"single_turn" \| "multi_turn"` | `"single_turn"` | Environment type (multi_turn requires sandbox) |
| `max_samples` | `int` | `None` | Limit number of samples from dataset |
| `max_turns` | `int` | `10` | Max turns for multi-turn environments |
| `sandbox_type` | `str` | `None` | Override sandbox type (`"docker"`, `"local"`) |
| `sandbox_config` | `str` | `None` | Path to sandbox config file |
| `include_bash` | `bool` | `True` | Include bash tool in sandbox environments |
| `include_submit` | `bool` | `None` | Include submit tool (auto: True for multi_turn) |
| `**task_kwargs` | `Any` | - | Arguments passed to the Inspect task function |

**Environment Selection:**

| env_type | sandbox | Result |
|----------|---------|--------|
| `single_turn` | No | `SingleTurnEnv` |
| `single_turn` | Yes | `InspectSandboxEnv(max_turns=1)` |
| `multi_turn` | No | `NotImplementedError` |
| `multi_turn` | Yes | `InspectSandboxEnv(max_turns=N)` with submit tool |

### `load_inspect_task`

Load and introspect an Inspect task without converting it.

```python
from inspect_verifiers_bridge.tasks import load_inspect_task

task_info = load_inspect_task(humaneval)
print(f"Task: {task_info.name}")
print(f"Sandbox: {task_info.sandbox_type}")
print(f"Scorers: {len(task_info.scorers)}")
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

For tasks that require code execution (like HumanEval, APPS), the bridge supports:

- **Docker sandbox**: Full isolation, recommended for untrusted code
- **Local sandbox**: Faster, runs code directly on host

```python
# Docker sandbox (default for tasks that specify sandbox="docker")
env = load_environment(humaneval, sandbox_type="docker")

# Local sandbox (faster, less isolated)
env = load_environment(humaneval, sandbox_type="local")
```

### Per-Rollout Sandbox Lifecycle

When using `InspectSandboxEnv`, each rollout gets a fresh sandbox:

1. **setup_state()**: Creates sandbox for this rollout
2. **Rollout loop**: Model interacts with bash/submit tools
3. **Scoring**: Scorer runs with sandbox context
4. **cleanup()**: Sandbox destroyed after scoring completes

This ensures no state contamination between rollouts and supports concurrent execution via `asyncio.gather()`.

## Multi-Turn Environments

For agentic tasks, use `env_type="multi_turn"`:

```python
env = load_environment(
    humaneval,
    env_type="multi_turn",
    max_turns=10,
    sandbox_type="local",
)
```

**Available tools:**

| Tool | Description |
|------|-------------|
| `_bash` | Execute bash commands in the sandbox |
| `_submit` | Submit final answer and end the rollout |

The model can use these tools iteratively until it calls `_submit` or reaches `max_turns`.

## Dataset Format

The bridge converts Inspect `Sample` objects to HuggingFace dataset rows:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `list[dict]` | List of messages (always includes system prompt) |
| `answer` | `str \| None` | Target answer (converted to string) |
| `id` | `str \| int` | Sample identifier (auto-generated if not set) |
| `info` | `dict` | All Inspect metadata preserved |

The `prompt` field is always a list of message dicts with `role` and `content` keys. For chat inputs with tool calls, it also preserves `tool_calls` and `tool_call_id`.

The `info` dict contains:
- `inspect_sample_id`: Original sample ID
- `inspect_input_raw`: Original input (pre-solver)
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
| Single-turn environments | ✅ | `SingleTurnEnv` or `InspectSandboxEnv` |
| Multi-turn with tools | ✅ | `InspectSandboxEnv` with bash/submit |
| Per-rollout sandbox lifecycle | ✅ | Fresh sandbox per rollout |
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
- Concurrent sandbox execution
- Edge cases

## Architecture

```
inspect_verifiers_bridge/
├── __init__.py      # Public API (load_environment)
├── loader.py        # Main loader and environment selection
├── environment.py   # InspectSandboxEnv with per-rollout lifecycle
├── tasks.py         # Task introspection utilities
├── dataset.py       # Sample → HuggingFace dataset conversion
├── ground_truth.py  # Solver execution for prompt construction
├── scoring.py       # Inspect scorer → Verifiers rubric bridge
└── sandbox.py       # Sandbox creation and context management

examples/
├── gsm8k_example.py              # Single turn, no sandbox
├── humaneval_example.py          # Single turn with sandbox
└── humaneval_multiturn_example.py # Multi-turn with tools
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
│     Extracts: scorers, sandbox_type, solver_has_tools                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DATASET CONVERSION                                                      │
│     inspect_dataset_to_hf(task, task_name) → HuggingFace Dataset            │
│     Runs solver pipeline (without model) to get ground truth prompts        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. RUBRIC CREATION                                                         │
│     build_rubric_from_scorers(scorers) → Verifiers Rubric                   │
│     Wraps Inspect scorers in reward functions with sandbox context          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. ENVIRONMENT CREATION                                                    │
│     Based on env_type and sandbox:                                          │
│     - SingleTurnEnv (no sandbox, single turn)                               │
│     - InspectSandboxEnv (sandbox, single or multi-turn)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Task Introspection

**Entry Point:** `loader.py`

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
```

**Returns:** `InspectTaskInfo` dataclass with:
- `task`: The Inspect Task object
- `name`: Task name (e.g., "humaneval")
- `dataset`: Inspect Dataset
- `scorers`: List of scorer functions
- `sandbox_type`: "docker" | "local" | None
- `solver_has_tools`: bool

---

### Step 2: Dataset Conversion

**Entry Point:** `loader.py`

```python
hf_dataset = ds.inspect_dataset_to_hf(
    task_info.task,
    task_name=task_info.name,
    max_samples=max_samples,
)
```

**What happens in `inspect_dataset_to_hf()`:**

For each sample, the solver pipeline is executed (without calling the model) to produce the ground truth prompt:

1. **Ground truth execution**: Runs all prompt-engineering solvers (`system_message`, `prompt_template`, `chain_of_thought`, etc.) stopping before `generate()`
2. **Message extraction**: Extracts the resulting `state.messages` list
3. **Serialization**: Converts `ChatMessage` objects to dicts

**Auto-ID generation:** Samples without IDs (like GSM8K) automatically get index-based IDs.

**Output row structure:**

```python
{
    "prompt": [                               # Always a list of messages
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a function to add two numbers..."},
    ],
    "answer": "def add(a, b): return a + b",  # String target
    "info": {
        "inspect_sample_id": "0",
        "inspect_input_raw": "Write a function...",  # Pre-solver input
        "inspect_target_raw": "def add...",
        "inspect_metadata": {},
        "inspect_sandbox": None,
        "inspect_files": {},
        "inspect_setup": None,
        "inspect_task_name": "humaneval",
    },
    "id": "0",
}
```

---

### Step 3: Rubric Creation

**Entry Point:** `loader.py`

```python
if scoring_mode == "live":
    rubric = scoring.build_rubric_from_scorers(task_info.scorers)
elif scoring_mode == "custom":
    rubric = vf.Rubric(funcs=[custom_reward_fn])
```

**Building reward functions from scorers:**

```python
# scoring.py
reward_funcs = []
for i, scorer in enumerate(scorers):
    # Wrap scorer in a partial function
    func = partial(reward_from_inspect_scorer, scorer=scorer)

    # Extract unique name from __qualname__
    # e.g., "verify.<locals>.score" → "verify"
    scorer_name = _get_scorer_name(scorer)

    # Add index for uniqueness (prevents metric overwriting)
    func.__name__ = f"inspect_{scorer_name}_{i}"
    reward_funcs.append(func)

return vf.Rubric(funcs=reward_funcs)
```

**Reward function flow (during training):**

```
reward_from_inspect_scorer(prompt, completion, answer, state)
    │
    ├── Build TaskState from Verifiers state
    │   - Uses state["info"]["inspect_input_raw"] for TaskState.input
    │   - Converts messages to Inspect ChatMessage objects
    │
    ├── Get sandbox context from state["_sandbox_envs"] (if present)
    │
    └── Call Inspect scorer within sandbox context
        │
        └── Convert Score to float (0.0-1.0)
```

**Critical: ContextVar Setup for Concurrent Rollouts**

When scoring with sandboxes, the `sandbox_context()` sets all three ContextVars that Inspect expects:

```python
# sandbox.py
async with sandbox_context(sandboxes):
    # Sets these ContextVars:
    sandbox_environments_context_var.set(sandboxes)
    sandbox_default_context_var.set(default_name)
    sandbox_with_environments_context_var.set({})

    # Now sandbox() calls inside scorer will work
    yield sandboxes
```

> **Why this matters:** Verifiers runs multiple rollouts concurrently via `asyncio.gather()`. Each coroutine has its own ContextVar context. Without setting all three ContextVars per-coroutine, only the first rollout succeeds.

---

### Step 4: Environment Creation

**Entry Point:** `loader.py`

```python
if env_type == "single_turn":
    if effective_sandbox_type:
        return InspectSandboxEnv(
            dataset=hf_dataset,
            rubric=rubric,
            sandbox_config=SandboxConfig(...),
            task_name=task_info.name,
            max_turns=1,
            include_bash=include_bash,
            include_submit=False,
        )
    return vf.SingleTurnEnv(dataset=hf_dataset, rubric=rubric)

elif env_type == "multi_turn":
    if not effective_sandbox_type:
        raise NotImplementedError("Multi-turn requires sandbox")
    return InspectSandboxEnv(
        dataset=hf_dataset,
        rubric=rubric,
        sandbox_config=SandboxConfig(...),
        task_name=task_info.name,
        max_turns=max_turns,
        include_bash=include_bash,
        include_submit=True,  # Always for multi-turn
    )
```

### InspectSandboxEnv Lifecycle

`InspectSandboxEnv` extends `vf.StatefulToolEnv` and manages per-rollout sandbox lifecycle:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Rollout Lifecycle                                                          │
│                                                                             │
│  1. setup_state(state)                                                      │
│     └── create_sandbox_for_sample() → SandboxInstance                       │
│     └── state["_sandbox_envs"] = instance.environments                      │
│                                                                             │
│  2. Rollout loop (until @vf.stop triggers)                                  │
│     ├── Model generates response                                            │
│     ├── If tool_calls in response:                                          │
│     │   └── env_response() → calls _bash/_submit via update_tool_args()     │
│     └── Check stop conditions:                                              │
│         ├── max_turns_reached                                               │
│         └── answer_submitted (if _submit was called)                        │
│                                                                             │
│  3. Scoring                                                                 │
│     └── Scorer accesses sandbox via state["_sandbox_envs"]                  │
│                                                                             │
│  4. @vf.cleanup: destroy_sandbox(state)                                     │
│     └── cleanup_sandbox(instance) → removes container/files                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tools implementation:**

```python
class InspectSandboxEnv(vf.StatefulToolEnv):
    async def _bash(self, command: str, state) -> str:
        """Execute bash command in sandbox."""
        sandbox = next(iter(state["_sandbox_envs"].values()))
        result = await sandbox.exec(cmd=["bash", "-c", command])
        return result.stdout or "(no output)"

    async def _submit(self, answer: str, state) -> str:
        """Submit answer and trigger rollout end."""
        state["_submitted_answer"] = answer
        return f"Answer submitted: {answer}"

    @vf.stop(priority=10)
    async def answer_submitted(self, state) -> bool:
        """Stop when model calls submit tool."""
        return "_submitted_answer" in state
```

---

### Complete Example

```python
from inspect_evals.humaneval import humaneval
from inspect_verifiers_bridge import load_environment

# This call triggers the entire flow above
env = load_environment(
    humaneval,                    # Step 1: Introspect task
    env_type="multi_turn",        # Step 4: Create InspectSandboxEnv
    scoring_mode="live",          # Step 3: Use Inspect scorers
    sandbox_type="local",         # Step 4: Configure sandbox
    max_samples=10,               # Step 2: Limit dataset
    max_turns=5,                  # Step 4: Configure turns
)

# Result:
# - env.dataset: HuggingFace Dataset with 10 samples
# - env.rubric: Verifiers Rubric wrapping humaneval's verify() scorer
# - env.oai_tools: [_bash, _submit] tools in OpenAI format
# - Each rollout gets fresh sandbox, cleaned up after scoring
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
