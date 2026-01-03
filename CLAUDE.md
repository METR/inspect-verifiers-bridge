# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bridge library that converts [Inspect AI](https://inspect.aisi.org.uk/) evaluation tasks into [Verifiers](https://verifiers.readthedocs.io/) environments for RL training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_dataset_conversion.py -v

# Run a specific test
uv run pytest tests/test_dataset_conversion.py::test_function_name -v

# Type checking
uv run basedpyright .

# Linting
uv run ruff check .

# Format code
uv run ruff format .
```

## Architecture

```
inspect_verifiers_bridge/
├── __init__.py      # Public API: exports load_environment()
├── loader.py        # Main entry point - orchestrates task loading
├── tasks.py         # Task introspection - extracts system prompts, templates, scorers
├── dataset.py       # Sample → HuggingFace dataset conversion with template application
├── scoring.py       # Inspect scorer → Verifiers rubric bridge
└── sandbox.py       # Sandbox lifecycle management for code execution tasks
```

### Data Flow

1. **Task Introspection** (`tasks.py`): `load_inspect_task()` invokes the task function and extracts system_prompt, prompt_template, multiple_choice_template, user_messages, scorers, and sandbox config from the solver chain

2. **Dataset Conversion** (`dataset.py`): `inspect_dataset_to_hf()` converts Inspect Samples to HuggingFace dataset rows, applying templates and embedding system prompts into the `prompt` list

3. **Sandbox Setup** (`sandbox.py`): `SandboxManager` handles Docker/local sandbox lifecycle. Critical: must set all three Inspect ContextVars (`sandbox_environments_context_var`, `sandbox_default_context_var`, `sandbox_with_environments_context_var`) for concurrent rollouts

4. **Rubric Creation** (`scoring.py`): `build_rubric_from_scorers()` wraps Inspect scorers in Verifiers reward functions with proper async sandbox context

5. **Environment Creation** (`loader.py`): Returns `vf.SingleTurnEnv` or `vf.ToolEnv` ready for training

### Key Implementation Details

- The `prompt` field in the HuggingFace dataset is always a list of message dicts (not a string)
- System prompts are embedded in each sample's prompt list, not passed separately to the environment
- Sandbox contexts use ContextVars for thread isolation - critical for `asyncio.gather()` concurrent rollouts
- Scorer names are extracted from `__qualname__` to avoid metric name collisions

## Testing

Tests use fake task fixtures in `tests/fake_tasks.py`. Key test files:
- `test_dataset_conversion.py`: Template application and message format tests
- `test_task_introspection.py`: Solver chain extraction tests
- `test_integration.py`: End-to-end environment creation
- `test_regressions.py`: Bug fix regression tests
