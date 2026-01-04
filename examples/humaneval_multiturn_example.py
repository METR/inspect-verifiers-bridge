"""
HumanEval Multi-Turn (Multi Turn, With Tools)

Demonstrates multi-turn agentic interaction with bash and submit tools.
Model can use the bash tool across multiple turns to test code iteratively,
then call submit when ready with the final answer.

Available tools:
- _bash(command): Execute bash commands in the sandbox
- _submit(answer): Submit final answer and end the rollout

Run with:
    vf-eval humaneval_multiturn_example -p examples/ -m gpt-4o-mini -n 5

Or with more turns:
    vf-eval humaneval_multiturn_example -p examples/ -m gpt-4o -n 10 -a '{"max_turns": 10}'
"""

import verifiers as vf
from inspect_evals.humaneval import humaneval

from inspect_verifiers_bridge import load_environment as bridge_load


def load_environment(
    max_samples: int = 20,
    max_turns: int = 5,
    sandbox_type: str = "local",
) -> vf.Environment:
    """
    Load HumanEval multi-turn environment for vf-eval.

    Args:
        max_samples: Maximum number of samples to include (default: 20)
        max_turns: Maximum conversation turns before forced stop (default: 5)
        sandbox_type: Sandbox type - "local" for testing, "docker" for production

    Returns:
        An InspectSandboxEnv with bash and submit tools for multi-turn coding.
    """
    return bridge_load(
        humaneval,
        env_type="multi_turn",
        max_turns=max_turns,
        sandbox_type=sandbox_type,
        max_samples=max_samples,
    )
