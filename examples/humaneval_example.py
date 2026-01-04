"""
HumanEval Code Generation (Single Turn, With Sandbox)

Demonstrates code generation with sandbox execution for verification.
Model generates code once, then it's executed in a sandbox to check correctness.

Run with:
    vf-eval humaneval_example -p examples/ -m gpt-4o-mini -n 10

Or with Docker sandbox:
    vf-eval humaneval_example -p examples/ -m gpt-4o -n 20 -a '{"sandbox_type": "docker"}'
"""

import verifiers as vf
from inspect_evals.humaneval import humaneval

from inspect_verifiers_bridge import load_environment as bridge_load


def load_environment(
    max_samples: int = 50,
    sandbox_type: str = "local",
) -> vf.Environment:
    """
    Load HumanEval environment for vf-eval.

    Args:
        max_samples: Maximum number of samples to include (default: 50)
        sandbox_type: Sandbox type - "local" for testing, "docker" for production

    Returns:
        An InspectSandboxEnv with max_turns=1 for single-turn code generation.
    """
    return bridge_load(
        humaneval,
        env_type="single_turn",
        sandbox_type=sandbox_type,
        max_samples=max_samples,
    )
