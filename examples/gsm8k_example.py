"""
GSM8K Math Reasoning (Single Turn, No Sandbox)

Demonstrates the simplest case: pure text generation evaluated by pattern matching.
No sandbox required - just generates an answer and checks if it matches.

Run with:
    vf-eval gsm8k_example -p examples/ -m gpt-4o-mini -n 10

Or with custom args:
    vf-eval gsm8k_example -p examples/ -m gpt-4o -n 50 -a '{"fewshot": 5}'
"""

import verifiers as vf
from inspect_evals.gsm8k import gsm8k

from inspect_verifiers_bridge import load_environment as bridge_load


def load_environment(
    max_samples: int = 100,
    fewshot: int = 3,
) -> vf.Environment:
    """
    Load GSM8K environment for vf-eval.

    Args:
        max_samples: Maximum number of samples to include (default: 100)
        fewshot: Number of few-shot examples in the prompt (default: 3)

    Returns:
        A SingleTurnEnv for math reasoning evaluation.
    """
    return bridge_load(
        gsm8k,
        env_type="single_turn",
        max_samples=max_samples,
        fewshot=fewshot,
    )
