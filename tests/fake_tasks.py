"""
Fake Inspect tasks for testing the bridge.

These tasks cover different features:
- Simple string input/output
- Different scorer types (exact, includes, pattern)
- Sandbox-based scoring
- Chat message inputs
- Multiple choice
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    exact,
    includes,
    match,
    scorer,
)
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import sandbox

# =============================================================================
# Task 1: Simple math problems with exact match scoring
# =============================================================================
MATH_SAMPLES = [
    Sample(
        input="What is 2 + 2?",
        target="4",
        id="math_1",
        metadata={"difficulty": "easy"},
    ),
    Sample(
        input="What is 10 * 5?",
        target="50",
        id="math_2",
        metadata={"difficulty": "easy"},
    ),
    Sample(
        input="What is 100 / 4?",
        target="25",
        id="math_3",
        metadata={"difficulty": "medium"},
    ),
]


@task
def simple_math() -> Task:
    """Simple math task with exact match scoring."""
    return Task(
        dataset=MATH_SAMPLES,
        solver=[
            system_message("Answer with just the number, nothing else."),
            generate(),
        ],
        scorer=exact(),
    )


# =============================================================================
# Task 2: Trivia with includes scoring
# =============================================================================
TRIVIA_SAMPLES = [
    Sample(
        input="What is the capital of France?",
        target="Paris",
        id="trivia_1",
    ),
    Sample(
        input="Who wrote Romeo and Juliet?",
        target="Shakespeare",
        id="trivia_2",
    ),
    Sample(
        input="What planet is known as the Red Planet?",
        target="Mars",
        id="trivia_3",
    ),
]


@task
def trivia_includes() -> Task:
    """Trivia with includes scoring (answer just needs to contain target)."""
    return Task(
        dataset=TRIVIA_SAMPLES,
        solver=[
            system_message("Answer the trivia question."),
            generate(),
        ],
        scorer=includes(),
    )


# =============================================================================
# Task 3: Multiple choice
# =============================================================================
MC_INPUT_1 = (
    "What color is the sky on a clear day?\nA) Red\nB) Blue\nC) Green\nD) Yellow"
)
MC_INPUT_2 = (
    "Which is the largest ocean?\nA) Atlantic\nB) Indian\nC) Pacific\nD) Arctic"
)

MULTIPLE_CHOICE_SAMPLES = [
    Sample(
        input=MC_INPUT_1,
        target="B",
        choices=["Red", "Blue", "Green", "Yellow"],
        id="mc_1",
    ),
    Sample(
        input=MC_INPUT_2,
        target="C",
        choices=["Atlantic", "Indian", "Pacific", "Arctic"],
        id="mc_2",
    ),
]


@task
def multiple_choice() -> Task:
    """Multiple choice task."""
    return Task(
        dataset=MULTIPLE_CHOICE_SAMPLES,
        solver=[
            system_message("Answer with just the letter (A, B, C, or D)."),
            generate(),
        ],
        scorer=match(),
    )


# =============================================================================
# Task 4: Code execution with sandbox scoring
# =============================================================================
CODE_INPUT_1 = (
    "Write a Python function called 'add' that takes two numbers and returns their sum."
)
CODE_TARGET_1 = [
    "assert add(1, 2) == 3",
    "assert add(0, 0) == 0",
    "assert add(-1, 1) == 0",
]

CODE_INPUT_2 = (
    "Write a Python function called 'double' that takes a number "
    "and returns it doubled."
)
CODE_TARGET_2 = [
    "assert double(5) == 10",
    "assert double(0) == 0",
    "assert double(-3) == -6",
]

CODE_SAMPLES = [
    Sample(
        input=CODE_INPUT_1,
        target=CODE_TARGET_1,
        id="code_1",
        metadata={"test_cases": CODE_TARGET_1},
    ),
    Sample(
        input=CODE_INPUT_2,
        target=CODE_TARGET_2,
        id="code_2",
        metadata={"test_cases": CODE_TARGET_2},
    ),
]


@scorer(metrics=[])
def code_execution_scorer() -> Scorer:
    """Score code by executing it in a sandbox."""

    async def score(state: TaskState, target: Target) -> Score:
        import re

        # Extract code from response
        completion = state.output.completion
        code_match = re.search(r"```python\n(.*?)```", completion, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = completion

        # Build test script
        raw_target = target.target
        test_cases = raw_target if isinstance(raw_target, list) else [raw_target]
        test_script = code + "\n\n" + "\n".join(test_cases)

        try:
            result = await sandbox().exec(
                cmd=["python", "-c", test_script],
                timeout=30,
            )
            return Score(
                value=CORRECT if result.success else INCORRECT,
                explanation=f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
        except Exception as e:
            return Score(value=INCORRECT, explanation=str(e))

    return score


@task
def code_execution() -> Task:
    """Code execution task with sandbox-based scoring."""
    return Task(
        dataset=CODE_SAMPLES,
        solver=[
            system_message(
                "Write Python code. Wrap your code in ```python ... ``` blocks."
            ),
            generate(),
        ],
        scorer=code_execution_scorer(),
        sandbox="docker",
    )


# =============================================================================
# Task 5: Chat message input format
# =============================================================================
CHAT_SAMPLES = [
    Sample(
        input=[
            ChatMessageSystem(content="You are a helpful assistant."),
            ChatMessageUser(content="Say hello!"),
        ],
        target="hello",
        id="chat_1",
    ),
    Sample(
        input=[
            ChatMessageSystem(content="You are a math tutor."),
            ChatMessageUser(content="What is 5 + 5?"),
        ],
        target="10",
        id="chat_2",
    ),
]


@task
def chat_input() -> Task:
    """Task with chat message input format."""
    return Task(
        dataset=CHAT_SAMPLES,
        solver=[generate()],
        scorer=includes(),
    )


# =============================================================================
# Task 6: Custom metadata
# =============================================================================
METADATA_SAMPLES = [
    Sample(
        input="Translate 'hello' to Spanish.",
        target="hola",
        id="meta_1",
        metadata={
            "source_lang": "en",
            "target_lang": "es",
            "category": "greeting",
            "difficulty": 1,
        },
    ),
    Sample(
        input="Translate 'goodbye' to French.",
        target="au revoir",
        id="meta_2",
        metadata={
            "source_lang": "en",
            "target_lang": "fr",
            "category": "greeting",
            "difficulty": 2,
        },
    ),
]


@task
def with_metadata() -> Task:
    """Task with rich metadata on samples."""
    return Task(
        dataset=METADATA_SAMPLES,
        solver=[
            system_message("Translate the phrase. Respond with just the translation."),
            generate(),
        ],
        scorer=includes(),
    )


# =============================================================================
# Task 7: Multi-turn conversation with tool calls
# =============================================================================
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolFunction


# Sample with assistant message containing tool calls
TOOL_CALL_SAMPLES = [
    Sample(
        input=[
            ChatMessageSystem(content="You are a helpful assistant with access to tools."),
            ChatMessageUser(content="What's the weather in Paris?"),
            ChatMessageAssistant(
                content="I'll check the weather for you.",
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        function="get_weather",
                        arguments={"location": "Paris"},
                        type="function",
                    )
                ],
            ),
            ChatMessageTool(
                content="Weather in Paris: 22°C, sunny",
                tool_call_id="call_123",
                function="get_weather",
            ),
            ChatMessageUser(content="Thanks! What about London?"),
        ],
        target="London",
        id="tool_1",
        metadata={"has_tool_calls": True},
    ),
    Sample(
        input=[
            ChatMessageSystem(content="You are a math assistant with a calculator."),
            ChatMessageUser(content="Calculate 15 * 23"),
            ChatMessageAssistant(
                content="Let me calculate that.",
                tool_calls=[
                    ToolCall(
                        id="calc_001",
                        function="calculator",
                        arguments={"expression": "15 * 23"},
                        type="function",
                    )
                ],
            ),
            ChatMessageTool(
                content="345",
                tool_call_id="calc_001",
                function="calculator",
            ),
        ],
        target="345",
        id="tool_2",
        metadata={"has_tool_calls": True},
    ),
]


@task
def with_tool_calls() -> Task:
    """Task with tool call messages in the input."""
    return Task(
        dataset=TOOL_CALL_SAMPLES,
        solver=[generate()],
        scorer=includes(),
    )


# =============================================================================
# Task 8: Assistant-only messages (no user content at start)
# =============================================================================
ASSISTANT_ONLY_SAMPLES = [
    Sample(
        input=[
            ChatMessageSystem(content="Continue the story."),
            ChatMessageAssistant(content="Once upon a time, there was a brave knight..."),
        ],
        target="knight",
        id="assistant_1",
    ),
]


@task
def assistant_only_input() -> Task:
    """Task where input starts with assistant message (continuation)."""
    return Task(
        dataset=ASSISTANT_ONLY_SAMPLES,
        solver=[generate()],
        scorer=includes(),
    )


# =============================================================================
# Task 9: Mixed message types
# =============================================================================
MIXED_MESSAGE_SAMPLES = [
    Sample(
        input=[
            ChatMessageSystem(content="You are a coding assistant."),
            ChatMessageUser(content="Write a hello world function"),
            ChatMessageAssistant(content="```python\ndef hello():\n    print('Hello')\n```"),
            ChatMessageUser(content="Now make it take a name parameter"),
        ],
        target="name",
        id="mixed_1",
    ),
    Sample(
        input=[
            ChatMessageUser(content="First message"),
            ChatMessageAssistant(content="Response 1"),
            ChatMessageUser(content="Second message"),
            ChatMessageAssistant(content="Response 2"),
            ChatMessageUser(content="Final question?"),
        ],
        target="answer",
        id="mixed_2",
        metadata={"turn_count": 5},
    ),
]


@task
def mixed_messages() -> Task:
    """Task with mixed user/assistant messages (multi-turn)."""
    return Task(
        dataset=MIXED_MESSAGE_SAMPLES,
        solver=[generate()],
        scorer=includes(),
    )
