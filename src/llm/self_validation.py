"""Prompt construction for the self-validation confidence pass (P(True) style)."""

from typing import Final

SELF_VALIDATION_QUESTION: Final[str] = (
    "Review your analysis and verdict above. Is your verdict correct for this code?\n"
    "Respond with exactly one JSON object and nothing else:\n"
    '  {"verdict_is_correct": true}\n'
    '  {"verdict_is_correct": false}'
)
# Appended after the generation prompt so the next token is the true/false answer,
# whose probability is read directly instead of sampling a reply.
SELF_VALIDATION_PREFILL: Final[str] = '{"verdict_is_correct":'


def build_self_validation_messages(
    system_prompt: str, user_prompt: str, response: str
) -> list[dict[str, str]]:
    """Replay the original exchange and ask the model whether its verdict is correct."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
        {"role": "user", "content": SELF_VALIDATION_QUESTION},
    ]
