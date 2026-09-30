"""Split a context-assembler sample into the target function and its reference context.

llm_scanner context datasets store one flat ``code`` snippet: the function under
analysis mixed with callers, callees and other related code, with comments and
blank lines stripped. Nothing marks which part is under test, so the model tends
to flag flaws in the surrounding code. Here the target is taken verbatim (with
comments) from the function-only dataset, its lines are removed from the context
snippet, and the rest is rendered after the target as reference-only context.
"""

import difflib
import io
import re
import tokenize

_CONTEXT_HEADER = (
    "The code above is the function under analysis. Related repository code "
    "(callers, callees and surrounding definitions) follows for reference only: "
    "use it to trace where inputs come from and what called code does, but judge "
    "only the function above. A flaw that exists only in the reference code does "
    "not make the function under analysis vulnerable."
)
_BLOCK_SEPARATOR = "\n\n"
_TOP_LEVEL_DEFINITION = re.compile(r"(async\s+def|def|class)\b|@")
_MIN_MATCH_LINES = 2
"""Shortest run of equal lines taken as the target; single lines match by chance."""


def strip_python_comments(code: str) -> list[str]:
    """Return ``code`` lines with ``#`` comments removed (strings are left intact).

    Falls back to the unchanged lines when ``code`` does not tokenize.
    """
    lines: list[str] = code.split("\n")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return lines
    for token in reversed(tokens):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col]
    return lines


def top_level_definitions(code: str) -> list[list[str]]:
    """Split ``code`` into top-level definitions as stripped, non-empty, comment-free lines.

    Multi-function targets appear in the context in a different order, so each
    definition is matched separately. Decorators stay with the definition they decorate.
    """
    chunks: list[list[str]] = [[]]
    for line in strip_python_comments(code):
        starts_definition: bool = bool(line) and not line[0].isspace() and bool(
            _TOP_LEVEL_DEFINITION.match(line)
        )
        follows_decorator: bool = bool(chunks[-1]) and chunks[-1][-1].startswith("@")
        if starts_definition and chunks[-1] and not follows_decorator:
            chunks.append([])
        if line.strip():
            chunks[-1].append(line.strip())
    return [chunk for chunk in chunks if chunk]


def remove_target_lines(context: str, target: str) -> tuple[str, float]:
    """Remove the target's lines from a context snippet.

    Lines are compared stripped and without comments, in runs of at least
    ``_MIN_MATCH_LINES`` equal lines per top-level target definition.

    Returns:
        The remaining context (blank lines dropped) and the fraction of the
        target's lines found in the context.
    """
    context_lines: list[str] = context.split("\n")
    stripped: list[str] = [line.strip() for line in context_lines]
    removed: set[int] = set()
    target_line_count: int = 0
    matched_line_count: int = 0
    for definition in top_level_definitions(target):
        target_line_count += len(definition)
        available: list[int] = [
            index for index, line in enumerate(stripped) if line and index not in removed
        ]
        matcher = difflib.SequenceMatcher(
            None, [stripped[index] for index in available], definition, autojunk=False
        )
        for block in matcher.get_matching_blocks():
            if block.size >= _MIN_MATCH_LINES:
                removed.update(available[block.a : block.a + block.size])
                matched_line_count += block.size
    remaining: str = "\n".join(
        line
        for index, line in enumerate(context_lines)
        if index not in removed and line.strip()
    )
    coverage: float = matched_line_count / target_line_count if target_line_count else 0.0
    return remaining, coverage


def render_context_block(context: str | None) -> str:
    """Render reference context to append after the target code; empty context renders ""."""
    if not context or not context.strip():
        return ""
    return _BLOCK_SEPARATOR + _CONTEXT_HEADER + _BLOCK_SEPARATOR + context
