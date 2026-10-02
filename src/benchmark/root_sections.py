"""Render context-assembler v2 samples as separate root and reference-context sections.

v2 datasets split each sample into roots (the changed functions, the code under
analysis) and, per root, the related repository code attributed to it. Their flat
``code`` field interleaves both behind ``# =====`` marker comments, which a small
model reads as ordinary code. Here the roots are rendered first, each in its own
labelled code block, then the reference context in a separate section. The prompt
template ends with a short scope reminder so the instruction to judge only the
roots sits right before the answer instead of thousands of tokens earlier.
"""

from pydantic import BaseModel

_ROOTS_HEADER = "## CODE UNDER ANALYSIS"
_CONTEXT_HEADER = "## REFERENCE CONTEXT (read-only)"
_CONTEXT_INTRO = (
    "Related repository code: callers, callees and definitions used by the code under "
    "analysis. Use it only to trace where inputs come from, what called code does and "
    "which checks already exist. Do not report flaws that exist only in this section."
)
_BLOCK_SEPARATOR = "\n\n"


class RootSection(BaseModel):
    """One root of a v2 sample: a changed function and the context attributed to it."""

    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    code: str
    context: str = ""
    """This root's own reference context; empty when none was selected."""


def _root_label(index: int, root: RootSection) -> str:
    label: str = f"ROOT {index}"
    if root.file_path is None:
        return label
    location: str = root.file_path
    if root.line_start is not None and root.line_end is not None:
        location += f", lines {root.line_start}-{root.line_end}"
    return f"{label}: {location}"


def _code_block(code: str) -> str:
    return f"```python\n{code.strip("\n")}\n```"


def render_root_sections(roots: list[RootSection]) -> str:
    """Render the roots, then the non-empty contexts in a separate section.

    Raises:
        ValueError: If ``roots`` is empty.
    """
    if not roots:
        raise ValueError("A root-section sample needs at least one root")
    count: int = len(roots)
    intro: str = (
        "The function under analysis:"
        if count == 1
        else f"{count} functions changed together in one commit. The code is vulnerable "
        "if any of them is vulnerable."
    )
    parts: list[str] = [_ROOTS_HEADER, intro]
    for index, root in enumerate(roots, start=1):
        parts.append(f"### {_root_label(index, root)}\n{_code_block(root.code)}")

    contexts: list[tuple[int, str]] = [
        (index, root.context) for index, root in enumerate(roots, start=1) if root.context.strip()
    ]
    if contexts:
        parts.extend([_CONTEXT_HEADER, _CONTEXT_INTRO])
        for index, context in contexts:
            parts.append(f"### Context for ROOT {index}\n{_code_block(context)}")
    return _BLOCK_SEPARATOR.join(parts)
