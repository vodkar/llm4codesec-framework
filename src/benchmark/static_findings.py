"""Root static-analysis findings: extraction from raw dataset samples and prompt rendering.

Datasets built by llm_scanner with ``--include-static-findings`` carry a
``static_findings`` list per sample. Findings with ``is_root`` lie inside the
function under analysis; only those are used here.
"""

from typing import Any

from pydantic import BaseModel

ROOT_FINDINGS_PLACEHOLDER = "{root_static_findings}"

_FINDINGS_HEADER = (
    "Static analyzer findings in the function under analysis "
    "(automated tools; may be false positives):"
)
_NO_FINDINGS_TEXT = "Static analyzer findings in the function under analysis: none reported."
_BLOCK_SEPARATOR = "\n\n"


class RootStaticFinding(BaseModel):
    """Analyzer finding located inside the function under analysis."""

    tool: str
    rule_id: str
    cwe_id: int | None = None
    severity: str | None = None
    message: str
    file_path: str
    repo_line: int
    flagged_line: str
    """Stripped text of the flagged line in the snippet the finding was resolved against."""


def root_findings_from_raw(
    code: str, raw_findings: list[dict[str, Any]]
) -> list[RootStaticFinding]:
    """Keep ``is_root`` findings, sorted by snippet position, with their flagged line text.

    Args:
        code: Snippet the findings' ``snippet_line`` values index into.
        raw_findings: Raw ``static_findings`` entries from a dataset sample.

    Returns:
        Root findings sorted by ``(snippet_line, tool, rule_id)``.

    Raises:
        ValueError: If a root finding's ``snippet_line`` lies outside ``code``.
    """
    code_lines: list[str] = code.split("\n")
    root_findings: list[dict[str, Any]] = sorted(
        (finding for finding in raw_findings if finding.get("is_root") is True),
        key=lambda finding: (
            int(finding["snippet_line"]),
            str(finding["tool"]),
            str(finding["rule_id"]),
        ),
    )
    findings: list[RootStaticFinding] = []
    for raw in root_findings:
        snippet_line: int = int(raw["snippet_line"])
        if not 1 <= snippet_line <= len(code_lines):
            raise ValueError(
                f"snippet_line {snippet_line} is outside a snippet of {len(code_lines)} lines"
            )
        findings.append(
            RootStaticFinding(
                tool=str(raw["tool"]),
                rule_id=str(raw["rule_id"]),
                cwe_id=raw.get("cwe_id"),
                severity=raw.get("severity"),
                message=str(raw["message"]),
                file_path=str(raw["file_path"]),
                repo_line=int(raw["repo_line"]),
                flagged_line=code_lines[snippet_line - 1].strip(),
            )
        )
    return findings


def render_root_findings_block(findings: list[RootStaticFinding]) -> str:
    """Render findings as the value of the ``{root_static_findings}`` placeholder.

    The value starts with a blank-line separator so it lays out cleanly whether
    the placeholder sits after or before ``{code}``.
    """
    if not findings:
        return _BLOCK_SEPARATOR + _NO_FINDINGS_TEXT
    lines: list[str] = [_FINDINGS_HEADER]
    for index, finding in enumerate(findings, start=1):
        tags: list[str] = [f"{finding.tool} {finding.rule_id}"]
        if finding.cwe_id is not None:
            tags.append(f"CWE-{finding.cwe_id}")
        if finding.severity:
            tags.append(finding.severity)
        message: str = " ".join(finding.message.split())
        lines.append(f"{index}. [{' | '.join(tags)}] {message}")
        lines.append(f"   Flagged line: {finding.flagged_line}")
    return _BLOCK_SEPARATOR + "\n".join(lines)
