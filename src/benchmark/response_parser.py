import re
from abc import ABC, abstractmethod
from typing import Final

from benchmark.enums import TaskType

_FINAL_ANSWER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[\[FINAL_ANSWER:\s*(.*?)\s*\]\]", re.IGNORECASE | re.DOTALL
)
_CWE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
_BINARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(VULNERABLE|SAFE)\b", re.IGNORECASE
)
_BINARY_FALLBACK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(YES|NO|TRUE|FALSE|FOUND|DETECTED|NONE|CLEAN)\b", re.IGNORECASE
)
_VULDETECTBENCH_BINARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(YES|NO)\b", re.IGNORECASE
)
_VULDETECTBENCH_CHOICE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:FINAL\s*ANSWER|ANSWER|OPTION|CHOICE|VERDICT)?\s*[:\-]?\s*([ABCDE])(?:[.:])?$",
    re.IGNORECASE,
)
_LINE_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:FINAL\s*ANSWER|ANSWER|VERDICT|DECISION)\s*[:\-]?\s*(.+?)\s*$",
    re.IGNORECASE,
)

_VULBENCH_LABELS: Final[dict[str, str]] = {
    "INTEGER-OVERFLOW": "Integer-Overflow",
    "BUFFER-OVERFLOW": "Buffer-Overflow",
    "NULL-POINTER-DEREFERENCE": "Null-Pointer-Dereference",
    "USE-AFTER-FREE": "Use-After-Free",
    "DOUBLE-FREE": "Double-Free",
    "MEMORY-LEAK": "Memory-Leak",
    "FORMAT-STRING": "Format-String",
    "RACE-CONDITION": "Race-Condition",
    "IMPROPER-ACCESS-CONTROL": "Improper-Access-Control",
    "NO-VULNERABILITY": "No-Vulnerability",
}
_VULBENCH_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(label) for label in sorted(_VULBENCH_LABELS, key=len, reverse=True)),
    re.IGNORECASE,
)


def _get_non_empty_lines(response: str) -> list[str]:
    """Return stripped non-empty lines from a model response."""
    return [line.strip() for line in response.splitlines() if line.strip()]


def _extract_final_answer_payload(response: str) -> str | None:
    """Extract the last explicit final-answer payload if present."""
    matches: list[str] = _FINAL_ANSWER_PATTERN.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def _extract_prefixed_payload(text: str) -> str | None:
    """Extract payloads from lines like 'Final Answer: ...'."""
    match: re.Match[str] | None = _LINE_PREFIX_PATTERN.match(text.strip())
    if match is None:
        return None
    return match.group(1).strip()


def _normalize_binary_label(text: str) -> int | None:
    """Normalize a binary prediction label to 1 or 0."""
    candidate: str = text.strip().strip("\"'`[](){}.,:; ").upper()
    if candidate in {"VULNERABLE", "YES", "TRUE", "FOUND", "DETECTED"}:
        return 1
    if candidate in {"SAFE", "NO", "FALSE", "NONE", "CLEAN"}:
        return 0
    return None


def _extract_binary_from_tail(response: str) -> int | None:
    """Extract a binary label from the last meaningful response lines."""
    lines: list[str] = _get_non_empty_lines(response)
    for line in reversed(lines[-12:]):
        explicit_payload: str | None = _extract_prefixed_payload(line)
        if explicit_payload is not None:
            normalized: int | None = _normalize_binary_label(explicit_payload)
            if normalized is not None:
                return normalized

        normalized = _normalize_binary_label(line)
        if normalized is not None:
            return normalized

    return None


def _extract_multiclass_from_tail(response: str) -> str | None:
    """Extract a multiclass CWE or SAFE label from the last meaningful lines."""
    lines: list[str] = _get_non_empty_lines(response)
    for line in reversed(lines[-12:]):
        explicit_payload: str | None = _extract_prefixed_payload(line)
        candidate: str = explicit_payload if explicit_payload is not None else line
        cwe_match: re.Match[str] | None = _CWE_PATTERN.search(candidate)
        if cwe_match is not None:
            return cwe_match.group(0).upper()

        normalized: str = candidate.strip().strip("\"'`[](){}.,:; ").upper()
        if normalized == "SAFE":
            return "SAFE"

    return None


def _normalize_vulbench_label(text: str) -> str | None:
    """Normalize VulBench label variants to canonical label names."""
    candidate: str = text.strip().strip("\"'`[](){}.,:; ").upper()
    if candidate == "SAFE":
        return "No-Vulnerability"
    if candidate in _VULBENCH_LABELS:
        return _VULBENCH_LABELS[candidate]
    return None


def _extract_vulbench_from_tail(response: str) -> str | None:
    """Extract a VulBench label from the last meaningful lines."""
    lines: list[str] = _get_non_empty_lines(response)
    for line in reversed(lines[-12:]):
        explicit_payload: str | None = _extract_prefixed_payload(line)
        candidate: str = explicit_payload if explicit_payload is not None else line

        cwe_match: re.Match[str] | None = _CWE_PATTERN.search(candidate)
        if cwe_match is not None:
            return cwe_match.group(0).upper()

        normalized: str | None = _normalize_vulbench_label(candidate)
        if normalized is not None:
            return normalized

    return None


def _extract_vuldetectbench_choice_from_tail(response: str) -> str | None:
    """Extract the final VulDetectBench task2 option from the last meaningful lines."""
    lines: list[str] = _get_non_empty_lines(response)
    for line in reversed(lines[-12:]):
        explicit_payload: str | None = _extract_prefixed_payload(line)
        candidate: str = explicit_payload if explicit_payload is not None else line
        match: re.Match[str] | None = _VULDETECTBENCH_CHOICE_PATTERN.match(
            candidate.strip().upper()
        )
        if match is not None:
            return match.group(1).upper()

    return None


class IResponseParser(ABC):
    """Interface for response parsers."""

    @abstractmethod
    def parse_response(self, response: str) -> int | str:
        """Parse model response into standardized format."""
        pass


class BinaryResponseParser(IResponseParser):
    """Parser for binary vulnerability classification."""

    def parse_response(self, response: str) -> int:
        """Parse binary classification response."""
        response_text: str = response.strip()

        explicit_payload: str | None = _extract_final_answer_payload(response_text)
        if explicit_payload is not None:
            normalized: int | None = _normalize_binary_label(explicit_payload)
            if normalized is not None:
                return normalized

        tail_match: int | None = _extract_binary_from_tail(response_text)
        if tail_match is not None:
            return tail_match

        keyword_matches: list[str] = _BINARY_PATTERN.findall(response_text)
        if keyword_matches:
            return 1 if keyword_matches[-1].upper() == "VULNERABLE" else 0

        fallback_matches: list[str] = _BINARY_FALLBACK_PATTERN.findall(response_text)
        if fallback_matches:
            normalized = _normalize_binary_label(fallback_matches[-1])
            if normalized is not None:
                return normalized

        return 0


class BinaryCweSpecificResponseParser(BinaryResponseParser):
    """Parser for CWE-specific binary classification."""

    pass


class BinaryVulnerabilitySpecificResponseParser(BinaryResponseParser):
    """Parser for vulnerability-specific binary classification."""

    pass


class MulticlassResponseParser(IResponseParser):
    """Parser for multiclass vulnerability classification."""

    def parse_response(self, response: str) -> str:
        """Parse multiclass response."""
        response_text: str = response.strip()

        explicit_payload: str | None = _extract_final_answer_payload(response_text)
        if explicit_payload is not None:
            parsed_explicit: str | None = _extract_multiclass_from_tail(explicit_payload)
            if parsed_explicit is not None:
                return parsed_explicit

        tail_match: str | None = _extract_multiclass_from_tail(response_text)
        if tail_match is not None:
            return tail_match

        label_matches: list[str] = [
            match.group(0).upper()
            for match in re.finditer(r"\bCWE-\d+\b|\bSAFE\b", response_text, re.IGNORECASE)
        ]
        if label_matches:
            return label_matches[-1]

        return "UNKNOWN"


class VulBenchMulticlassResponseParser(IResponseParser):
    """Parser for VulBench multiclass vulnerability classification."""

    def parse_response(self, response: str) -> str:
        """Parse multiclass response for VulBench datasets."""
        response_text: str = response.strip()

        explicit_payload: str | None = _extract_final_answer_payload(response_text)
        if explicit_payload is not None:
            explicit_cwe: re.Match[str] | None = _CWE_PATTERN.search(explicit_payload)
            if explicit_cwe is not None:
                return explicit_cwe.group(0).upper()

            normalized_explicit: str | None = _normalize_vulbench_label(explicit_payload)
            if normalized_explicit is not None:
                return normalized_explicit

        tail_match: str | None = _extract_vulbench_from_tail(response_text)
        if tail_match is not None:
            return tail_match

        matches: list[re.Match[str]] = list(_VULBENCH_PATTERN.finditer(response_text))
        if matches:
            normalized_match: str | None = _normalize_vulbench_label(matches[-1].group(0))
            if normalized_match is not None:
                return normalized_match

        safe_matches: list[str] = re.findall(r"\b(?:SAFE|NO-VULNERABILITY|CLEAN)\b", response_text, re.IGNORECASE)
        if safe_matches:
            return "No-Vulnerability"

        return "UNKNOWN"


class VulDetectBenchResponseParser(IResponseParser):
    """Custom response parser for VulDetectBench tasks."""

    def __init__(self, task_type: str):
        self.task_type: str = task_type

    def parse_response(self, response: str) -> str:
        """Parse response based on VulDetectBench task type."""
        response_text = response.strip()
        explicit_payload: str | None = _extract_final_answer_payload(response_text)

        if self.task_type == "task1":
            if explicit_payload is not None:
                normalized_explicit: int | None = _normalize_binary_label(explicit_payload)
                if normalized_explicit is not None:
                    return "YES" if normalized_explicit == 1 else "NO"

            tail_match: int | None = _extract_binary_from_tail(response_text)
            if tail_match is not None:
                return "YES" if tail_match == 1 else "NO"

            matches: list[str] = _VULDETECTBENCH_BINARY_PATTERN.findall(response_text)
            if matches:
                return matches[-1].upper()

            return "NO"
        elif self.task_type == "task2":
            if explicit_payload is not None:
                explicit_choice: str | None = _extract_vuldetectbench_choice_from_tail(
                    explicit_payload
                )
                if explicit_choice is not None:
                    return explicit_choice

            tail_choice: str | None = _extract_vuldetectbench_choice_from_tail(response_text)
            if tail_choice is not None:
                return tail_choice

            return "A"
        else:
            return response_text


class ResponseParserFactory:
    """Factory for creating response parsers based on task type and dataset."""

    @staticmethod
    def create_parser(task_type: TaskType) -> IResponseParser:
        """
        Create appropriate response parser based on task type.

        Args:
            task_type: The type of task

        Returns:
            IResponseParser: Appropriate parser instance
        """
        if task_type == TaskType.BINARY_VULNERABILITY:
            return BinaryResponseParser()
        elif task_type == TaskType.BINARY_CWE_SPECIFIC:
            return BinaryCweSpecificResponseParser()
        elif task_type == TaskType.BINARY_VULNERABILITY_SPECIFIC:
            return BinaryVulnerabilitySpecificResponseParser()
        elif task_type == TaskType.MULTICLASS_VULNERABILITY:
            return MulticlassResponseParser()
        elif task_type == TaskType.VULBENCH_MULTICLASS:
            return VulBenchMulticlassResponseParser()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
