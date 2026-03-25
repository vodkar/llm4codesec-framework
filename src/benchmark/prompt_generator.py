import re
from abc import ABC, abstractmethod
from typing import Final

from pydantic import BaseModel

from benchmark.config import ExperimentConfig
from benchmark.enums import TaskType

_BINARY_RESPONSE_CONTRACT: Final[str] = (
    "\nIMPORTANT output rule:\n"
    "- You may think step by step if needed.\n"
    "- DO NOT USE the final label anywhere except the last non-empty line.\n"
    "- The last non-empty line must be EXACTLY ONE OF:\n"
    "  [[FINAL_ANSWER: VULNERABLE]]\n"
    "  [[FINAL_ANSWER: SAFE]]"
)

_MULTICLASS_RESPONSE_CONTRACT: Final[str] = (
    "\nIMPORTANT output rule:\n"
    "- You may think step by step if needed.\n"
    "- DO NOT USE the final label anywhere except the last non-empty line.\n"
    "- The last non-empty line must be EXACTLY ONE OF:\n"
    "  [[FINAL_ANSWER: CWE-<number>]]\n"
    "  [[FINAL_ANSWER: SAFE]]"
)

_VULBENCH_RESPONSE_CONTRACT: Final[str] = (
    "\nIMPORTANT output rule:\n"
    "- You may think step by step if needed.\n"
    "- DO NOT USE the final label anywhere except the last non-empty line.\n"
    "- The last non-empty line must be EXACTLY ONE OF:\n"
    "  [[FINAL_ANSWER: Integer-Overflow]]\n"
    "  [[FINAL_ANSWER: Buffer-Overflow]]\n"
    "  [[FINAL_ANSWER: Null-Pointer-Dereference]]\n"
    "  [[FINAL_ANSWER: Use-After-Free]]\n"
    "  [[FINAL_ANSWER: Double-Free]]\n"
    "  [[FINAL_ANSWER: Memory-Leak]]\n"
    "  [[FINAL_ANSWER: Format-String]]\n"
    "  [[FINAL_ANSWER: Race-Condition]]\n"
    "  [[FINAL_ANSWER: Improper-Access-Control]]\n"
    "  [[FINAL_ANSWER: No-Vulnerability]]"
)

_VULDETECTBENCH_TASK1_RESPONSE_CONTRACT: Final[str] = (
    "\nIMPORTANT output rule:\n"
    "- You may think step by step if needed.\n"
    "- DO NOT USE the final label anywhere except the last non-empty line.\n"
    "- The last non-empty line must be EXACTLY ONE OF:\n"
    "  [[FINAL_ANSWER: YES]]\n"
    "  [[FINAL_ANSWER: NO]]"
)

_VULDETECTBENCH_TASK2_RESPONSE_CONTRACT: Final[str] = (
    "\nIMPORTANT output rule:\n"
    "- You may think step by step if needed.\n"
    "- DO NOT USE the final label anywhere except the last non-empty line.\n"
    "- The last non-empty line must be EXACTLY ONE OF:\n"
    "  [[FINAL_ANSWER: A]]\n"
    "  [[FINAL_ANSWER: B]]\n"
    "  [[FINAL_ANSWER: C]]\n"
    "  [[FINAL_ANSWER: D]]\n"
    "  [[FINAL_ANSWER: E]]"
)

_REDUNDANT_RESPONSE_LINE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^\s*-\s*Respond with only\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Response Format:\s*Respond with only\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Final Answer:\s*After your analysis, respond\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Your answer should either be ['\"]YES['\"] or ['\"]NO['\"] only\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Output ['\"]A\.['\"] or ['\"]B\.['\"] or ['\"]C\.['\"] or ['\"]D\.['\"] or ['\"]E\.['\"] only\.?\s*$", re.IGNORECASE),
)


def _get_response_contract(
    task_type: TaskType | None, prompt_identifier: str | None
) -> str:
    """Return a task-specific response contract appended to the user prompt."""
    if task_type in {
        TaskType.BINARY_VULNERABILITY,
        TaskType.BINARY_CWE_SPECIFIC,
        TaskType.BINARY_VULNERABILITY_SPECIFIC,
    }:
        return _BINARY_RESPONSE_CONTRACT

    if task_type == TaskType.MULTICLASS_VULNERABILITY:
        return _MULTICLASS_RESPONSE_CONTRACT

    if task_type == TaskType.VULBENCH_MULTICLASS:
        return _VULBENCH_RESPONSE_CONTRACT

    if task_type == TaskType.VULDETECTBENCH_SPECIFIC:
        if prompt_identifier == "task1_specific":
            return _VULDETECTBENCH_TASK1_RESPONSE_CONTRACT
        if prompt_identifier == "task2_specific":
            return _VULDETECTBENCH_TASK2_RESPONSE_CONTRACT

    return ""


def _strip_redundant_response_instructions(text: str) -> str:
    """Remove legacy response-format lines now handled centrally."""
    cleaned_lines: list[str] = []
    previous_was_blank: bool = False

    for line in text.splitlines():
        if any(pattern.match(line) for pattern in _REDUNDANT_RESPONSE_LINE_PATTERNS):
            continue

        stripped_line: str = line.strip()
        if stripped_line == "":
            if previous_was_blank:
                continue
            previous_was_blank = True
        else:
            previous_was_blank = False

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


class IPromptGenerator(ABC, BaseModel):
    system_prompt_template: str
    user_prompt_template: str
    template_values: dict[str, str]

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Generate the system prompt.

        Returns:
            str: Rendered system prompt.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_prompt(self, template_values: dict[str, str]) -> str:
        """Generate the user prompt.

        Args:
            template_values (dict[str, str]): Template values to override.

        Returns:
            str: Rendered user prompt.
        """
        raise NotImplementedError


class DefaultPromptGenerator(IPromptGenerator):
    task_type: TaskType | None = None
    prompt_identifier: str | None = None

    def get_system_prompt(self) -> str:
        """Generate the system prompt.

        Returns:
            str: Rendered system prompt.
        """
        cleaned_template: str = _strip_redundant_response_instructions(
            self.system_prompt_template
        )
        return cleaned_template.format(**self.template_values)

    def get_user_prompt(self, template_values: dict[str, str]) -> str:
        """Generate the user prompt.

        Args:
            template_values (dict[str, str]): Template values to override.

        Returns:
            str: Rendered user prompt.
        """
        cleaned_template: str = _strip_redundant_response_instructions(
            self.user_prompt_template
        )
        prompt: str = cleaned_template.format(
            **self.template_values, **template_values
        )
        response_contract: str = _get_response_contract(
            self.task_type, self.prompt_identifier
        )
        return prompt + response_contract


def get_prompt_generator(
    config: ExperimentConfig, template_values: dict[str, str]
) -> IPromptGenerator:
    return DefaultPromptGenerator(
        system_prompt_template=config.system_prompt_template,
        user_prompt_template=config.user_prompt_template,
        template_values=template_values,
        task_type=config.task_type,
        prompt_identifier=config.prompt_identifier,
    )
