import re
from abc import ABC, abstractmethod

from benchmark.enums import TaskType


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
        response = response.strip().upper()

        if "VULNERABLE" in response:
            return 1
        elif "SAFE" in response:
            return 0
        else:
            # Try to extract decision from longer responses
            if any(word in response for word in ["YES", "TRUE", "FOUND", "DETECTED"]):
                return 1
            elif any(word in response for word in ["NO", "FALSE", "NONE", "CLEAN"]):
                return 0
            else:
                # Default to safe if unclear
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
        response = response.strip().upper()

        # Look for CWE patterns
        cwe_pattern = r"CWE-\d+"
        cwe_match = re.search(cwe_pattern, response)

        if cwe_match:
            return cwe_match.group()
        elif "SAFE" in response:
            return "SAFE"
        else:
            return "UNKNOWN"


class VulBenchMulticlassResponseParser(IResponseParser):
    """Parser for VulBench multiclass vulnerability classification."""

    def parse_response(self, response: str) -> str:
        """Parse multiclass response for VulBench datasets."""
        response = response.strip()

        # First check for CWE patterns (in case someone uses CWE format)
        cwe_pattern = r"CWE-\d+"
        cwe_match = re.search(cwe_pattern, response)
        if cwe_match:
            return cwe_match.group()

        # Convert to uppercase for case-insensitive matching
        response_upper = response.upper()

        # Look for VulBench vulnerability type patterns
        vulbench_patterns = {
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

        for pattern_upper, pattern_original in vulbench_patterns.items():
            if pattern_upper in response_upper or pattern_original in response:
                return pattern_original

        # Check for common safe indicators
        if any(
            safe_word in response_upper
            for safe_word in ["SAFE", "NO-VULNERABILITY", "CLEAN"]
        ):
            return "No-Vulnerability"
        else:
            return "UNKNOWN"


class ResponseParserFactory:
    """Factory for creating response parsers based on task type and dataset."""

    @staticmethod
    def create_parser(
        task_type: TaskType, is_vulbench: bool = False
    ) -> IResponseParser:
        """
        Create appropriate response parser based on task type.

        Args:
            task_type: The type of task
            is_vulbench: Whether this is for VulBench dataset

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
            if is_vulbench:
                return VulBenchMulticlassResponseParser()
            else:
                return MulticlassResponseParser()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
