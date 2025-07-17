import re

from benchmark.enums import TaskType


class ResponseParser:
    """Parses and normalizes model responses."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type

    def parse_response(self, response: str) -> int | str:
        """
        Parse model response into standardized format.

        Args:
            response (str): Raw model response

        Returns:
            int | str: Parsed response
        """
        response = response.strip().upper()

        if self.task_type == TaskType.BINARY_VULNERABILITY:
            return self._parse_binary_response(response)
        elif self.task_type == TaskType.BINARY_CWE_SPECIFIC:
            return self._parse_binary_response(response)
        elif self.task_type == TaskType.MULTICLASS_VULNERABILITY:
            return self._parse_multiclass_response(response)

        return response

    def _parse_binary_response(self, response: str) -> int:
        """Parse binary classification response."""
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

    def _parse_multiclass_response(self, response: str) -> str:
        """Parse multiclass response."""
        # Look for CWE patterns
        cwe_pattern = r"CWE-\d+"
        cwe_match = re.search(cwe_pattern, response)

        if cwe_match:
            return cwe_match.group()
        elif "SAFE" in response:
            return "SAFE"
        else:
            return "UNKNOWN"
