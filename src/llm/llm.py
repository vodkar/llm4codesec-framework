from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InferenceResult:
    """Result from a single LLM inference call."""

    response_text: str
    tokens_used: int
    duration: float
    confidence: float | None = field(default=None)
    """Geometric-mean per-token probability (exp of mean log-prob).
    None when logprobs are not enabled or not supported."""
    binary_label_confidence: float | None = field(default=None)
    """Binary P(VULNERABLE) derived from final-answer label-position logprobs."""


class ILLMInference(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> InferenceResult:
        """
        Generate response from the model.

        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt

        Returns:
            InferenceResult with response text, token count, duration, and optional confidence.
        """
        pass

    @abstractmethod
    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[InferenceResult]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list[str]): List of formatted prompts

        Returns:
            list[InferenceResult]: One result per prompt.
        """
        pass

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[InferenceResult]:
        """
        Generate responses for multiple system/user prompt pairs with batch optimization.

        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts (must be same length as system_prompts)

        Returns:
            List of InferenceResult objects, one per prompt pair.
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        # Default implementation — subclasses should override for GPU-backed batching
        results: list[InferenceResult] = []
        for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
            result = self.generate_response(sys_prompt, user_prompt)
            results.append(result)
        return results

    @abstractmethod
    def count_input_tokens(self, text: str) -> int:
        """Count input tokens for the provided text using backend tokenizer."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass
