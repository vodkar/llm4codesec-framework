from abc import ABC, abstractmethod


class ILLMInference(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int, float]:
        """
        Generate response from the model.

        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt

        Returns:
            tuple[str, int, float]: Response text, token count, and duration
        """
        pass

    @abstractmethod
    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list[str]): List of formatted prompts

        Returns:
            list[tuple[str, int, float]]: List of (response_text, token_count, duration) tuples
        """
        pass

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for multiple system/user prompt pairs with batch optimization.

        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts (must be same length as system_prompts)

        Returns:
            List of (response_text, token_count, duration) tuples
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        # This is a default implementation - can be overridden by subclasses
        results: list[tuple[str, int, float]] = []
        for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
            result = self.generate_response(sys_prompt, user_prompt)
            results.append(result)
        return results

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass
