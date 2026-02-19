from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Final

import torch
from transformers import PreTrainedTokenizerBase

from benchmark.config import ExperimentConfig
from llm.llm import ILLMInference

if TYPE_CHECKING:
    from vllm import LLM, RequestOutput, SamplingParams

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


class VllmLLM(ILLMInference):
    """vLLM-based LLM inference backend."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the vLLM backend.

        Args:
            config: Benchmark configuration for model and generation settings.
        """
        self.config: ExperimentConfig = config
        self.llm: LLM | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the vLLM model and tokenizer."""
        try:
            from vllm import LLM
        except ImportError as exc:
            LOGGER.exception("vLLM is not installed")
            raise RuntimeError(
                "vLLM backend selected but vLLM is not installed. "
                "Install with `pip install vllm`."
            ) from exc

        LOGGER.info("Loading vLLM model: %s", self.config.model_identifier)

        self.llm = LLM(
            model=self.config.model_identifier,
            trust_remote_code=True,
            max_num_seqs=max(self.config.batch_size, 1),
        )
        self.tokenizer = self.llm.get_tokenizer()

        if self.tokenizer is None:
            raise RuntimeError("vLLM tokenizer failed to load")

    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int, float]:
        """
        Generate a single response using vLLM.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.

        Returns:
            tuple[str, int, float]: Response text, token count, and duration.
        """
        formatted_prompt: str = self._format_prompt(system_prompt, user_prompt)
        batch_results = self.generate_batch_responses([formatted_prompt])
        return batch_results[0]

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for multiple system/user prompt pairs with batch optimization.

        Args:
            system_prompts: List of system prompts.
            user_prompts: List of user prompts.

        Returns:
            List of (response_text, token_count, duration) tuples.
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        formatted_prompts: list[str] = [
            self._format_prompt(system_prompt, user_prompt)
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]

        return self.generate_batch_responses(formatted_prompts)

    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of formatted prompts.

        Returns:
            List of (response_text, token_count, duration) tuples.
        """
        if not self.llm:
            raise RuntimeError("vLLM model not loaded")

        if not prompts:
            return []

        try:
            from vllm import SamplingParams
        except ImportError as exc:
            LOGGER.exception("vLLM SamplingParams import failed")
            raise RuntimeError("vLLM is not installed") from exc

        hard_batch_size: int = int(
            os.getenv("HARD_BATCH_SIZE", str(self.config.batch_size))
        )
        batch_size: int = max(min(hard_batch_size, len(prompts)), 1)
        results: list[tuple[str, int, float]] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            sampling_params = self._create_sampling_params(SamplingParams)

            start_time: float = time.time()
            batch_outputs: list[RequestOutput] = self.llm.generate(
                batch_prompts, sampling_params
            )
            duration: float = time.time() - start_time

            results.extend(
                self._collect_batch_results(
                    batch_outputs=batch_outputs,
                    duration=duration,
                    batch_size=len(batch_prompts),
                )
            )

            LOGGER.info(
                "Processed vLLM batch %d/%d",
                i // batch_size + 1,
                (len(prompts) - 1) // batch_size + 1,
            )

        return results

    def _create_sampling_params(
        self, sampling_params_cls: type[SamplingParams]
    ) -> SamplingParams:
        """
        Create vLLM sampling parameters from config.

        Args:
            sampling_params_cls: vLLM SamplingParams class.

        Returns:
            SamplingParams: Configured sampling parameters.
        """
        return sampling_params_cls(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    def _collect_batch_results(
        self, batch_outputs: list[RequestOutput], duration: float, batch_size: int
    ) -> list[tuple[str, int, float]]:
        """
        Collect generation results for a batch.

        Args:
            batch_outputs: vLLM outputs for each prompt.
            duration: Total batch duration in seconds.
            batch_size: Number of prompts in this batch.

        Returns:
            List of (response_text, token_count, duration) tuples.
        """
        results: list[tuple[str, int, float]] = []
        per_item_duration: float = duration / max(batch_size, 1)

        for output in batch_outputs:
            response_text: str = ""
            token_count: int = 0

            if output.outputs:
                response_text = output.outputs[0].text.strip()
                token_count = self._count_tokens(output)

            results.append((response_text, token_count, per_item_duration))

        return results

    @staticmethod
    def _count_tokens(output: RequestOutput) -> int:
        """
        Count prompt and output tokens for a vLLM response.

        Args:
            output: vLLM request output.

        Returns:
            Total token count for prompt and generated text.
        """
        prompt_tokens: int = (
            len(output.prompt_token_ids) if output.prompt_token_ids else 0
        )
        generated_tokens: int = (
            len(output.outputs[0].token_ids)
            if output.outputs and output.outputs[0].token_ids
            else 0
        )
        return prompt_tokens + generated_tokens

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format prompt using tokenizer chat template when available.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.

        Returns:
            Formatted prompt string.
        """
        if not self.tokenizer:
            raise RuntimeError("vLLM tokenizer not loaded")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                if self.config.is_thinking_enabled:
                    formatted_prompt = apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        is_thinking_enabled=True,
                    )
                else:
                    formatted_prompt = apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                return str(formatted_prompt)
            except Exception:
                LOGGER.exception(
                    "vLLM chat template failed for %s, falling back",
                    self.config.model_identifier,
                )

        return self._format_prompt_fallback(system_prompt, user_prompt)

    def _format_prompt_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """
        Fallback prompt formatting for vLLM models without chat templates.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.

        Returns:
            Formatted prompt string.
        """
        model_name_lower: str = self.config.model_identifier.lower()

        if "llama" in model_name_lower:
            if "llama-3" in model_name_lower:
                return (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            return (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            )

        if "qwen" in model_name_lower:
            if "qwen3" in model_name_lower:
                return (
                    "<|im_start|>system\n"
                    f"{system_prompt}<|im_end|>\n<|im_start|>user\n"
                    f"{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                )
            return (
                "<|system|>\n"
                f"{system_prompt}<|endofsystem|>\n<|user|>\n"
                f"{user_prompt}<|endofuser|>\n<|assistant|>\n"
            )

        if "deepseek" in model_name_lower:
            return (
                "### System:\n"
                f"{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
            )

        if "wizard" in model_name_lower:
            return (
                "### Instruction:\n"
                f"{system_prompt}\n\n### Input:\n{user_prompt}\n\n### Response:\n"
            )

        if "gemma" in model_name_lower:
            return (
                "<bos><start_of_turn>user\n"
                f"{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
            )

        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def count_input_tokens(self, text: str) -> int:
        """Count tokenizer tokens for input text."""
        if not self.tokenizer:
            raise RuntimeError("vLLM tokenizer not loaded")

        return len(self.tokenizer.encode(text))

    def cleanup(self) -> None:
        """Clean up vLLM model resources."""
        if self.llm:
            del self.llm
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
