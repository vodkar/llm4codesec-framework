from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Final

from benchmark.config import ExperimentConfig
from llm.llm import ILLMInference, InferenceResult

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_PROGRESS_LOG_STEPS: Final[int] = 10


class BaseOnlineLLM(ILLMInference):
    """Shared utilities for remote-provider LLM backends."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config: ExperimentConfig = config
        self._warned_sampling_params: set[str] = set()
        self._token_count_cache: dict[str, int] = {}

        self._load_client()

    @abstractmethod
    def _load_client(self) -> None:
        """Load provider-specific SDK clients."""

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[InferenceResult]:
        """Generate responses sequentially for multiple prompt pairs."""
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        total_generations: int = len(system_prompts)
        self._log_generation_batch_start(
            total_generations,
            "system/user prompt pairs",
        )

        results: list[InferenceResult] = []
        started_at: float = time.time()
        for index, (system_prompt, user_prompt) in enumerate(
            zip(system_prompts, user_prompts),
            start=1,
        ):
            results.append(self.generate_response(system_prompt, user_prompt))
            self._maybe_log_generation_progress(
                completed=index,
                total=total_generations,
                started_at=started_at,
            )

        return results

    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[InferenceResult]:
        """Generate responses for preformatted prompts sequentially."""
        total_generations: int = len(prompts)
        self._log_generation_batch_start(total_generations, "formatted prompts")

        results: list[InferenceResult] = []
        started_at: float = time.time()
        for index, prompt in enumerate(prompts, start=1):
            results.append(self.generate_response("", prompt))
            self._maybe_log_generation_progress(
                completed=index,
                total=total_generations,
                started_at=started_at,
            )

        return results

    @staticmethod
    def _is_configured_value(value: object) -> bool:
        """Return whether a config value should be treated as explicitly configured."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return True

    def _warn_unsupported_sampling_param(self, param_name: str, reason: str) -> None:
        """Warn once when a configured parameter is unsupported."""
        if param_name in self._warned_sampling_params:
            return

        LOGGER.warning(
            "Ignoring sampling parameter '%s' for %s backend: %s",
            param_name,
            self.__class__.__name__,
            reason,
        )
        self._warned_sampling_params.add(param_name)

    @staticmethod
    def _log_generation_batch_start(total: int, batch_kind: str) -> None:
        """Log the start of a sequential online generation batch."""
        if total <= 0:
            return

        LOGGER.info(
            "Starting %d online generations for %s",
            total,
            batch_kind,
        )

    @staticmethod
    def _should_log_generation_progress(completed: int, total: int) -> bool:
        """Determine whether to emit an online-generation progress update."""
        if total <= 0:
            return False
        if completed >= total:
            return True

        progress_interval: int = max(1, total // _PROGRESS_LOG_STEPS)
        return completed == 1 or completed % progress_interval == 0

    def _maybe_log_generation_progress(
        self,
        *,
        completed: int,
        total: int,
        started_at: float,
    ) -> None:
        """Log online generation progress at a controlled cadence."""
        if not self._should_log_generation_progress(completed, total):
            return

        elapsed_seconds: float = time.time() - started_at
        average_seconds_per_generation: float = (
            elapsed_seconds / completed if completed else 0.0
        )
        eta_seconds: float = average_seconds_per_generation * max(total - completed, 0)

        LOGGER.info(
            "online generation progress: %d/%d (%.1f%%) elapsed=%.1fs eta=%.1fs",
            completed,
            total,
            (completed / total) * 100 if total else 0.0,
            elapsed_seconds,
            eta_seconds,
        )