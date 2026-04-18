from __future__ import annotations

import logging
import os
import time
from importlib import import_module
from typing import Any, Final, cast

from benchmark.config import ExperimentConfig
from llm.base_online_backend import BaseOnlineLLM
from llm.llm import InferenceResult

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_BATCH_ERROR_TEXT_PREFIX: Final[str] = "ERROR: "
_DEFAULT_ANTHROPIC_MAX_RETRIES: Final[int] = 5
_DEFAULT_BATCH_MAX_WAIT_SECONDS: Final[float] = 3600.0
_DEFAULT_BATCH_POLL_INTERVAL_SECONDS: Final[float] = 15.0
_DEFAULT_ANTHROPIC_KEY_ENV_VAR: Final[str] = "ANTHROPIC_API_KEY"


class AnthropicLLM(BaseOnlineLLM):
    """Anthropic Messages API backend."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._anthropic_client: Any | None = None
        super().__init__(config)

    def _load_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            anthropic_module: Any = import_module("anthropic")
            anthropic_client_cls: Any = anthropic_module.Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend selected, but the Anthropic SDK is not installed. "
                "Install the optional dependency group: `uv sync --extra online`."
            ) from exc

        api_key_env_var: str = (
            self.config.api_key_env_var or _DEFAULT_ANTHROPIC_KEY_ENV_VAR
        )
        api_key: str | None = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Missing required environment variable for Anthropic backend: {api_key_env_var}"
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.config.api_base_url:
            client_kwargs["base_url"] = self.config.api_base_url
        if self.config.api_timeout_seconds is not None:
            client_kwargs["timeout"] = self.config.api_timeout_seconds
        max_retries: int = (
            self.config.api_max_retries
            if self.config.api_max_retries is not None
            else _DEFAULT_ANTHROPIC_MAX_RETRIES
        )
        client_kwargs["max_retries"] = max_retries

        self._anthropic_client = anthropic_client_cls(**client_kwargs)

    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> InferenceResult:
        """Generate one response from Anthropic."""
        start_time: float = time.time()
        response_text, tokens_used = self._generate_anthropic_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return InferenceResult(
            response_text=response_text,
            tokens_used=tokens_used,
            duration=time.time() - start_time,
            confidence=None,
            binary_label_confidence=None,
        )

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[InferenceResult]:
        """Generate responses, using Anthropic Message Batches when enabled."""
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        if not self._should_use_batch_api(len(system_prompts)):
            return super().generate_responses_batch_optimized(system_prompts, user_prompts)

        request_payloads: list[dict[str, object]] = [
            self._build_request_kwargs(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        return self._execute_batch_requests(request_payloads)

    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[InferenceResult]:
        """Generate formatted prompts, optionally through Anthropic Message Batches."""
        if not self._should_use_batch_api(len(prompts)):
            return super().generate_batch_responses(prompts)

        request_payloads: list[dict[str, object]] = [
            self._build_request_kwargs(system_prompt="", user_prompt=prompt)
            for prompt in prompts
        ]
        return self._execute_batch_requests(request_payloads)

    def _should_use_batch_api(self, request_count: int) -> bool:
        """Return whether Message Batches should be used for this request group."""
        return bool(self.config.api_use_batch) and request_count > 1

    def _build_request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        """Build Anthropic Messages API request parameters."""
        request_kwargs: dict[str, object] = {
            "model": self.config.model_identifier,
            "max_tokens": self.config.max_output_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt

        thinking_config: dict[str, object] | None = self._build_anthropic_thinking_config()
        if thinking_config is not None:
            request_kwargs["thinking"] = thinking_config

        output_config: dict[str, object] | None = self._build_anthropic_output_config()
        if output_config is not None:
            request_kwargs["output_config"] = output_config

        if thinking_config is None:
            request_kwargs["temperature"] = self.config.temperature
            if self.config.top_k is not None:
                request_kwargs["top_k"] = self.config.top_k
        else:
            self._warn_if_anthropic_param_unsupported(
                "temperature",
                self.config.temperature,
                "Anthropic thinking mode does not support temperature overrides",
            )
            self._warn_if_anthropic_param_unsupported(
                "top_k",
                self.config.top_k,
                "Anthropic thinking mode does not support top_k overrides",
            )

        if self.config.top_p is not None:
            if thinking_config is None or self.config.top_p >= 0.95:
                request_kwargs["top_p"] = self.config.top_p
            else:
                self._warn_if_anthropic_param_unsupported(
                    "top_p",
                    self.config.top_p,
                    "Anthropic thinking mode only supports top_p values in [0.95, 1.0]",
                )

        self._warn_if_anthropic_param_unsupported(
            "presence_penalty",
            self.config.presence_penalty,
            "Anthropic Messages API does not expose presence_penalty",
        )
        self._warn_if_anthropic_param_unsupported(
            "repetition_penalty",
            self.config.repetition_penalty,
            "Anthropic Messages API does not expose repetition_penalty",
        )
        self._warn_if_anthropic_param_unsupported(
            "enable_logprobs",
            self.config.enable_logprobs,
            "Anthropic Messages API does not expose token logprobs",
        )

        return request_kwargs

    def _execute_batch_requests(
        self,
        request_payloads: list[dict[str, object]],
    ) -> list[InferenceResult]:
        """Execute a group of Anthropic requests through the Message Batches API."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not loaded")
        if not request_payloads:
            return []

        self._warn_if_anthropic_param_unsupported(
            "api_batch_completion_window",
            self.config.api_batch_completion_window,
            "Anthropic Message Batches use a fixed 24-hour expiration and do not accept a completion window parameter",
        )

        started_at: float = time.time()
        self._log_generation_batch_start(
            len(request_payloads),
            "Anthropic Message Batches",
        )
        batch_requests: list[dict[str, object]] = [
            {
                "custom_id": self._build_batch_custom_id(index),
                "params": request_payload,
            }
            for index, request_payload in enumerate(request_payloads)
        ]
        message_batch: Any = self._anthropic_client.messages.batches.create(
            requests=batch_requests,
        )
        finished_batch: Any = self._wait_for_batch_completion(
            batch_id=str(getattr(message_batch, "id")),
            expected_requests=len(request_payloads),
        )
        return self._collect_batch_results(
            batch_id=str(getattr(finished_batch, "id")),
            request_count=len(request_payloads),
            started_at=started_at,
        )

    @staticmethod
    def _build_batch_custom_id(index: int) -> str:
        """Build a stable batch request identifier."""
        return f"request_{index}"

    def _wait_for_batch_completion(self, batch_id: str, expected_requests: int) -> Any:
        """Poll an Anthropic message batch until it reaches the ended state."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not loaded")

        poll_interval_seconds: float = (
            self.config.api_batch_poll_interval_seconds
            or _DEFAULT_BATCH_POLL_INTERVAL_SECONDS
        )
        max_wait_seconds: float = (
            self.config.api_batch_max_wait_seconds
            or _DEFAULT_BATCH_MAX_WAIT_SECONDS
        )
        deadline: float = time.monotonic() + max_wait_seconds

        while True:
            batch: Any = self._anthropic_client.messages.batches.retrieve(batch_id)
            processing_status: str = str(
                getattr(batch, "processing_status", "unknown")
            )
            self._log_batch_status(batch, expected_requests)

            if processing_status == "ended":
                return batch
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Anthropic batch {batch_id} after {max_wait_seconds:.0f}s"
                )
            time.sleep(max(poll_interval_seconds, 0.0))

    def _log_batch_status(self, batch: Any, expected_requests: int) -> None:
        """Log Anthropic batch progress from the retrieved batch object."""
        processing_status: str = str(getattr(batch, "processing_status", "unknown"))
        request_counts: Any = getattr(batch, "request_counts", None)
        processing: int = int(getattr(request_counts, "processing", 0) or 0)
        succeeded: int = int(getattr(request_counts, "succeeded", 0) or 0)
        errored: int = int(getattr(request_counts, "errored", 0) or 0)
        canceled: int = int(getattr(request_counts, "canceled", 0) or 0)
        expired: int = int(getattr(request_counts, "expired", 0) or 0)
        total: int = processing + succeeded + errored + canceled + expired
        if total == 0:
            total = expected_requests

        LOGGER.info(
            "Anthropic batch %s status=%s processing=%d succeeded=%d errored=%d canceled=%d expired=%d total=%d",
            str(getattr(batch, "id", "<unknown>")),
            processing_status,
            processing,
            succeeded,
            errored,
            canceled,
            expired,
            total,
        )

    def _collect_batch_results(
        self,
        *,
        batch_id: str,
        request_count: int,
        started_at: float,
    ) -> list[InferenceResult]:
        """Read and order Anthropic batch results."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not loaded")

        per_item_duration: float = (time.time() - started_at) / max(request_count, 1)
        parsed_results: dict[str, InferenceResult] = {}
        for raw_result in self._anthropic_client.messages.batches.results(batch_id):
            raw_custom_id: str | None = self._safe_str(
                getattr(raw_result, "custom_id", None)
            )
            if raw_custom_id is None:
                continue

            result_payload: Any = getattr(raw_result, "result", None)
            result_type: str = str(getattr(result_payload, "type", ""))
            if result_type == "succeeded":
                message: Any = getattr(result_payload, "message", None)
                response_text: str = self._extract_anthropic_response_text(message)
                token_count: int = self._extract_anthropic_total_tokens(message)
                parsed_results[raw_custom_id] = InferenceResult(
                    response_text=response_text,
                    tokens_used=token_count,
                    duration=per_item_duration,
                    confidence=None,
                    binary_label_confidence=None,
                )
                continue

            error_text: str = self._extract_anthropic_batch_error_text(
                result_payload=result_payload,
                result_type=result_type,
            )
            parsed_results[raw_custom_id] = InferenceResult(
                response_text=f"{_BATCH_ERROR_TEXT_PREFIX}{error_text}",
                tokens_used=0,
                duration=per_item_duration,
                confidence=None,
                binary_label_confidence=None,
            )

        ordered_results: list[InferenceResult] = []
        for index in range(request_count):
            custom_id: str = self._build_batch_custom_id(index)
            result: InferenceResult | None = parsed_results.get(custom_id)
            if result is None:
                result = InferenceResult(
                    response_text=f"{_BATCH_ERROR_TEXT_PREFIX}Missing batch result for {custom_id}",
                    tokens_used=0,
                    duration=per_item_duration,
                    confidence=None,
                    binary_label_confidence=None,
                )
            ordered_results.append(result)

        return ordered_results

    def _extract_anthropic_batch_error_text(
        self,
        *,
        result_payload: Any,
        result_type: str,
    ) -> str:
        """Extract a readable error string for a non-success Anthropic batch result."""
        if result_type == "errored":
            error_payload: Any = getattr(result_payload, "error", None)
            error_type: str | None = self._safe_str(getattr(error_payload, "type", None))
            error_message: str | None = self._safe_str(
                getattr(error_payload, "message", None)
            )
            if error_type and error_message:
                return f"{error_type}: {error_message}"
            if error_message:
                return error_message
            if error_type:
                return error_type
            return "Anthropic batch request errored"
        if result_type == "canceled":
            return "Anthropic batch request canceled"
        if result_type == "expired":
            return "Anthropic batch request expired"
        return f"Anthropic batch request ended with result type '{result_type or 'unknown'}'"

    @staticmethod
    def _safe_str(value: object) -> str | None:
        """Convert a possibly-empty object to a stripped string."""
        if value is None:
            return None
        text: str = str(value).strip()
        return text if text else None

    def _generate_anthropic_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int]:
        """Generate a response with the Anthropic Messages API."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not loaded")

        request_kwargs: dict[str, object] = {
            "model": self.config.model_identifier,
            "max_tokens": self.config.max_output_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        request_kwargs = self._build_request_kwargs(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        response: Any = self._anthropic_client.messages.create(**request_kwargs)
        response_text: str = self._extract_anthropic_response_text(response)
        token_count: int = self._extract_anthropic_total_tokens(response)
        return response_text, token_count

    def _build_anthropic_thinking_config(self) -> dict[str, object] | None:
        """Build Anthropic thinking configuration when enabled."""
        if not self.config.is_thinking_enabled:
            return None

        thinking_type: str = self._resolve_anthropic_thinking_type()
        if thinking_type == "disabled":
            return None

        thinking_config: dict[str, object] = {"type": thinking_type}
        if thinking_type == "enabled":
            budget_tokens: int | None = self._resolve_anthropic_thinking_budget_tokens()
            if budget_tokens is None:
                return None
            thinking_config["budget_tokens"] = budget_tokens
        else:
            self._warn_if_anthropic_param_unsupported(
                "api_reasoning_budget_tokens",
                self.config.api_reasoning_budget_tokens,
                "Anthropic adaptive thinking does not accept a token budget; use api_reasoning_effort to control reasoning spend",
            )
            self._warn_if_anthropic_param_unsupported(
                "api_thinking_budget_tokens",
                self.config.api_thinking_budget_tokens,
                "Anthropic adaptive thinking does not accept a token budget; use api_reasoning_effort to control reasoning spend",
            )

        display: str | None = self.config.api_thinking_display
        if display:
            thinking_config["display"] = display

        return thinking_config

    def _build_anthropic_output_config(self) -> dict[str, object] | None:
        """Build Anthropic output configuration for effort control."""
        effort: str | None = self._normalize_reasoning_effort(
            self.config.api_reasoning_effort
        )
        if effort is None:
            return None
        return {"effort": effort}

    def _resolve_anthropic_thinking_type(self) -> str:
        """Resolve the Anthropic thinking type for the configured model."""
        configured_type: str | None = self.config.api_thinking_type
        if configured_type:
            return configured_type.strip().lower()

        if self.config.api_reasoning_effort and self._supports_adaptive_thinking():
            return "adaptive"
        return "enabled"

    def _supports_adaptive_thinking(self) -> bool:
        """Return whether the configured Claude model supports adaptive thinking."""
        model_identifier: str = self.config.model_identifier.strip().lower()
        return any(
            candidate in model_identifier
            for candidate in (
                "claude-sonnet-4-6",
                "claude-opus-4-6",
                "claude-opus-4-7",
                "mythos",
            )
        )

    @staticmethod
    def _normalize_reasoning_effort(effort: str | None) -> str | None:
        """Normalize configured reasoning effort to Anthropic's lowercase wire format."""
        if effort is None:
            return None
        normalized_effort: str = effort.strip().lower()
        return normalized_effort if normalized_effort else None

    def _resolve_anthropic_thinking_budget_tokens(self) -> int | None:
        """Resolve a valid Anthropic thinking budget, or disable thinking if impossible."""
        max_tokens: int = self.config.max_output_tokens
        if max_tokens <= 1024:
            self._warn_unsupported_sampling_param(
                "is_thinking_enabled",
                "Anthropic thinking requires max_output_tokens greater than 1024",
            )
            return None

        configured_budget: int | None = (
            self.config.api_thinking_budget_tokens
            or self.config.api_reasoning_budget_tokens
        )
        default_budget: int = min(max_tokens - 1, max(1024, max_tokens // 2))
        budget_tokens: int = configured_budget or default_budget
        if budget_tokens >= max_tokens:
            budget_tokens = max_tokens - 1
        if budget_tokens < 1024:
            self._warn_unsupported_sampling_param(
                "api_thinking_budget_tokens",
                "Anthropic thinking budget must be at least 1024 tokens",
            )
            return None

        return budget_tokens

    def _extract_anthropic_response_text(self, response: Any) -> str:
        """Extract readable text blocks from an Anthropic Messages response."""
        raw_content_blocks: Any = getattr(response, "content", None)
        if not isinstance(raw_content_blocks, list):
            return ""
        content_blocks: list[object] = cast(list[object], raw_content_blocks)

        text_fragments: list[str] = []
        for block in content_blocks:
            block_type: str = str(getattr(block, "type", ""))
            if block_type == "thinking":
                thinking_text: str = str(getattr(block, "thinking", "")).strip()
                if thinking_text:
                    text_fragments.append(thinking_text)
            elif block_type == "text":
                block_text: str = str(getattr(block, "text", "")).strip()
                if block_text:
                    text_fragments.append(block_text)

        return "\n\n".join(text_fragments)

    def _extract_anthropic_total_tokens(self, response: Any) -> int:
        """Extract combined token usage from an Anthropic Messages response."""
        usage: Any = getattr(response, "usage", None)
        input_tokens_raw: Any = getattr(usage, "input_tokens", 0)
        output_tokens_raw: Any = getattr(usage, "output_tokens", 0)

        input_tokens: int = int(input_tokens_raw) if input_tokens_raw else 0
        output_tokens: int = int(output_tokens_raw) if output_tokens_raw else 0
        return input_tokens + output_tokens

    def _warn_if_anthropic_param_unsupported(
        self,
        param_name: str,
        value: object,
        reason: str,
    ) -> None:
        """Warn when an Anthropic parameter is configured but unsupported."""
        if not self._is_configured_value(value):
            return
        self._warn_unsupported_sampling_param(param_name, reason)

    def count_input_tokens(self, text: str) -> int:
        """Count approximate Anthropic tokens for the provided text."""
        cached_token_count: int | None = self._token_count_cache.get(text)
        if cached_token_count is not None:
            return cached_token_count

        token_count: int = self._count_anthropic_input_tokens(text)
        self._token_count_cache[text] = token_count
        return token_count

    def _count_anthropic_input_tokens(self, text: str) -> int:
        """Count Anthropic input tokens, falling back to a rough estimate on failure."""
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client not loaded")

        try:
            count_response: Any = self._anthropic_client.messages.count_tokens(
                model=self.config.model_identifier,
                messages=[{"role": "user", "content": text}],
            )
            input_tokens_raw: Any = getattr(count_response, "input_tokens", 0)
            return int(input_tokens_raw) if input_tokens_raw else 0
        except Exception:
            LOGGER.exception(
                "Anthropic token counting failed for %s; using rough estimate",
                self.config.model_identifier,
            )
            return max(1, len(text) // 4)

    def cleanup(self) -> None:
        """Release Anthropic backend client references."""
        self._anthropic_client = None