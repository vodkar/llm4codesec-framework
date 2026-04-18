from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Final, cast

import tiktoken

from benchmark.config import ExperimentConfig
from llm.base_online_backend import BaseOnlineLLM
from llm.llm import InferenceResult

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_BATCH_COMPLETED_STATUS: Final[str] = "completed"
_BATCH_ENDPOINT: Final[str] = "/v1/responses"
_BATCH_ERROR_TEXT_PREFIX: Final[str] = "ERROR: "
_DEFAULT_BATCH_COMPLETION_WINDOW: Final[str] = "24h"
_DEFAULT_BATCH_MAX_WAIT_SECONDS: Final[float] = 3600.0
_DEFAULT_BATCH_POLL_INTERVAL_SECONDS: Final[float] = 15.0
_DEFAULT_OPENAI_KEY_ENV_VAR: Final[str] = "OPENAI_API_KEY"


class OpenAILLM(BaseOnlineLLM):
    """OpenAI Responses API backend with optional Batch API support."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._openai_client: Any | None = None
        self._openai_encoding: tiktoken.Encoding | None = None
        super().__init__(config)

    def _load_client(self) -> None:
        """Initialize the OpenAI client and tokenizer."""
        try:
            openai_module: Any = import_module("openai")
            openai_client_cls: Any = openai_module.OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend selected, but the OpenAI SDK is not installed. "
                "Install the optional dependency group: `uv sync --extra online`."
            ) from exc

        api_key_env_var: str = self.config.api_key_env_var or _DEFAULT_OPENAI_KEY_ENV_VAR
        api_key: str | None = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Missing required environment variable for OpenAI backend: {api_key_env_var}"
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.config.api_base_url:
            client_kwargs["base_url"] = self.config.api_base_url
        if self.config.api_timeout_seconds is not None:
            client_kwargs["timeout"] = self.config.api_timeout_seconds
        if self.config.api_max_retries is not None:
            client_kwargs["max_retries"] = self.config.api_max_retries

        self._openai_client = openai_client_cls(**client_kwargs)
        self._openai_encoding = self._resolve_openai_encoding()

    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> InferenceResult:
        """Generate one response from OpenAI."""
        start_time: float = time.time()
        response_text, tokens_used = self._generate_openai_response(
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
        """Generate responses, using OpenAI Batch API when enabled."""
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
        """Generate responses for preformatted prompts, optionally through Batch API."""
        if not self._should_use_batch_api(len(prompts)):
            return super().generate_batch_responses(prompts)

        request_payloads: list[dict[str, object]] = [
            self._build_request_kwargs(system_prompt="", user_prompt=prompt)
            for prompt in prompts
        ]
        return self._execute_batch_requests(request_payloads)

    def _generate_openai_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int]:
        """Generate a response with the OpenAI Responses API."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not loaded")

        request_kwargs: dict[str, object] = self._build_request_kwargs(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        response: Any = self._openai_client.responses.create(**request_kwargs)
        response_text: str = self._extract_openai_response_text(response)
        token_count: int = self._extract_openai_total_tokens(response)
        return response_text, token_count

    def _build_request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        """Build OpenAI Responses API request parameters."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "developer", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        request_kwargs: dict[str, object] = {
            "model": self.config.model_identifier,
            "input": messages,
            "max_output_tokens": self._resolve_openai_max_output_tokens(),
            "temperature": self.config.temperature,
        }

        if self.config.top_p is not None:
            request_kwargs["top_p"] = self.config.top_p

        reasoning_config: dict[str, str] | None = self._build_openai_reasoning_config()
        if reasoning_config is not None:
            request_kwargs["reasoning"] = reasoning_config

        self._warn_if_openai_param_unsupported("top_k", self.config.top_k)
        self._warn_if_openai_param_unsupported(
            "presence_penalty",
            self.config.presence_penalty,
        )
        self._warn_if_openai_param_unsupported(
            "repetition_penalty",
            self.config.repetition_penalty,
        )
        self._warn_if_openai_param_unsupported(
            "enable_logprobs",
            self.config.enable_logprobs,
        )

        return request_kwargs

    def _build_openai_reasoning_config(self) -> dict[str, str] | None:
        """Build OpenAI reasoning config when enabled for the model."""
        effort: str | None = self.config.api_reasoning_effort
        reasoning_budget_tokens: int | None = self.config.api_reasoning_budget_tokens
        if (
            not self.config.is_thinking_enabled
            and not effort
            and reasoning_budget_tokens is None
        ):
            return None
        return {"effort": effort or "medium"}

    def _resolve_openai_max_output_tokens(self) -> int:
        """Resolve the combined OpenAI output cap including optional reasoning budget.

        OpenAI exposes only a single combined `max_output_tokens` limit covering both
        reasoning tokens and visible output tokens. When an explicit reasoning budget
        is configured, reserve that much additional room on top of the visible output
        target from `config.max_output_tokens`.
        """
        reasoning_budget_tokens: int | None = self.config.api_reasoning_budget_tokens
        if reasoning_budget_tokens is None:
            return self.config.max_output_tokens
        if reasoning_budget_tokens <= 0:
            raise ValueError("api_reasoning_budget_tokens must be greater than 0")

        return self.config.max_output_tokens + reasoning_budget_tokens

    def _should_use_batch_api(self, request_count: int) -> bool:
        """Return whether Batch API should be used for this request group."""
        return bool(self.config.api_use_batch) and request_count > 1

    def _execute_batch_requests(
        self,
        request_payloads: list[dict[str, object]],
    ) -> list[InferenceResult]:
        """Execute a group of OpenAI requests through the Batch API and wait for completion."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not loaded")
        if not request_payloads:
            return []

        started_at: float = time.time()
        self._log_generation_batch_start(len(request_payloads), "OpenAI Batch API")
        input_file_path: Path = self._write_batch_input_file(request_payloads)

        try:
            with input_file_path.open("rb") as input_file:
                uploaded_file: Any = self._openai_client.files.create(
                    file=input_file,
                    purpose="batch",
                )

            batch: Any = self._openai_client.batches.create(
                input_file_id=str(getattr(uploaded_file, "id")),
                endpoint=_BATCH_ENDPOINT,
                completion_window=(
                    self.config.api_batch_completion_window
                    or _DEFAULT_BATCH_COMPLETION_WINDOW
                ),
            )
            finished_batch: Any = self._wait_for_batch_completion(
                batch_id=str(getattr(batch, "id")),
                expected_requests=len(request_payloads),
            )
            return self._collect_batch_results(
                finished_batch=finished_batch,
                request_count=len(request_payloads),
                started_at=started_at,
            )
        finally:
            input_file_path.unlink(missing_ok=True)

    def _write_batch_input_file(
        self,
        request_payloads: list[dict[str, object]],
    ) -> Path:
        """Write OpenAI batch requests to a temporary JSONL file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            for index, request_payload in enumerate(request_payloads):
                json.dump(
                    {
                        "custom_id": self._build_batch_custom_id(index),
                        "method": "POST",
                        "url": _BATCH_ENDPOINT,
                        "body": request_payload,
                    },
                    temp_file,
                )
                temp_file.write("\n")

        return Path(temp_file.name)

    @staticmethod
    def _build_batch_custom_id(index: int) -> str:
        """Build a stable batch request identifier."""
        return f"request-{index}"

    def _wait_for_batch_completion(self, batch_id: str, expected_requests: int) -> Any:
        """Poll an OpenAI batch until it reaches a terminal state."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not loaded")

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
            batch: Any = self._openai_client.batches.retrieve(batch_id)
            status: str = str(getattr(batch, "status", "unknown"))
            self._log_batch_status(batch, expected_requests)

            if status == _BATCH_COMPLETED_STATUS:
                return batch
            if status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(
                    f"OpenAI batch {batch_id} ended with terminal status '{status}'"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for OpenAI batch {batch_id} after {max_wait_seconds:.0f}s"
                )

            time.sleep(max(poll_interval_seconds, 0.0))

    def _log_batch_status(self, batch: Any, expected_requests: int) -> None:
        """Log OpenAI batch progress from the retrieved batch object."""
        status: str = str(getattr(batch, "status", "unknown"))
        request_counts: Any = getattr(batch, "request_counts", None)
        completed: int = int(getattr(request_counts, "completed", 0) or 0)
        failed: int = int(getattr(request_counts, "failed", 0) or 0)
        total: int = int(getattr(request_counts, "total", expected_requests) or expected_requests)

        LOGGER.info(
            "OpenAI batch %s status=%s completed=%d failed=%d total=%d",
            str(getattr(batch, "id", "<unknown>")),
            status,
            completed,
            failed,
            total,
        )

    def _collect_batch_results(
        self,
        *,
        finished_batch: Any,
        request_count: int,
        started_at: float,
    ) -> list[InferenceResult]:
        """Download and order OpenAI batch results."""
        output_file_id: str | None = self._safe_str(getattr(finished_batch, "output_file_id", None))
        if output_file_id is None:
            raise RuntimeError("OpenAI batch completed without an output_file_id")

        output_text: str = self._read_file_text(output_file_id)
        error_file_id: str | None = self._safe_str(getattr(finished_batch, "error_file_id", None))
        error_text: str = self._read_file_text(error_file_id) if error_file_id else ""

        per_item_duration: float = (time.time() - started_at) / max(request_count, 1)
        successful_results: dict[str, InferenceResult] = self._parse_successful_batch_results(
            output_text=output_text,
            per_item_duration=per_item_duration,
        )
        failed_results: dict[str, InferenceResult] = self._parse_failed_batch_results(
            error_text=error_text,
            per_item_duration=per_item_duration,
        )

        ordered_results: list[InferenceResult] = []
        for index in range(request_count):
            custom_id: str = self._build_batch_custom_id(index)
            result: InferenceResult | None = successful_results.get(custom_id)
            if result is None:
                result = failed_results.get(custom_id)
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

    def _parse_successful_batch_results(
        self,
        *,
        output_text: str,
        per_item_duration: float,
    ) -> dict[str, InferenceResult]:
        """Parse successful result lines from an OpenAI batch output file."""
        parsed_results: dict[str, InferenceResult] = {}
        for line in output_text.splitlines():
            if not line.strip():
                continue

            payload: dict[str, Any] = json.loads(line)
            custom_id: str | None = self._safe_str(payload.get("custom_id"))
            response_payload: Any = payload.get("response")
            response_payload_dict: dict[str, Any] | None = (
                cast(dict[str, Any], response_payload)
                if isinstance(response_payload, dict)
                else None
            )
            response_body_obj: object = (
                response_payload_dict.get("body")
                if response_payload_dict is not None
                else None
            )
            if custom_id is None or not isinstance(response_body_obj, dict):
                continue
            response_body: dict[str, Any] = cast(dict[str, Any], response_body_obj)

            response_text: str = self._extract_openai_response_text_from_dict(response_body)
            token_count: int = self._extract_openai_total_tokens_from_dict(response_body)
            parsed_results[custom_id] = InferenceResult(
                response_text=response_text,
                tokens_used=token_count,
                duration=per_item_duration,
                confidence=None,
                binary_label_confidence=None,
            )

        return parsed_results

    def _parse_failed_batch_results(
        self,
        *,
        error_text: str,
        per_item_duration: float,
    ) -> dict[str, InferenceResult]:
        """Parse failed result lines from an OpenAI batch error file."""
        parsed_results: dict[str, InferenceResult] = {}
        for line in error_text.splitlines():
            if not line.strip():
                continue

            payload: dict[str, Any] = json.loads(line)
            custom_id: str | None = self._safe_str(payload.get("custom_id"))
            error_payload: Any = payload.get("error")
            if custom_id is None or not isinstance(error_payload, dict):
                continue
            error_payload_dict: dict[str, Any] = cast(dict[str, Any], error_payload)

            error_message: str = str(
                error_payload_dict.get("message", "Unknown batch error")
            )
            parsed_results[custom_id] = InferenceResult(
                response_text=f"{_BATCH_ERROR_TEXT_PREFIX}{error_message}",
                tokens_used=0,
                duration=per_item_duration,
                confidence=None,
                binary_label_confidence=None,
            )

        return parsed_results

    def _read_file_text(self, file_id: str | None) -> str:
        """Read an OpenAI file payload as text using several supported SDK shapes."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not loaded")
        if file_id is None:
            return ""

        file_response: Any = self._openai_client.files.content(file_id)

        text_attr: Any = getattr(file_response, "text", None)
        if callable(text_attr):
            return str(text_attr())
        if isinstance(text_attr, str):
            return text_attr

        read_attr: Any = getattr(file_response, "read", None)
        if callable(read_attr):
            raw_value: Any = read_attr()
            if isinstance(raw_value, bytes):
                return raw_value.decode("utf-8")
            return str(raw_value)

        content_attr: Any = getattr(file_response, "content", None)
        if isinstance(content_attr, bytes):
            return content_attr.decode("utf-8")
        if isinstance(content_attr, str):
            return content_attr

        if isinstance(file_response, bytes):
            return file_response.decode("utf-8")
        return str(file_response)

    @staticmethod
    def _safe_str(value: object) -> str | None:
        """Convert a possibly-empty object to a stripped string."""
        if value is None:
            return None
        text: str = str(value).strip()
        return text if text else None

    def _resolve_openai_encoding(self) -> tiktoken.Encoding:
        """Resolve a tiktoken encoding for the configured OpenAI model."""
        try:
            return tiktoken.encoding_for_model(self.config.model_identifier)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")

    def _extract_openai_response_text(self, response: Any) -> str:
        """Extract visible text from an OpenAI Responses API result."""
        output_text: Any = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()

        raw_output_items: Any = getattr(response, "output", None)
        if not isinstance(raw_output_items, list):
            return ""
        return self._extract_openai_response_text_from_dict({"output": raw_output_items})

    def _extract_openai_response_text_from_dict(self, response_body: dict[str, Any]) -> str:
        """Extract visible text from a dict-shaped OpenAI response payload."""
        output_text: object = response_body.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()

        raw_output_items: object = response_body.get("output")
        if not isinstance(raw_output_items, list):
            return ""
        output_items: list[object] = cast(list[object], raw_output_items)

        text_fragments: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_dict: dict[str, Any] = cast(dict[str, Any], item)
            item_type: str = str(item_dict.get("type", ""))
            if item_type != "message":
                continue

            raw_content_blocks: object = item_dict.get("content")
            if not isinstance(raw_content_blocks, list):
                continue
            content_blocks: list[object] = cast(list[object], raw_content_blocks)

            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_dict: dict[str, Any] = cast(dict[str, Any], block)
                block_type: str = str(block_dict.get("type", ""))
                if block_type == "output_text":
                    block_text: str = str(block_dict.get("text", "")).strip()
                    if block_text:
                        text_fragments.append(block_text)

        return "\n\n".join(text_fragments)

    def _extract_openai_total_tokens(self, response: Any) -> int:
        """Extract total token usage from an OpenAI response."""
        usage: Any = getattr(response, "usage", None)
        total_tokens: Any = getattr(usage, "total_tokens", None)
        if isinstance(total_tokens, int):
            return total_tokens
        if isinstance(total_tokens, float):
            return int(total_tokens)
        return 0

    def _extract_openai_total_tokens_from_dict(self, response_body: dict[str, Any]) -> int:
        """Extract total token usage from a dict-shaped OpenAI response payload."""
        usage: object = response_body.get("usage")
        if not isinstance(usage, dict):
            return 0
        usage_dict: dict[str, Any] = cast(dict[str, Any], usage)
        total_tokens: object = usage_dict.get("total_tokens")
        if isinstance(total_tokens, int):
            return total_tokens
        if isinstance(total_tokens, float):
            return int(total_tokens)
        return 0

    def _warn_if_openai_param_unsupported(
        self,
        param_name: str,
        value: object,
    ) -> None:
        """Warn when an OpenAI parameter is configured but unsupported."""
        if not self._is_configured_value(value):
            return

        reason: str = "OpenAI Responses API does not expose this parameter"
        if param_name == "enable_logprobs":
            reason = "OpenAI logprobs are not implemented in this backend"
        self._warn_unsupported_sampling_param(param_name, reason)

    def count_input_tokens(self, text: str) -> int:
        """Count approximate OpenAI tokens for the provided text."""
        cached_token_count: int | None = self._token_count_cache.get(text)
        if cached_token_count is not None:
            return cached_token_count

        if self._openai_encoding is None:
            raise RuntimeError("OpenAI tokenizer not loaded")
        token_count: int = len(self._openai_encoding.encode(text))
        self._token_count_cache[text] = token_count
        return token_count

    def cleanup(self) -> None:
        """Release OpenAI backend client references."""
        self._openai_client = None
        self._openai_encoding = None