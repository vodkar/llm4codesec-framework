from __future__ import annotations

import gc
import fnmatch
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import PreTrainedTokenizerBase

from benchmark.config import ExperimentConfig
from llm.llm import ILLMInference

if TYPE_CHECKING:
    from vllm import LLM, RequestOutput, SamplingParams

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_DEFAULT_MODEL_DIR: Final[str] = "~/.cache/huggingface"


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
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        try:
            from vllm import LLM
        except ImportError as exc:
            LOGGER.exception("vLLM is not installed")
            raise RuntimeError(
                "vLLM backend selected but vLLM is not installed. "
                "Install with `pip install vllm`."
            ) from exc

        LOGGER.info("Loading vLLM model: %s", self.config.model_identifier)

        model_identifier, inferred_hf_config_path = self._resolve_model_identifier()
        model_dtype: str = self._resolve_model_dtype(model_identifier)
        self._ensure_multimodal_gguf_artifacts(model_dtype)

        configured_quantization = self.config.vllm_quantization
        quantization: str | None
        if configured_quantization == "none":
            quantization = None
        elif configured_quantization:
            quantization = configured_quantization
        else:
            quantization = "fp8" if self.config.use_quantization else None

        configured_kv_cache_dtype = self.config.kv_cache_dtype
        if configured_kv_cache_dtype == "none":
            kv_cache_dtype = None
        elif configured_kv_cache_dtype:
            kv_cache_dtype = configured_kv_cache_dtype
        else:
            kv_cache_dtype = "fp8" if quantization == "fp8" else "auto"

        max_num_seqs: int = self.config.max_num_seqs or int(
            os.getenv("MAX_NUM_SEQS", "32")
        )
        gpu_memory_utilization: float = self.config.gpu_memory_utilization or 0.82
        is_gguf_model: bool = self._is_gguf_model_reference(model_identifier)
        enforce_eager: bool = (
            self.config.enforce_eager
            if self.config.enforce_eager is not None
            else is_gguf_model
        )
        max_num_batched_tokens: int | None = self.config.max_num_batched_tokens
        if max_num_batched_tokens is None and is_gguf_model:
            max_num_batched_tokens = min(2048, self.config.model_context_length_tokens)

        llm_kwargs: dict[str, object] = dict(
            model=model_identifier,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.config.model_context_length_tokens,
            dtype=model_dtype,
            enable_prefix_caching=True if self.config.enable_prefix_caching is None else self.config.enable_prefix_caching,
            enforce_eager=enforce_eager,
        )

        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

        if quantization is not None:
            llm_kwargs["quantization"] = quantization

        if kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

        if self.config.tokenizer_identifier:
            llm_kwargs["tokenizer"] = self.config.tokenizer_identifier

        hf_config_path: str | None = self.config.hf_config_path or inferred_hf_config_path
        if hf_config_path:
            llm_kwargs["hf_config_path"] = hf_config_path

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        if self.tokenizer is None:
            raise RuntimeError("vLLM tokenizer failed to load")

    @staticmethod
    def _is_hf_reference(model_ref: str) -> bool:
        """Check whether a model reference uses the hf:// shorthand."""
        return model_ref.startswith("hf://") or model_ref.startswith("hf:")

    @staticmethod
    def _parse_hf_reference(model_ref: str) -> tuple[str, str]:
        """Parse an hf://owner/repo/filename reference into repo and filename."""
        normalized: str = model_ref.removeprefix("hf://").removeprefix("hf:").strip(
            "/"
        )
        parts: list[str] = [part for part in normalized.split("/") if part]

        if len(parts) < 3:
            raise ValueError(
                f"HF reference must be hf://owner/repo/filename, got: {model_ref}"
            )

        filename: str = parts[-1]
        repo_id: str = "/".join(parts[:2])
        return repo_id, filename

    def _resolve_model_identifier(self) -> tuple[str, str | None]:
        """Resolve the configured model identifier for vLLM loading."""
        model_ref: str = self.config.model_identifier
        if not self._is_hf_reference(model_ref):
            return model_ref, None

        repo_id, filename_pattern = self._parse_hf_reference(model_ref)
        resolved_filename: str = self._resolve_hf_filename(repo_id, filename_pattern)
        cache_dir: Path = Path(os.getenv("HF_HOME", _DEFAULT_MODEL_DIR))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_model_path: Path = cache_dir / resolved_filename

        if cached_model_path.is_file():
            LOGGER.info(
                "Using pre-downloaded vLLM HF model file %s",
                cached_model_path,
            )
            local_model_path = str(cached_model_path)
        else:
            local_model_path = hf_hub_download(
                repo_id=repo_id,
                filename=resolved_filename,
                local_dir=str(cache_dir),
            )
        inferred_hf_config_path: str = (
            self.config.tokenizer_identifier or self.config.hf_config_path or repo_id
        )

        LOGGER.info(
            "Resolved vLLM HF model %s to local file %s",
            model_ref,
            local_model_path,
        )
        return local_model_path, inferred_hf_config_path

    def _ensure_multimodal_gguf_artifacts(self, model_dtype: str) -> None:
        """Download required sidecar artifacts for multimodal GGUF models."""
        if not self._requires_mmproj_artifact():
            return

        if not self._is_hf_reference(self.config.model_identifier):
            return

        repo_id, _ = self._parse_hf_reference(self.config.model_identifier)
        mmproj_filename: str = self._resolve_mmproj_filename(repo_id, model_dtype)
        cache_dir: Path = Path(os.getenv("HF_HOME", _DEFAULT_MODEL_DIR))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_mmproj_path: Path = cache_dir / mmproj_filename

        if cached_mmproj_path.is_file():
            local_mmproj_path = str(cached_mmproj_path)
        else:
            local_mmproj_path = hf_hub_download(
                repo_id=repo_id,
                filename=mmproj_filename,
                local_dir=str(cache_dir),
            )
        LOGGER.info(
            "Resolved multimodal GGUF sidecar %s to local file %s",
            mmproj_filename,
            local_mmproj_path,
        )

    def _resolve_model_dtype(self, model_identifier: str) -> str:
        """Choose a vLLM dtype compatible with the configured model format."""
        if self._requires_float32_dtype(model_identifier):
            return "float32"

        if self._requires_bfloat16_dtype():
            return "bfloat16"

        if self._is_gguf_model_reference(model_identifier):
            return "float16"

        return "bfloat16"

    def _requires_bfloat16_dtype(self) -> bool:
        """Check whether the configured model should prefer bfloat16 in vLLM."""
        model_type_value: str = getattr(self.config.model_type, "value", "")
        normalized_model_type: str = str(model_type_value).lower()
        if normalized_model_type == "gemma-3":
            return True

        candidate_identifiers: list[str] = [
            self.config.model_identifier,
            self.config.tokenizer_identifier or "",
            self.config.hf_config_path or "",
        ]
        return any("gemma" in identifier.lower() for identifier in candidate_identifiers)

    def _requires_float32_dtype(self, model_identifier: str) -> bool:
        """Check whether the configured model must use float32 in vLLM."""
        return self._is_gguf_model_reference(model_identifier) and self._requires_bfloat16_dtype()

    def _requires_mmproj_artifact(self) -> bool:
        """Check whether the configured model requires an mmproj GGUF sidecar."""
        if not self._is_gguf_model_reference(self.config.model_identifier):
            return False

        model_type_value: str = getattr(self.config.model_type, "value", "")
        return str(model_type_value).lower() == "gemma-3"

    def _is_gguf_model_reference(self, model_identifier: str) -> bool:
        """Check whether the resolved model identifier points to a GGUF model."""
        normalized_identifier: str = model_identifier.lower()

        if normalized_identifier.endswith(".gguf"):
            return True

        original_identifier: str = self.config.model_identifier.lower()
        if original_identifier.endswith(".gguf"):
            return True

        if self._is_hf_reference(self.config.model_identifier):
            _, filename = self._parse_hf_reference(self.config.model_identifier)
            return filename.lower().endswith(".gguf")

        return False

    @staticmethod
    def _resolve_hf_filename(repo_id: str, filename_pattern: str) -> str:
        """Resolve an HF filename or glob pattern to a single repo file."""
        repo_files: list[str] = list_repo_files(repo_id=repo_id)

        if any(char in filename_pattern for char in "*?[]"):
            matches: list[str] = [
                repo_file
                for repo_file in repo_files
                if fnmatch.fnmatch(repo_file, filename_pattern)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"No files matched pattern '{filename_pattern}' in repo '{repo_id}'"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Pattern '{filename_pattern}' matched multiple files in '{repo_id}': "
                    f"{matches}"
                )
            return matches[0]

        if filename_pattern not in repo_files:
            raise FileNotFoundError(
                f"File '{filename_pattern}' not found in repo '{repo_id}'"
            )

        return filename_pattern

    @classmethod
    def _resolve_mmproj_filename(cls, repo_id: str, model_dtype: str) -> str:
        """Resolve the matching mmproj sidecar file for a multimodal GGUF model."""
        dtype_suffix_by_dtype: dict[str, str] = {
            "float16": "F16",
            "float32": "F32",
            "bfloat16": "BF16",
        }
        dtype_suffix: str = dtype_suffix_by_dtype.get(model_dtype, "F16")
        preferred_pattern: str = f"mmproj-{dtype_suffix}.gguf"

        try:
            return cls._resolve_hf_filename(repo_id, preferred_pattern)
        except FileNotFoundError:
            return cls._resolve_hf_filename(repo_id, "mmproj-*.gguf")

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

        sampling_params = self._create_sampling_params(SamplingParams)

        LOGGER.info("Submitting %d prompts to vLLM scheduler", len(prompts))
        start_time: float = time.time()
        all_outputs: list[RequestOutput] = self.llm.generate(prompts, sampling_params)
        duration: float = time.time() - start_time

        return self._collect_batch_results(
            batch_outputs=all_outputs,
            duration=duration,
            batch_size=len(prompts),
        )

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
        sampling_kwargs: dict[str, int | float] = {
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }

        optional_sampling_values: dict[str, int | float | None] = {
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "presence_penalty": self.config.presence_penalty,
            "repetition_penalty": self.config.repetition_penalty,
        }
        for key, value in optional_sampling_values.items():
            if value is not None:
                sampling_kwargs[key] = value

        return sampling_params_cls(**sampling_kwargs)

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
        has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        if callable(apply_chat_template) and has_chat_template:
            try:
                if self.config.is_thinking_enabled:
                    formatted_prompt = apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
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
            self.llm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
