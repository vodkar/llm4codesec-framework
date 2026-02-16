from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Final

from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)

from benchmark.config import BenchmarkConfig
from llm.llm import ILLMInference

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# Default directory for downloaded GGUF models
_DEFAULT_MODEL_DIR: Final[str] = "llm_models"


class LlamaCppLLM(ILLMInference):
    """llama.cpp-based LLM inference backend.

    Supports loading GGUF models from:
    - Local file paths (e.g., ``./models/model.gguf``)
    - Hugging Face repos via ``Llama.from_pretrained()``
      (e.g., ``hf://unsloth/Qwen3-Coder-Next-GGUF/Q4_K_M``)
    - Pre-downloading with ``llama-cli`` through ``download_model_via_cli()``.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Initialize the llama.cpp backend.

        Args:
            config: Benchmark configuration for model and generation settings.
        """
        self.config: BenchmarkConfig = config
        self.model: Llama | None = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the llama.cpp model from a local path or HuggingFace repo."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            LOGGER.exception("llama-cpp-python is not installed")
            raise RuntimeError(
                "llama.cpp backend selected but llama-cpp-python is not installed. "
                "Install with `pip install llama-cpp-python`."
            ) from exc

        model_ref: str = self.config.model_name
        if not model_ref:
            raise ValueError("model_name must be set for llama.cpp backend")

        n_ctx: int = int(os.getenv("LLAMA_CPP_N_CTX", "4096"))
        n_threads: int | None = self._get_int_env("LLAMA_CPP_N_THREADS")
        n_batch: int | None = self._get_int_env("LLAMA_CPP_N_BATCH")
        n_gpu_layers: int | None = self._get_int_env("LLAMA_CPP_N_GPU_LAYERS")

        common_kwargs: dict[str, Any] = {
            k: v
            for k, v in {
                "n_ctx": n_ctx,
                "n_threads": n_threads,
                "n_batch": n_batch,
                "n_gpu_layers": n_gpu_layers,
            }.items()
            if v is not None
        }

        if self._is_hf_reference(model_ref):
            self.model = self._load_from_huggingface(model_ref, common_kwargs)
        else:
            resolved_path: str = str(Path(model_ref).expanduser())
            LOGGER.info("Loading llama.cpp model from local path: %s", resolved_path)
            self.model = Llama(model_path=resolved_path, **common_kwargs)

    @staticmethod
    def _is_hf_reference(model_ref: str) -> bool:
        """Check whether ``model_ref`` points to a Hugging Face repo."""
        return model_ref.startswith("hf://") or model_ref.startswith("hf:")

    def _load_from_huggingface(self, model_ref: str, kwargs: dict[str, Any]) -> Llama:
        """Load a GGUF model directly from Hugging Face via ``Llama.from_pretrained``.

        Args:
            model_ref: HF reference in ``hf://owner/repo/filename.gguf`` or
                ``hf://owner/repo/*pattern*`` format.
            kwargs: Extra keyword arguments forwarded to ``Llama``.

        Returns:
            Loaded ``Llama`` model instance.
        """
        from llama_cpp import Llama

        repo_id, filename = self.parse_hf_reference(model_ref)

        model_dir: Path = Path(os.getenv("LLAMA_CPP_MODEL_DIR", _DEFAULT_MODEL_DIR))
        model_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Loading GGUF from HuggingFace: repo=%s  file=%s  cache=%s",
            repo_id,
            filename,
            model_dir,
        )
        return Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            **kwargs,
        )

    @staticmethod
    def parse_hf_reference(model_ref: str) -> tuple[str, str]:
        """Parse an ``hf://owner/repo/filename`` reference.

        Args:
            model_ref: HF-style reference string.

        Returns:
            ``(repo_id, filename)`` tuple.  ``filename`` may contain
            glob patterns (e.g. ``*Q4_K_M*``).

        Raises:
            ValueError: If the reference cannot be parsed.
        """
        normalized: str = model_ref.removeprefix("hf://").removeprefix("hf:").strip("/")

        parts: list[str] = [p for p in normalized.split("/") if p]
        if len(parts) < 3:
            raise ValueError(
                f"HF reference must be hf://owner/repo/filename, got: {model_ref}"
            )

        filename: str = parts[-1]
        repo_id: str = "/".join(parts[:2])
        return repo_id, filename

    # ------------------------------------------------------------------
    # CLI-based model downloading (llama-cli / huggingface-cli)
    # ------------------------------------------------------------------

    @staticmethod
    def download_model_via_cli(
        repo_id: str,
        filename: str,
        *,
        local_dir: str | Path | None = None,
    ) -> Path:
        """Download a GGUF from Hugging Face using the ``llama-cli`` binary.

        Falls back to ``huggingface-cli download`` when ``llama-cli`` is not
        available on ``$PATH``.

        This method performs **download only** — the model is NOT loaded into
        memory.  Call ``_load_model`` or ``Llama()`` afterwards.

        Args:
            repo_id: Hugging Face repository (e.g. ``unsloth/Qwen3-Coder-Next-GGUF``).
            filename: GGUF file name (e.g. ``Qwen3-Coder-Next-Q4_K_M.gguf``).
            local_dir: Directory to store the downloaded GGUF.
                Defaults to ``$LLAMA_CPP_MODEL_DIR`` or ``llm_models``.

        Returns:
            Path to the downloaded GGUF file.

        Raises:
            FileNotFoundError: If neither ``llama-cli`` nor ``huggingface-cli``
                is found.
            subprocess.CalledProcessError: If the download command fails.
        """
        dest: Path = Path(
            local_dir or os.getenv("LLAMA_CPP_MODEL_DIR", _DEFAULT_MODEL_DIR)
        )
        dest.mkdir(parents=True, exist_ok=True)

        llama_cli: str | None = shutil.which("llama-cli")
        hf_cli: str | None = shutil.which("huggingface-cli")

        if llama_cli:
            return LlamaCppLLM._download_with_llama_cli(
                llama_cli, repo_id, filename, dest
            )
        if hf_cli:
            return LlamaCppLLM._download_with_hf_cli(hf_cli, repo_id, filename, dest)

        raise FileNotFoundError(
            "Neither `llama-cli` nor `huggingface-cli` found on $PATH.  "
            "Install llama.cpp (`brew install llama.cpp` / build from source) "
            "or `pip install huggingface-hub[cli]`."
        )

    @staticmethod
    def _download_with_llama_cli(
        cli_path: str,
        repo_id: str,
        filename: str,
        dest: Path,
    ) -> Path:
        """Download a GGUF using ``llama-cli --hf-repo``.

        ``llama-cli`` supports ``--hf-repo`` and ``--hf-file`` flags that
        automatically download the model from Hugging Face before running
        inference.  We use ``-n 0`` to exit immediately after download.

        Args:
            cli_path: Absolute path to the ``llama-cli`` binary.
            repo_id: HF repo id.
            filename: GGUF filename inside the repo.
            dest: Target directory.

        Returns:
            Path to the downloaded file.
        """
        LOGGER.info("Downloading via llama-cli: %s/%s -> %s", repo_id, filename, dest)
        cmd: list[str] = [
            cli_path,
            "--hf-repo",
            repo_id,
            "--hf-file",
            filename,
            "-n",
            "0",  # generate 0 tokens → download only
            "--model",
            str(dest / filename),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        downloaded: Path = dest / filename
        if not downloaded.exists():
            # llama-cli may cache in HF cache; search common locations
            hf_cache: Path = Path.home() / ".cache" / "huggingface" / "hub"
            candidates: list[Path] = list(hf_cache.rglob(filename))
            if candidates:
                downloaded = candidates[0]
        return downloaded

    @staticmethod
    def _download_with_hf_cli(
        cli_path: str,
        repo_id: str,
        filename: str,
        dest: Path,
    ) -> Path:
        """Download a GGUF using ``huggingface-cli download``.

        Args:
            cli_path: Absolute path to ``huggingface-cli``.
            repo_id: HF repo id.
            filename: GGUF filename.
            dest: Target directory.

        Returns:
            Path to the downloaded file.
        """
        LOGGER.info(
            "Downloading via huggingface-cli: %s/%s -> %s",
            repo_id,
            filename,
            dest,
        )
        cmd: list[str] = [
            cli_path,
            "download",
            repo_id,
            filename,
            "--local-dir",
            str(dest),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return dest / filename

    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int, float]:
        """
        Generate a single response using llama.cpp.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.

        Returns:
            tuple[str, int, float]: Response text, token count, and duration.
        """
        if not self.model:
            raise RuntimeError("llama.cpp model not loaded")

        start_time: float = time.time()
        response_text: str
        token_count: int

        try:
            response_text, token_count = self._generate_chat(system_prompt, user_prompt)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            LOGGER.warning(
                "Chat completion failed (%s); falling back to text prompt",
                type(exc).__name__,
            )
            response_text, token_count = self._generate_completion(
                system_prompt, user_prompt
            )

        duration: float = time.time() - start_time
        return response_text, token_count, duration

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for multiple system/user prompt pairs.

        Args:
            system_prompts: List of system prompts.
            user_prompts: List of user prompts.

        Returns:
            List of (response_text, token_count, duration) tuples.
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        results: list[tuple[str, int, float]] = []
        for system_prompt, user_prompt in zip(system_prompts, user_prompts):
            results.append(self.generate_response(system_prompt, user_prompt))
        return results

    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for a batch of formatted prompts.

        Args:
            prompts: List of formatted prompts.

        Returns:
            List of (response_text, token_count, duration) tuples.
        """
        if not self.model:
            raise RuntimeError("llama.cpp model not loaded")

        results: list[tuple[str, int, float]] = []
        for prompt in prompts:
            start_time: float = time.time()
            output: CreateCompletionResponse = self.model(  # type: ignore[assignment]
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            duration: float = time.time() - start_time

            response_text: str = self._extract_completion_text(output)
            token_count: int = self._extract_usage_tokens(output)
            results.append((response_text, token_count, duration))

        return results

    def _generate_chat(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Generate response using chat completion if supported."""
        if not self.model:
            raise RuntimeError("llama.cpp model not loaded")

        messages: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response: CreateChatCompletionResponse = self.model.create_chat_completion(  # type: ignore[assignment]
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        choices = response.get("choices", [])
        content: str | None = None
        if choices:
            message = choices[0].get("message")
            if message:
                content = message.get("content")
        response_text: str = (content or "").strip()
        token_count: int = self._extract_usage_tokens(response)
        return response_text.strip(), token_count

    def _generate_completion(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int]:
        """Generate response using a formatted text prompt."""
        if not self.model:
            raise RuntimeError("llama.cpp model not loaded")

        prompt: str = self._format_prompt(system_prompt, user_prompt)
        response: CreateCompletionResponse = self.model(  # type: ignore[assignment]
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        response_text: str = self._extract_completion_text(response)
        token_count: int = self._extract_usage_tokens(response)
        return response_text, token_count

    @staticmethod
    def _extract_completion_text(response: CreateCompletionResponse) -> str:
        """Extract text from llama.cpp completion output."""
        choices = response.get("choices", [])
        if choices:
            first_choice = choices[0]
            text_value = first_choice.get("text", "")
            return str(text_value).strip()
        return ""

    @staticmethod
    def _extract_usage_tokens(
        response: CreateCompletionResponse | CreateChatCompletionResponse,
    ) -> int:
        """Extract token usage from llama.cpp output.

        Returns:
            Total token count, or 0 if usage info is unavailable.
        """
        usage = response.get("usage")
        if usage is not None:
            return int(usage.get("total_tokens", 0))
        return 0

    @staticmethod
    def _format_prompt(system_prompt: str, user_prompt: str) -> str:
        """Format a generic prompt for text completion."""
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    @staticmethod
    def _get_int_env(env_name: str) -> int | None:
        """Fetch an integer from environment variables when present."""
        raw_value = os.getenv(env_name)
        if raw_value is None:
            return None
        return int(raw_value)

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
            LOGGER.info("llama.cpp model resources released")
