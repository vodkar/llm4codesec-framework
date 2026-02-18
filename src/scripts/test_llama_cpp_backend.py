#!/usr/bin/env python3
"""End-to-end test for the llama.cpp backend with Qwen/Qwen3-8B-GGUF.

This script validates:
1. CLI-based model downloading (llama-cli / huggingface-cli)
2. Model loading via ``Llama.from_pretrained()``
3. Chat completion and text completion inference
4. Batch response generation
5. Resource cleanup

Usage
-----
Run directly::

    python -m scripts.test_llama_cpp_backend

Override quant (default Q4_K_M — 5.03 GB)::

    GGUF_QUANT=Q8_0 python -m scripts.test_llama_cpp_backend

Override context length for constrained hardware::

    LLAMA_CPP_N_CTX=2048 python -m scripts.test_llama_cpp_backend

Skip download and use existing GGUF::

    GGUF_LOCAL_PATH=/path/to/model.gguf python -m scripts.test_llama_cpp_backend
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on the import path when running as a script
# ---------------------------------------------------------------------------
_SRC_DIR: Final[Path] = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from benchmark.config import ExperimentConfig  # noqa: E402
from benchmark.enums import ModelType, TaskType  # noqa: E402
from llm.llama_cpp_backend import LlamaCppLLM  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID: Final[str] = "Qwen/Qwen3-8B-GGUF"

# Map quant labels → GGUF filenames published by Qwen
_QUANT_MAP: Final[dict[str, str]] = {
    "Q4_K_M": "Qwen3-8B-Q4_K_M.gguf",
    "Q5_0": "Qwen3-8B-Q5_0.gguf",
    "Q5_K_M": "Qwen3-8B-Q5_K_M.gguf",
    "Q6_K": "Qwen3-8B-Q6_K.gguf",
    "Q8_0": "Qwen3-8B-Q8_0.gguf",
}

DEFAULT_QUANT: Final[str] = "Q4_K_M"


def _resolve_gguf_filename() -> str:
    """Resolve the GGUF filename from ``$GGUF_QUANT`` env variable.

    Returns:
        GGUF filename string.

    Raises:
        ValueError: If the requested quant is not in ``_QUANT_MAP``.
    """
    quant: str = os.getenv("GGUF_QUANT", DEFAULT_QUANT).upper()
    if quant not in _QUANT_MAP:
        raise ValueError(
            f"Unknown quant '{quant}'. Choose from: {sorted(_QUANT_MAP.keys())}"
        )
    return _QUANT_MAP[quant]


# ---------------------------------------------------------------------------
# Test: CLI download
# ---------------------------------------------------------------------------


def test_cli_download() -> Path | None:
    """Attempt to download the GGUF via llama-cli / huggingface-cli.

    Returns:
        Path to the downloaded GGUF, or ``None`` if no CLI is available.
    """
    filename: str = _resolve_gguf_filename()
    LOGGER.info("=== Test: CLI-based download ===")
    LOGGER.info("Repo: %s  File: %s", REPO_ID, filename)

    try:
        downloaded: Path = LlamaCppLLM.download_model_via_cli(
            repo_id=REPO_ID,
            filename=filename,
        )
        LOGGER.info("CLI download succeeded: %s", downloaded)
        return downloaded
    except FileNotFoundError:
        LOGGER.warning(
            "Neither llama-cli nor huggingface-cli found on $PATH — "
            "skipping CLI download test.  Model will be fetched via "
            "Llama.from_pretrained() during load."
        )
        return None
    except Exception:
        LOGGER.exception("CLI download failed")
        return None


# ---------------------------------------------------------------------------
# Test: Load + inference via from_pretrained
# ---------------------------------------------------------------------------


def test_load_and_infer() -> None:
    """Load the model and run inference tests."""
    local_path: str | None = os.getenv("GGUF_LOCAL_PATH")
    filename: str = _resolve_gguf_filename()

    if local_path:
        model_name: str = local_path
        LOGGER.info("Using local GGUF: %s", model_name)
    else:
        model_name = f"hf://{REPO_ID}/{filename}"
        LOGGER.info("Using HuggingFace reference: %s", model_name)

    config: ExperimentConfig = ExperimentConfig(
        model_name=model_name,
        model_type=ModelType.CUSTOM,
        task_type=TaskType.BINARY_VULNERABILITY,
        description=f"Qwen3-8B GGUF test ({filename})",
        dataset_path=Path("datasets_processed/castle/castle_binary.json"),
        output_dir=Path("results/llama_cpp_test"),
        backend="llama_cpp",
        batch_size=1,
        max_tokens=128,
        temperature=0.1,
        use_quantization=False,
        system_prompt_template=(
            "You are a security expert. Analyze the following code snippet "
            "and determine if it contains a vulnerability."
        ),
        user_prompt_template="Code:\n```c\n{code}\n```\nIs this code vulnerable? Answer YES or NO.",
    )

    LOGGER.info("=== Test: Model loading ===")
    start: float = time.time()
    backend: LlamaCppLLM = LlamaCppLLM(config)
    load_time: float = time.time() - start
    LOGGER.info("Model loaded in %.2f s", load_time)
    assert backend.model is not None, "Model should not be None after loading"

    # --- Single inference (chat completion) ---
    LOGGER.info("=== Test: Single inference (chat) ===")
    system_prompt: str = (
        "You are a helpful assistant specialized in code security analysis."
    )
    user_prompt: str = (
        "Is the following C code vulnerable?\n"
        "```c\n"
        "void copy(char *dst, char *src) {\n"
        "    strcpy(dst, src);\n"
        "}\n"
        "```\n"
        "Answer YES or NO and briefly explain."
    )
    response_text, token_count, duration = backend.generate_response(
        system_prompt, user_prompt
    )
    LOGGER.info(
        "Response (%d tokens, %.2f s):\n%s",
        token_count,
        duration,
        response_text[:500],
    )
    assert len(response_text) > 0, "Response should not be empty"
    assert token_count >= 0, "Token count should be non-negative"
    assert duration > 0, "Duration should be positive"

    # --- Batch inference ---
    LOGGER.info("=== Test: Batch inference ===")
    prompts: list[str] = [
        "System: You are a code reviewer.\n\nUser: Is `gets()` safe?\n\nAssistant:",
        "System: You are a code reviewer.\n\nUser: Is `fgets()` safe?\n\nAssistant:",
    ]
    batch_results: list[tuple[str, int, float]] = backend.generate_batch_responses(
        prompts
    )
    assert len(batch_results) == 2, "Should have 2 batch results"
    for i, (text, tokens, dur) in enumerate(batch_results):
        LOGGER.info("Batch[%d]: %d tokens, %.2f s — %s", i, tokens, dur, text[:200])

    # --- Batch-optimized inference ---
    LOGGER.info("=== Test: Batch-optimized inference ===")
    sys_prompts: list[str] = [system_prompt, system_prompt]
    usr_prompts: list[str] = [
        "Is `malloc()` without `free()` a vulnerability?",
        "Is `printf(user_input)` a vulnerability?",
    ]
    opt_results: list[tuple[str, int, float]] = (
        backend.generate_responses_batch_optimized(sys_prompts, usr_prompts)
    )
    assert len(opt_results) == 2, "Should have 2 optimized results"
    for i, (text, tokens, dur) in enumerate(opt_results):
        LOGGER.info("Opt[%d]: %d tokens, %.2f s — %s", i, tokens, dur, text[:200])

    # --- Cleanup ---
    LOGGER.info("=== Test: Cleanup ===")
    backend.cleanup()
    assert backend.model is None, "Model should be None after cleanup"
    LOGGER.info("Cleanup verified — model reference is None")


# ---------------------------------------------------------------------------
# Test: _parse_hf_reference
# ---------------------------------------------------------------------------


def test_parse_hf_reference() -> None:
    """Unit-test the HF reference parser."""
    LOGGER.info("=== Test: _parse_hf_reference ===")

    # Standard three-part reference
    repo, fname = LlamaCppLLM.parse_hf_reference(
        "hf://Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
    )
    assert repo == "Qwen/Qwen3-8B-GGUF", f"Unexpected repo: {repo}"
    assert fname == "Qwen3-8B-Q4_K_M.gguf", f"Unexpected filename: {fname}"

    # With glob pattern
    repo2, fname2 = LlamaCppLLM.parse_hf_reference("hf://Qwen/Qwen3-8B-GGUF/*Q4_K_M*")
    assert repo2 == "Qwen/Qwen3-8B-GGUF"
    assert fname2 == "*Q4_K_M*"

    # hf: prefix (no double slash)
    repo3, fname3 = LlamaCppLLM.parse_hf_reference(
        "hf:Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf"
    )
    assert repo3 == "Qwen/Qwen3-8B-GGUF"
    assert fname3 == "Qwen3-8B-Q8_0.gguf"

    # Too few parts should raise
    raised: bool = False
    try:
        LlamaCppLLM.parse_hf_reference("hf://repo_only")
    except ValueError:
        raised = True
    assert raised, "Should raise ValueError for malformed reference"

    LOGGER.info("All parser tests passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all tests."""
    LOGGER.info("=" * 72)
    LOGGER.info("llama.cpp backend test — Qwen/Qwen3-8B-GGUF")
    LOGGER.info("=" * 72)

    # 1. Unit tests (no model required)
    test_parse_hf_reference()

    # 2. CLI download test (optional — depends on CLI availability)
    cli_result: Path | None = test_cli_download()
    if cli_result:
        LOGGER.info("CLI download artifact: %s", cli_result)

    # 3. Full load + inference test
    test_load_and_infer()

    LOGGER.info("=" * 72)
    LOGGER.info("ALL TESTS PASSED")
    LOGGER.info("=" * 72)


if __name__ == "__main__":
    main()
