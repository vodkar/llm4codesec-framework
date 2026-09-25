"""Plain-python checks: vLLM realized prompt token count is None when unknown.

``torch`` is not installed on the host, so a bare stub module stands in for it
(only after the real ``transformers`` import, which probes torch's spec).

Run: PYTHONPATH=src uv run python tests/test_vllm_prompt_tokens.py
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import transformers  # noqa: F401  (must be imported before the torch stub)

sys.modules.setdefault("torch", types.ModuleType("torch"))

from llm.vllm_backend import VllmLLM  # noqa: E402  (after the torch stub)


def _collect(output: SimpleNamespace, prompts: list[str] | None = None) -> int | None:
    backend = VllmLLM.__new__(VllmLLM)
    results = VllmLLM._collect_batch_results(backend, [output], 0.0, 1, prompts=prompts)
    return results[0].prompt_tokens


def test_prompt_tokens_none_when_token_ids_missing() -> None:
    output = SimpleNamespace(prompt="P", prompt_token_ids=None, outputs=[])
    assert _collect(output) is None


def test_prompt_tokens_counts_token_ids() -> None:
    output = SimpleNamespace(prompt=None, prompt_token_ids=[1, 2, 3], outputs=[])
    assert _collect(output, prompts=["X"]) == 3


def test_prompt_tokens_zero_for_empty_token_ids() -> None:
    output = SimpleNamespace(prompt="P", prompt_token_ids=[], outputs=[])
    assert _collect(output) == 0


if __name__ == "__main__":
    test_prompt_tokens_none_when_token_ids_missing()
    test_prompt_tokens_counts_token_ids()
    test_prompt_tokens_zero_for_empty_token_ids()
    print("ALL PASSED")
