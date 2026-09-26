"""Plain-python checks: per-model vLLM overrides (files to skip, banned output strings).

``torch`` is not installed on the host, so a bare stub module stands in for it
(only after the real ``transformers`` import, which probes torch's spec).

Run: PYTHONPATH=src uv run python tests/test_vllm_model_overrides.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import transformers  # noqa: F401  (must be imported before the torch stub)

sys.modules.setdefault("torch", types.ModuleType("torch"))

from benchmark.config import ExperimentConfig  # noqa: E402
from llm.vllm_backend import VllmLLM  # noqa: E402


def _experiment(model_extra: dict) -> ExperimentConfig:
    config = {
        "models": {
            "m": {
                "model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                "use_quantization": False, **model_extra,
            }
        },
        "datasets": {"d": {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
    }
    return ExperimentConfig.from_file(
        config=config, model_key="m", dataset_key="d", prompt_key="p", experiment_name="plan"
    )


def test_ignore_patterns_default_none() -> None:
    assert _experiment({}).vllm_ignore_patterns is None


def test_ignore_patterns_copied_from_model_config() -> None:
    patterns = ["original/**/*", "adapter_model.safetensors"]
    assert _experiment({"vllm_ignore_patterns": patterns}).vllm_ignore_patterns == patterns


def _sampling_kwargs(model_extra: dict) -> dict:
    backend = VllmLLM.__new__(VllmLLM)
    object.__setattr__(backend, "config", _experiment(model_extra))
    return VllmLLM._create_sampling_params(backend, lambda **kwargs: kwargs, seed=1)


def test_bad_words_default_not_sent() -> None:
    assert _experiment({}).vllm_bad_words is None
    assert "bad_words" not in _sampling_kwargs({})


def test_bad_words_passed_to_sampling_params() -> None:
    kwargs = _sampling_kwargs({"vllm_bad_words": ["<|tool_call_start|>"]})
    assert kwargs["bad_words"] == ["<|tool_call_start|>"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
