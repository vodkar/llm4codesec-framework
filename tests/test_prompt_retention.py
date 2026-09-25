"""Plain-python checks: prompts, per-draw P(VULNERABLE) and filtered ids are retained.

Run: PYTHONPATH=src uv run python tests/test_prompt_retention.py
"""
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.config import ExperimentConfig
from benchmark.models import BenchmarkSample, PredictionResult, SampleCollection
from benchmark.prompt_generator import get_prompt_generator
from benchmark.response_parser import ResponseParserFactory
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import MetricsResult, SampleInferenceData
from datasets.loaders.base import JsonDatasetLoader
from llm.llm import ILLMInference, InferenceResult


def test_inference_result_has_prompt_fields() -> None:
    result = InferenceResult(response_text="x", tokens_used=1, duration=0.0, prompt_text="P", prompt_tokens=3)
    assert (result.prompt_text, result.prompt_tokens) == ("P", 3)


def test_prediction_fields_round_trip_into_record() -> None:
    prediction = PredictionResult(
        sample_id="p_vuln", predicted_label=1, true_label=1, confidence=None,
        binary_label_confidence=0.7, response_text="r", processing_time=0.1, is_success=True,
        error_message=None, prompt_text="PROMPT", prompt_tokens=12, p_vulnerable_per_draw=[0.6, 0.8],
    )
    record = BenchmarkResultProcessor._to_prediction_record(
        BenchmarkResultProcessor.model_construct(), prediction
    )
    data: SampleInferenceData = record.inference_data
    assert data.prompt_text == "PROMPT"
    assert data.prompt_tokens == 12
    assert data.p_vulnerable_per_draw == [0.6, 0.8]


def _config() -> dict:
    """Minimal experiment config dict, following tests/test_confidence_methods.py's helper."""
    return {
        "models": {
            "m": {
                "model_identifier": "org/model",
                "model_type": "qwen-3",
                "backend": "vllm",
                "max_output_tokens": 16,
                "batch_size": 1,
                "temperature": 0.0,
                "use_quantization": False,
                "self_consistency_samples": 2,
            }
        },
        # The dataset file only has to exist for config validation.
        "datasets": {"d": {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {
            "plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}
        },
    }


class _StubLLM(ILLMInference):
    """Fake backend cycling through a fixed list of binary_label_confidence values."""

    def __init__(self, p_values: list[float]) -> None:
        self._p_values = p_values
        self._call_index = 0

    def generate_response(self, system_prompt: str, user_prompt: str) -> InferenceResult:
        raise NotImplementedError

    def generate_batch_responses(self, prompts: list[str]) -> list[InferenceResult]:
        raise NotImplementedError

    def generate_responses_batch_optimized(
        self,
        system_prompts: list[str],
        user_prompts: list[str],
        seeds: list[int] | None = None,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for _ in system_prompts:
            p = self._p_values[self._call_index % len(self._p_values)]
            self._call_index += 1
            results.append(
                InferenceResult(
                    response_text='{"is_vulnerable": true}',
                    tokens_used=5,
                    duration=0.0,
                    binary_label_confidence=p,
                    prompt_text="FMT",
                    prompt_tokens=4,
                )
            )
        return results

    def count_input_tokens(self, text: str) -> int:
        return 0

    def cleanup(self) -> None:
        pass


def test_runner_retains_prompt_and_per_draw_p_vulnerable() -> None:
    experiment_config = ExperimentConfig.from_file(
        config=_config(),
        model_key="m",
        dataset_key="d",
        prompt_key="p",
        experiment_name="plan",
    )
    runner = BenchmarkRunner(config=experiment_config, dataset_loader=JsonDatasetLoader())

    samples = SampleCollection(
        [BenchmarkSample(id="s1_vuln", code="int x;", label=1, metadata={})]
    )
    prompt_generator = get_prompt_generator(experiment_config, template_values={})
    response_parser = ResponseParserFactory.create_parser(experiment_config.task_type)
    llm = _StubLLM([0.2, 0.9])

    predictions = runner._process_samples_with_batch_optimization(
        samples, llm, prompt_generator, response_parser
    )

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.p_vulnerable_per_draw == [0.2, 0.9]
    assert prediction.prompt_text == "FMT"
    assert prediction.prompt_tokens == 4


def test_report_records_sampling_seed_prompt_identity_and_utc_timestamp() -> None:
    config = _config()
    config["models"]["m"]["sampling_seed"] = 11
    experiment_config = ExperimentConfig.from_file(
        config=config,
        model_key="m",
        dataset_key="d",
        prompt_key="p",
        experiment_name="plan",
    )
    processor = BenchmarkResultProcessor(config=experiment_config)
    metrics = MetricsResult.model_validate(
        {"task_type": "binary", "accuracy": 0.0, "summary": {}, "details": {}}
    )

    report = processor.build_report(
        metrics=metrics, predictions=[], total_time=0.0, total_samples=0
    )

    info = report.benchmark_info
    assert info.model.sampling_seed == 11
    assert info.prompt_identifier == "p"
    assert info.prompt_template_sha256 == hashlib.sha256(b"sys\0{code}").hexdigest()
    assert info.timestamp_utc is not None
    assert datetime.fromisoformat(info.timestamp_utc).utcoffset() == timedelta(0)


if __name__ == "__main__":
    test_inference_result_has_prompt_fields()
    test_prediction_fields_round_trip_into_record()
    test_runner_retains_prompt_and_per_draw_p_vulnerable()
    test_report_records_sampling_seed_prompt_identity_and_utc_timestamp()
    print("ALL PASSED")
