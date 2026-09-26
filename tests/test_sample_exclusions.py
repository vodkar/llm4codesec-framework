"""Plain-python checks: audited samples are excluded by (source_row_ids, label).

Run: PYTHONPATH=src python tests/test_sample_exclusions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.config import ExperimentConfig
from benchmark.models import BenchmarkSample, SampleCollection
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import MetricsResult
from benchmark.sample_exclusions import SampleExclusion, apply_sample_exclusions, sample_key
from datasets.loaders.base import JsonDatasetLoader


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _sample(sample_id: str, row_ids: list[int] | None, label: int) -> BenchmarkSample:
    metadata = {} if row_ids is None else {"source_row_ids": row_ids}
    return BenchmarkSample(id=sample_id, code="x", label=label, metadata=metadata)


def _config(exclusions: list[dict] | None = None) -> dict:
    dataset: dict = {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}
    if exclusions is not None:
        dataset["exclude_samples"] = exclusions
    return {
        "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                         "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                         "use_quantization": False}},
        "datasets": {"d": dataset},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {"plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}},
    }


def _experiment(exclusions: list[dict] | None = None) -> ExperimentConfig:
    return ExperimentConfig.from_file(
        config=_config(exclusions), model_key="m", dataset_key="d", prompt_key="p",
        experiment_name="plan",
    )


def test_key_uses_row_ids_and_label() -> None:
    assert sample_key(_sample("a", [3, 1], 1)) == ((3, 1), 1)
    assert SampleExclusion(source_row_ids=[3, 1], label=1).key == ((3, 1), 1)


def test_excludes_matching_samples_and_reports_unmatched() -> None:
    samples = [_sample("a", [1], 1), _sample("b", [1], 0), _sample("c", [2, 3], 1)]
    result = apply_sample_exclusions(samples, [
        SampleExclusion(source_row_ids=[1], label=1, reason="wrong"),
        SampleExclusion(source_row_ids=[9], label=0),
    ])
    assert [s.id for s in result.kept] == ["b", "c"]
    assert [e.key for e in result.excluded] == [((1,), 1)]
    assert [e.key for e in result.unmatched] == [((9,), 0)]


def test_missing_row_ids_raises_only_when_exclusions_configured() -> None:
    samples = [_sample("a", None, 1)]
    assert [s.id for s in apply_sample_exclusions(samples, []).kept] == ["a"]
    _raises(ValueError, lambda: apply_sample_exclusions(samples, [SampleExclusion(source_row_ids=[1], label=1)]))


def test_config_carries_exclusions() -> None:
    assert _experiment().exclude_samples == []
    config = _experiment([{"source_row_ids": [789, 2754], "label": 1, "reason": "not_a_security_fix"}])
    assert [e.key for e in config.exclude_samples] == [((789, 2754), 1)]


def test_runner_prepare_samples_applies_exclusions() -> None:
    runner = BenchmarkRunner(
        config=_experiment([{"source_row_ids": [1], "label": 1}, {"source_row_ids": [5], "label": 0}]),
        dataset_loader=JsonDatasetLoader(),
    )
    kept, metadata = runner._prepare_samples(
        SampleCollection([_sample("a", [1], 1), _sample("b", [1], 0)])
    )
    assert [s.id for s in kept] == ["b"]
    assert metadata["excluded_samples"] == [{"source_row_ids": [1], "label": 1, "reason": ""}]
    assert metadata["unmatched_exclusions"] == [{"source_row_ids": [5], "label": 0, "reason": ""}]


def test_report_records_run_metadata() -> None:
    processor = BenchmarkResultProcessor(config=_experiment())
    metrics = MetricsResult.model_validate({"task_type": "binary", "accuracy": 0.0, "summary": {}, "details": {}})
    report = processor.build_report(
        metrics=metrics, predictions=[], total_time=0.0, total_samples=0,
        run_metadata={"excluded_samples": [{"source_row_ids": [1], "label": 1, "reason": ""}]},
    )
    assert report.benchmark_info.extra_metadata["excluded_samples"][0]["source_row_ids"] == [1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
