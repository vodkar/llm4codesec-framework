"""Plain-python checks: saved reports are recomputed under exclusions and merged.

Run: PYTHONPATH=src python tests/test_merge_results.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.config import ExperimentConfig
from benchmark.merge_results import (
    latest_condition_reports,
    merge_plan_results,
    recompute_report,
    record_to_prediction,
    render_summary_markdown,
)
from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import BenchmarkReport
from benchmark.sample_exclusions import SampleExclusion
from benchmark.static_findings import RootStaticFinding

FINDINGS = [
    RootStaticFinding(tool="bandit", rule_id="B101", message="m", file_path="a.py", repo_line=1, flagged_line="x"),
    RootStaticFinding(tool="bandit", rule_id="B602", message="m", file_path="a.py", repo_line=2, flagged_line="y"),
]


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _prediction(index: int, predicted: int, true: int, probability: float,
                row_ids: list[int] | None) -> PredictionResult:
    return PredictionResult(
        sample_id=f"s{index}", predicted_label=predicted, true_label=true, confidence=None,
        binary_label_confidence=probability if predicted == 1 else 1 - probability,
        answer_probability=probability, response_text="r", processing_time=0.1, tokens_used=3,
        is_success=True, error_message=None, all_responses=["r"], vote_counts={str(predicted): 1},
        source_row_ids=row_ids, root_static_findings=list(FINDINGS), root_findings_in_prompt=True,
    )


PREDICTIONS = [
    _prediction(0, 1, 1, 0.9, [10]), _prediction(1, 1, 0, 0.8, [10]),
    _prediction(2, 0, 0, 0.7, [11]), _prediction(3, 0, 1, 0.6, [11]),
]


def _record(prediction: PredictionResult):
    return BenchmarkResultProcessor._to_prediction_record(BenchmarkResultProcessor.model_construct(), prediction)


def _report(predictions: list[PredictionResult]) -> BenchmarkReport:
    """Build a report with the real processor so it matches the saved-report schema."""
    config = ExperimentConfig.from_file(
        config={
            "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                             "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                             "use_quantization": False}},
            "datasets": {"d": {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}},
            "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
            "output_settings": {"base_output_dir": "results/x"},
            "experiment_plans": {"plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}},
        },
        model_key="m", dataset_key="d", prompt_key="p", experiment_name="plan",
    )
    metrics = BinaryMetricsCalculator().calculate(predictions)
    return BenchmarkResultProcessor(config=config).build_report(
        metrics=metrics, predictions=predictions, total_time=1.0, total_samples=len(predictions))


def test_record_round_trip() -> None:
    record = _record(PREDICTIONS[0])
    assert _record(record_to_prediction(record)) == record


def test_recompute_without_changes_matches_direct_metrics() -> None:
    report = _report(PREDICTIONS)
    recomputed = recompute_report(report, [], [], (0.5, 1.0), Path("src.json"))
    direct = BinaryMetricsCalculator((0.5, 1.0)).calculate(PREDICTIONS)
    assert recomputed.metrics.summary == direct.summary
    assert recomputed.benchmark_info.extra_metadata["recomputed_from"] == "src.json"


def test_recompute_applies_exclusions_and_rule_filter() -> None:
    report = _report(PREDICTIONS)
    recomputed = recompute_report(
        report, [SampleExclusion(source_row_ids=[10], label=0)], ["B101"], (1.0,), Path("x"))
    assert [p.sample_id for p in recomputed.predictions] == ["s0", "s2", "s3"]
    assert all([f.rule_id for f in p.root_static_findings] == ["B602"] for p in recomputed.predictions)
    assert recomputed.benchmark_info.stats.total_samples == 3
    meta = recomputed.benchmark_info.extra_metadata
    assert meta["excluded_samples"] == [{"source_row_ids": [10], "label": 0}]
    assert meta["analysis_exclude_finding_rules"] == ["B101"]
    expected = BinaryMetricsCalculator((1.0,)).calculate([PREDICTIONS[0], PREDICTIONS[2], PREDICTIONS[3]])
    assert recomputed.metrics.summary == expected.summary


def test_recompute_without_row_ids_raises_when_excluding() -> None:
    report = _report([_prediction(0, 1, 1, 0.9, None)])
    recompute_report(report, [], [], (1.0,), Path("x"))  # fine without exclusions
    _raises(ValueError, lambda: recompute_report(
        report, [SampleExclusion(source_row_ids=[1], label=1)], [], (1.0,), Path("x")))


def test_latest_condition_reports_picks_newest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan = Path(tmp) / "plan_a"
        condition = plan / "ds_a" / "m" / "p"
        condition.mkdir(parents=True)
        for stamp in ("20260925_100000", "20260926_090000"):
            (condition / f"benchmark_report_{stamp}.json").write_text("{}")
        latest = latest_condition_reports(plan)
        assert latest == {Path("ds_a/m/p"): condition / "benchmark_report_20260926_090000.json"}
        _raises(RuntimeError, lambda: latest_condition_reports(Path(tmp) / "empty_missing"))


def test_summary_bolds_best_and_handles_none() -> None:
    rows = {
        "a": {"accuracy": 0.6, "fpr": 0.3, "pr_auc": None},
        "b": {"accuracy": 0.6, "fpr": 0.4, "pr_auc": 0.7},
    }
    text = render_summary_markdown(rows, ())
    line_a = next(line for line in text.splitlines() if line.startswith("| a "))
    line_b = next(line for line in text.splitlines() if line.startswith("| b "))
    assert "**0.600**" in line_a and "**0.600**" in line_b  # tie -> both bold
    assert "**0.300**" in line_a and "0.400" in line_b and "**0.400**" not in line_b  # lower FPR wins
    assert "—" in line_a and "**0.700**" in line_b


def test_merge_plan_results_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for plan, dataset in (("plan_a", "ds_a"), ("plan_b", "ds_b")):
            condition = root / plan / dataset / "m" / "p"
            condition.mkdir(parents=True)
            (condition / "benchmark_report_20260926_000000.json").write_text(
                json.dumps(_report(PREDICTIONS).model_dump(), default=str))
        exclusion = {"source_row_ids": [10], "label": 0, "reason": "wrong"}
        datasets = root / "datasets.json"
        datasets.write_text(json.dumps({"datasets": {
            "ds_a": {"dataset_path": "x", "task_type": "binary_vulnerability", "description": "a",
                     "exclude_samples": [exclusion]},
            "ds_b": {"dataset_path": "y", "task_type": "binary_vulnerability", "description": "b",
                     "exclude_samples": [exclusion]},
        }}))
        out = root / "merged"
        summary_path = merge_plan_results([root / "plan_a", root / "plan_b"], datasets, (0.5, 1.0), ["B101"], out)
        assert summary_path == out / "summary.md"
        text = summary_path.read_text()
        assert "ds_a · p" in text and "ds_b · p" in text
        assert (out / "experiment_plan_results.json").exists()
        merged = BenchmarkReport.model_validate(json.loads(
            (out / "plan_a" / "ds_a" / "m" / "p" / "benchmark_report_20260926_000000.json").read_text()))
        assert len(merged.predictions) == 3


def test_merge_unknown_dataset_key_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        condition = root / "plan_a" / "ds_unknown" / "m" / "p"
        condition.mkdir(parents=True)
        (condition / "benchmark_report_20260926_000000.json").write_text(
            json.dumps(_report(PREDICTIONS).model_dump(), default=str))
        datasets = root / "datasets.json"
        datasets.write_text(json.dumps({"datasets": {}}))
        _raises(KeyError, lambda: merge_plan_results([root / "plan_a"], datasets, (1.0,), [], root / "out"))


def test_merge_model_filter_and_duplicate_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for model in ("m1", "m2"):
            condition = root / "plan_a" / "ds_a" / model / "p"
            condition.mkdir(parents=True)
            (condition / "benchmark_report_20260926_000000.json").write_text(
                json.dumps(_report(PREDICTIONS).model_dump(), default=str))
        datasets = root / "datasets.json"
        datasets.write_text(json.dumps({"datasets": {
            "ds_a": {"dataset_path": "x", "task_type": "binary_vulnerability", "description": "a"}}}))
        # Two models give the same "ds_a · p" row: refuse instead of overwriting.
        _raises(ValueError, lambda: merge_plan_results([root / "plan_a"], datasets, (1.0,), [], root / "o1"))
        summary = merge_plan_results([root / "plan_a"], datasets, (1.0,), [], root / "o2", models=["m2"])
        assert (root / "o2" / "plan_a" / "ds_a" / "m2").exists()
        assert not (root / "o2" / "plan_a" / "ds_a" / "m1").exists()
        assert summary.read_text().count("ds_a · p") == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
