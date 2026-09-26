"""Plain-python checks: full metrics at configurable coverage levels.

Run: PYTHONPATH=src python tests/test_coverage_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.config import ExperimentConfig, ExperimentsPlanConfig
from benchmark.coverage import DEFAULT_COVERAGE_LEVELS, validate_coverage_levels
from benchmark.enums import TaskType
from benchmark.metrics_calculator import BinaryMetricsCalculator, MetricsCalculatorFactory
from benchmark.models import PredictionResult


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _pred(index: int, predicted: int, true: int, probability: float) -> PredictionResult:
    return PredictionResult(sample_id=f"s{index}", predicted_label=predicted, true_label=true,
                            confidence=None, answer_probability=probability, response_text="",
                            processing_time=0.0, is_success=True, error_message=None)


# Ranked by answer probability: TP, FP, TN, FN.
PREDICTIONS = [_pred(0, 1, 1, 0.9), _pred(1, 1, 0, 0.8), _pred(2, 0, 0, 0.7), _pred(3, 0, 1, 0.6)]


def _close(a, b) -> bool:
    return a is not None and abs(a - b) < 1e-9


def test_validation() -> None:
    assert validate_coverage_levels([0.5, 1]) == (0.5, 1.0)
    _raises(ValueError, lambda: validate_coverage_levels([]))
    _raises(ValueError, lambda: validate_coverage_levels([0.0]))
    _raises(ValueError, lambda: validate_coverage_levels([1.5]))


def test_default_levels_include_full_coverage() -> None:
    assert DEFAULT_COVERAGE_LEVELS == (0.25, 0.5, 0.75, 1.0)
    summary = BinaryMetricsCalculator().calculate(PREDICTIONS).summary
    for pct in (25, 50, 75, 100):
        for name in ("accuracy", "precision", "recall", "f1_score", "fpr", "fnr"):
            assert f"{name}_at_coverage_{pct}" in summary, (name, pct)


def test_full_metrics_at_half_and_full_coverage() -> None:
    summary = BinaryMetricsCalculator().calculate(PREDICTIONS).summary
    # Top 2 = TP, FP.
    assert _close(summary["accuracy_at_coverage_50"], 0.5)
    assert _close(summary["precision_at_coverage_50"], 0.5)
    assert _close(summary["recall_at_coverage_50"], 1.0)
    assert _close(summary["f1_score_at_coverage_50"], 2 / 3)
    assert _close(summary["fpr_at_coverage_50"], 1.0)
    assert _close(summary["fnr_at_coverage_50"], 0.0)
    # All 4 = TP, FP, TN, FN.
    for name in ("accuracy", "precision", "recall", "f1_score", "fpr", "fnr"):
        assert _close(summary[f"{name}_at_coverage_100"], 0.5), name


def test_undefined_ratios_are_none() -> None:
    # Top 25% = one TP only: no negatives -> FPR undefined.
    result = BinaryMetricsCalculator().calculate(PREDICTIONS)
    assert result.summary["fpr_at_coverage_25"] is None
    assert _close(result.summary["precision_at_coverage_25"], 1.0)
    level = result.details["selective_prediction"]["levels"][0]
    assert level["selected_samples"] == 1 and level["fpr"] is None


def test_single_class_selection_never_divides_by_zero() -> None:
    only_negatives = [_pred(0, 0, 0, 0.9), _pred(1, 0, 0, 0.8)]
    summary = BinaryMetricsCalculator().calculate(only_negatives).summary
    assert summary["precision_at_coverage_100"] is None
    assert summary["recall_at_coverage_100"] is None
    assert summary["f1_score_at_coverage_100"] is None
    assert _close(summary["fpr_at_coverage_100"], 0.0)


def test_custom_levels_change_keys() -> None:
    summary = BinaryMetricsCalculator([0.5]).calculate(PREDICTIONS).summary
    assert "accuracy_at_coverage_50" in summary
    assert "accuracy_at_coverage_25" not in summary and "accuracy_at_coverage_100" not in summary


def test_factory_passes_levels() -> None:
    calculator = MetricsCalculatorFactory.create_calculator(
        TaskType.BINARY_VULNERABILITY, coverage_levels=[0.3])
    assert calculator.coverage_levels == (0.3,)


def _config(plan_levels: list[float] | None) -> dict:
    plan: dict = {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}
    if plan_levels is not None:
        plan["coverage_levels"] = plan_levels
    return {
        "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                         "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                         "use_quantization": False}},
        "datasets": {"d": {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {"plan": plan},
    }


def test_plan_coverage_levels_flow_into_experiments() -> None:
    assert ExperimentsPlanConfig.from_file(_config(None), "plan").experiments[0].coverage_levels == [
        0.25, 0.5, 0.75, 1.0]
    assert ExperimentsPlanConfig.from_file(_config([0.1, 0.9]), "plan").experiments[0].coverage_levels == [
        0.1, 0.9]
    _raises(ValueError, lambda: ExperimentsPlanConfig.from_file(_config([2.0]), "plan"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
