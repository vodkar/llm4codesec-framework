"""Plain-python checks for MCC.

Run: PYTHONPATH=src uv run python tests/test_mcc.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult


def _pred(i: int, truth: int, predicted: int) -> PredictionResult:
    return PredictionResult(
        sample_id=f"s{i}",
        predicted_label=predicted,
        true_label=truth,
        confidence=None,
        response_text="",
        processing_time=0.0,
        is_success=True,
        error_message=None,
    )


def test_mcc_hand_computed() -> None:
    # tp=2 fn=1 fp=1 tn=2 -> (4-1)/sqrt(3*3*3*3) = 3/9
    preds = [
        _pred(0, 1, 1),
        _pred(1, 1, 1),
        _pred(2, 1, 0),
        _pred(3, 0, 1),
        _pred(4, 0, 0),
        _pred(5, 0, 0),
    ]
    mcc = BinaryMetricsCalculator().calculate(preds).summary["mcc"]
    assert isinstance(mcc, float)
    assert math.isclose(mcc, 1 / 3)


def test_mcc_perfect_and_inverted() -> None:
    assert (
        BinaryMetricsCalculator()
        .calculate([_pred(0, 1, 1), _pred(1, 0, 0)])
        .summary["mcc"]
        == 1.0
    )
    assert (
        BinaryMetricsCalculator()
        .calculate([_pred(0, 1, 0), _pred(1, 0, 1)])
        .summary["mcc"]
        == -1.0
    )


if __name__ == "__main__":
    test_mcc_hand_computed()
    test_mcc_perfect_and_inverted()
    print("ALL PASSED")
