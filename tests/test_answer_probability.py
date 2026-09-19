"""Plain-python checks for per-draw and sample-level answer probabilities.

Run: PYTHONPATH=src uv run python tests/test_answer_probability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import _answer_probability


def _close(actual, expected):
    return actual is not None and abs(actual - expected) < 1e-9


def test_vulnerable_answer_keeps_p_vulnerable():
    assert _close(_answer_probability(1, 0.8), 0.8)
    print("test_vulnerable_answer_keeps_p_vulnerable PASSED")


def test_safe_answer_uses_complement():
    assert _close(_answer_probability(0, 0.8), 0.2)
    print("test_safe_answer_uses_complement PASSED")


def test_missing_p_vulnerable_gives_none():
    assert _answer_probability(1, None) is None
    print("test_missing_p_vulnerable_gives_none PASSED")


def test_non_binary_label_gives_none():
    # Multiclass labels have no SAFE/VULNERABLE token probability to map onto.
    assert _answer_probability("CWE-79", 0.8) is None
    print("test_non_binary_label_gives_none PASSED")


if __name__ == "__main__":
    test_vulnerable_answer_keeps_p_vulnerable()
    test_safe_answer_uses_complement()
    test_missing_p_vulnerable_gives_none()
    test_non_binary_label_gives_none()
    print("ALL PASSED")
