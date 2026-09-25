"""Plain-python checks for deterministic per-draw sampling seeds.

Run: PYTHONPATH=src uv run python tests/test_sampling_seeds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.sampling_seeds import draw_seed


def test_seed_is_deterministic_and_bounded() -> None:
    first = draw_seed(7, "abc_vuln", 0)
    assert first == draw_seed(7, "abc_vuln", 0)
    assert 0 <= first < 2**31 - 1


def test_seed_varies_by_draw_sample_and_global_seed() -> None:
    seeds = {draw_seed(7, "abc_vuln", d) for d in range(7)}
    assert len(seeds) == 7
    assert draw_seed(7, "abc_vuln", 0) != draw_seed(7, "abc_safe", 0)
    assert draw_seed(7, "abc_vuln", 0) != draw_seed(8, "abc_vuln", 0)


def test_seed_is_independent_of_dataset_position() -> None:
    assert draw_seed(7, "abc_vuln", 3) == draw_seed(7, "abc_vuln", 3)


if __name__ == "__main__":
    test_seed_is_deterministic_and_bounded()
    test_seed_varies_by_draw_sample_and_global_seed()
    test_seed_is_independent_of_dataset_position()
    print("ALL PASSED")
