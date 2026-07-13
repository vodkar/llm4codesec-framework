"""Plain-python checks for CleanVul token/score filtering.

Run: PYTHONPATH=src uv run python tests/test_cleanvul_filters.py
Uses a stub tokenizer (whitespace split) so no model download is needed.
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.enums import TaskType
from datasets.loaders.cleanvul import CleanVulDatasetLoader


class _StubLoader(CleanVulDatasetLoader):
    """Counts tokens as whitespace-separated words (no HF download)."""
    def _count_tokens(self, code: str) -> int:
        return len(code.split())


_HEADER = [
    "func_before", "func_after", "commit_msg", "commit_url", "cve_id",
    "cwe_id", "file_name", "vulnerability_score", "extension", "is_test", "date",
]


def _row(before, after, url, score):
    return {
        "func_before": before, "func_after": after, "commit_msg": "m",
        "commit_url": url, "cve_id": "", "cwe_id": "CWE-79", "file_name": "f.py",
        "vulnerability_score": score, "extension": "py", "is_test": "false", "date": "",
    }


def _write_csv(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(fh, fieldnames=_HEADER)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    fh.close()
    return Path(fh.name)


def test_token_range_filters_per_sample():
    # before=4 words, after=2 words for group A; before=7 words for group B
    rows = [
        _row("a b c d", "x y", "https://h/commit/aaaaaaa", "4"),
        _row("a b c d e f g", "z", "https://h/commit/bbbbbbb", "4"),
    ]
    src = _write_csv(rows)
    loader = _StubLoader(source_path=src, programming_language="Python",
                         min_tokens=3, max_tokens=6, tokenizer_id="stub")
    samples = list(loader.load_dataset(task_type=TaskType.BINARY_VULNERABILITY))
    counts = sorted(s.metadata["code_tokens"] for s in samples)
    # kept: group A before (4) only. after(2) below min, groupB before(7) >= max.
    assert counts == [4], counts
    assert all(3 <= c < 6 for c in counts), counts
    print("test_token_range_filters_per_sample PASSED")


def test_score_range_excludes_low_scores():
    rows = [
        _row("a b c", "x y", "https://h/commit/ccccccc", "2"),
        _row("a b c", "x y", "https://h/commit/ddddddd", "4"),
    ]
    src = _write_csv(rows)
    loader = _StubLoader(source_path=src, programming_language="Python",
                         min_score=3, tokenizer_id="stub")
    samples = list(loader.load_dataset(task_type=TaskType.BINARY_VULNERABILITY))
    urls = {s.metadata["commit_url"] for s in samples}
    assert urls == {"https://h/commit/ddddddd"}, urls
    print("test_score_range_excludes_low_scores PASSED")


if __name__ == "__main__":
    test_token_range_filters_per_sample()
    test_score_range_excludes_low_scores()
    print("ALL PASSED")
