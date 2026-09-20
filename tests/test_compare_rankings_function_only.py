"""Plain-python checks for the function-only CleanVul baseline of the compare-rankings study.

Run: PYTHONPATH=src uv run python tests/test_compare_rankings_function_only.py
The baseline must pair sample-for-sample (id, label, commit) with the ranking variants.
"""
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entrypoints.loaders.run_setup_context_assembler_compare_rankings import (
    write_function_only_dataset,
)


def _entry(
    sample_id: str, url: str, func_code: str, is_vulnerable: bool
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "commit_url": url,
        "func_code": func_code,
        "commit_msg": f"fix {sample_id}",
        "is_vulnerable": is_vulnerable,
    }


def _reference_sample(sample_id: str, url: str, label: int) -> dict[str, Any]:
    return {
        "id": sample_id,
        "code": f"# assembled context for {sample_id}",
        "label": str(label),
        "metadata": {"commit_url": url, "description": "", "cwe_number": 0},
    }


def _write_sources(
    tmp_dir: Path,
    entries: list[dict[str, Any]],
    reference_samples: list[dict[str, Any]],
) -> tuple[Path, Path]:
    entries_path: Path = tmp_dir / "cleanvul_entries.json"
    reference_path: Path = tmp_dir / "cleanvul_context_benchmark_cpg_structural.json"
    entries_path.write_text(json.dumps(entries), encoding="utf-8")
    reference_path.write_text(
        json.dumps({"metadata": {}, "samples": reference_samples}), encoding="utf-8"
    )
    return entries_path, reference_path


def test_function_only_dataset_pairs_with_reference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        entries_path, reference_path = _write_sources(
            tmp_dir,
            [
                _entry("S-1", "https://h/commit/a", "def vuln(): pass", True),
                _entry("S-2", "https://h/commit/a", "def fixed(): pass", False),
            ],
            [
                _reference_sample("S-1", "https://h/commit/a", 1),
                _reference_sample("S-2", "https://h/commit/a", 0),
            ],
        )
        output_path = tmp_dir / "out" / "function_only.json"
        write_function_only_dataset(
            entries_path=entries_path,
            reference_path=reference_path,
            output_path=output_path,
            sample_limit=None,
        )

        dataset = json.loads(output_path.read_text(encoding="utf-8"))
        samples = dataset["samples"]
        assert [s["id"] for s in samples] == ["S-1", "S-2"], samples
        assert [s["label"] for s in samples] == [1, 0], samples
        assert [s["code"] for s in samples] == ["def vuln(): pass", "def fixed(): pass"]
        assert samples[0]["metadata"]["commit_url"] == "https://h/commit/a"
        assert dataset["metadata"]["task_type"] == "binary_vulnerability"
    print("test_function_only_dataset_pairs_with_reference PASSED")


def test_sample_limit_keeps_reference_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        entries_path, reference_path = _write_sources(
            tmp_dir,
            [
                _entry("S-2", "https://h/commit/a", "def fixed(): pass", False),
                _entry("S-1", "https://h/commit/a", "def vuln(): pass", True),
            ],
            [
                _reference_sample("S-1", "https://h/commit/a", 1),
                _reference_sample("S-2", "https://h/commit/a", 0),
            ],
        )
        output_path = tmp_dir / "function_only.json"
        write_function_only_dataset(
            entries_path=entries_path,
            reference_path=reference_path,
            output_path=output_path,
            sample_limit=1,
        )
        samples = json.loads(output_path.read_text(encoding="utf-8"))["samples"]
        assert [s["id"] for s in samples] == ["S-1"], samples
    print("test_sample_limit_keeps_reference_order PASSED")


def _assert_misaligned(
    entries: list[dict[str, Any]],
    reference_samples: list[dict[str, Any]],
    expected_fragment: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        entries_path, reference_path = _write_sources(tmp_dir, entries, reference_samples)
        try:
            write_function_only_dataset(
                entries_path=entries_path,
                reference_path=reference_path,
                output_path=tmp_dir / "function_only.json",
                sample_limit=None,
            )
        except ValueError as error:
            assert expected_fragment in str(error), str(error)
            return
        raise AssertionError("expected ValueError for misaligned entries")


def test_label_mismatch_is_rejected() -> None:
    _assert_misaligned(
        [_entry("S-1", "https://h/commit/a", "def f(): pass", False)],
        [_reference_sample("S-1", "https://h/commit/a", 1)],
        "S-1",
    )
    print("test_label_mismatch_is_rejected PASSED")


def test_commit_mismatch_is_rejected() -> None:
    # Sample ids are re-randomized per dataset generation, so a stale entries
    # file reuses the same ids for different commits.
    _assert_misaligned(
        [_entry("S-1", "https://h/commit/stale", "def f(): pass", True)],
        [_reference_sample("S-1", "https://h/commit/a", 1)],
        "S-1",
    )
    print("test_commit_mismatch_is_rejected PASSED")


def test_missing_entry_is_rejected() -> None:
    _assert_misaligned(
        [_entry("S-1", "https://h/commit/a", "def f(): pass", True)],
        [
            _reference_sample("S-1", "https://h/commit/a", 1),
            _reference_sample("S-2", "https://h/commit/a", 0),
        ],
        "S-2",
    )
    print("test_missing_entry_is_rejected PASSED")


if __name__ == "__main__":
    test_function_only_dataset_pairs_with_reference()
    test_sample_limit_keeps_reference_order()
    test_label_mismatch_is_rejected()
    test_commit_mismatch_is_rejected()
    test_missing_entry_is_rejected()
    print("ALL PASSED")
