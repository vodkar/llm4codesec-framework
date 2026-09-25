"""Plain-python checks for analysis input alignment and pair survival.

Run: PYTHONPATH=src uv run python tests/test_reference_context_inputs.py
"""

import hashlib
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.reference_context.inputs import Condition, load_inputs, load_reports
from analysis.reference_context.run_config import ReferenceRunConfig

CONDITIONS = [c.value for c in Condition]
_PAIRS: tuple[str, str] = ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
_SIDES: tuple[tuple[str, int], tuple[str, int]] = (("vuln", 1), ("safe", 0))


def _report(dataset: str, preds: list[dict], filtered: list[str]) -> dict:
    return {
        "benchmark_info": {
            "experiment_name": "x",
            "task_type": "binary_vulnerability",
            "dataset_path": f"datasets_processed/reference_context/{dataset}",
            "description": "",
            "cwe_type": None,
            "batch_size": 1,
            "timestamp": "2026-09-23T05:00:00",
            "timestamp_utc": "2026-09-23T00:00:00+00:00",
            "prompt_identifier": "p",
            "prompt_template_sha256": "promptsha",
            "model": {
                "model_name": "m",
                "model_type": "gemma-4",
                "backend": "vllm",
                "context_length": 1,
                "max_output_tokens": 1,
                "temperature": 1.0,
                "use_quantization": False,
                "is_thinking_enabled": True,
                "self_consistency_samples": 1,
                "enable_logprobs": True,
                "binary_decision_mode": "text",
                "sampling_seed": 7,
            },
            "stats": {
                "total_samples": len(preds),
                "total_time_seconds": 0.0,
                "avg_time_per_sample": 0.0,
                "tokens_used_total": 0,
                "tokens_used_avg": 0.0,
                "processing_time_stats": {},
                "tokens_used_stats": {},
                "confidence_stats": None,
            },
            "extra_metadata": {},
        },
        "metrics": {
            "task_type": "binary",
            "accuracy": 0.0,
            "summary": {},
            "details": {},
        },
        "predictions": preds,
        "is_success": True,
        "filtered_sample_ids": filtered,
    }


def _pred(sample_id: str, label: int, score: float | None) -> dict:
    return {
        "sample_id": sample_id,
        "predicted_label": int((score or 0) > 0.5),
        "true_label": label,
        "is_success": True,
        "error_message": None,
        "inference_data": {
            "responses": ["r"],
            "vote_counts": {},
            "tokens_used": 10,
            "processing_time": 0.0,
            "confidence": None,
            "binary_label_confidence": score,
            "prompt_tokens": 100,
            "p_vulnerable_per_draw": [score],
        },
    }


_STATS_ROW_CONTEXT_TOKEN_COUNT: int = 40


def _stats_row(
    pair: str,
    suffix: str,
    label: int,
    cond: str,
    *,
    include_context_token_count: bool = True,
) -> dict:
    row: dict = {
        "pair_id": pair,
        "item_id": f"{pair}_{suffix}",
        "condition": cond,
        "label": label,
        "repo_url": f"https://github.com/o/{pair}",
        "token_count": 50,
        "budget": 50,
        "node_count": 2,
        "file_count": 1,
        "symbol_count": 2,
        "underfill": False,
        "symbols": ["a.py::t"],
        "symbol_names_hash": "h",
    }
    if include_context_token_count:
        row["context_token_count"] = _STATS_ROW_CONTEXT_TOKEN_COUNT
    return row


def _all_stats_rows() -> list[dict]:
    return [
        _stats_row(pair, suffix, label, cond)
        for pair in _PAIRS
        for suffix, label in _SIDES
        for cond in CONDITIONS
    ]


def _all_strata_rows() -> list[dict]:
    return [
        {"item_id": f"{pair}_{suffix}", "pair_id": pair, "stratum": "r100"}
        for pair in _PAIRS
        for suffix, _ in _SIDES
    ]


def _write_scanner_artifacts(
    scanner: Path, stats: list[dict], strata: list[dict]
) -> None:
    (scanner / "overlap").mkdir(parents=True, exist_ok=True)
    (scanner / "item_stats.jsonl").write_text(
        "\n".join(json.dumps(s) for s in stats) + "\n"
    )
    (scanner / "overlap" / "strata.jsonl").write_text(
        "\n".join(json.dumps(s) for s in strata) + "\n"
    )
    (scanner / "coverage.json").write_text(json.dumps({"eligible_pairs": 5}))
    _write_manifest(scanner, {})


_MANIFEST_CREATED_UTC: str = "2026-09-22T00:00:00+00:00"


def _write_manifest(scanner: Path, extra: dict) -> None:
    manifest = {"config_sha256": "c", "created_utc": _MANIFEST_CREATED_UTC, **extra}
    (scanner / "run_manifest.json").write_text(json.dumps(manifest))


def _write_reports(
    results: Path,
    filtered_sample_ids: dict[str, list[str]] | None = None,
    patch: Callable[[str, dict], None] | None = None,
) -> None:
    filtered_by_condition = filtered_sample_ids or {}
    for cond in CONDITIONS:
        exp = results / f"exp_{cond}"
        exp.mkdir(parents=True, exist_ok=True)
        preds = [
            _pred(f"{p}_{s}", lbl, 0.6 if lbl else 0.4)
            for p in _PAIRS
            for s, lbl in _SIDES
        ]
        filtered = filtered_by_condition.get(cond, [])
        preds = [p for p in preds if p["sample_id"] not in filtered]
        report = _report(f"cleanvul_cond_{cond}.json", preds, filtered)
        if patch is not None:
            patch(cond, report)
        (exp / "benchmark_report_20260923_000000.json").write_text(json.dumps(report))


def test_pair_dropped_everywhere_when_one_item_filtered() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results, scanner = root / "results", root / "scanner"
        _write_scanner_artifacts(scanner, _all_stats_rows(), _all_strata_rows())
        _write_reports(
            results, filtered_sample_ids={"reference": ["bbbbbbbbbbbb_safe"]}
        )

        inputs = load_inputs(results, scanner)

        assert inputs.pair_ids == ["aaaaaaaaaaaa"]
        assert inputs.inference_exclusions == {"filtered_by_token_limit": 1}
        assert len(inputs.items) == 2 * len(CONDITIONS)
        assert {i.stratum for i in inputs.items} == {"r100"}


def test_pair_dropped_when_item_stats_row_missing_for_one_condition() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results, scanner = root / "results", root / "scanner"
        stats = [
            row
            for row in _all_stats_rows()
            if not (
                row["pair_id"] == "bbbbbbbbbbbb"
                and row["item_id"] == "bbbbbbbbbbbb_safe"
                and row["condition"] == "cpg"
            )
        ]
        _write_scanner_artifacts(scanner, stats, _all_strata_rows())
        _write_reports(results)

        inputs = load_inputs(results, scanner)

        assert inputs.pair_ids == ["aaaaaaaaaaaa"]
        assert inputs.inference_exclusions == {"missing_scanner_artifacts": 1}
        assert len(inputs.items) == 2 * len(CONDITIONS)
        assert all(item.pair_id == "aaaaaaaaaaaa" for item in inputs.items)


def test_pair_dropped_when_strata_row_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results, scanner = root / "results", root / "scanner"
        strata = [
            row for row in _all_strata_rows() if row["item_id"] != "bbbbbbbbbbbb_safe"
        ]
        _write_scanner_artifacts(scanner, _all_stats_rows(), strata)
        _write_reports(results)

        inputs = load_inputs(results, scanner)

        assert inputs.pair_ids == ["aaaaaaaaaaaa"]
        assert inputs.inference_exclusions == {"missing_scanner_artifacts": 1}
        assert len(inputs.items) == 2 * len(CONDITIONS)
        assert all(item.pair_id == "aaaaaaaaaaaa" for item in inputs.items)


def test_load_reports_missing_condition_names_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        results.mkdir()

        try:
            load_reports(results)
        except FileNotFoundError as exc:
            assert "none" in str(exc)
            assert "mismatched" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")


def _expect_value_error(
    root: Path, *needles: str, dataset_root: Path | None = None
) -> None:
    kwargs = {} if dataset_root is None else {"dataset_root": dataset_root}
    try:
        load_inputs(root / "results", root / "scanner", **kwargs)
    except ValueError as exc:
        for needle in needles:
            assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError("expected ValueError")


def _setup(root: Path, patch: Callable[[str, dict], None] | None = None) -> None:
    _write_scanner_artifacts(root / "scanner", _all_stats_rows(), _all_strata_rows())
    _write_reports(root / "results", patch=patch)


def test_context_token_count_is_read_from_item_stats() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        inputs = load_inputs(root / "results", root / "scanner")
        assert {item.context_token_count for item in inputs.items} == {
            _STATS_ROW_CONTEXT_TOKEN_COUNT
        }


def test_context_token_count_missing_key_reads_as_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stats = [
            _stats_row(
                pair,
                suffix,
                label,
                cond,
                include_context_token_count=not (
                    pair == "aaaaaaaaaaaa" and suffix == "vuln" and cond == "cpg"
                ),
            )
            for pair in _PAIRS
            for suffix, label in _SIDES
            for cond in CONDITIONS
        ]
        _write_scanner_artifacts(root / "scanner", stats, _all_strata_rows())
        _write_reports(root / "results")

        inputs = load_inputs(root / "results", root / "scanner")

        target = next(
            item
            for item in inputs.items
            if item.item_id == "aaaaaaaaaaaa_vuln" and item.condition.value == "cpg"
        )
        assert target.context_token_count is None
        assert all(
            item.context_token_count == _STATS_ROW_CONTEXT_TOKEN_COUNT
            for item in inputs.items
            if item is not target
        )


def test_consistent_reports_load_with_draw_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        inputs = load_inputs(root / "results", root / "scanner")
        assert inputs.draws_per_item == 1
        assert {item.scored_draws for item in inputs.items} == {1}


def test_mixed_model_config_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        if cond == "cpg":
            report["benchmark_info"]["model"]["temperature"] = 0.5

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "model run config", "cpg=")


def test_null_sampling_seed_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        report["benchmark_info"]["model"]["sampling_seed"] = None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "sampling_seed")


def test_mixed_prompt_identity_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        if cond == "reference":
            report["benchmark_info"]["prompt_identifier"] = "other"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "prompt identity", "reference=")


def test_missing_prompt_identity_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        del report["benchmark_info"]["prompt_identifier"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "prompt identity")


def _write_datasets(dataset_root: Path) -> dict[str, str]:
    directory = dataset_root / "datasets_processed" / "reference_context"
    directory.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for cond in CONDITIONS:
        payload = f"dataset {cond}".encode()
        (directory / f"cleanvul_cond_{cond}.json").write_bytes(payload)
        hashes[cond] = hashlib.sha256(payload).hexdigest()
    return hashes


def test_dataset_sha256_matching_manifest_accepted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        hashes = _write_datasets(root / "app")
        _write_manifest(root / "scanner", {"condition_dataset_sha256": hashes})
        inputs = load_inputs(root / "results", root / "scanner", root / "app")
        assert len(inputs.pair_ids) == 2


def test_dataset_sha256_mismatch_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        hashes = _write_datasets(root / "app")
        hashes["random"] = "0" * 64
        _write_manifest(root / "scanner", {"condition_dataset_sha256": hashes})
        _expect_value_error(
            root, "condition 'random'", "sha256", dataset_root=root / "app"
        )


def test_dataset_file_missing_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        _write_manifest(
            root / "scanner", {"condition_dataset_sha256": {"none": "0" * 64}}
        )
        _expect_value_error(root, "not found", dataset_root=root / "empty")


def test_report_older_than_manifest_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        if cond == "mismatched":
            report["benchmark_info"]["timestamp_utc"] = "2026-09-21T23:59:59+00:00"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "older than the llm_scanner manifest", "mismatched")


def test_report_without_aware_timestamp_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        del report["benchmark_info"]["timestamp_utc"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "no timezone")


def test_manifest_without_created_utc_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        (root / "scanner" / "run_manifest.json").write_text(json.dumps({}))
        _expect_value_error(root, "created_utc")


def test_item_stats_label_mismatch_names_item() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stats = _all_stats_rows()
        for row in stats:
            if row["item_id"] == "aaaaaaaaaaaa_safe" and row["condition"] == "cpg":
                row["label"] = 1
        _write_scanner_artifacts(root / "scanner", stats, _all_strata_rows())
        _write_reports(root / "results")
        _expect_value_error(root, "aaaaaaaaaaaa_safe", "item_stats")


def test_report_true_label_mismatch_names_item() -> None:
    def patch(cond: str, report: dict) -> None:
        for prediction in report["predictions"]:
            if prediction["sample_id"] == "bbbbbbbbbbbb_vuln":
                prediction["true_label"] = 0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        _expect_value_error(root, "bbbbbbbbbbbb_vuln", "true_label")


def test_only_latest_report_per_condition_is_parsed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root)
        stale = root / "results" / "exp_none" / "benchmark_report_20200101_000000.json"
        stale.write_text("not json")
        inputs = load_inputs(root / "results", root / "scanner")
        assert len(inputs.pair_ids) == 2


def test_latest_report_with_wrong_dataset_rejected() -> None:
    def patch(cond: str, report: dict) -> None:
        if cond == "cpg":
            report["benchmark_info"]["dataset_path"] = "x/cleanvul_cond_none.json"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _setup(root, patch)
        try:
            load_reports(root / "results")
        except ValueError as exc:
            assert "'cpg'" in str(exc)
        else:
            raise AssertionError("expected ValueError")


_SMALL_YAML = """
pins:
  llm_scanner_path: /home/somen/llm_scanner
  llm_scanner_git_sha: abc123
llm_scanner:
  dataset_path: data/vulnerability_score_4.csv
  token_budget: 2048
framework:
  experiments_config: src/configs/cleanvul_experiments.json
  datasets_config: src/configs/cleanvul_datasets.json
  results_dir: results/reference_context
  datasets_dir: datasets_processed/reference_context
  bootstrap_seed: 1
  probe_seed: 2
"""


def test_reference_run_config_from_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(_SMALL_YAML)
        config = ReferenceRunConfig.from_yaml(path)
        assert config.pins.llm_scanner_git_sha == "abc123"
        assert config.llm_scanner["token_budget"] == 2048
        assert config.framework.plan == "reference_context_sweep"
        assert config.framework.bootstrap_resamples == 2000
        assert config.framework.bootstrap_seed == 1
        assert config.framework.mismatch_tolerance == 0.02
        digest = ReferenceRunConfig.sha256(path)
        assert isinstance(digest, str)
        assert len(digest) == 64


if __name__ == "__main__":
    test_pair_dropped_everywhere_when_one_item_filtered()
    test_pair_dropped_when_item_stats_row_missing_for_one_condition()
    test_pair_dropped_when_strata_row_missing()
    test_load_reports_missing_condition_names_it()
    test_context_token_count_is_read_from_item_stats()
    test_context_token_count_missing_key_reads_as_none()
    test_reference_run_config_from_yaml()
    test_consistent_reports_load_with_draw_counts()
    test_mixed_model_config_rejected()
    test_null_sampling_seed_rejected()
    test_mixed_prompt_identity_rejected()
    test_missing_prompt_identity_rejected()
    test_dataset_sha256_matching_manifest_accepted()
    test_dataset_sha256_mismatch_rejected()
    test_dataset_file_missing_rejected()
    test_report_older_than_manifest_rejected()
    test_report_without_aware_timestamp_rejected()
    test_manifest_without_created_utc_rejected()
    test_item_stats_label_mismatch_names_item()
    test_report_true_label_mismatch_names_item()
    test_only_latest_report_per_condition_is_parsed()
    test_latest_report_with_wrong_dataset_rejected()
    print("ALL PASSED")
