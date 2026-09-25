"""End-to-end regression tests for the ``analyze-reference-context`` CLI command.

Builds synthetic llm_scanner artifacts and five synthetic benchmark reports
(builder helpers copied/adapted from ``tests/test_reference_context_inputs.py``),
writes a temp config YAML, and invokes the real Typer command through
``typer.testing.CliRunner`` -- exercising config loading, ``output_dir``
resolution (both absolute and relative-to-``pins.llm_scanner_path``), the
``--plan`` override, the bootstrap/metrics/acceptance pipeline and the
report writer, end to end.

Run: PYTHONPATH=src uv run python tests/test_reference_context_cli.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from click.testing import Result
from typer.testing import CliRunner

import cli
from analysis.reference_context.inputs import Condition
from analysis.reference_context.run_config import ReferenceRunConfig

CONDITIONS: list[str] = [c.value for c in Condition]
_PAIRS: tuple[str, str] = ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
_SIDES: tuple[tuple[str, int], tuple[str, int]] = (("vuln", 1), ("safe", 0))
_LLM_SCANNER_GIT_SHA: str = "abc123deadbeef"
_FRAMEWORK_GIT_SHA: str = "framework-sha-test"
_BOOTSTRAP_RESAMPLES: int = 50


def _report(dataset: str, preds: list[dict], filtered: list[str]) -> dict:
    """One synthetic ``benchmark_report_*.json`` payload for one condition."""

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
    """One synthetic prediction record; ``score`` is fixed per label."""

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


def _stats_row(pair: str, suffix: str, label: int, cond: str) -> dict:
    """One synthetic ``item_stats.jsonl`` row.

    Every row shares the same token/context-token/node/file/symbol counts
    and ``symbol_names_hash`` across pairs, sides and conditions, so the
    leakage probe's constant-feature shortcut and the symmetry/budget
    checks are satisfied deterministically (no reliance on ML variance).
    """

    return {
        "pair_id": pair,
        "item_id": f"{pair}_{suffix}",
        "condition": cond,
        "label": label,
        "repo_url": f"https://github.com/o/{pair}",
        "token_count": 50,
        "context_token_count": 50,
        "budget": 50,
        "node_count": 2,
        "file_count": 1,
        "symbol_count": 2,
        "underfill": False,
        "symbols": ["a.py::t"],
        "symbol_names_hash": "h",
    }


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


def _write_scanner_artifacts(scanner: Path, config_sha256: str) -> None:
    """Write ``item_stats.jsonl``, ``overlap/strata.jsonl``, ``coverage.json``
    and a complete ``run_manifest.json`` (matching ``config_sha256`` and the
    pinned llm_scanner git sha) so ``reproducibility_check`` can pass.
    """

    (scanner / "overlap").mkdir(parents=True, exist_ok=True)
    stats = _all_stats_rows()
    strata = _all_strata_rows()
    (scanner / "item_stats.jsonl").write_text(
        "\n".join(json.dumps(s) for s in stats) + "\n"
    )
    (scanner / "overlap" / "strata.jsonl").write_text(
        "\n".join(json.dumps(s) for s in strata) + "\n"
    )
    (scanner / "coverage.json").write_text(
        json.dumps(
            {
                "csv_rows": 10,
                "eligible_pairs": 4,
                "s1_survivors": 3,
                "s1_excluded": {"checkout_failed": 1},
                "s2_input_pairs": 3,
                "s2_excluded": {},
                "evaluated_pairs": 2,
                "s2_mismatched_targets_only": 0,
            }
        )
    )
    (scanner / "run_manifest.json").write_text(
        json.dumps(
            {
                "llm_scanner_git_sha": _LLM_SCANNER_GIT_SHA,
                "llm_scanner_dirty": False,
                "config_sha256": config_sha256,
                "created_utc": "2026-09-22T00:00:00+00:00",
            }
        )
    )


def _write_reports(results_for_plan: Path, filtered: list[str] | None = None) -> None:
    """Write one benchmark report per condition under ``results_for_plan``.

    Sample ids in ``filtered`` are recorded as token-limit filtered in every
    condition and left out of the predictions.
    """

    dropped: list[str] = filtered or []
    for cond in CONDITIONS:
        exp = results_for_plan / f"exp_{cond}"
        exp.mkdir(parents=True, exist_ok=True)
        preds = [
            _pred(f"{p}_{s}", lbl, 0.6 if lbl else 0.4)
            for p in _PAIRS
            for s, lbl in _SIDES
            if f"{p}_{s}" not in dropped
        ]
        (exp / "benchmark_report_20260923_000000.json").write_text(
            json.dumps(_report(f"cleanvul_cond_{cond}.json", preds, dropped))
        )


def _write_config(
    config_path: Path,
    llm_scanner_path: Path,
    llm_scanner_output_dir: str,
    results_dir: Path,
    datasets_dir: Path,
) -> None:
    config_path.write_text(
        "\n".join(
            [
                "pins:",
                f"  llm_scanner_path: {llm_scanner_path}",
                f"  llm_scanner_git_sha: {_LLM_SCANNER_GIT_SHA}",
                "llm_scanner:",
                f"  output_dir: {llm_scanner_output_dir}",
                "framework:",
                "  experiments_config: e.json",
                "  datasets_config: d.json",
                f"  results_dir: {results_dir}",
                f"  datasets_dir: {datasets_dir}",
                f"  bootstrap_resamples: {_BOOTSTRAP_RESAMPLES}",
                "  bootstrap_seed: 1",
                "  probe_seed: 2",
                "",
            ]
        )
    )


def _invoke(
    tmp_root: Path,
    scanner_dir: Path,
    llm_scanner_path: Path,
    llm_scanner_output_dir: str,
    plan_dir_name: str,
    cli_args_extra: list[str],
    framework_dirty: bool = False,
    filtered: list[str] | None = None,
) -> Result:
    """Write artifacts + config and invoke the CLI; return the Typer result."""

    results_dir = tmp_root / "results"
    datasets_dir = tmp_root / "datasets_processed"
    report_out_dir = tmp_root / "report_out"

    _write_reports(results_dir / plan_dir_name, filtered)
    scanner_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_root / "config.yaml"
    _write_config(
        config_path, llm_scanner_path, llm_scanner_output_dir, results_dir, datasets_dir
    )
    config_sha256: str = ReferenceRunConfig.sha256(config_path)
    _write_scanner_artifacts(scanner_dir, config_sha256)

    original_git_state = cli._framework_git_state
    cli._framework_git_state = lambda: (_FRAMEWORK_GIT_SHA, framework_dirty)
    try:
        runner = CliRunner()
        return runner.invoke(
            cli.app,
            [
                "analyze-reference-context",
                "--config",
                str(config_path),
                "--output-dir",
                str(report_out_dir),
                *cli_args_extra,
            ],
        )
    finally:
        cli._framework_git_state = original_git_state


def _invoke_and_check(
    tmp_root: Path,
    scanner_dir: Path,
    llm_scanner_path: Path,
    llm_scanner_output_dir: str,
    plan_dir_name: str,
    cli_args_extra: list[str],
    framework_dirty: bool = False,
) -> Result:
    """Shared plumbing: write artifacts + config, invoke the CLI, assert on it."""

    result = _invoke(
        tmp_root,
        scanner_dir,
        llm_scanner_path,
        llm_scanner_output_dir,
        plan_dir_name,
        cli_args_extra,
        framework_dirty,
    )
    report_out_dir = tmp_root / "report_out"
    assert result.exit_code in (0, 1), (
        f"unexpected exit code {result.exit_code}; output={result.output!r}; "
        f"exception={result.exception!r}"
    )

    report_md = report_out_dir / "report.md"
    report_json_path = report_out_dir / "report.json"
    acceptance_json_path = report_out_dir / "acceptance.json"
    assert report_md.exists(), result.output
    assert report_json_path.exists(), result.output
    assert acceptance_json_path.exists(), result.output

    acceptance = json.loads(acceptance_json_path.read_text(encoding="utf-8"))
    run_valid = acceptance["run_valid"]
    assert isinstance(run_valid, bool)
    assert (result.exit_code == 1) == (run_valid is False), (
        f"exit_code={result.exit_code} but run_valid={run_valid}"
    )
    assert len(acceptance["checks"]) == 6

    report_json = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert report_json["evaluated_pairs_in_analysis"] == 2
    assert report_json["manifest"]["framework"]["framework_sha"] == _FRAMEWORK_GIT_SHA
    assert report_json["coverage"]["s2_excluded"] == {}

    report_text = report_md.read_text(encoding="utf-8")
    assert "## Coverage" in report_text
    # coverage.json's s2_excluded is an empty dict; it must render as a
    # single "none" line, never a dangling bullet with no children.
    assert "- s2_excluded: none" in report_text

    redraws = report_json["bootstrap_redraws"]
    assert isinstance(redraws["overall"], int)
    assert set(redraws["strata"]) == {"no_reference", "r0", "r0_50", "r50_100", "r100"}
    assert report_json["manifest"]["framework"]["bootstrap_redraws"] == redraws
    assert "Bootstrap redraws" in report_text
    assert report_json["items_with_fewer_scored_draws"] == {
        "draws_per_item": 1,
        "per_condition": dict.fromkeys(CONDITIONS, 0),
    }
    assert "scored draws" in report_text
    assert report_json["coverage_reconciliation"] == {
        "coverage_evaluated_pairs": 2,
        "analysed_plus_excluded": 2,
        "matches": True,
    }
    assert "Coverage reconciliation (informational)" in report_text
    return result


def test_cli_end_to_end_absolute_llm_scanner_output_dir() -> None:
    """``llm_scanner.output_dir`` given as an absolute path is used as-is."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        scanner_dir = tmp_root / "scanner_abs"
        _invoke_and_check(
            tmp_root=tmp_root,
            scanner_dir=scanner_dir,
            llm_scanner_path=tmp_root / "llm_scanner_root_unused",
            llm_scanner_output_dir=str(scanner_dir),
            plan_dir_name="reference_context_sweep",
            cli_args_extra=[],
        )


def test_cli_end_to_end_relative_output_dir_with_plan_override() -> None:
    """A relative ``llm_scanner.output_dir`` resolves against ``pins.llm_scanner_path``,
    and ``--plan`` overrides ``framework.plan`` for locating the results dir.
    """

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        llm_scanner_path = tmp_root / "llm_scanner_root"
        scanner_dir = llm_scanner_path / "scanner_rel"
        override_plan = "custom_plan_override"
        _invoke_and_check(
            tmp_root=tmp_root,
            scanner_dir=scanner_dir,
            llm_scanner_path=llm_scanner_path,
            llm_scanner_output_dir="scanner_rel",
            plan_dir_name=override_plan,
            cli_args_extra=["--plan", override_plan],
        )


def test_cli_exits_1_and_marks_run_invalid_when_a_check_fails() -> None:
    """A dirty framework tree fails reproducibility_check: exit 1, RUN INVALID."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        scanner_dir = tmp_root / "scanner_abs"
        result = _invoke_and_check(
            tmp_root=tmp_root,
            scanner_dir=scanner_dir,
            llm_scanner_path=tmp_root / "llm_scanner_root_unused",
            llm_scanner_output_dir=str(scanner_dir),
            plan_dir_name="reference_context_sweep",
            cli_args_extra=[],
            framework_dirty=True,
        )
        assert result.exit_code == 1, result.output
        acceptance = json.loads(
            (tmp_root / "report_out" / "acceptance.json").read_text(encoding="utf-8")
        )
        assert acceptance["run_valid"] is False
        failed = [c["name"] for c in acceptance["checks"] if not c["passed"]]
        assert "reproducibility_check" in failed
        report_text = (tmp_root / "report_out" / "report.md").read_text(
            encoding="utf-8"
        )
        assert report_text.startswith("# RUN INVALID")


def test_cli_zero_surviving_pairs_writes_invalid_report_and_exits_1() -> None:
    """Every pair filtered: no crash, a RUN INVALID report, exit code 1."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        scanner_dir = tmp_root / "scanner_abs"
        result = _invoke(
            tmp_root=tmp_root,
            scanner_dir=scanner_dir,
            llm_scanner_path=tmp_root / "llm_scanner_root_unused",
            llm_scanner_output_dir=str(scanner_dir),
            plan_dir_name="reference_context_sweep",
            cli_args_extra=[],
            filtered=[f"{pair}_safe" for pair in _PAIRS],
        )
        assert result.exit_code == 1, (result.output, result.exception)
        assert not isinstance(result.exception, Exception) or isinstance(
            result.exception, SystemExit
        ), result.exception
        out_dir = tmp_root / "report_out"
        acceptance = json.loads((out_dir / "acceptance.json").read_text())
        assert acceptance["run_valid"] is False
        assert [c["name"] for c in acceptance["checks"]] == ["evaluated_pairs"]
        report_json = json.loads((out_dir / "report.json").read_text())
        assert report_json["evaluated_pairs_in_analysis"] == 0
        assert report_json["inference_exclusions"] == {"filtered_by_token_limit": 2}
        assert "No pair survived" in report_json["note"]
        report_text = (out_dir / "report.md").read_text()
        assert report_text.startswith("# RUN INVALID")
        assert "No pair survived" in report_text


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e.x", *args],
        check=True,
        capture_output=True,
    )


def _with_temp_repo_root(repo: Path) -> tuple[str, bool]:
    original_root = cli._FRAMEWORK_REPO_ROOT
    cli._FRAMEWORK_REPO_ROOT = repo
    try:
        return cli._framework_git_state()
    finally:
        cli._FRAMEWORK_REPO_ROOT = original_root


def test_framework_git_state_dirty_scope() -> None:
    """Only modified tracked files under src/config/scripts make the tree dirty."""

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        for directory in ("src", "config", "scripts", "docs"):
            (repo / directory).mkdir()
            (repo / directory / "f.txt").write_text("x\n")
        _git(repo, "init", "-q")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "init")

        sha, dirty = _with_temp_repo_root(repo)
        assert len(sha) == 40 and dirty is False

        (repo / "results" / "reference_context").mkdir(parents=True)
        (repo / "results" / "reference_context" / "report.json").write_text("{}")
        (repo / "src" / "untracked.py").write_text("y = 1\n")
        (repo / "stray.txt").write_text("z\n")
        assert _with_temp_repo_root(repo)[1] is False

        (repo / "docs" / "f.txt").write_text("changed\n")
        assert _with_temp_repo_root(repo)[1] is False

        (repo / "src" / "f.txt").write_text("changed\n")
        assert _with_temp_repo_root(repo)[1] is True


if __name__ == "__main__":
    test_cli_end_to_end_absolute_llm_scanner_output_dir()
    test_cli_end_to_end_relative_output_dir_with_plan_override()
    test_cli_exits_1_and_marks_run_invalid_when_a_check_fails()
    test_cli_zero_surviving_pairs_writes_invalid_report_and_exits_1()
    test_framework_git_state_dirty_scope()
    print("ALL PASSED")
