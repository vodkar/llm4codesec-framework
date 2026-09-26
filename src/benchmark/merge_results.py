"""Recompute metrics of saved benchmark reports and merge several plans into one result set.

No inference is run: predictions stored in each report are re-scored after
dropping excluded samples (dataset config ``exclude_samples``) and filtering the
stored root findings with analysis-time rule patterns.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult
from benchmark.results import BenchmarkReport, PredictionRecord
from benchmark.run_experiment import rebuild_experiment_plan_results
from benchmark.sample_exclusions import SampleExclusion, SampleKey
from benchmark.static_findings import filter_root_findings
from entrypoints.utils import load_config_dict, normalize_config_schema

# (summary key, column label, higher is better)
_BASE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "Acc", True),
    ("precision", "P", True),
    ("recall", "R", True),
    ("f1_score", "F1", True),
    ("fpr", "FPR", False),
    ("fnr", "FNR", False),
    ("pr_auc", "AP", True),
)
_COVERAGE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "Acc", True),
    ("f1_score", "F1", True),
    ("fpr", "FPR", False),
)


def record_to_prediction(record: PredictionRecord) -> PredictionResult:
    """Rebuild the runner's PredictionResult from a saved PredictionRecord."""
    data = record.inference_data
    return PredictionResult(
        sample_id=record.sample_id,
        predicted_label=record.predicted_label,
        true_label=record.true_label,
        confidence=data.confidence,
        binary_label_confidence=data.binary_label_confidence,
        answer_probability=data.answer_probability,
        stated_confidence=data.stated_confidence,
        self_validation_probability=data.self_validation_probability,
        prompt_text=data.prompt_text,
        prompt_tokens=data.prompt_tokens,
        p_vulnerable_per_draw=data.p_vulnerable_per_draw,
        response_text=data.responses[0] if data.responses else "",
        processing_time=data.processing_time,
        tokens_used=data.tokens_used,
        is_success=record.is_success,
        error_message=record.error_message,
        all_responses=data.responses,
        answer_probabilities=data.answer_probabilities,
        stated_confidences=data.stated_confidences,
        self_validation_probabilities=data.self_validation_probabilities,
        vote_counts=data.vote_counts,
        source_row_ids=record.source_row_ids,
        root_static_findings=record.root_static_findings,
        root_findings_in_prompt=record.root_findings_in_prompt,
    )


def latest_condition_reports(plan_dir: Path) -> dict[Path, Path]:
    """Map each condition directory (relative to ``plan_dir``) to its newest report.

    Raises:
        RuntimeError: If ``plan_dir`` holds no ``benchmark_report_*.json``.
    """
    latest: dict[Path, Path] = {}
    # Report names embed a sortable timestamp, so the last one per directory is the newest.
    for path in sorted(plan_dir.rglob("benchmark_report_*.json")):
        latest[path.parent.relative_to(plan_dir)] = path
    if not latest:
        raise RuntimeError(f"No benchmark_report_*.json under {plan_dir}")
    return latest


def recompute_report(
    report: BenchmarkReport,
    exclusions: list[SampleExclusion],
    analysis_exclude_rules: list[str],
    coverage_levels: Sequence[float],
    source_path: Path,
) -> BenchmarkReport:
    """Drop excluded samples, filter stored findings and recompute binary metrics.

    Raises:
        ValueError: If exclusions are configured but a prediction has no source_row_ids.
    """
    excluded_keys: set[SampleKey] = {exclusion.key for exclusion in exclusions}
    kept: list[PredictionRecord] = []
    dropped: list[SampleKey] = []
    for record in report.predictions:
        if exclusions and record.source_row_ids is None:
            raise ValueError(
                f"{source_path}: prediction {record.sample_id} has no source_row_ids; "
                "re-run it or drop exclude_samples for this condition"
            )
        if record.source_row_ids is not None:
            key: SampleKey = (tuple(record.source_row_ids), int(record.true_label))
            if key in excluded_keys:
                dropped.append(key)
                continue
        kept.append(
            record.model_copy(
                update={
                    "root_static_findings": filter_root_findings(
                        record.root_static_findings, analysis_exclude_rules
                    )
                }
            )
        )

    metrics = BinaryMetricsCalculator(coverage_levels).calculate(
        [record_to_prediction(record) for record in kept]
    )
    info = report.benchmark_info
    benchmark_info = info.model_copy(
        update={
            "stats": info.stats.model_copy(update={"total_samples": len(kept)}),
            "extra_metadata": {
                **info.extra_metadata,
                "recomputed_from": str(source_path),
                "excluded_samples": [
                    {"source_row_ids": list(row_ids), "label": label}
                    for row_ids, label in dropped
                ],
                "analysis_exclude_finding_rules": list(analysis_exclude_rules),
                "coverage_levels": list(coverage_levels),
            },
        }
    )
    return report.model_copy(
        update={"benchmark_info": benchmark_info, "metrics": metrics, "predictions": kept}
    )


def summary_row(
    summary: dict[str, Any], coverage_levels: Sequence[float]
) -> dict[str, float | None]:
    """Pick the merged-table metrics from a metrics summary."""
    row: dict[str, float | None] = {
        key: summary.get(key)
        for key in ("accuracy", "precision", "recall", "f1_score", "pr_auc")
    }
    specificity: float | None = summary.get("specificity")
    row["fpr"] = None if specificity is None else 1.0 - specificity
    recall: float | None = row["recall"]
    row["fnr"] = None if recall is None else 1.0 - recall
    for level in coverage_levels:
        pct: int = round(level * 100)
        for name, _, _ in _COVERAGE_COLUMNS:
            row[f"{name}_at_coverage_{pct}"] = summary.get(f"{name}_at_coverage_{pct}")
    return row


def render_summary_markdown(
    rows: dict[str, dict[str, float | None]], coverage_levels: Sequence[float]
) -> str:
    """Render the merged table; the best value per column is bold (ties all bold)."""
    columns: list[tuple[str, str, bool]] = list(_BASE_COLUMNS)
    for level in coverage_levels:
        pct = round(level * 100)
        columns.extend(
            (f"{name}_at_coverage_{pct}", f"{label}@{pct}%", higher)
            for name, label, higher in _COVERAGE_COLUMNS
        )
    best: dict[str, float | None] = {}
    for key, _, higher in columns:
        values: list[float] = [
            round(value, 3) for row in rows.values() if (value := row.get(key)) is not None
        ]
        best[key] = (max(values) if higher else min(values)) if values else None

    lines: list[str] = [
        "| Condition | " + " | ".join(label for _, label, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name, row in rows.items():
        cells: list[str] = []
        for key, _, _ in columns:
            value: float | None = row.get(key)
            if value is None:
                cells.append("—")
            elif round(value, 3) == best[key]:
                cells.append(f"**{value:.3f}**")
            else:
                cells.append(f"{value:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def merge_plan_results(
    plan_dirs: list[Path],
    datasets_config: Path,
    coverage_levels: Sequence[float],
    analysis_exclude_rules: list[str],
    output_dir: Path,
    models: list[str] | None = None,
) -> Path:
    """Recompute the newest report of every condition and merge them.

    Writes recomputed reports under ``output_dir/<plan>/<condition>/``,
    ``output_dir/experiment_plan_results.json`` and ``output_dir/summary.md``.

    Args:
        models: If given, only conditions whose model directory is listed are merged.

    Raises:
        KeyError: If a condition's dataset key is missing from the datasets config.
        ValueError: If two merged conditions share a summary row name.
    """
    entries: dict[str, dict[str, Any]] = normalize_config_schema(
        load_config_dict(datasets_config)
    )["datasets"]
    rows: dict[str, dict[str, float | None]] = {}
    for plan_dir in plan_dirs:
        for condition, report_path in latest_condition_reports(plan_dir).items():
            # Condition directories are <dataset_key>/<model>/<prompt>.
            if models is not None and condition.parts[1] not in models:
                continue
            dataset_key: str = condition.parts[0]
            if dataset_key not in entries:
                raise KeyError(f"Dataset key {dataset_key!r} not in {datasets_config}")
            exclusions: list[SampleExclusion] = [
                SampleExclusion.model_validate(entry)
                for entry in entries[dataset_key].get("exclude_samples", [])
            ]
            report = BenchmarkReport.model_validate(
                json.loads(report_path.read_text(encoding="utf-8"))
            )
            recomputed: BenchmarkReport = recompute_report(
                report, exclusions, analysis_exclude_rules, coverage_levels, report_path
            )
            target: Path = output_dir / plan_dir.name / condition / report_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(recomputed.model_dump(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            row_name: str = f"{dataset_key} · {condition.parts[-1]}"
            if row_name in rows:
                raise ValueError(
                    f"Two conditions map to summary row {row_name!r}; pass --model to pick one"
                )
            rows[row_name] = summary_row(recomputed.metrics.summary, coverage_levels)

    rebuild_experiment_plan_results(
        input_path=output_dir,
        plan_name="merged",
        description="Merged recomputed results of: " + ", ".join(str(p) for p in plan_dirs),
    )
    summary_path: Path = output_dir / "summary.md"
    summary_path.write_text(render_summary_markdown(rows, coverage_levels), encoding="utf-8")
    return summary_path
