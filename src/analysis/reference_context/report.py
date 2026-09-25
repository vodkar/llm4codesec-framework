"""Report writer for the reference-context oracle evaluation (design spec §10).

Writes three artifacts into an output directory for one analysis run:

- ``report.md``: a human-readable report in the section order fixed by the
  task brief -- an optional **RUN INVALID** banner, the coverage funnel, the
  acceptance table, per-condition metrics, base rate and realized lengths,
  the retrieval/reasoning decomposition, and one metrics table per recall
  stratum.
- ``report.json``: every number behind the report, as strict JSON
  (``allow_nan=False``).
- ``acceptance.json``: the acceptance checks plus ``"run_valid"``, as strict
  JSON.

:func:`write_invalid_report` writes the same three files, headed **RUN
INVALID**, for a run that cannot be analysed at all (no surviving pairs).
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple, TypeAlias

from analysis.reference_context.acceptance import CheckResult
from analysis.reference_context.bootstrap import ConfidenceInterval
from analysis.reference_context.inputs import AnalysisInputs, Condition, ItemRecord
from analysis.reference_context.metrics import (
    STRATA,
    ConditionTable,
    MetricName,
    base_rate,
)

DecompositionTable: TypeAlias = dict[MetricName, dict[str, ConfidenceInterval]]
LengthsTable: TypeAlias = dict[Condition, dict[str, float | None]]
StrataTables: TypeAlias = dict[str, ConditionTable]

_REPORT_MD_FILENAME: Final[str] = "report.md"
_REPORT_JSON_FILENAME: Final[str] = "report.json"
_ACCEPTANCE_JSON_FILENAME: Final[str] = "acceptance.json"
_LEAKAGE_PROBE_CHECK_NAME: Final[str] = "leakage_probe"
_JSON_INDENT: Final[int] = 2
_COVERAGE_EVALUATED_KEY: Final[str] = "evaluated_pairs"
_NO_PAIRS_CHECK_NAME: Final[str] = "evaluated_pairs"


class BootstrapRedraws(NamedTuple):
    """Draws discarded by the cluster bootstraps (one-class resamples).

    Attributes:
        overall: Discarded draws of the overall bootstrap.
        strata: Stratum to its bootstrap's discarded draws; ``None`` when the
            stratum was too small to bootstrap.
    """

    overall: int
    strata: dict[str, int | None]

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the redraw counts."""

        return {"overall": self.overall, "strata": dict(self.strata)}

    def markdown_line(self) -> str:
        """Return the one-line markdown summary of the redraw counts."""

        strata: str = ", ".join(
            f"{stratum}={'n/a' if count is None else count}"
            for stratum, count in self.strata.items()
        )
        return f"Bootstrap redraws (discarded one-class resamples): overall={self.overall}; strata: {strata}"


_COVERAGE_FUNNEL_KEYS: Final[tuple[str, ...]] = (
    "csv_rows",
    "eligible_pairs",
    "s1_survivors",
    "s1_excluded",
    "s2_input_pairs",
    "s2_excluded",
    "evaluated_pairs",
    "s2_mismatched_targets_only",
)

_LENGTH_STAT_KEYS: Final[tuple[str, ...]] = (
    "prompt_tokens_mean",
    "prompt_tokens_median",
    "prompt_tokens_q1",
    "prompt_tokens_q3",
    "context_tokens_mean",
    "context_tokens_median",
    "context_tokens_q1",
    "context_tokens_q3",
)


def _format_number(value: float | None, digits: int = 3) -> str:
    """Format a float to a fixed number of decimals, or ``"n/a"`` for ``None``.

    Args:
        value: The value to format.
        digits: Number of decimal places.

    Returns:
        ``"n/a"`` when ``value`` is ``None``; otherwise the formatted number.
    """

    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_ci(interval: ConfidenceInterval) -> str:
    """Format a confidence interval as ``"0.712 [0.680, 0.744]"``.

    Args:
        interval: The interval to format.

    Returns:
        ``"n/a"`` when the point estimate is undefined. Otherwise the point
        estimate to three decimals, with the percentile bounds appended in
        brackets when both are known, and a trailing ``" (dropped k)"`` note
        appended only when ``interval.dropped`` is greater than zero.
    """

    if interval.point is None:
        return "n/a"
    text: str = f"{interval.point:.3f}"
    if interval.low is not None and interval.high is not None:
        text = f"{text} [{interval.low:.3f}, {interval.high:.3f}]"
    if interval.dropped > 0:
        text = f"{text} (dropped {interval.dropped})"
    return text


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a GitHub-flavored markdown table.

    Args:
        headers: Column headers.
        rows: Table rows; each must have the same length as ``headers``.

    Returns:
        The table's lines, including the header separator row.
    """

    lines: list[str] = [
        f"| {' | '.join(headers)} |",
        f"|{'|'.join(['---'] * len(headers))}|",
    ]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def _invalid_banner(checks: Sequence[CheckResult]) -> list[str]:
    """Build the ``# RUN INVALID`` banner, or an empty list when nothing failed.

    Args:
        checks: This run's acceptance checks.

    Returns:
        The banner's lines (first line exactly ``"# RUN INVALID"``), or an
        empty list when every check passed.
    """

    failed: list[str] = [check.name for check in checks if not check.passed]
    if not failed:
        return []
    return ["# RUN INVALID", "", f"Failed checks: {', '.join(failed)}", ""]


def _render_coverage_entry(key: str, value: object) -> list[str]:
    """Render one coverage.json entry, expanding a reason-count breakdown."""

    if isinstance(value, dict):
        if not value:
            return [f"- {key}: none"]
        lines: list[str] = [f"- {key}:"]
        lines.extend(
            f"  - {reason}: {count}" for reason, count in sorted(value.items())
        )
        return lines
    return [f"- {key}: {value}"]


def _coverage_section(
    coverage: dict[str, object],
    inference_exclusions: dict[str, int],
    evaluated_in_analysis: int,
) -> list[str]:
    """Render the coverage funnel: ``coverage.json`` counts plus inference exclusions.

    Args:
        coverage: llm_scanner's parsed ``coverage.json``.
        inference_exclusions: Pair counts dropped by :func:`load_inputs`'s
            report-based and scanner-artifact checks, keyed by reason.
        evaluated_in_analysis: Final count of pairs surviving into this
            analysis (``len(inputs.pair_ids)``).

    Returns:
        The section's markdown lines, ending with the evaluated-pairs count.
    """

    lines: list[str] = ["## Coverage", ""]
    for key in _COVERAGE_FUNNEL_KEYS:
        if key in coverage:
            lines.extend(_render_coverage_entry(key, coverage[key]))
    extra_keys: list[str] = sorted(set(coverage) - set(_COVERAGE_FUNNEL_KEYS))
    for key in extra_keys:
        lines.extend(_render_coverage_entry(key, coverage[key]))
    lines.append("")
    lines.append("### Inference-time exclusions")
    lines.append("")
    if inference_exclusions:
        lines.extend(
            f"- {reason}: {count}"
            for reason, count in sorted(inference_exclusions.items())
        )
    else:
        lines.append("- none")
    lines.append("")
    lines.append(f"**Evaluated pairs in analysis: {evaluated_in_analysis}**")
    lines.append("")
    lines.append(
        _reconciliation_line(
            coverage_reconciliation(
                coverage, inference_exclusions, evaluated_in_analysis
            )
        )
    )
    lines.append("")
    return lines


def coverage_reconciliation(
    coverage: Mapping[str, object],
    inference_exclusions: Mapping[str, int],
    evaluated_in_analysis: int,
) -> dict[str, object]:
    """Compare ``coverage.evaluated_pairs`` with analysed plus excluded pairs.

    Informational only: a smoke run (``sample_limit``) legitimately evaluates
    fewer pairs than llm_scanner built.

    Args:
        coverage: llm_scanner's parsed ``coverage.json``.
        inference_exclusions: Pairs dropped by :func:`load_inputs`, by reason.
        evaluated_in_analysis: Pairs surviving into the analysis.

    Returns:
        ``coverage_evaluated_pairs`` (``None`` when absent or not an int),
        ``analysed_plus_excluded`` and ``matches`` (``None`` when unknown).
    """

    recorded: object = coverage.get(_COVERAGE_EVALUATED_KEY)
    expected: int | None = (
        recorded
        if isinstance(recorded, int) and not isinstance(recorded, bool)
        else None
    )
    accounted: int = evaluated_in_analysis + sum(inference_exclusions.values())
    return {
        "coverage_evaluated_pairs": expected,
        "analysed_plus_excluded": accounted,
        "matches": None if expected is None else expected == accounted,
    }


def _reconciliation_line(reconciliation: Mapping[str, object]) -> str:
    """Render the coverage reconciliation as one informational markdown line."""

    matches: object = reconciliation["matches"]
    verdict: str = (
        "unknown (coverage.json has no evaluated_pairs)"
        if matches is None
        else ("match" if matches else "MISMATCH")
    )
    return (
        "Coverage reconciliation (informational): coverage.evaluated_pairs="
        f"{reconciliation['coverage_evaluated_pairs']} vs analysed + inference "
        f"exclusions={reconciliation['analysed_plus_excluded']}: {verdict}"
    )


def draw_shortfall(inputs: AnalysisInputs) -> dict[str, object]:
    """Count, per condition, items with fewer than ``n`` scored draws.

    A draw is scored when its ``p_vulnerable_per_draw`` entry is non-null;
    ``n`` is :attr:`AnalysisInputs.draws_per_item`.

    Args:
        inputs: Aligned analysis inputs.

    Returns:
        ``{"draws_per_item": n, "per_condition": {condition: count}}``; the
        counts are ``None`` when ``n`` is unknown.
    """

    draws: int | None = inputs.draws_per_item
    per_condition: dict[str, int | None] = {
        condition.value: None
        if draws is None
        else sum(
            item.scored_draws is None or item.scored_draws < draws
            for item in inputs.items
            if item.condition is condition
        )
        for condition in Condition
    }
    return {"draws_per_item": draws, "per_condition": per_condition}


def _draw_shortfall_line(shortfall: Mapping[str, object]) -> str:
    """Render the per-condition scored-draw shortfall as one markdown line."""

    per_condition: object = shortfall["per_condition"]
    counts: str = (
        ", ".join(
            f"{condition}={'n/a' if count is None else count}"
            for condition, count in per_condition.items()
        )
        if isinstance(per_condition, dict)
        else "n/a"
    )
    return (
        f"Items with fewer than n={shortfall['draws_per_item']} scored draws: {counts}"
    )


def _leakage_diagnostic_lines(check: CheckResult) -> list[str]:
    """Render the leakage probe's non-gating per-condition length diagnostic."""

    ranking: object = check.detail.get("length_ranking_accuracy", {})
    deltas: object = check.detail.get("mean_token_delta", {})
    if not isinstance(ranking, dict) or not isinstance(deltas, dict):
        return []
    lines: list[str] = [
        "Non-gating diagnostic (does raw context length alone rank vuln above safe?):",
        "",
    ]
    lines.extend(
        f"- {condition.value}: length_ranking_accuracy="
        f"{_format_number(ranking.get(condition.value))}, mean_token_delta="
        f"{_format_number(deltas.get(condition.value), digits=2)}"
        for condition in Condition
    )
    lines.append("")
    return lines


def _acceptance_section(checks: Sequence[CheckResult]) -> list[str]:
    """Render the acceptance table plus the leakage probe's length diagnostic.

    Args:
        checks: This run's acceptance checks.

    Returns:
        The section's markdown lines.
    """

    lines: list[str] = ["## Acceptance checks", ""]
    rows: list[list[str]] = [
        [
            check.name,
            "PASS" if check.passed else "FAIL",
            _format_number(check.value),
            check.threshold,
        ]
        for check in checks
    ]
    lines.extend(_markdown_table(("check", "result", "value", "threshold"), rows))
    lines.append("")
    leakage_check: CheckResult | None = next(
        (check for check in checks if check.name == _LEAKAGE_PROBE_CHECK_NAME), None
    )
    if leakage_check is not None:
        lines.extend(_leakage_diagnostic_lines(leakage_check))
    return lines


def _condition_table_section(title: str, table: ConditionTable) -> list[str]:
    """Render one metrics-per-condition table under ``title``.

    Args:
        title: Markdown heading line for the table (e.g. ``"## Metrics per
            condition"``).
        table: Condition to metric to confidence interval.

    Returns:
        The section's markdown lines.
    """

    lines: list[str] = [title, ""]
    headers: tuple[str, ...] = ("condition", *(metric.value for metric in MetricName))
    rows: list[list[str]] = [
        [
            condition.value,
            *(_format_ci(table[condition][metric]) for metric in MetricName),
        ]
        for condition in table
    ]
    lines.extend(_markdown_table(headers, rows))
    lines.append("")
    return lines


def _base_rate_by_condition(inputs: AnalysisInputs) -> dict[Condition, float | None]:
    """Fraction of positive items per condition, ``None`` for an empty condition."""

    by_condition: dict[Condition, list[ItemRecord]] = {
        condition: [item for item in inputs.items if item.condition is condition]
        for condition in Condition
    }
    return {
        condition: base_rate(items) if items else None
        for condition, items in by_condition.items()
    }


def _base_rate_and_lengths_section(
    inputs: AnalysisInputs, lengths: LengthsTable
) -> list[str]:
    """Render the base-rate and realized-length table.

    Args:
        inputs: Aligned analysis inputs, used to compute the base rate.
        lengths: Realized prompt/context length stats per condition.

    Returns:
        The section's markdown lines.
    """

    lines: list[str] = ["## Base rate and realized lengths", ""]
    rates: dict[Condition, float | None] = _base_rate_by_condition(inputs)
    headers: tuple[str, ...] = ("condition", "base_rate", *_LENGTH_STAT_KEYS)
    rows: list[list[str]] = [
        [
            condition.value,
            _format_number(rates.get(condition)),
            *(_format_number(lengths[condition].get(key)) for key in _LENGTH_STAT_KEYS),
        ]
        for condition in Condition
        if condition in lengths
    ]
    lines.extend(_markdown_table(headers, rows))
    lines.append("")
    return lines


def _decomposition_section(decomposition: DecompositionTable) -> list[str]:
    """Render the retrieval-vs-reasoning decomposition table.

    Args:
        decomposition: Metric to ``{"retrieval_loss": ..., "reasoning_loss": ...}``.

    Returns:
        The section's markdown lines.
    """

    lines: list[str] = ["## Retrieval vs reasoning decomposition", ""]
    headers: tuple[str, str, str] = ("metric", "retrieval_loss", "reasoning_loss")
    rows: list[list[str]] = [
        [
            metric.value,
            _format_ci(values["retrieval_loss"]),
            _format_ci(values["reasoning_loss"]),
        ]
        for metric, values in decomposition.items()
    ]
    lines.extend(_markdown_table(headers, rows))
    lines.append("")
    return lines


def _strata_section(strata: StrataTables) -> list[str]:
    """Render one metrics-per-condition table per recall stratum.

    Args:
        strata: Recall stratum to its condition table.

    Returns:
        The section's markdown lines.
    """

    lines: list[str] = ["## Metrics per recall stratum", ""]
    for stratum in STRATA:
        if stratum not in strata:
            continue
        lines.extend(
            _condition_table_section(f"### Stratum: {stratum}", strata[stratum])
        )
    return lines


def _ci_to_json(interval: ConfidenceInterval) -> dict[str, object]:
    """Convert a confidence interval to a JSON-safe mapping."""

    return {
        "point": interval.point,
        "low": interval.low,
        "high": interval.high,
        "dropped": interval.dropped,
    }


def _condition_table_to_json(
    table: ConditionTable,
) -> dict[str, dict[str, dict[str, object]]]:
    """Convert a condition table to a JSON-safe, string-keyed mapping."""

    return {
        condition.value: {
            metric.value: _ci_to_json(interval) for metric, interval in metrics.items()
        }
        for condition, metrics in table.items()
    }


def _decomposition_to_json(
    decomposition: DecompositionTable,
) -> dict[str, dict[str, dict[str, object]]]:
    """Convert the decomposition table to a JSON-safe, string-keyed mapping."""

    return {
        metric.value: {name: _ci_to_json(interval) for name, interval in values.items()}
        for metric, values in decomposition.items()
    }


def _lengths_to_json(lengths: LengthsTable) -> dict[str, dict[str, float | None]]:
    """Convert the lengths table to a JSON-safe, string-keyed mapping."""

    return {condition.value: dict(stats) for condition, stats in lengths.items()}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as strict, sorted, indented JSON.

    Args:
        path: Destination file.
        payload: JSON-safe mapping to serialize.

    Raises:
        ValueError: If ``payload`` contains a NaN or infinite float.
    """

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, allow_nan=False, sort_keys=True, indent=_JSON_INDENT)
        )
        handle.write("\n")


def write_report(
    out_dir: Path,
    inputs: AnalysisInputs,
    tables: ConditionTable,
    strata: StrataTables,
    lengths: LengthsTable,
    decomposition: DecompositionTable,
    checks: list[CheckResult],
    manifest_extra: dict[str, object],
    bootstrap_redraws: BootstrapRedraws | None = None,
) -> Path:
    """Write ``report.md``, ``report.json`` and ``acceptance.json`` for one run.

    Args:
        out_dir: Directory to write into; created if missing.
        inputs: Aligned analysis inputs (coverage, manifest, inference
            exclusions and the surviving pair/item table).
        tables: Overall per-condition metrics table (:func:`condition_table`).
        strata: Per-recall-stratum metrics tables (:func:`stratum_tables`).
        lengths: Realized prompt/context length stats per condition
            (:func:`length_stats`).
        decomposition: Retrieval/reasoning loss decomposition
            (:func:`decomposition`).
        checks: This run's acceptance checks, in design spec §11 order.
        manifest_extra: Framework-side run metadata not already carried by
            ``inputs.manifest`` (framework SHA/dirty flag, config path and
            hash, plan name, bootstrap/probe settings, ...).
        bootstrap_redraws: Discarded-draw counts of the overall and
            per-stratum bootstraps, when known.

    Returns:
        The path to the written ``report.md``.

    Raises:
        ValueError: If a number destined for ``report.json`` or
            ``acceptance.json`` is NaN or infinite.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    run_valid: bool = all(check.passed for check in checks)

    lines: list[str] = []
    lines.extend(_invalid_banner(checks))
    lines.extend(
        _coverage_section(
            inputs.coverage, inputs.inference_exclusions, len(inputs.pair_ids)
        )
    )
    lines.extend(_acceptance_section(checks))
    lines.extend(_condition_table_section("## Metrics per condition", tables))
    shortfall: dict[str, object] = draw_shortfall(inputs)
    lines.extend([_draw_shortfall_line(shortfall), ""])
    if bootstrap_redraws is not None:
        lines.extend([bootstrap_redraws.markdown_line(), ""])
    lines.extend(_base_rate_and_lengths_section(inputs, lengths))
    lines.extend(_decomposition_section(decomposition))
    lines.extend(_strata_section(strata))

    report_path: Path = out_dir / _REPORT_MD_FILENAME
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip("\n") + "\n")

    report_json: dict[str, object] = {
        "manifest": {"llm_scanner": inputs.manifest, "framework": manifest_extra},
        "coverage": inputs.coverage,
        "inference_exclusions": inputs.inference_exclusions,
        "evaluated_pairs_in_analysis": len(inputs.pair_ids),
        "coverage_reconciliation": coverage_reconciliation(
            inputs.coverage, inputs.inference_exclusions, len(inputs.pair_ids)
        ),
        "items_with_fewer_scored_draws": shortfall,
        "bootstrap_redraws": None
        if bootstrap_redraws is None
        else bootstrap_redraws.to_json(),
        "base_rate": {
            condition.value: rate
            for condition, rate in _base_rate_by_condition(inputs).items()
        },
        "lengths": _lengths_to_json(lengths),
        "condition_table": _condition_table_to_json(tables),
        "stratum_tables": {
            stratum: _condition_table_to_json(table)
            for stratum, table in strata.items()
        },
        "decomposition": _decomposition_to_json(decomposition),
    }
    _write_json(out_dir / _REPORT_JSON_FILENAME, report_json)

    acceptance_json: dict[str, object] = {
        "run_valid": run_valid,
        "checks": [check.model_dump() for check in checks],
    }
    _write_json(out_dir / _ACCEPTANCE_JSON_FILENAME, acceptance_json)

    return report_path


def write_invalid_report(
    out_dir: Path,
    inputs: AnalysisInputs,
    note: str,
    manifest_extra: dict[str, object],
) -> Path:
    """Write a **RUN INVALID** report for a run that cannot be analysed.

    Used when no pair survives into the analysis: there is nothing to
    bootstrap, so only the coverage funnel and ``note`` are reported, and
    ``acceptance.json`` carries a single failed ``evaluated_pairs`` check.

    Args:
        out_dir: Directory to write into; created if missing.
        inputs: Aligned analysis inputs (coverage, manifest, exclusions).
        note: Why the run cannot be analysed.
        manifest_extra: Framework-side run metadata (see :func:`write_report`).

    Returns:
        The path to the written ``report.md``.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    check = CheckResult(
        name=_NO_PAIRS_CHECK_NAME,
        passed=False,
        value=float(len(inputs.pair_ids)),
        threshold="at least one pair survives into the analysis",
        detail={"note": note},
    )
    lines: list[str] = [*_invalid_banner([check]), note, ""]
    lines.extend(
        _coverage_section(
            inputs.coverage, inputs.inference_exclusions, len(inputs.pair_ids)
        )
    )
    report_path: Path = out_dir / _REPORT_MD_FILENAME
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip("\n") + "\n")

    _write_json(
        out_dir / _REPORT_JSON_FILENAME,
        {
            "manifest": {"llm_scanner": inputs.manifest, "framework": manifest_extra},
            "coverage": inputs.coverage,
            "inference_exclusions": inputs.inference_exclusions,
            "evaluated_pairs_in_analysis": len(inputs.pair_ids),
            "coverage_reconciliation": coverage_reconciliation(
                inputs.coverage, inputs.inference_exclusions, len(inputs.pair_ids)
            ),
            "note": note,
        },
    )
    _write_json(
        out_dir / _ACCEPTANCE_JSON_FILENAME,
        {"run_valid": False, "checks": [check.model_dump()]},
    )
    return report_path
