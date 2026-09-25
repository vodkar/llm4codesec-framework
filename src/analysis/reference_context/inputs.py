"""Analysis inputs: aligned item table and pair survival.

Joins the framework's per-condition benchmark reports with llm_scanner's
per-item artifacts (``item_stats.jsonl``, ``overlap/strata.jsonl``,
``coverage.json``, ``run_manifest.json``). A pair is analysed only if both
its items are present, successful, unfiltered and scored in all five
conditions, and both items have a complete set of scanner artifacts (an
``item_stats.jsonl`` row per condition and a ``strata.jsonl`` row); every
dropped pair is counted by reason.

Before any alignment, the five chosen reports are checked to come from one
consistent, fresh run (see :func:`validate_reports`): identical model run
config and prompt identity, one non-null pinned sampling seed, dataset files
whose sha256 matches llm_scanner's manifest, and report timestamps no older
than the manifest.
"""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, NamedTuple

from pydantic import BaseModel, ConfigDict

from benchmark.results import BenchmarkReport, PredictionRecord

_ITEM_STATS_FILENAME: Final[str] = "item_stats.jsonl"
_STRATA_PATH: Final[tuple[str, str]] = ("overlap", "strata.jsonl")
_COVERAGE_FILENAME: Final[str] = "coverage.json"
_MANIFEST_FILENAME: Final[str] = "run_manifest.json"

_FILTERED_BY_TOKEN_LIMIT: Final[str] = "filtered_by_token_limit"
_INFERENCE_FAILED: Final[str] = "inference_failed"
_UNSCORED: Final[str] = "unscored"
_MISSING_FROM_REPORT: Final[str] = "missing_from_report"
_MISSING_SCANNER_ARTIFACTS: Final[str] = "missing_scanner_artifacts"
_REPORT_GLOB: Final[str] = "benchmark_report_*.json"
_CONDITION_DATASET_SHA_KEY: Final[str] = "condition_dataset_sha256"
_CREATED_UTC_KEY: Final[str] = "created_utc"
_CONTAINER_APP_ROOT: Final[PurePosixPath] = PurePosixPath("/app")
FRAMEWORK_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
"""Framework repository root; the container's ``/app`` working dir maps onto it
for ``datasets_processed/`` (a bind mount)."""
_SUFFIX_LABELS: Final[MappingProxyType[str, int]] = MappingProxyType(
    {"_vuln": 1, "_safe": 0}
)
_FAILURE_PRIORITY: Final[tuple[str, str, str, str, str]] = (
    _FILTERED_BY_TOKEN_LIMIT,
    _INFERENCE_FAILED,
    _UNSCORED,
    _MISSING_FROM_REPORT,
    _MISSING_SCANNER_ARTIFACTS,
)


class Condition(StrEnum):
    """Evaluation context conditions, in reporting order (matches llm_scanner)."""

    NONE = "none"
    RANDOM = "random"
    CPG = "cpg"
    REFERENCE = "reference"
    MISMATCHED = "mismatched"


class ItemRecord(BaseModel):
    """One item's aligned model output and context-size stats for one condition."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    pair_id: str
    condition: Condition
    label: int
    repo_url: str
    score: float | None
    predicted_label: int | None
    success: bool
    prompt_tokens: int | None
    token_count: int
    budget: int | None
    node_count: int
    file_count: int
    symbol_count: int
    symbol_names_hash: str
    underfill: bool
    stratum: str
    scored_draws: int | None = None
    """Draws with a non-null P(VULNERABLE) in ``p_vulnerable_per_draw``;
    ``None`` when the item has no prediction."""
    context_token_count: int | None = None
    """Tokens of the rendered non-target context only (excludes the target
    function(s), whose length legitimately changes with the fix); read from
    ``item_stats.jsonl``. ``None`` when the row predates this field or the
    condition does not distinguish target from context (e.g. ``cpg``)."""


class AnalysisInputs(BaseModel):
    """Aligned item table plus pair-survival bookkeeping for one analysis run."""

    items: list[ItemRecord]
    pair_ids: list[str]
    coverage: dict[str, object]
    manifest: dict[str, object]
    inference_exclusions: dict[str, int]
    draws_per_item: int | None = None
    """Self-consistency draws per item (``self_consistency_samples``) shared by
    all five reports; ``None`` when unknown."""


class _ConditionView(NamedTuple):
    """One condition's report, indexed for constant-time item lookups."""

    filtered: frozenset[str]
    predictions: dict[str, PredictionRecord]


class _ItemStatsRow(BaseModel):
    """One row of llm_scanner's ``item_stats.jsonl``."""

    pair_id: str
    item_id: str
    condition: Condition
    label: int
    repo_url: str
    token_count: int
    context_token_count: int | None = None
    budget: int | None
    node_count: int
    file_count: int
    symbol_count: int
    underfill: bool
    symbol_names_hash: str


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a newline-delimited JSON file into a list of row dicts.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        One dict per non-blank line, in file order.
    """

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        raw: object = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return raw


def _in_condition_dir(path: Path, results_dir: Path, condition: Condition) -> bool:
    """Whether a report lives under a directory named for ``condition``.

    Run-plan output is ``<plan>/<dataset key>/<model>/<prompt>/``, and the
    dataset keys are ``reference_context_<condition>``.
    """

    suffix: str = f"_{condition.value}"
    return any(
        part == condition.value or part.endswith(suffix)
        for part in path.relative_to(results_dir).parent.parts
    )


def _latest_report_for(results_dir: Path, condition: Condition) -> BenchmarkReport:
    """Return the most recent benchmark report for one condition.

    The report is chosen by path alone (latest filename among the reports
    under a ``*_<condition>`` directory; the embedded ``%Y%m%d_%H%M%S``
    timestamp sorts chronologically), so only the chosen file is parsed.

    Args:
        results_dir: Root directory searched recursively for
            ``benchmark_report_*.json`` files.
        condition: Condition whose report to load.

    Returns:
        The parsed :class:`BenchmarkReport`.

    Raises:
        FileNotFoundError: If no report lives under a directory for this
            condition.
        ValueError: If the chosen report's ``dataset_path`` does not
            reference this condition's dataset file.
    """

    candidates: list[Path] = [
        path
        for path in results_dir.rglob(_REPORT_GLOB)
        if _in_condition_dir(path, results_dir, condition)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No benchmark report under {results_dir} for condition {condition.value!r}"
        )
    latest: Path = max(candidates, key=lambda path: (path.name, str(path)))
    report: BenchmarkReport = BenchmarkReport.model_validate(_read_json(latest))
    expected: str = f"cleanvul_cond_{condition.value}.json"
    if not report.benchmark_info.dataset_path.endswith(expected):
        raise ValueError(
            f"Latest report {latest} for condition {condition.value!r} references "
            f"dataset {report.benchmark_info.dataset_path!r}, expected *{expected}"
        )
    return report


def load_reports(results_dir: Path) -> dict[Condition, BenchmarkReport]:
    """Load the latest benchmark report for each of the five conditions.

    Args:
        results_dir: Root directory searched recursively for
            ``benchmark_report_*.json`` files.

    Returns:
        Mapping of :class:`Condition` to its latest :class:`BenchmarkReport`.

    Raises:
        FileNotFoundError: If any condition has no matching report; the
            message names every missing condition.
    """

    reports: dict[Condition, BenchmarkReport] = {}
    missing: list[str] = []
    for condition in Condition:
        try:
            reports[condition] = _latest_report_for(results_dir, condition)
        except FileNotFoundError:
            missing.append(condition.value)
    if missing:
        raise FileNotFoundError(
            f"No benchmark report found under {results_dir} for condition(s): "
            f"{', '.join(missing)}"
        )
    return reports


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _resolve_dataset_path(raw: str, dataset_root: Path) -> Path:
    """Map a report's (container-side) ``dataset_path`` onto the host.

    Args:
        raw: ``benchmark_info.dataset_path`` as recorded by the run.
        dataset_root: Host directory the container's ``/app`` maps onto.

    Returns:
        ``dataset_root / raw`` for a relative path, ``dataset_root / rest``
        for ``/app/rest``, and ``raw`` unchanged for any other absolute path.
    """

    path = PurePosixPath(raw)
    if not path.is_absolute():
        return dataset_root / path
    if path.is_relative_to(_CONTAINER_APP_ROOT):
        return dataset_root / path.relative_to(_CONTAINER_APP_ROOT)
    return Path(raw)


def _parse_aware(raw: str, what: str) -> datetime:
    """Parse an ISO timestamp that must carry a UTC offset.

    Raises:
        ValueError: If ``raw`` is not ISO 8601 or has no UTC offset.
    """

    parsed: datetime = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{what} {raw!r} has no timezone; cannot order it")
    return parsed


def _report_time(report: BenchmarkReport, condition: Condition) -> datetime:
    """Return a report's creation time as an aware datetime.

    Uses ``timestamp_utc``; falls back to ``timestamp`` only when that is
    itself timezone-aware (the container's naive local ``timestamp`` cannot
    be ordered against the manifest).

    Raises:
        ValueError: If the report records no timezone-aware timestamp.
    """

    info = report.benchmark_info
    raw: str = info.timestamp_utc if info.timestamp_utc is not None else info.timestamp
    return _parse_aware(raw, f"Report timestamp for condition {condition.value!r}")


def _require_identical(values: Mapping[Condition, object], what: str) -> None:
    """Raise unless every condition's value is identical.

    Raises:
        ValueError: Naming each condition's differing value.
    """

    if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
        detail: str = "; ".join(
            f"{condition.value}={value!r}" for condition, value in values.items()
        )
        raise ValueError(
            f"The five condition reports do not share one {what} (mixed or stale "
            f"runs under the results dir?): {detail}"
        )


def _check_dataset_hashes(
    reports: Mapping[Condition, BenchmarkReport],
    manifest: Mapping[str, object],
    dataset_root: Path,
) -> None:
    """Match each report's dataset file against the manifest's sha256, when recorded.

    Raises:
        ValueError: If a recorded sha256 is not a string, the dataset file
            is missing, or its sha256 differs from the manifest's.
    """

    expected_by_condition: object = manifest.get(_CONDITION_DATASET_SHA_KEY)
    if not isinstance(expected_by_condition, dict):
        return
    for condition, report in reports.items():
        expected: object = expected_by_condition.get(condition.value)
        if expected is None:
            continue
        if not isinstance(expected, str):
            raise ValueError(
                f"{_CONDITION_DATASET_SHA_KEY}[{condition.value!r}] is not a string: "
                f"{expected!r}"
            )
        path: Path = _resolve_dataset_path(
            report.benchmark_info.dataset_path, dataset_root
        )
        if not path.is_file():
            raise ValueError(
                f"Dataset file for condition {condition.value!r} not found at {path}"
            )
        actual: str = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Dataset for condition {condition.value!r} ({path}) has sha256 "
                f"{actual}, but the llm_scanner manifest records {expected}: the "
                "report was run on a different condition dataset"
            )


def _check_freshness(
    reports: Mapping[Condition, BenchmarkReport], manifest: Mapping[str, object]
) -> None:
    """Require every report to be no older than the llm_scanner manifest.

    Raises:
        ValueError: If the manifest lacks ``created_utc``, a report has no
            timezone-aware timestamp, or a report predates the manifest.
    """

    raw_created: object = manifest.get(_CREATED_UTC_KEY)
    if not isinstance(raw_created, str):
        raise ValueError(
            f"run_manifest.json lacks a string {_CREATED_UTC_KEY!r}; cannot rule "
            "out reports older than the condition datasets"
        )
    created: datetime = _parse_aware(raw_created, f"Manifest {_CREATED_UTC_KEY}")
    stale: list[str] = [
        f"{condition.value} ({_report_time(report, condition).isoformat()})"
        for condition, report in reports.items()
        if _report_time(report, condition) < created
    ]
    if stale:
        raise ValueError(
            f"Report(s) older than the llm_scanner manifest ({raw_created}): "
            f"{', '.join(stale)}; re-run the benchmark sweep"
        )


def validate_reports(
    reports: Mapping[Condition, BenchmarkReport],
    manifest: Mapping[str, object],
    dataset_root: Path = FRAMEWORK_REPO_ROOT,
) -> None:
    """Reject a mixed or stale set of condition reports before any analysis.

    Args:
        reports: The chosen report per condition.
        manifest: llm_scanner's parsed ``run_manifest.json``.
        dataset_root: Host directory that report ``dataset_path`` values
            (container-relative) resolve against.

    Raises:
        ValueError: Unless all reports share one ``benchmark_info.model``
            (including a non-null ``sampling_seed``) and one prompt identity
            (``prompt_identifier`` + ``prompt_template_sha256``, non-null),
            every dataset file's sha256 matches
            ``manifest["condition_dataset_sha256"][<condition>]`` (when
            recorded), and no report predates ``manifest["created_utc"]``.
    """

    _require_identical(
        {
            condition: report.benchmark_info.model.model_dump(mode="json")
            for condition, report in reports.items()
        },
        "model run config",
    )
    unseeded: list[str] = [
        condition.value
        for condition, report in reports.items()
        if report.benchmark_info.model.sampling_seed is None
    ]
    if unseeded:
        raise ValueError(
            f"Report(s) without a pinned sampling_seed: {', '.join(unseeded)}; "
            "draws are not paired across conditions"
        )
    prompts: dict[Condition, object] = {
        condition: [
            report.benchmark_info.prompt_identifier,
            report.benchmark_info.prompt_template_sha256,
        ]
        for condition, report in reports.items()
    }
    unidentified: list[str] = [
        condition.value
        for condition, report in reports.items()
        if report.benchmark_info.prompt_identifier is None
        or report.benchmark_info.prompt_template_sha256 is None
    ]
    if unidentified:
        raise ValueError(
            f"Report(s) without a recorded prompt identity: {', '.join(unidentified)}"
            "; re-run with the current framework"
        )
    _require_identical(prompts, "prompt identity")
    _check_dataset_hashes(reports, manifest, dataset_root)
    _check_freshness(reports, manifest)


def _condition_failure(item_id: str, view: _ConditionView) -> str | None:
    """Return why one item fails to survive in one condition's report."""

    if item_id in view.filtered:
        return _FILTERED_BY_TOKEN_LIMIT
    prediction: PredictionRecord | None = view.predictions.get(item_id)
    if prediction is None:
        return _MISSING_FROM_REPORT
    if not prediction.is_success:
        return _INFERENCE_FAILED
    if prediction.inference_data.binary_label_confidence is None:
        return _UNSCORED
    return None


def _first_failure(
    pair_id: str, views: Mapping[Condition, _ConditionView]
) -> str | None:
    """Return the highest-priority report-based reason a pair fails to survive.

    Checks ``<pair_id>_vuln`` and ``<pair_id>_safe`` against all five
    conditions' reports and returns the failure reason with the highest
    priority among ``filtered_by_token_limit``, ``inference_failed`` and
    ``unscored`` and ``missing_from_report``. Scanner-artifact completeness
    (``missing_scanner_artifacts``) is checked separately by
    :func:`_scanner_artifacts_complete`, only once a pair passes this
    report-based check.

    Args:
        pair_id: Pair identifier.
        views: Indexed reports keyed by condition.

    Returns:
        The highest-priority failure reason, or ``None`` if the pair
        survives in all five conditions' reports.
    """

    reasons: set[str] = {
        reason
        for item_id in (f"{pair_id}_vuln", f"{pair_id}_safe")
        for view in views.values()
        if (reason := _condition_failure(item_id, view)) is not None
    }
    return next((reason for reason in _FAILURE_PRIORITY if reason in reasons), None)


def _scanner_artifacts_complete(
    pair_id: str,
    stats_index: Mapping[str, frozenset[tuple[str, Condition]]],
    strata: Mapping[str, str],
) -> bool:
    """Whether a pair's two items have a complete set of scanner artifacts.

    Args:
        pair_id: Pair identifier.
        stats_index: Pair id to the set of ``(item_id, condition)`` pairs
            present for it in ``item_stats.jsonl``.
        strata: Item id to stratum, from ``overlap/strata.jsonl``.

    Returns:
        True if both ``<pair_id>_vuln`` and ``<pair_id>_safe`` have an
        ``item_stats.jsonl`` row for every :class:`Condition` and a
        ``strata.jsonl`` row.
    """

    item_ids: tuple[str, str] = (f"{pair_id}_vuln", f"{pair_id}_safe")
    present: frozenset[tuple[str, Condition]] = stats_index.get(pair_id, frozenset())
    has_all_stats_rows: bool = all(
        (item_id, condition) in present
        for item_id in item_ids
        for condition in Condition
    )
    has_all_strata_rows: bool = all(item_id in strata for item_id in item_ids)
    return has_all_stats_rows and has_all_strata_rows


def _predicted_label_int(prediction: PredictionRecord) -> int:
    """Return a prediction's ``predicted_label`` coerced to int.

    Args:
        prediction: The prediction record to read ``predicted_label`` from.

    Returns:
        The predicted label as an int.

    Raises:
        ValueError: If ``predicted_label`` cannot be converted to int.
    """

    try:
        return int(prediction.predicted_label)
    except ValueError as exc:
        raise ValueError(
            f"Cannot convert predicted_label to int for sample "
            f"{prediction.sample_id!r}: {prediction.predicted_label!r}"
        ) from exc


def _suffix_label(item_id: str) -> int:
    """Return the label implied by an item id's ``_vuln``/``_safe`` suffix.

    Raises:
        ValueError: If the id has neither suffix.
    """

    label: int | None = next(
        (value for suffix, value in _SUFFIX_LABELS.items() if item_id.endswith(suffix)),
        None,
    )
    if label is None:
        raise ValueError(f"Item id {item_id!r} ends with neither _vuln nor _safe")
    return label


def _check_labels(row: _ItemStatsRow, prediction: PredictionRecord | None) -> None:
    """Require item_stats, report and id-suffix labels to agree.

    Raises:
        ValueError: Naming the item and each label source on any mismatch.
    """

    expected: int = _suffix_label(row.item_id)
    labels: dict[str, object] = {"item_stats": row.label}
    if prediction is not None:
        labels["report true_label"] = prediction.true_label
    mismatched: dict[str, object] = {
        source: value
        for source, value in labels.items()
        if str(value).strip() != str(expected)
    }
    if mismatched:
        raise ValueError(
            f"Label mismatch for item {row.item_id!r} (condition "
            f"{row.condition.value}): suffix implies {expected}, got {mismatched}"
        )


def _merge(
    row: _ItemStatsRow, prediction: PredictionRecord | None, stratum: str
) -> ItemRecord:
    """Combine one item_stats row with its prediction and stratum into an ItemRecord.

    Args:
        row: One parsed row from ``item_stats.jsonl``.
        prediction: The matching prediction record from this row's
            condition's report, when the item is present there.
        stratum: This item's pair's recall stratum from ``strata.jsonl``.

    Returns:
        The combined, frozen :class:`ItemRecord`.

    Raises:
        ValueError: If ``prediction.predicted_label`` cannot be converted
            to int, or the item_stats label, the report's ``true_label`` and
            the ``_vuln``/``_safe`` id suffix disagree.
    """

    _check_labels(row, prediction)
    score: float | None = None
    prompt_tokens: int | None = None
    scored_draws: int | None = None
    if prediction is not None:
        score = prediction.inference_data.binary_label_confidence
        prompt_tokens = prediction.inference_data.prompt_tokens
        scored_draws = sum(
            value is not None
            for value in prediction.inference_data.p_vulnerable_per_draw
        )
    return ItemRecord(
        item_id=row.item_id,
        pair_id=row.pair_id,
        condition=row.condition,
        label=row.label,
        repo_url=row.repo_url,
        score=score,
        predicted_label=_predicted_label_int(prediction)
        if prediction is not None
        else None,
        success=prediction.is_success if prediction is not None else False,
        prompt_tokens=prompt_tokens,
        token_count=row.token_count,
        context_token_count=row.context_token_count,
        budget=row.budget,
        node_count=row.node_count,
        file_count=row.file_count,
        symbol_count=row.symbol_count,
        symbol_names_hash=row.symbol_names_hash,
        underfill=row.underfill,
        stratum=stratum,
        scored_draws=scored_draws,
    )


def load_inputs(
    results_dir: Path, scanner_out: Path, dataset_root: Path = FRAMEWORK_REPO_ROOT
) -> AnalysisInputs:
    """Load and align llm_scanner's per-item artifacts with the framework's reports.

    A pair is kept only if both its items are present, successful,
    unfiltered and scored in all five conditions, and both items have a
    complete set of scanner artifacts (an ``item_stats.jsonl`` row for
    every condition and a ``strata.jsonl`` row). Dropped pairs are counted
    in ``inference_exclusions`` under the first failing reason
    (``filtered_by_token_limit`` > ``inference_failed`` > ``unscored`` >
    ``missing_from_report`` > ``missing_scanner_artifacts``); report-based
    reasons take precedence over the scanner-artifact check.

    Args:
        results_dir: Root directory holding the five conditions'
            experiment result trees (searched recursively for report files).
        scanner_out: llm_scanner's ``output_dir`` for this run.
        dataset_root: Host directory report ``dataset_path`` values resolve
            against (see :func:`validate_reports`).

    Returns:
        The aligned :class:`AnalysisInputs`.

    Raises:
        ValueError: If the reports are mixed or stale
            (:func:`validate_reports`) or an item's labels disagree.
        RuntimeError: If the final item count does not equal
            ``2 * len(pair_ids) * len(Condition)``, indicating a bug in
            pair-survival gating or duplicate/malformed scanner artifacts.
    """

    reports: dict[Condition, BenchmarkReport] = load_reports(results_dir)
    manifest: dict[str, object] = _read_json(scanner_out / _MANIFEST_FILENAME)
    validate_reports(reports, manifest, dataset_root)
    stats_rows: list[_ItemStatsRow] = [
        _ItemStatsRow.model_validate(row)
        for row in _read_jsonl(scanner_out / _ITEM_STATS_FILENAME)
    ]
    strata_dir, strata_file = _STRATA_PATH
    strata: dict[str, str] = {
        str(row["item_id"]): str(row["stratum"])
        for row in _read_jsonl(scanner_out / strata_dir / strata_file)
    }
    coverage: dict[str, object] = _read_json(scanner_out / _COVERAGE_FILENAME)

    views: dict[Condition, _ConditionView] = {
        condition: _ConditionView(
            filtered=frozenset(report.filtered_sample_ids),
            predictions={
                prediction.sample_id: prediction for prediction in report.predictions
            },
        )
        for condition, report in reports.items()
    }

    stats_by_pair: defaultdict[str, set[tuple[str, Condition]]] = defaultdict(set)
    for row in stats_rows:
        stats_by_pair[row.pair_id].add((row.item_id, row.condition))
    stats_index: dict[str, frozenset[tuple[str, Condition]]] = {
        pair_id: frozenset(entries) for pair_id, entries in stats_by_pair.items()
    }

    pair_ids: list[str] = sorted({row.pair_id for row in stats_rows})
    exclusions: Counter[str] = Counter()
    surviving_pairs: list[str] = []
    for pair_id in pair_ids:
        reason: str | None = _first_failure(pair_id, views)
        if reason is None and not _scanner_artifacts_complete(
            pair_id, stats_index, strata
        ):
            reason = _MISSING_SCANNER_ARTIFACTS
        if reason is None:
            surviving_pairs.append(pair_id)
        else:
            exclusions[reason] += 1

    surviving: frozenset[str] = frozenset(surviving_pairs)
    items: list[ItemRecord] = [
        _merge(
            row,
            views[row.condition].predictions.get(row.item_id),
            strata[row.item_id],
        )
        for row in stats_rows
        if row.pair_id in surviving
    ]

    if len(items) != 2 * len(surviving_pairs) * len(Condition):
        raise RuntimeError(
            "Aligned item count mismatch: "
            f"len(items)={len(items)} != 2 * len(pair_ids)={len(surviving_pairs)} "
            f"* len(Condition)={len(Condition)}"
        )

    draws_per_item: int = next(
        iter(reports.values())
    ).benchmark_info.model.self_consistency_samples
    return AnalysisInputs(
        items=items,
        pair_ids=surviving_pairs,
        coverage=coverage,
        manifest=manifest,
        inference_exclusions=dict(exclusions),
        draws_per_item=draws_per_item,
    )
