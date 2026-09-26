"""Acceptance checks for one reference-context oracle run (design spec §11).

Each function computes exactly one row of ``acceptance.json``. Any
:class:`CheckResult` with ``passed is False`` marks the whole run headed
**RUN INVALID**; a check that cannot be validated (a condition with no
items, too few repositories, a single-label condition, a missing manifest
key) also fails rather than being silently skipped or vacuously passing,
and never raises. ``leakage_probe`` and ``sanity_check`` always evaluate
every :class:`Condition`, not just the ones present in the input, so a
run that is missing a whole condition's data fails loudly instead of
passing on an empty ``per_condition`` table.
"""

from collections import defaultdict
from collections.abc import Sequence
from statistics import median
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import precision_recall_curve, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold, cross_val_predict  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from analysis.reference_context.bootstrap import ClusterBootstrap, ConfidenceInterval
from analysis.reference_context.inputs import AnalysisInputs, Condition, ItemRecord
from analysis.reference_context.metrics import MetricName
from analysis.reference_context.run_config import Pins, ReferenceRunConfig
from analysis.reference_context.scoring import FloatArray, IntArray, score_arrays

_LEAKAGE_AUC_THRESHOLD: Final[float] = 0.55
_SYMMETRY_TOKEN_RATIO_THRESHOLD: Final[float] = 0.05
_MAX_LISTED_VIOLATORS: Final[int] = 50
_SANITY_ABS_TOL: Final[float] = 1e-12
_MAX_GROUP_FOLDS: Final[int] = 5
_MIN_REPOS_FOR_PROBE: Final[int] = 2
_NO_ITEMS_NOTE: Final[str] = "no items for this condition"
_SYMMETRY_CONDITIONS: Final[tuple[Condition, ...]] = (
    Condition.REFERENCE,
    Condition.RANDOM,
    Condition.MISMATCHED,
)


class CheckResult(BaseModel):
    """One row of ``acceptance.json``.

    Attributes:
        name: Check identifier, e.g. ``"leakage_probe"``.
        passed: Whether the check's threshold is met. ``False`` for any
            check marks the whole run **RUN INVALID**.
        value: The quantity the threshold is applied to, or ``None`` when
            it could not be computed (never ``NaN``).
        threshold: Human-readable statement of the pass condition.
        detail: JSON-safe supporting detail (per-condition values, violator
            ids, notes explaining why a check could not be validated).
    """

    name: str
    passed: bool
    value: float | None
    threshold: str
    detail: dict[str, object]


def _by_condition(items: Sequence[ItemRecord]) -> dict[Condition, list[ItemRecord]]:
    """Group items by condition, keeping only conditions present in ``items``."""

    grouped: defaultdict[Condition, list[ItemRecord]] = defaultdict(list)
    for item in items:
        grouped[item.condition].append(item)
    return dict(grouped)


def _probe_features(items: Sequence[ItemRecord]) -> FloatArray:
    """Build the leakage probe's feature matrix, one row per item.

    Args:
        items: Items of a single condition.

    Returns:
        An ``(n_items, 4)`` array of
        ``[token_count, node_count, file_count, symbol_count]``.
    """

    return np.array(
        [
            [
                float(item.token_count),
                float(item.node_count),
                float(item.file_count),
                float(item.symbol_count),
            ]
            for item in items
        ],
        dtype=np.float64,
    )


def _length_ranking_stats(
    items: Sequence[ItemRecord],
) -> tuple[float | None, float | None]:
    """Non-gating diagnostic: does raw ``token_count`` alone rank vuln above safe?

    A within-pair length difference (e.g. an off-by-a-few-tokens artifact of
    how a condition is rendered) does not necessarily show up in
    :func:`_condition_leakage_auc`, because that fits an unpaired classifier
    over absolute feature values across every repository. This diagnostic
    checks the paired signal directly; it never affects ``passed``.

    Args:
        items: One condition's items (any subset of pairs, any labels).

    Returns:
        ``(length_ranking_accuracy, mean_token_delta)`` over the pairs with
        both halves present in ``items``: the fraction where
        ``vuln.token_count > safe.token_count`` (an exact tie counts 0.5),
        and the mean of ``vuln.token_count - safe.token_count``. Both are
        ``None`` when no pair has both halves.
    """

    halves: defaultdict[str, dict[int, ItemRecord]] = defaultdict(dict)
    for item in items:
        halves[item.pair_id][item.label] = item

    deltas: list[float] = []
    wins: float = 0.0
    for pair_halves in halves.values():
        vuln: ItemRecord | None = pair_halves.get(1)
        safe: ItemRecord | None = pair_halves.get(0)
        if vuln is None or safe is None:
            continue
        delta: float = float(vuln.token_count - safe.token_count)
        deltas.append(delta)
        if delta > 0:
            wins += 1.0
        elif delta == 0:
            wins += 0.5
    if not deltas:
        return None, None
    return wins / len(deltas), sum(deltas) / len(deltas)


def _condition_leakage_auc(
    items: Sequence[ItemRecord], seed: int
) -> tuple[float | None, str | None]:
    """Cross-validated leakage AUC for one condition's items.

    Args:
        items: This condition's items (both labels, all repositories); may
            be empty.
        seed: Seed for the pinned logistic regression.

    Returns:
        A ``(auc, note)`` pair. ``note`` is set, and ``auc`` is ``None``,
        when there are no items, fewer than two repositories to
        group-fold over, or only one label is present (any of these would
        crash ``GroupKFold``/``cross_val_predict`` or make the AUC
        meaningless). ``auc`` is ``0.5`` without fitting when every
        feature is exactly constant across all items.
    """

    if not items:
        return None, _NO_ITEMS_NOTE

    repos: list[str] = [item.repo_url for item in items]
    n_repos: int = len(set(repos))
    if n_repos < _MIN_REPOS_FOR_PROBE:
        return None, f"cannot validate leakage: only {n_repos} repo(s)"

    if len({item.label for item in items}) < 2:
        return None, "cannot validate leakage: single label present"

    features: FloatArray = _probe_features(items)
    if bool(np.all(np.ptp(features, axis=0) == 0)):
        return 0.5, None

    labels: IntArray = np.array([item.label for item in items], dtype=np.int64)
    groups: npt.NDArray[np.str_] = np.array(repos)
    pipeline: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    cv: GroupKFold = GroupKFold(n_splits=min(_MAX_GROUP_FOLDS, n_repos))
    probabilities = cross_val_predict(
        pipeline, features, labels, groups=groups, cv=cv, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, probabilities)), None


def leakage_probe(inputs: AnalysisInputs, seed: int) -> CheckResult:
    """Detect whether context-size features alone predict the label.

    For every :class:`Condition` (not only the ones present in ``inputs``),
    fits ``Pipeline(StandardScaler(), LogisticRegression(...))`` on
    ``[token_count, node_count, file_count, symbol_count]`` under
    ``GroupKFold`` grouped by ``repo_url`` and scores the out-of-fold
    predictions by ROC-AUC.

    Args:
        inputs: Aligned analysis inputs.
        seed: Seed for the pinned logistic regression.

    Returns:
        ``value`` is the max AUC over conditions that could be computed
        (``None`` if none could be). ``passed`` requires every condition to
        be both computable and ``<= 0.55``; a condition that is missing,
        has fewer than two repositories, or has a single label fails the
        check instead of being skipped. ``detail["per_condition"]`` holds
        every condition's AUC (``None`` where it could not be validated),
        ``detail["notes"]`` explains any such condition, and
        ``detail["length_ranking_accuracy"]``/``detail["mean_token_delta"]``
        report the non-gating paired-length diagnostic per condition.
    """

    per_condition: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    length_ranking_accuracy: dict[str, float | None] = {}
    mean_token_delta: dict[str, float | None] = {}
    passed = True
    by_condition = _by_condition(inputs.items)
    for condition in Condition:
        items: list[ItemRecord] = by_condition.get(condition, [])
        ranking, delta = _length_ranking_stats(items)
        length_ranking_accuracy[condition.value] = ranking
        mean_token_delta[condition.value] = delta

        auc, note = _condition_leakage_auc(items, seed)
        per_condition[condition.value] = auc
        if note is not None:
            notes[condition.value] = note
            passed = False
        elif auc is None or auc > _LEAKAGE_AUC_THRESHOLD:
            passed = False
    computed: list[float] = [auc for auc in per_condition.values() if auc is not None]
    value: float | None = max(computed) if computed else None
    detail: dict[str, object] = {
        "per_condition": per_condition,
        "length_ranking_accuracy": length_ranking_accuracy,
        "mean_token_delta": mean_token_delta,
    }
    if notes:
        detail["notes"] = notes
    return CheckResult(
        name="leakage_probe",
        passed=passed,
        value=value,
        threshold=f"AUC <= {_LEAKAGE_AUC_THRESHOLD} for every condition",
        detail=detail,
    )


def mismatched_control(bootstrap: ClusterBootstrap, tolerance: float) -> CheckResult:
    """Confirm the mismatched-donor control is indistinguishable from no context.

    Args:
        bootstrap: Cluster bootstrap covering the ``MISMATCHED`` and
            ``NONE`` conditions.
        tolerance: Maximum acceptable ``|ROC-AUC(mismatched) - ROC-AUC(none)|``
            when the confidence interval does not contain zero.

    Returns:
        ``value`` is the point difference (``None`` if undefined).
        ``passed`` is true when the interval contains 0 or the point
        difference is within ``tolerance``.

    Raises:
        KeyError: If ``bootstrap`` lacks the ``MISMATCHED`` or ``NONE``
            condition.
    """

    diff: ConfidenceInterval = bootstrap.difference(
        Condition.MISMATCHED, Condition.NONE, MetricName.ROC_AUC
    )
    threshold: str = (
        f"CI contains 0, or |ROC-AUC(mismatched) - ROC-AUC(none)| <= {tolerance}"
    )
    interval_detail: dict[str, object] = {
        "point": diff.point,
        "low": diff.low,
        "high": diff.high,
        "dropped": diff.dropped,
    }
    if diff.point is None:
        return CheckResult(
            name="mismatched_control",
            passed=False,
            value=None,
            threshold=threshold,
            detail={
                "interval": interval_detail,
                "note": "ROC-AUC difference is undefined for MISMATCHED vs NONE",
            },
        )
    ci_contains_zero: bool = (
        diff.low is not None and diff.high is not None and diff.low <= 0.0 <= diff.high
    )
    within_tolerance: bool = abs(diff.point) <= tolerance
    return CheckResult(
        name="mismatched_control",
        passed=ci_contains_zero or within_tolerance,
        value=diff.point,
        threshold=threshold,
        detail={
            "interval": interval_detail,
            "ci_contains_zero": ci_contains_zero,
            "within_tolerance": within_tolerance,
        },
    )


def _symmetric(vuln: ItemRecord, safe: ItemRecord) -> tuple[bool, str | None]:
    """Whether one pair's vulnerable/fixed halves are symmetric in one condition.

    The token-ratio half of the check is measured on ``context_token_count``
    (the rendered non-target context only), not ``token_count``: every
    condition contains the target function(s), whose length legitimately
    changes with the fix, so comparing full ``token_count`` would penalize
    that expected difference instead of an asymmetry in the surrounding
    context (design decision 2026-09-24).

    Args:
        vuln: The pair's vulnerable (parent) half.
        safe: The pair's fixed half.

    Returns:
        ``(symmetric, note)``. ``note`` explains a failure that is not a
        simple ratio overshoot (a missing ``context_token_count``), else
        ``None``.
    """

    if vuln.symbol_names_hash != safe.symbol_names_hash:
        return False, None
    if vuln.context_token_count is None or safe.context_token_count is None:
        return False, "missing context_token_count"
    denom: int = max(vuln.context_token_count, safe.context_token_count)
    if denom == 0:
        return True, None
    ratio: float = abs(vuln.context_token_count - safe.context_token_count) / denom
    return ratio <= _SYMMETRY_TOKEN_RATIO_THRESHOLD, None


def symmetry_check(inputs: AnalysisInputs) -> CheckResult:
    """Check that reference/random/mismatched contexts are pair-symmetric.

    For each pair and each of ``{REFERENCE, RANDOM, MISMATCHED}``, the
    vulnerable and fixed halves must both be present, share
    ``symbol_names_hash``, and have a relative deviation of at most 0.05 in
    ``context_token_count`` — the rendered non-target context tokens, which
    excludes the target function(s) since their length legitimately changes
    with the fix. A pair missing either half in a covered condition, or
    either half's ``context_token_count``, is a violator, the same as a pair
    whose halves are present but asymmetric — incomplete data is never
    silently skipped or passed vacuously.

    Args:
        inputs: Aligned analysis inputs.

    Returns:
        ``value`` is the fraction of ``pair_ids x {REFERENCE, RANDOM,
        MISMATCHED}`` combinations that are symmetric (``None`` if
        ``inputs`` has no pairs). ``passed`` requires ``value == 1.0``.
        ``detail["violators"]`` lists the offending pair ids, each listed
        once. ``detail["notes"]`` explains non-ratio failures (e.g. a
        missing ``context_token_count``), keyed by ``"<pair_id>:<condition>"``.
    """

    halves_by_key: defaultdict[tuple[str, Condition], dict[int, ItemRecord]] = (
        defaultdict(dict)
    )
    for item in inputs.items:
        if item.condition in _SYMMETRY_CONDITIONS:
            halves_by_key[(item.pair_id, item.condition)][item.label] = item

    total = 0
    passed_count = 0
    violators: list[str] = []
    seen_violators: set[str] = set()
    notes: dict[str, str] = {}
    for pair_id in inputs.pair_ids:
        for condition in _SYMMETRY_CONDITIONS:
            total += 1
            halves = halves_by_key.get((pair_id, condition))
            if halves is not None and 0 in halves and 1 in halves:
                symmetric, note = _symmetric(halves[1], halves[0])
            else:
                symmetric, note = False, "missing half"
            if symmetric:
                passed_count += 1
                continue
            if pair_id not in seen_violators:
                violators.append(pair_id)
                seen_violators.add(pair_id)
            if note is not None:
                notes[f"{pair_id}:{condition.value}"] = note

    value: float | None = passed_count / total if total else None
    detail: dict[str, object] = {"violators": violators, "n_checked": total}
    if notes:
        detail["notes"] = notes
    if total == 0:
        detail["note"] = "no pairs to check"
    return CheckResult(
        name="symmetry_check",
        passed=value == 1.0,
        value=value,
        threshold="100% of pairs: both halves present, equal symbol_names_hash, "
        "context-token ratio <= 0.05",
        detail=detail,
    )


class _BudgetItem(NamedTuple):
    """One REFERENCE item's budget comparison against its matching CPG item."""

    item_id: str
    violates: bool
    shortfall: float


def _budget_item(item: ItemRecord, cpg_token_count: int) -> _BudgetItem:
    """Compare one REFERENCE item with its (nonzero-token) CPG item.

    The item violates the cap iff it is longer than the cpg item **and** holds
    non-target context (``context_token_count > 0``): a targets-only
    reference cannot be shortened. A missing ``context_token_count`` is
    treated as non-empty context, so an overshoot is never vacuously excused.
    """

    overshoots: bool = item.token_count > cpg_token_count
    has_context: bool = item.context_token_count is None or item.context_token_count > 0
    shortfall: float = max(0, cpg_token_count - item.token_count) / cpg_token_count
    return _BudgetItem(item.item_id, overshoots and has_context, shortfall)


def budget_check(inputs: AnalysisInputs) -> CheckResult:
    """Check that reference contexts never exceed the cpg condition's token count.

    The cpg item's ``token_count`` is a cap, not a target (decision
    2026-09-26): a REFERENCE item **violates** iff
    ``reference.token_count > cpg.token_count`` and
    ``reference.context_token_count > 0`` (an overshoot by a targets-only
    reference cannot be shortened and is not a violation; a missing
    ``context_token_count`` counts as context). Underfill never fails.

    Args:
        inputs: Aligned analysis inputs.

    Returns:
        ``value`` is the fraction of REFERENCE items with a matching,
        nonzero-token CPG item that do not violate (``None`` if none
        matched). ``passed`` requires ``value == 1.0`` **and** every
        REFERENCE item to have a matching, nonzero-token CPG item; unmatched
        or zero-token items fail the check with a detail note. ``detail``
        reports the median relative shortfall ``max(0, cpg - ref) / cpg``,
        the ``underfill_rate`` of REFERENCE items, ``n_violations`` and up to
        ``_MAX_LISTED_VIOLATORS`` violating item ids.
    """

    by_condition = _by_condition(inputs.items)
    cpg_tokens: dict[str, int] = {
        item.item_id: item.token_count for item in by_condition.get(Condition.CPG, [])
    }
    reference_items: list[ItemRecord] = by_condition.get(Condition.REFERENCE, [])
    compared: list[_BudgetItem] = [
        _budget_item(item, cpg_tokens[item.item_id])
        for item in reference_items
        if cpg_tokens.get(item.item_id)
    ]
    unmatched_or_zero: int = len(reference_items) - len(compared)
    violators: list[str] = sorted(entry.item_id for entry in compared if entry.violates)
    value: float | None = (
        (len(compared) - len(violators)) / len(compared) if compared else None
    )
    underfill_rate: float | None = (
        sum(item.underfill for item in reference_items) / len(reference_items)
        if reference_items
        else None
    )
    detail: dict[str, object] = {
        "median_relative_shortfall": (
            median(entry.shortfall for entry in compared) if compared else None
        ),
        "underfill_rate": underfill_rate,
        "n_matched_items": len(compared),
        "n_unmatched_or_zero_cpg": unmatched_or_zero,
        "n_violations": len(violators),
        "violators": violators[:_MAX_LISTED_VIOLATORS],
    }
    has_unmatched: bool = unmatched_or_zero > 0
    if has_unmatched:
        detail["note"] = (
            f"{unmatched_or_zero} REFERENCE item(s) lack a matching, nonzero-token "
            "CPG item"
        )
    elif not compared:
        detail["note"] = "no item_id matched between REFERENCE and CPG"
    return CheckResult(
        name="budget_check",
        passed=value == 1.0 and not has_unmatched,
        value=value,
        threshold="no REFERENCE item with context exceeds its CPG item's tokens, "
        "every REFERENCE item matched",
        detail=detail,
    )


def _sanity_endpoint(items: Sequence[ItemRecord]) -> tuple[float | None, float | None]:
    """Return ``(endpoint_precision, base_rate)`` for one condition, or ``(None, None)``."""

    arrays = score_arrays(items)
    if not arrays.has_both_classes:
        return None, None
    precisions, _, _ = precision_recall_curve(arrays.score_labels, arrays.scores)
    return float(precisions[0]), float(arrays.score_labels.mean())


def sanity_check(inputs: AnalysisInputs) -> CheckResult:
    """Confirm the PR-curve endpoint equals the base rate, per condition.

    sklearn's ``precision_recall_curve`` places the lowest-threshold point
    (every item predicted positive) first, whose precision must equal the
    base rate exactly, up to floating-point error.

    Args:
        inputs: Aligned analysis inputs.

    Returns:
        ``value`` is the largest absolute deviation across conditions that
        could be checked (``None`` if none could be). ``passed`` requires
        every :class:`Condition` (not only the ones present in ``inputs``)
        to be checkable and within ``1e-12``; a missing or single-label
        condition fails the check instead of being skipped.
    """

    by_condition = _by_condition(inputs.items)
    per_condition: dict[str, dict[str, float | None]] = {}
    notes: dict[str, str] = {}
    deviations: list[float] = []
    passed = True
    for condition in Condition:
        items: list[ItemRecord] = by_condition.get(condition, [])
        if not items:
            per_condition[condition.value] = {
                "endpoint_precision": None,
                "base_rate": None,
            }
            notes[condition.value] = _NO_ITEMS_NOTE
            passed = False
            continue
        endpoint, rate = _sanity_endpoint(items)
        per_condition[condition.value] = {
            "endpoint_precision": endpoint,
            "base_rate": rate,
        }
        if endpoint is None or rate is None:
            notes[condition.value] = "cannot validate: fewer than two classes scored"
            passed = False
            continue
        deviation = abs(endpoint - rate)
        deviations.append(deviation)
        if deviation > _SANITY_ABS_TOL:
            passed = False
    value: float | None = max(deviations) if deviations else None
    detail: dict[str, object] = {"per_condition": per_condition}
    if notes:
        detail["notes"] = notes
    return CheckResult(
        name="sanity_check",
        passed=passed,
        value=value,
        threshold=f"PR-curve endpoint precision == base rate (abs tol {_SANITY_ABS_TOL})",
        detail=detail,
    )


_REQUIRED_MANIFEST_KEYS: Final[tuple[str, str, str]] = (
    "llm_scanner_git_sha",
    "llm_scanner_dirty",
    "config_sha256",
)


def reproducibility_check(
    manifest: dict[str, object],
    pins: Pins,
    config_sha256: str,
    framework_sha: str,
    framework_dirty: bool,
) -> CheckResult:
    """Confirm the run's manifest matches the pinned config and both trees are clean.

    Args:
        manifest: llm_scanner's ``run_manifest.json``, already parsed.
        pins: The pinned llm_scanner checkout for this run.
        config_sha256: sha256 of the config file actually used for this run.
        framework_sha: Git SHA of the framework repository at run time.
        framework_dirty: Whether the framework working tree has uncommitted
            changes.

    Returns:
        ``passed`` requires ``manifest["llm_scanner_git_sha"] ==
        pins.llm_scanner_git_sha``, ``manifest["llm_scanner_dirty"] is
        False``, ``manifest["config_sha256"] == config_sha256`` and
        ``framework_dirty is False``. A missing manifest key fails the
        check with a detail note instead of raising. ``framework_sha`` is
        always recorded in ``detail``.
    """

    threshold = "manifest SHAs/hashes match pins; framework tree clean"
    detail: dict[str, object] = {"framework_sha": framework_sha}
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        detail["note"] = f"manifest missing key(s): {', '.join(missing)}"
        return CheckResult(
            name="reproducibility_check",
            passed=False,
            value=None,
            threshold=threshold,
            detail=detail,
        )

    sha_matches: bool = manifest["llm_scanner_git_sha"] == pins.llm_scanner_git_sha
    llm_scanner_clean: bool = manifest["llm_scanner_dirty"] is False
    config_matches: bool = manifest["config_sha256"] == config_sha256
    framework_clean: bool = framework_dirty is False
    passed: bool = (
        sha_matches and llm_scanner_clean and config_matches and framework_clean
    )
    detail.update(
        {
            "llm_scanner_git_sha_matches": sha_matches,
            "llm_scanner_clean": llm_scanner_clean,
            "config_sha256_matches": config_matches,
            "framework_clean": framework_clean,
        }
    )
    return CheckResult(
        name="reproducibility_check",
        passed=passed,
        value=float(passed),
        threshold=threshold,
        detail=detail,
    )


def run_all(
    inputs: AnalysisInputs,
    bootstrap: ClusterBootstrap,
    config: ReferenceRunConfig,
    config_sha256: str,
    framework_sha: str,
    framework_dirty: bool,
) -> list[CheckResult]:
    """Run every acceptance check for one reference-context oracle run.

    Args:
        inputs: Aligned analysis inputs.
        bootstrap: Cluster bootstrap covering at least the ``MISMATCHED``
            and ``NONE`` conditions.
        config: Pinned run configuration (probe seed, mismatch tolerance,
            pins).
        config_sha256: sha256 of the config file actually used for this run.
        framework_sha: Git SHA of the framework repository at run time.
        framework_dirty: Whether the framework working tree has uncommitted
            changes.

    Returns:
        One :class:`CheckResult` per check, in design spec §11 order
        (leakage probe, mismatched control, symmetry, budget, sanity,
        reproducibility). The run is **RUN INVALID** if any result's
        ``passed`` is ``False``.
    """

    return [
        leakage_probe(inputs, config.framework.probe_seed),
        mismatched_control(bootstrap, config.framework.mismatch_tolerance),
        symmetry_check(inputs),
        budget_check(inputs),
        sanity_check(inputs),
        reproducibility_check(
            inputs.manifest, config.pins, config_sha256, framework_sha, framework_dirty
        ),
    ]
