"""Study tables: per-condition and per-stratum metrics, lengths, decomposition.

Point metrics live in :mod:`analysis.reference_context.scoring` and the
repository bootstrap in :mod:`analysis.reference_context.bootstrap`; both are
re-exported here so callers import one module.
"""

from collections.abc import Sequence
from typing import Final, TypeAlias

import numpy as np

from analysis.reference_context.bootstrap import ClusterBootstrap, ConfidenceInterval
from analysis.reference_context.inputs import AnalysisInputs, Condition, ItemRecord
from analysis.reference_context.scoring import MetricName, point_metrics

__all__ = [
    "AUC_TYPE_METRICS",
    "STRATA",
    "ClusterBootstrap",
    "ConditionTable",
    "ConfidenceInterval",
    "MetricName",
    "base_rate",
    "condition_table",
    "decomposition",
    "length_stats",
    "point_metrics",
    "stratum_tables",
    "stratum_tables_with_redraws",
]

ConditionTable: TypeAlias = dict[Condition, dict[MetricName, ConfidenceInterval]]

STRATA: Final[tuple[str, ...]] = ("no_reference", "r0", "r0_50", "r50_100", "r100")
"""Recall strata in reporting order (llm_scanner's ``RecallStratum`` values)."""

AUC_TYPE_METRICS: Final[tuple[MetricName, ...]] = (
    MetricName.ROC_AUC,
    MetricName.PR_AUC,
    MetricName.PAIRED_RANKING_ACCURACY,
)
"""Metrics whose ceiling is 1.0, used by the retrieval/reasoning decomposition."""

_MIN_BOOTSTRAP_REPOS: Final[int] = 2
_QUARTILES: Final[tuple[float, float, float]] = (25.0, 50.0, 75.0)
_RETRIEVAL_LOSS: Final[str] = "retrieval_loss"
_REASONING_LOSS: Final[str] = "reasoning_loss"
_NO_INTERVAL: Final[ConfidenceInterval] = ConfidenceInterval(None, None, None)


def _conditions_in(items: Sequence[ItemRecord]) -> tuple[Condition, ...]:
    present: frozenset[Condition] = frozenset(item.condition for item in items)
    return tuple(condition for condition in Condition if condition in present)


def _items_by_condition(
    items: Sequence[ItemRecord],
) -> dict[Condition, list[ItemRecord]]:
    return {
        condition: [item for item in items if item.condition is condition]
        for condition in _conditions_in(items)
    }


def condition_table(
    inputs: AnalysisInputs, bootstrap: ClusterBootstrap
) -> ConditionTable:
    """Every metric with its bootstrap interval, for each condition in the inputs.

    Args:
        inputs: Aligned analysis inputs; their conditions select the rows.
        bootstrap: Bootstrap built over the same items.

    Returns:
        Condition (in :class:`Condition` order) to metric (in
        :class:`MetricName` order) to :class:`ConfidenceInterval`.

    Raises:
        ValueError: If a condition in ``inputs`` is missing from ``bootstrap``.
    """

    conditions: tuple[Condition, ...] = _conditions_in(inputs.items)
    missing: list[str] = [
        condition.value
        for condition in conditions
        if condition not in bootstrap.conditions
    ]
    if missing:
        raise ValueError(f"Bootstrap lacks condition(s): {', '.join(missing)}")
    return {
        condition: {
            metric: bootstrap.interval(condition, metric) for metric in MetricName
        }
        for condition in conditions
    }


def _full_shape(table: ConditionTable) -> ConditionTable:
    """Add every missing condition as an all-``None`` row, in :class:`Condition` order."""

    return {
        condition: table.get(condition, dict.fromkeys(MetricName, _NO_INTERVAL))
        for condition in Condition
    }


def _point_only_table(
    by_condition: dict[Condition, list[ItemRecord]],
) -> ConditionTable:
    """Point estimates without intervals, for strata too small to bootstrap."""

    return {
        condition: {
            metric: ConfidenceInterval(value, None, None)
            for metric, value in point_metrics(items).items()
        }
        for condition, items in by_condition.items()
    }


def _can_bootstrap(by_condition: dict[Condition, list[ItemRecord]]) -> bool:
    """Whether a stratum has enough repositories and both classes everywhere."""

    repos: frozenset[str] = frozenset(
        item.repo_url for items in by_condition.values() for item in items
    )
    both_classes: bool = all(
        {item.label for item in items if item.score is not None} == {0, 1}
        for items in by_condition.values()
    )
    return len(repos) >= _MIN_BOOTSTRAP_REPOS and both_classes


def _stratum_table(
    items: list[ItemRecord], resamples: int, seed: int
) -> tuple[ConditionTable, int | None]:
    """One stratum's table from its own bootstrap, degrading to ``None`` values.

    Returns:
        The table and its bootstrap's discarded-draw count (``None`` when the
        stratum was too small to bootstrap).
    """

    if not items:
        return _full_shape({}), None
    by_condition: dict[Condition, list[ItemRecord]] = _items_by_condition(items)
    if not _can_bootstrap(by_condition):
        return _full_shape(_point_only_table(by_condition)), None
    bootstrap: ClusterBootstrap = ClusterBootstrap(by_condition, resamples, seed)
    table: ConditionTable = _full_shape(
        {
            condition: {
                metric: bootstrap.interval(condition, metric) for metric in MetricName
            }
            for condition in bootstrap.conditions
        }
    )
    return table, bootstrap.redraws


def stratum_tables_with_redraws(
    inputs: AnalysisInputs, resamples: int, seed: int
) -> tuple[dict[str, ConditionTable], dict[str, int | None]]:
    """Per-stratum tables (see :func:`stratum_tables`) plus bootstrap redraws.

    Args:
        inputs: Aligned analysis inputs.
        resamples: Number of bootstrap resamples per stratum.
        seed: Seed for each stratum's bootstrap.

    Returns:
        Stratum to its condition table, and stratum to the number of draws
        its bootstrap discarded (``None`` when the stratum was not
        bootstrapped), both in :data:`STRATA` order.

    Raises:
        ValueError: If an item carries a stratum outside :data:`STRATA`.
    """

    unknown: set[str] = {item.stratum for item in inputs.items} - set(STRATA)
    if unknown:
        raise ValueError(f"Unknown stratum value(s): {', '.join(sorted(unknown))}")
    results: dict[str, tuple[ConditionTable, int | None]] = {
        stratum: _stratum_table(
            [item for item in inputs.items if item.stratum == stratum],
            resamples,
            seed,
        )
        for stratum in STRATA
    }
    return (
        {stratum: table for stratum, (table, _) in results.items()},
        {stratum: redraws for stratum, (_, redraws) in results.items()},
    )


def stratum_tables(
    inputs: AnalysisInputs, resamples: int, seed: int
) -> dict[str, ConditionTable]:
    """One metrics table per recall stratum, each from its own cluster bootstrap.

    Each stratum's bootstrap resamples only that stratum's repositories, with
    the same ``seed``. Every table has a row for every :class:`Condition`
    and every :class:`MetricName`. A stratum with no items, or a condition
    absent from a stratum, yields ``None`` everywhere; a stratum with fewer
    than two repositories (or a condition lacking a class) yields point
    estimates with ``None`` bounds.

    Args:
        inputs: Aligned analysis inputs.
        resamples: Number of bootstrap resamples per stratum.
        seed: Seed for each stratum's bootstrap.

    Returns:
        Stratum (in :data:`STRATA` order) to its condition table.

    Raises:
        ValueError: If an item carries a stratum outside :data:`STRATA`.
    """

    tables, _ = stratum_tables_with_redraws(inputs, resamples, seed)
    return tables


def _summary(prefix: str, values: Sequence[int]) -> dict[str, float | None]:
    """Mean, median and quartiles (numpy linear interpolation); ``None`` when empty."""

    keys: tuple[str, ...] = tuple(
        f"{prefix}_{name}" for name in ("mean", "median", "q1", "q3")
    )
    if not values:
        return dict.fromkeys(keys, None)
    q1, median, q3 = np.percentile(values, _QUARTILES)
    stats: tuple[float | None, ...] = (
        float(np.mean(values)),
        float(median),
        float(q1),
        float(q3),
    )
    return dict(zip(keys, stats, strict=True))


def length_stats(
    inputs: AnalysisInputs,
) -> dict[Condition, dict[str, float | None]]:
    """Realized prompt and context lengths per condition.

    Args:
        inputs: Aligned analysis inputs.

    Returns:
        Condition to ``prompt_tokens_{mean,median,q1,q3}`` (over items with a
        recorded prompt length; ``None`` if none, so the result is strict-JSON safe) and
        ``context_tokens_{mean,median,q1,q3}`` (context ``token_count``).
    """

    return {
        condition: _summary(
            "prompt_tokens",
            [item.prompt_tokens for item in items if item.prompt_tokens is not None],
        )
        | _summary("context_tokens", [item.token_count for item in items])
        for condition, items in _items_by_condition(inputs.items).items()
    }


def base_rate(items: Sequence[ItemRecord]) -> float:
    """Share of positive (vulnerable) items.

    Args:
        items: Items to summarise.

    Returns:
        Fraction of items with ``label == 1``.

    Raises:
        ValueError: If ``items`` is empty.
    """

    if not items:
        raise ValueError("Base rate is undefined for zero items")
    return sum(item.label for item in items) / len(items)


def _one_minus(interval: ConfidenceInterval) -> ConfidenceInterval:
    """``1 - interval``; the bounds swap because subtraction reverses order."""

    def flip(value: float | None) -> float | None:
        return None if value is None else 1.0 - value

    return ConfidenceInterval(
        flip(interval.point), flip(interval.high), flip(interval.low), interval.dropped
    )


def decomposition(
    bootstrap: ClusterBootstrap,
) -> dict[MetricName, dict[str, ConfidenceInterval]]:
    """Split the gap to a perfect score into retrieval and reasoning loss.

    For each AUC-type metric (ceiling 1.0): ``retrieval_loss`` is the paired
    difference ``metric(reference) - metric(cpg)`` and ``reasoning_loss`` is
    ``1 - metric(reference)``.

    Args:
        bootstrap: Bootstrap covering the reference and cpg conditions.

    Returns:
        Metric (in :data:`AUC_TYPE_METRICS` order) to
        ``{"retrieval_loss": ..., "reasoning_loss": ...}``.

    Raises:
        KeyError: If the bootstrap lacks the reference or cpg condition.
    """

    return {
        metric: {
            _RETRIEVAL_LOSS: bootstrap.difference(
                Condition.REFERENCE, Condition.CPG, metric
            ),
            _REASONING_LOSS: _one_minus(
                bootstrap.interval(Condition.REFERENCE, metric)
            ),
        }
        for metric in AUC_TYPE_METRICS
    }
