"""Point estimates of the reference-context study's discrimination metrics.

Every metric mirrors :class:`benchmark.metrics_calculator.BinaryMetricsCalculator`
so study numbers agree with the framework's own reports on the same data:

- ROC-AUC (``roc_auc_score``) and PR-AUC (``average_precision_score``) on
  ``score`` (P(VULNERABLE));
- precision at recall ``r`` as ``max(precision[recall >= r])`` from
  ``precision_recall_curve``;
- MCC on the model's own ``predicted_label`` (no tuned threshold);
- paired ranking accuracy: per pre/post pair, 1 if the vulnerable half
  outscores the fixed half, 0.5 on an exact tie, 0 otherwise, averaged.

Metrics are computed from :class:`ScoreArrays`, a flat array form that can be
concatenated. The cluster bootstrap concatenates per-repository arrays, so a
repository drawn twice contributes its pairs twice as distinct pairs, which
keying pairs by ``pair_id`` could not express.
"""

from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, NamedTuple, Self, TypeAlias

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

from analysis.reference_context.inputs import ItemRecord

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


class MetricName(StrEnum):
    """Metrics reported per condition and per stratum, in reporting order."""

    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    PAIRED_RANKING_ACCURACY = "paired_ranking_accuracy"
    MCC = "mcc"
    P_AT_R_50 = "precision_at_recall_50"
    P_AT_R_80 = "precision_at_recall_80"
    P_AT_R_90 = "precision_at_recall_90"
    P_AT_R_95 = "precision_at_recall_95"


MetricValues: TypeAlias = dict[MetricName, float | None]

_PRECISION_AT_RECALL: Final[tuple[tuple[MetricName, float], ...]] = (
    (MetricName.P_AT_R_50, 0.5),
    (MetricName.P_AT_R_80, 0.8),
    (MetricName.P_AT_R_90, 0.9),
    (MetricName.P_AT_R_95, 0.95),
)
_TIE_CREDIT: Final[float] = 0.5


class ScoreArrays(NamedTuple):
    """Flat, concatenable inputs for :func:`metrics_from_arrays`.

    Attributes:
        score_labels: True labels of the items that carry a score.
        scores: P(VULNERABLE) of those items, aligned with ``score_labels``.
        verdict_labels: True labels of the items that carry a verdict.
        verdicts: The model's own predicted labels, aligned with ``verdict_labels``.
        margins: ``score_vuln - score_safe`` for each pair with both halves scored.
    """

    score_labels: IntArray
    scores: FloatArray
    verdict_labels: IntArray
    verdicts: IntArray
    margins: FloatArray

    @classmethod
    def concat(cls, parts: Sequence[Self]) -> Self:
        """Concatenate several arrays field by field, keeping duplicates.

        Args:
            parts: Arrays to join, in order; must not be empty.

        Returns:
            One :class:`ScoreArrays` holding every part's entries.

        Raises:
            ValueError: If ``parts`` is empty.
        """

        if not parts:
            raise ValueError("Cannot concatenate zero ScoreArrays")
        return cls(
            score_labels=np.concatenate([part.score_labels for part in parts]),
            scores=np.concatenate([part.scores for part in parts]),
            verdict_labels=np.concatenate([part.verdict_labels for part in parts]),
            verdicts=np.concatenate([part.verdicts for part in parts]),
            margins=np.concatenate([part.margins for part in parts]),
        )

    @property
    def has_both_classes(self) -> bool:
        """Whether the scored items include both a positive and a negative."""

        return bool(np.any(self.score_labels == 1) and np.any(self.score_labels == 0))


def _pair_margins(items: Sequence[ItemRecord]) -> list[float]:
    """Return ``score_vuln - score_safe`` for every pair with both halves scored.

    Args:
        items: Items with unique ``item_id`` values.

    Returns:
        One margin per complete, fully scored pair, in first-seen pair order.

    Raises:
        ValueError: If a pair has two items with the same label.
    """

    halves: defaultdict[str, dict[int, ItemRecord]] = defaultdict(dict)
    for item in items:
        if item.label in halves[item.pair_id]:
            raise ValueError(
                f"Pair {item.pair_id!r} has more than one item with label {item.label}"
            )
        halves[item.pair_id][item.label] = item
    return [
        vulnerable.score - fixed.score
        for pair in halves.values()
        if (vulnerable := pair.get(1)) is not None
        and (fixed := pair.get(0)) is not None
        and vulnerable.score is not None
        and fixed.score is not None
    ]


def score_arrays(items: Sequence[ItemRecord]) -> ScoreArrays:
    """Flatten items into :class:`ScoreArrays`, pairing halves by ``pair_id``.

    Items without a score are left out of the ranking inputs and items
    without a verdict are left out of MCC, as the framework does.

    Args:
        items: Items of one condition with unique ``item_id`` values.

    Returns:
        The flattened arrays.

    Raises:
        ValueError: If ``item_id`` values repeat or a pair has two items
            with the same label.
    """

    item_ids: list[str] = [item.item_id for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Items must have unique item_id values")
    scored: list[ItemRecord] = [item for item in items if item.score is not None]
    judged: list[ItemRecord] = [
        item for item in items if item.predicted_label is not None
    ]
    return ScoreArrays(
        score_labels=np.array([item.label for item in scored], dtype=np.int64),
        scores=np.array([item.score for item in scored], dtype=np.float64),
        verdict_labels=np.array([item.label for item in judged], dtype=np.int64),
        verdicts=np.array([item.predicted_label for item in judged], dtype=np.int64),
        margins=np.array(_pair_margins(items), dtype=np.float64),
    )


def _ranking_metrics(arrays: ScoreArrays) -> MetricValues:
    """ROC-AUC, PR-AUC and precision-at-recall; all ``None`` for a single class."""

    names: tuple[MetricName, ...] = (
        MetricName.ROC_AUC,
        MetricName.PR_AUC,
        *(name for name, _ in _PRECISION_AT_RECALL),
    )
    if not arrays.has_both_classes:
        return dict.fromkeys(names, None)
    precisions, recalls, _ = precision_recall_curve(arrays.score_labels, arrays.scores)
    values: MetricValues = {
        MetricName.ROC_AUC: float(roc_auc_score(arrays.score_labels, arrays.scores)),
        MetricName.PR_AUC: float(
            average_precision_score(arrays.score_labels, arrays.scores)
        ),
    }
    values.update(
        {
            name: float(precisions[recalls >= level].max())
            for name, level in _PRECISION_AT_RECALL
        }
    )
    return values


def _mcc(arrays: ScoreArrays) -> float | None:
    """MCC on the model's verdicts; ``None`` unless both true classes are present."""

    if np.unique(arrays.verdict_labels).size < 2:
        return None
    return float(matthews_corrcoef(arrays.verdict_labels, arrays.verdicts))


def _paired_ranking_accuracy(margins: FloatArray) -> float | None:
    """Share of pairs ranked correctly, a tie counting half (as the framework does)."""

    if margins.size == 0:
        return None
    wins: int = int(np.count_nonzero(margins > 0))
    ties: int = int(np.count_nonzero(margins == 0))
    return (wins + _TIE_CREDIT * ties) / int(margins.size)


def metrics_from_arrays(arrays: ScoreArrays) -> MetricValues:
    """Compute every :class:`MetricName` from flattened arrays.

    Args:
        arrays: Flattened scores, verdicts and pair margins.

    Returns:
        Every metric in :class:`MetricName` order; a metric is ``None`` when
        it is undefined on these arrays (single class, or no complete pairs).
    """

    values: MetricValues = _ranking_metrics(arrays)
    values[MetricName.MCC] = _mcc(arrays)
    values[MetricName.PAIRED_RANKING_ACCURACY] = _paired_ranking_accuracy(
        arrays.margins
    )
    return {name: values[name] for name in MetricName}


def point_metrics(items: Sequence[ItemRecord]) -> MetricValues:
    """Compute every :class:`MetricName` on one condition's items.

    Args:
        items: Items of one condition with unique ``item_id`` values.

    Returns:
        Every metric in :class:`MetricName` order; ``None`` where undefined.

    Raises:
        ValueError: If ``item_id`` values repeat or a pair has two items
            with the same label.
    """

    return metrics_from_arrays(score_arrays(items))
