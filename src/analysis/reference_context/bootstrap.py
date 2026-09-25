"""Paired cluster bootstrap over repositories.

Items from one repository are correlated (shared code base, shared style,
often several commits), so resampling items or pairs independently would
understate the uncertainty. Each resample instead draws repositories with
replacement, and every pair of a drawn repository enters the resample
(a repository drawn ``k`` times contributes its pairs ``k`` times).

The repository draws are made once from a single seeded generator and reused
for every condition, which makes between-condition differences paired.
A draw in which some condition lacks a positive or a negative item is
discarded and redrawn from the same generator; the number of discarded
draws is reported as :attr:`ClusterBootstrap.redraws`.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple

import numpy as np

from analysis.reference_context.inputs import Condition, ItemRecord
from analysis.reference_context.scoring import (
    IntArray,
    MetricName,
    MetricValues,
    ScoreArrays,
    metrics_from_arrays,
    score_arrays,
)

_PERCENTILES: Final[tuple[float, float]] = (2.5, 97.5)
_MAX_DRAWS_PER_RESAMPLE: Final[int] = 1000


class ConfidenceInterval(NamedTuple):
    """A point estimate with a 95% percentile bootstrap interval.

    Any bound is ``None`` when it is undefined (for example, a metric that
    needs both classes, or an interval with no valid bootstrap replicates).
    ``dropped`` counts the bootstrap replicates left out of the percentiles
    because the metric was undefined (``None``) on that resample.
    """

    point: float | None
    low: float | None
    high: float | None
    dropped: int = 0


def _percentile_interval(
    point: float | None, replicates: Sequence[float | None]
) -> ConfidenceInterval:
    """Wrap a point estimate with the 2.5/97.5 percentiles of its non-``None`` replicates."""

    values: list[float] = [value for value in replicates if value is not None]
    dropped: int = len(replicates) - len(values)
    if not values:
        return ConfidenceInterval(point, None, None, dropped)
    low, high = np.percentile(values, _PERCENTILES)
    return ConfidenceInterval(point, float(low), float(high), dropped)


def _repo_of_each_pair(items: Sequence[ItemRecord]) -> dict[str, str]:
    """Map each pair to its repository, rejecting pairs split across repositories."""

    repo_by_pair: dict[str, str] = {}
    for item in items:
        known: str = repo_by_pair.setdefault(item.pair_id, item.repo_url)
        if known != item.repo_url:
            raise ValueError(
                f"Pair {item.pair_id!r} spans repositories {known!r} and "
                f"{item.repo_url!r}"
            )
    return repo_by_pair


def _arrays_by_repo(items: Sequence[ItemRecord]) -> dict[str, ScoreArrays]:
    """Group one condition's items by repository and flatten each group."""

    _repo_of_each_pair(items)
    grouped: defaultdict[str, list[ItemRecord]] = defaultdict(list)
    for item in items:
        grouped[item.repo_url].append(item)
    return {repo: score_arrays(repo_items) for repo, repo_items in grouped.items()}


class ClusterBootstrap:
    """Paired, repository-level bootstrap of every :class:`MetricName`.

    Metric replicates are computed lazily, once per condition, and cached.

    Attributes:
        redraws: Number of draws discarded because some condition's resample
            lacked a positive or a negative item.
    """

    def __init__(
        self,
        items_by_condition: Mapping[Condition, Sequence[ItemRecord]],
        resamples: int,
        seed: int,
    ) -> None:
        """Draw the shared repository resamples.

        Args:
            items_by_condition: Each condition's items; every condition must
                cover the same set of repositories.
            resamples: Number of bootstrap resamples ``B``.
            seed: Seed for ``np.random.default_rng``.

        Raises:
            ValueError: If ``resamples`` is not positive, no condition is
                given, conditions cover different repositories, a pair spans
                repositories, or some condition lacks one of the classes
                (no valid resample could ever be drawn).
            RuntimeError: If a single resample needs more than
                ``_MAX_DRAWS_PER_RESAMPLE`` draws to become valid.
        """

        if resamples < 1:
            raise ValueError(f"resamples must be positive, got {resamples}")
        if not items_by_condition:
            raise ValueError("At least one condition is required")
        self.__items: dict[Condition, tuple[ItemRecord, ...]] = {
            condition: tuple(items) for condition, items in items_by_condition.items()
        }
        self.__by_repo: dict[Condition, dict[str, ScoreArrays]] = {
            condition: _arrays_by_repo(items)
            for condition, items in self.__items.items()
        }
        self.__repos: tuple[str, ...] = self.__shared_repos()
        self.__check_both_classes()
        self.redraws: int = 0
        self.__draws: tuple[IntArray, ...] = self.__draw(resamples, seed)
        self.__point: dict[Condition, MetricValues] = {}
        self.__replicates: dict[Condition, tuple[MetricValues, ...]] = {}

    def __shared_repos(self) -> tuple[str, ...]:
        """Return the sorted repository ids, identical across all conditions."""

        repo_sets: dict[Condition, frozenset[str]] = {
            condition: frozenset(by_repo)
            for condition, by_repo in self.__by_repo.items()
        }
        reference: frozenset[str] = next(iter(repo_sets.values()))
        mismatched: list[str] = [
            condition.value
            for condition, repos in repo_sets.items()
            if repos != reference
        ]
        if mismatched:
            raise ValueError(
                "Conditions must cover the same repositories for a paired "
                f"bootstrap; differing: {', '.join(mismatched)}"
            )
        if not reference:
            raise ValueError("No items to bootstrap")
        return tuple(sorted(reference))

    def __check_both_classes(self) -> None:
        """Fail fast when a condition can never yield a two-class resample."""

        single_class: list[str] = [
            condition.value
            for condition, by_repo in self.__by_repo.items()
            if not ScoreArrays.concat(list(by_repo.values())).has_both_classes
        ]
        if single_class:
            raise ValueError(
                "Scored items lack a positive or a negative in condition(s): "
                f"{', '.join(single_class)}"
            )

    def __is_valid(self, draw: IntArray) -> bool:
        """Whether every condition's resample has both classes among scored items."""

        return all(
            ScoreArrays.concat(
                [by_repo[self.__repos[index]] for index in draw]
            ).has_both_classes
            for by_repo in self.__by_repo.values()
        )

    def __draw(self, resamples: int, seed: int) -> tuple[IntArray, ...]:
        """Draw ``resamples`` valid repository index lists from one generator."""

        rng: np.random.Generator = np.random.default_rng(seed)
        size: int = len(self.__repos)
        draws: list[IntArray] = []
        while len(draws) < resamples:
            for _ in range(_MAX_DRAWS_PER_RESAMPLE):
                draw: IntArray = rng.choice(size, size=size, replace=True)
                if self.__is_valid(draw):
                    draws.append(draw)
                    break
                self.redraws += 1
            else:
                raise RuntimeError(
                    f"No two-class resample in {_MAX_DRAWS_PER_RESAMPLE} draws "
                    f"(resample {len(draws)}, {size} repositories)"
                )
        return tuple(draws)

    @property
    def conditions(self) -> tuple[Condition, ...]:
        """Conditions covered by this bootstrap, in :class:`Condition` order."""

        return tuple(condition for condition in Condition if condition in self.__items)

    @property
    def repo_draws(self) -> tuple[tuple[str, ...], ...]:
        """The repository ids of each resample, duplicates kept, in draw order."""

        return tuple(
            tuple(self.__repos[index] for index in draw) for draw in self.__draws
        )

    def __require(self, condition: Condition) -> None:
        if condition not in self.__items:
            raise KeyError(f"Condition {condition.value!r} is not in this bootstrap")

    def __point_values(self, condition: Condition) -> MetricValues:
        self.__require(condition)
        if condition not in self.__point:
            self.__point[condition] = metrics_from_arrays(
                score_arrays(self.__items[condition])
            )
        return self.__point[condition]

    def __replicate_values(self, condition: Condition) -> tuple[MetricValues, ...]:
        self.__require(condition)
        if condition not in self.__replicates:
            by_repo: dict[str, ScoreArrays] = self.__by_repo[condition]
            self.__replicates[condition] = tuple(
                metrics_from_arrays(
                    ScoreArrays.concat([by_repo[self.__repos[index]] for index in draw])
                )
                for draw in self.__draws
            )
        return self.__replicates[condition]

    def replicates(
        self, condition: Condition, metric: MetricName
    ) -> tuple[float | None, ...]:
        """Return one metric's value on every resample of one condition.

        Args:
            condition: Condition to evaluate.
            metric: Metric to read.

        Returns:
            One value per resample, in draw order; ``None`` where undefined.

        Raises:
            KeyError: If ``condition`` is not in this bootstrap.
        """

        return tuple(values[metric] for values in self.__replicate_values(condition))

    def interval(self, condition: Condition, metric: MetricName) -> ConfidenceInterval:
        """Point estimate and 95% percentile interval of one metric.

        Args:
            condition: Condition to evaluate.
            metric: Metric to summarise.

        Returns:
            The point estimate on the original items, the 2.5/97.5
            percentiles over the non-``None`` replicates, and the number of
            ``None`` replicates left out.

        Raises:
            KeyError: If ``condition`` is not in this bootstrap.
        """

        return _percentile_interval(
            self.__point_values(condition)[metric], self.replicates(condition, metric)
        )

    def difference(
        self, cond_a: Condition, cond_b: Condition, metric: MetricName
    ) -> ConfidenceInterval:
        """Paired difference ``metric(cond_a) - metric(cond_b)``.

        Each replicate subtracts the two conditions' values on the same
        repository resample.

        Args:
            cond_a: Minuend condition.
            cond_b: Subtrahend condition.
            metric: Metric to compare.

        Returns:
            The point difference, the 2.5/97.5 percentiles over resamples
            where both values are defined, and the number of resamples left
            out because either value was ``None``.

        Raises:
            KeyError: If either condition is not in this bootstrap.
        """

        point_a: float | None = self.__point_values(cond_a)[metric]
        point_b: float | None = self.__point_values(cond_b)[metric]
        point: float | None = (
            point_a - point_b if point_a is not None and point_b is not None else None
        )
        diffs: list[float | None] = [
            a - b if a is not None and b is not None else None
            for a, b in zip(
                self.replicates(cond_a, metric),
                self.replicates(cond_b, metric),
                strict=True,
            )
        ]
        return _percentile_interval(point, diffs)
