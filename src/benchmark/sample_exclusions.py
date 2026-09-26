"""Exclude audited samples (for example wrong labels) from a run by a stable key.

Sample ids are regenerated whenever a dataset is rebuilt, so samples are matched
on ``(metadata.source_row_ids, label)`` instead.
"""

from pydantic import BaseModel

from benchmark.models import BenchmarkSample

SampleKey = tuple[tuple[int, ...], int]


class SampleExclusion(BaseModel):
    """One sample to drop, as written in a dataset config's ``exclude_samples``."""

    source_row_ids: list[int]
    label: int
    reason: str = ""

    @property
    def key(self) -> SampleKey:
        return tuple(self.source_row_ids), self.label


class ExclusionResult(BaseModel):
    """Samples kept, exclusions that matched, and exclusions that matched nothing."""

    kept: list[BenchmarkSample]
    excluded: list[SampleExclusion]
    unmatched: list[SampleExclusion]


def sample_key(sample: BenchmarkSample) -> SampleKey:
    """Return the ``(source_row_ids, label)`` key of a sample.

    Raises:
        ValueError: If the sample has no ``metadata.source_row_ids``.
    """
    row_ids: list[int] | None = sample.metadata.get("source_row_ids")
    if not row_ids:
        raise ValueError(
            f"Sample {sample.id} has no metadata.source_row_ids; sample exclusions need it"
        )
    return tuple(int(row_id) for row_id in row_ids), int(sample.label)


def apply_sample_exclusions(
    samples: list[BenchmarkSample], exclusions: list[SampleExclusion]
) -> ExclusionResult:
    """Drop samples whose key matches an exclusion; report unmatched exclusions."""
    if not exclusions:
        return ExclusionResult(kept=list(samples), excluded=[], unmatched=[])
    excluded_keys: set[SampleKey] = {exclusion.key for exclusion in exclusions}
    matched: set[SampleKey] = set()
    kept: list[BenchmarkSample] = []
    for sample in samples:
        key: SampleKey = sample_key(sample)
        if key in excluded_keys:
            matched.add(key)
        else:
            kept.append(sample)
    return ExclusionResult(
        kept=kept,
        excluded=[exclusion for exclusion in exclusions if exclusion.key in matched],
        unmatched=[exclusion for exclusion in exclusions if exclusion.key not in matched],
    )
