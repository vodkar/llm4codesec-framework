"""Exclude audited samples (for example wrong labels) from a run by a stable key.

Sample ids are regenerated whenever a dataset is rebuilt, so samples are matched
on ``(metadata.source_row_ids, label)`` instead. Datasets mixing CleanVul score
files (e.g. score 3 + 4) reuse row ids across files, so an exclusion may also name
``metadata.source_file``; without it, it matches the key in any file.
"""

from pydantic import BaseModel

from benchmark.models import BenchmarkSample

SampleKey = tuple[tuple[int, ...], int]


class SampleExclusion(BaseModel):
    """One sample to drop, as written in a dataset config's ``exclude_samples``."""

    source_row_ids: list[int]
    label: int
    source_file: str | None = None
    """CleanVul CSV the row ids index into (``metadata.source_file``); None matches any file."""
    reason: str = ""

    @property
    def key(self) -> SampleKey:
        return tuple(self.source_row_ids), self.label

    def matches(self, source_file: str | None, key: SampleKey) -> bool:
        """Whether this exclusion applies to a sample from ``source_file`` with ``key``."""
        return self.key == key and self.source_file in (None, source_file)


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


def sample_source_file(sample: BenchmarkSample) -> str | None:
    """Return the CleanVul CSV a sample's row ids index into, when the dataset records it."""
    source_file: object = sample.metadata.get("source_file")
    return str(source_file) if source_file is not None else None


def apply_sample_exclusions(
    samples: list[BenchmarkSample], exclusions: list[SampleExclusion]
) -> ExclusionResult:
    """Drop samples whose key matches an exclusion; report unmatched exclusions."""
    if not exclusions:
        return ExclusionResult(kept=list(samples), excluded=[], unmatched=[])
    matched: set[int] = set()
    kept: list[BenchmarkSample] = []
    for sample in samples:
        key: SampleKey = sample_key(sample)
        source_file: str | None = sample_source_file(sample)
        hits: list[int] = [
            index
            for index, exclusion in enumerate(exclusions)
            if exclusion.matches(source_file, key)
        ]
        if hits:
            matched.update(hits)
        else:
            kept.append(sample)
    return ExclusionResult(
        kept=kept,
        excluded=[exclusion for index, exclusion in enumerate(exclusions) if index in matched],
        unmatched=[
            exclusion for index, exclusion in enumerate(exclusions) if index not in matched
        ],
    )
