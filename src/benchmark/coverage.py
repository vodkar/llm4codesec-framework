"""Coverage levels for selective prediction (metrics on the most confident samples)."""

from collections.abc import Sequence

DEFAULT_COVERAGE_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

COVERAGE_METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "fpr",
    "fnr",
)


def validate_coverage_levels(levels: Sequence[float]) -> tuple[float, ...]:
    """Return levels as floats.

    Raises:
        ValueError: If no level is given or a level lies outside (0, 1].
    """
    values: tuple[float, ...] = tuple(float(level) for level in levels)
    if not values:
        raise ValueError("coverage_levels must not be empty")
    invalid: list[float] = [value for value in values if not 0.0 < value <= 1.0]
    if invalid:
        raise ValueError(f"coverage_levels must lie in (0, 1]: {invalid}")
    return values
