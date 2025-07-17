from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkSample:
    """Data structure for a single benchmark sample."""

    id: str
    code: str
    label: int | str
    metadata: dict[str, Any]
    cwe_types: list[str] | None = None
    severity: str | None = None


@dataclass
class PredictionResult:
    """Data structure for model prediction results."""

    sample_id: str
    predicted_label: int | str
    true_label: int | str
    confidence: float | None
    response_text: str
    processing_time: float
    tokens_used: int | None = None
