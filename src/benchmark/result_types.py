from typing import Any

from pydantic import BaseModel, ConfigDict

from benchmark.models import PredictionResult


class MetricsResult(BaseModel):
    """Standardized metrics result returned by calculators."""

    task_type: str
    accuracy: float
    summary: dict[str, float | int | str | None]
    details: dict[str, Any]


class PredictionRecord(BaseModel):
    """Serializable prediction record for reporting."""

    sample_id: str
    predicted_label: int | str
    true_label: int | str
    confidence: float | None
    response_text: str
    processing_time: float
    tokens_used: int | None


class BenchmarkInfo(BaseModel):
    """Standardized benchmark metadata for reports."""

    experiment_name: str | None
    model_name: str
    model_type: str
    task_type: str
    dataset_path: str
    description: str
    cwe_type: str | None
    batch_size: int
    max_tokens: int
    temperature: float
    use_quantization: bool
    is_thinking_enabled: bool
    total_samples: int
    total_time_seconds: float
    avg_time_per_sample: float
    tokens_used_total: int | None
    tokens_used_avg: float | None
    processing_time_stats: dict[str, float | int]
    tokens_used_stats: dict[str, float | int] | None
    extra_metadata: dict[str, Any]
    timestamp: str


class BenchmarkReport(BaseModel):
    """Standardized benchmark report format."""

    benchmark_info: BenchmarkInfo
    metrics: MetricsResult
    predictions: list[PredictionRecord]


class ResultArtifacts(BaseModel):
    """Saved artifact paths for a benchmark run."""

    report_json: str
    metrics_json: str
    predictions_csv: str
    predictions_json: str


class BenchmarkRunResult(BaseModel):
    """Raw run data returned by benchmark runners before saving."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: MetricsResult
    predictions: list[PredictionResult]
    total_samples: int
    total_time: float
