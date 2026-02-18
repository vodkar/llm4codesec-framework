from datetime import datetime
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

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
    tokens_used: int


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
    tokens_used_total: int
    tokens_used_avg: float
    processing_time_stats: dict[str, float | int]
    tokens_used_stats: dict[str, float | int]
    extra_metadata: dict[str, Any]
    timestamp: str


class ShortExperimentReport(BaseModel):
    """Short summary report for quick reference."""

    benchmark_info: BenchmarkInfo
    metrics: MetricsResult
    is_success: bool


class BenchmarkReport(ShortExperimentReport):
    """Standardized benchmark report format."""

    predictions: list[PredictionRecord]

    @property
    def short_summary(self) -> ShortExperimentReport:
        """Generate a short summary report from the full benchmark report."""
        return ShortExperimentReport(
            benchmark_info=self.benchmark_info,
            metrics=self.metrics,
            is_success=self.is_success,
        )


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


class ExperimentPlanSummary(BaseModel):
    """Summary of benchmark results for quick reference."""

    total_experiments: int
    successful_experiments: int
    failed_experiments: int

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_experiments == 0:
            return 0.0
        return (self.successful_experiments / self.total_experiments) * 100


class ExperimentPlanResult(BaseModel):
    """Result of an entire experiment plan execution."""

    plan_name: str
    description: str
    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: datetime | None
    experiments: list[ShortExperimentReport]
    summary: ExperimentPlanSummary
    output_dir: str


class DescribeResult(Protocol):
    """Protocol describing the result from scipy.stats.describe."""

    @property
    def nobs(self) -> float: ...

    @property
    def minmax(
        self,
    ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]: ...

    @property
    def mean(self) -> float | NDArray[np.float64]: ...

    @property
    def variance(self) -> float | NDArray[np.float64]: ...
