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


class SampleInferenceData(BaseModel):
    """Raw inference data for a benchmark sample across all self-consistency draws."""

    responses: list[str]
    """All N raw response texts. Single-element list when self_consistency_samples=1."""
    vote_counts: dict[str, int]
    """Maps label string to number of votes; empty when self_consistency_samples=1."""
    tokens_used: int
    """Total token count across all N draws (prompt + generated)."""
    processing_time: float
    """Mean inference duration across all N draws in seconds."""
    confidence: float | None
    """Geometric-mean per-token probability averaged over N draws; None if not enabled."""


class PredictionRecord(BaseModel):
    """Serializable prediction record for reporting."""

    sample_id: str
    predicted_label: int | str
    true_label: int | str
    is_success: bool
    error_message: str | None
    inference_data: SampleInferenceData


class ModelRunConfig(BaseModel):
    """Model configuration metadata captured at run time."""

    model_name: str
    model_type: str
    backend: str
    max_output_tokens: int
    temperature: float
    use_quantization: bool
    is_thinking_enabled: bool
    self_consistency_samples: int
    enable_logprobs: bool


class RunStats(BaseModel):
    """Runtime and resource statistics for a benchmark run."""

    total_samples: int
    total_time_seconds: float
    avg_time_per_sample: float
    tokens_used_total: int
    tokens_used_avg: float
    processing_time_stats: dict[str, float | int]
    tokens_used_stats: dict[str, float | int]
    confidence_stats: dict[str, float] | None
    """Per-run confidence summary; None when enable_logprobs=False."""


class BenchmarkInfo(BaseModel):
    """Standardized benchmark metadata for reports."""

    experiment_name: str | None
    task_type: str
    dataset_path: str
    description: str
    cwe_type: str | None
    batch_size: int
    timestamp: str
    model: ModelRunConfig
    stats: RunStats
    extra_metadata: dict[str, Any]


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
