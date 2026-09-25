from datetime import datetime
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from benchmark.enums import BinaryDecisionMode
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
    binary_label_confidence: float | None = None
    """Mean final-answer-position P(VULNERABLE) across draws when enabled."""
    answer_probability: float | None = None
    """Probability of the final predicted label, from the mean P(VULNERABLE) across draws."""
    answer_probabilities: list[float | None] = Field(default_factory=list)
    """Probability of each draw's own answered label, aligned with responses."""
    stated_confidence: float | None = None
    """Model-stated confidence in the final predicted label, scaled to [0, 1]."""
    stated_confidences: list[float | None] = Field(default_factory=list)
    """Stated confidence of each draw in its own verdict, aligned with responses."""
    self_validation_probability: float | None = None
    """Self-validation P(correct) for the final predicted label."""
    self_validation_probabilities: list[float | None] = Field(default_factory=list)
    """Self-validation P(correct) of each draw's own verdict, aligned with responses."""
    prompt_text: str | None = None
    """Realized formatted prompt text actually sent to the model, when available."""
    prompt_tokens: int | None = None
    """Realized prompt token count, when available."""
    p_vulnerable_per_draw: list[float | None] = Field(default_factory=list)
    """Per-draw P(VULNERABLE) from binary_label_confidence, aligned with responses."""


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
    context_length: int
    max_output_tokens: int
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    use_quantization: bool
    is_thinking_enabled: bool
    self_consistency_samples: int
    enable_logprobs: bool
    binary_decision_mode: BinaryDecisionMode
    binary_logprob_threshold: float | None = None
    confidence_methods: list[str] = Field(default_factory=list)
    """Enabled optional confidence methods; 'stated_confidence' changes the response contract."""
    sampling_seed: int | None = None
    """Global pinned sampling seed (per-draw vLLM seeds derive from it); None when unpinned."""


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
    binary_label_confidence_stats: dict[str, float] | None = None
    """Per-run final-answer-position P(VULNERABLE) summary when enabled and available."""


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
    prompt_identifier: str | None = None
    """Prompt config key (slug) the run used; None in reports predating this field."""
    prompt_template_sha256: str | None = None
    """sha256 of the system and user prompt templates; None in older reports."""
    timestamp_utc: str | None = None
    """Timezone-aware UTC ISO timestamp of report creation; None in older reports."""


class ShortExperimentReport(BaseModel):
    """Short summary report for quick reference."""

    benchmark_info: BenchmarkInfo
    metrics: MetricsResult
    is_success: bool


class BenchmarkReport(ShortExperimentReport):
    """Standardized benchmark report format."""

    predictions: list[PredictionRecord]
    filtered_sample_ids: list[str] = Field(default_factory=list)
    """IDs of samples dropped by the runner's token-limit filter before inference."""

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
    filtered_sample_ids: list[str] = Field(default_factory=list)
    """IDs of samples dropped by the runner's token-limit filter before inference."""


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
