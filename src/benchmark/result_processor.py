import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import stats

from benchmark.config import ExperimentConfig
from benchmark.models import PredictionResult
from benchmark.results import (
    BenchmarkInfo,
    BenchmarkReport,
    MetricsResult,
    ModelRunConfig,
    PredictionRecord,
    ResultArtifacts,
    RunStats,
    SampleInferenceData,
)

_LOGGER = logging.getLogger(__name__)


class BenchmarkResultProcessor(BaseModel):
    """Build and persist standardized benchmark results."""

    config: ExperimentConfig

    def build_report(
        self,
        metrics: MetricsResult,
        predictions: list[PredictionResult],
        total_time: float,
        total_samples: int,
    ) -> BenchmarkReport:
        """
        Build a standardized report with metadata, metrics, and predictions.
        """
        prediction_records: list[PredictionRecord] = [
            self._to_prediction_record(prediction) for prediction in predictions
        ]

        processing_stats: dict[str, float | int] = self._describe_processing_times(
            prediction_records
        )
        token_stats: dict[str, float | int] = self._describe_token_usage(
            prediction_records
        )
        confidence_stats: dict[str, float] | None = self._describe_confidence_scores(
            prediction_records
        )

        benchmark_info: BenchmarkInfo = self._build_benchmark_info(
            total_time=total_time,
            total_samples=total_samples,
            processing_stats=processing_stats,
            token_stats=token_stats,
            confidence_stats=confidence_stats,
        )

        report: BenchmarkReport = BenchmarkReport(
            benchmark_info=benchmark_info,
            metrics=metrics,
            predictions=prediction_records,
            is_success=all(prediction.is_success for prediction in predictions),
        )

        return report

    def save_report(self, report: BenchmarkReport) -> ResultArtifacts:
        """
        Persist report, metrics, and predictions to the output directory.
        """
        output_dir: Path = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_path: Path = output_dir / f"benchmark_report_{timestamp}.json"
        metrics_path: Path = output_dir / f"metrics_summary_{timestamp}.json"

        report_payload: dict[str, Any] = report.model_dump()
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                report_payload,
                report_file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        metrics_summary: dict[str, Any] = {
            "benchmark_info": report.benchmark_info.model_dump(),
            "metrics": report.metrics.model_dump(),
        }
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics_summary, metrics_file, indent=2, ensure_ascii=False)

        artifacts: ResultArtifacts = ResultArtifacts(
            report_json=str(report_path),
            metrics_json=str(metrics_path),
        )

        _LOGGER.info("Saved benchmark report to %s", report_path)
        return artifacts

    def build_and_save(
        self,
        metrics: MetricsResult,
        predictions: list[PredictionResult],
        total_time: float,
        total_samples: int,
    ) -> tuple[BenchmarkReport, ResultArtifacts]:
        """Build a standardized report and persist it to disk."""
        report: BenchmarkReport = self.build_report(
            metrics=metrics,
            predictions=predictions,
            total_time=total_time,
            total_samples=total_samples,
        )
        artifacts: ResultArtifacts = self.save_report(report)
        return report, artifacts

    def _to_prediction_record(self, prediction: PredictionResult) -> PredictionRecord:
        """Convert PredictionResult to a serializable PredictionRecord."""
        predicted_label: int | str = prediction.predicted_label
        true_label: int | str = prediction.true_label
        inference_data = SampleInferenceData(
            responses=prediction.all_responses if prediction.all_responses else [prediction.response_text],
            vote_counts=prediction.vote_counts,
            tokens_used=prediction.tokens_used or 0,
            processing_time=prediction.processing_time,
            confidence=prediction.confidence,
        )
        return PredictionRecord(
            sample_id=str(prediction.sample_id),
            predicted_label=predicted_label,
            true_label=true_label,
            is_success=prediction.is_success,
            error_message=prediction.error_message,
            inference_data=inference_data,
        )

    def _describe_processing_times(
        self, predictions: list[PredictionRecord]
    ) -> dict[str, float | int]:
        """Compute descriptive statistics for processing times."""
        times: list[float] = [p.inference_data.processing_time for p in predictions]
        if not times:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "variance": 0.0, "std": 0.0}

        time_array: NDArray[np.float64] = np.array(times, dtype=float)
        summary = stats.describe(time_array)

        return {
            "count": int(summary.nobs),
            "min": float(summary.minmax[0]),
            "max": float(summary.minmax[1]),
            "mean": float(summary.mean),
            "variance": float(summary.variance),
            "std": float(np.sqrt(summary.variance)),
        }

    def _describe_token_usage(
        self, predictions: list[PredictionRecord]
    ) -> dict[str, float | int]:
        """Compute descriptive statistics for token usage."""
        tokens: list[int] = [p.inference_data.tokens_used for p in predictions]
        if not tokens:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "variance": 0.0, "std": 0.0}

        token_array: NDArray[np.float64] = np.array(tokens, dtype=float)
        summary = stats.describe(token_array)

        return {
            "count": int(summary.nobs),
            "min": float(summary.minmax[0]),
            "max": float(summary.minmax[1]),
            "mean": float(summary.mean),
            "variance": float(summary.variance),
            "std": float(np.sqrt(summary.variance)),
        }

    def _describe_confidence_scores(
        self, predictions: list[PredictionRecord]
    ) -> dict[str, float] | None:
        """Compute descriptive statistics for confidence scores when available."""
        scores: list[float] = [
            p.inference_data.confidence
            for p in predictions
            if p.inference_data.confidence is not None
        ]
        if not scores:
            return None
        return {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": float(len(scores)),
        }

    def _build_benchmark_info(
        self,
        total_time: float,
        total_samples: int,
        processing_stats: dict[str, float | int],
        token_stats: dict[str, float | int],
        confidence_stats: dict[str, float] | None,
    ) -> BenchmarkInfo:
        """Construct benchmark metadata for the report."""
        avg_time_per_sample: float = (
            total_time / total_samples if total_samples > 0 else 0.0
        )
        tokens_used_total: int = int(token_stats.get("count", 0) * token_stats.get("mean", 0.0))
        tokens_used_avg: float = float(token_stats.get("mean", 0.0))

        extra_metadata: dict[str, Any] = {}
        if hasattr(self.config, "vulnerability_type"):
            extra_metadata["vulnerability_type"] = getattr(
                self.config, "vulnerability_type"
            )

        model_run_config = ModelRunConfig(
            model_name=self.config.model_name,
            model_type=self.config.model_type.value,
            backend=self.config.backend.value,
            context_length=int(self.config.model_context_length_tokens),
            max_output_tokens=int(self.config.max_output_tokens),
            temperature=float(self.config.temperature),
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            presence_penalty=self.config.presence_penalty,
            repetition_penalty=self.config.repetition_penalty,
            use_quantization=bool(self.config.use_quantization),
            is_thinking_enabled=bool(self.config.is_thinking_enabled),
            self_consistency_samples=int(self.config.self_consistency_samples),
            enable_logprobs=bool(self.config.enable_logprobs),
        )

        run_stats = RunStats(
            total_samples=int(total_samples),
            total_time_seconds=float(total_time),
            avg_time_per_sample=float(avg_time_per_sample),
            tokens_used_total=tokens_used_total,
            tokens_used_avg=tokens_used_avg,
            processing_time_stats=processing_stats,
            tokens_used_stats=token_stats,
            confidence_stats=confidence_stats,
        )

        return BenchmarkInfo(
            experiment_name=self.config.experiment_name,
            task_type=self.config.task_type.value,
            dataset_path=str(self.config.dataset_path),
            description=self.config.description,
            cwe_type=self.config.cwe_type,
            batch_size=int(self.config.batch_size),
            timestamp=datetime.now().isoformat(),
            model=model_run_config,
            stats=run_stats,
            extra_metadata=extra_metadata,
        )
