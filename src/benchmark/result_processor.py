import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import stats

from benchmark.config import ExperimentConfig
from benchmark.models import PredictionResult
from benchmark.results import (
    BenchmarkInfo,
    BenchmarkReport,
    DescribeResult,
    MetricsResult,
    PredictionRecord,
    ResultArtifacts,
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

        Args:
            metrics: Calculated metrics result
            predictions: Raw prediction results
            total_time: Total benchmark time in seconds
            total_samples: Number of evaluated samples

        Returns:
            BenchmarkReport: Standardized report payload
        """
        # SECTION: Prepare prediction records
        prediction_records: list[PredictionRecord] = [
            self._to_prediction_record(prediction) for prediction in predictions
        ]

        # SECTION: Aggregate runtime and token statistics
        processing_stats: dict[str, float | int] = self._describe_processing_times(
            prediction_records
        )
        token_stats: dict[str, float | int] | None = self._describe_token_usage(
            prediction_records
        )

        benchmark_info: BenchmarkInfo = self._build_benchmark_info(
            total_time=total_time,
            total_samples=total_samples,
            processing_stats=processing_stats,
            token_stats=token_stats,
        )

        report: BenchmarkReport = BenchmarkReport(
            benchmark_info=benchmark_info,
            metrics=metrics,
            predictions=prediction_records,
            # Only If all predictions are marked as success, we consider the overall benchmark a success
            is_success=all(prediction.is_success for prediction in predictions),
        )

        return report

    def save_report(self, report: BenchmarkReport) -> ResultArtifacts:
        """
        Persist report, metrics, and predictions to the output directory.

        Args:
            report: Standardized benchmark report payload

        Returns:
            ResultArtifacts: Paths of saved artifacts
        """
        # SECTION: Resolve output directory and filenames
        output_dir: Path = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_path: Path = output_dir / f"benchmark_report_{timestamp}.json"
        metrics_path: Path = output_dir / f"metrics_summary_{timestamp}.json"
        predictions_csv_path: Path = output_dir / f"predictions_{timestamp}.csv"
        predictions_json_path: Path = output_dir / f"predictions_{timestamp}.json"

        # SECTION: Serialize report payload
        report_payload: dict[str, Any] = report.model_dump()
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                report_payload,
                report_file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        # SECTION: Persist metrics summary
        metrics_summary: dict[str, Any] = {
            "benchmark_info": report.benchmark_info.model_dump(),
            "metrics": report.metrics.model_dump(),
        }
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics_summary, metrics_file, indent=2, ensure_ascii=False)

        # SECTION: Persist predictions as CSV and JSON
        predictions_payload: list[dict[str, Any]] = [
            prediction.model_dump() for prediction in report.predictions
        ]
        predictions_df: pd.DataFrame = pd.DataFrame(predictions_payload)
        predictions_df.to_csv(predictions_csv_path, index=False)
        with open(predictions_json_path, "w", encoding="utf-8") as predictions_file:
            json.dump(
                predictions_payload,
                predictions_file,
                indent=2,
                ensure_ascii=False,
            )

        artifacts: ResultArtifacts = ResultArtifacts(
            report_json=str(report_path),
            metrics_json=str(metrics_path),
            predictions_csv=str(predictions_csv_path),
            predictions_json=str(predictions_json_path),
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
        """
        Build a standardized report and persist it to disk.

        Args:
            metrics: Calculated metrics result
            predictions: Raw prediction results
            total_time: Total benchmark time in seconds
            total_samples: Number of evaluated samples

        Returns:
            tuple: Report payload and saved artifact paths
        """
        report: BenchmarkReport = self.build_report(
            metrics=metrics,
            predictions=predictions,
            total_time=total_time,
            total_samples=total_samples,
        )
        artifacts: ResultArtifacts = self.save_report(report)
        return report, artifacts

    def _to_prediction_record(self, prediction: PredictionResult) -> PredictionRecord:
        """Convert PredictionResult to a serializable prediction record."""
        prediction_dict: dict[str, Any] = prediction.model_dump()
        predicted_label_raw: Any = prediction_dict.get("predicted_label")
        true_label_raw: Any = prediction_dict.get("true_label")
        predicted_label: int | str = (
            predicted_label_raw
            if isinstance(predicted_label_raw, (int, str))
            else str(predicted_label_raw)
            if predicted_label_raw is not None
            else "UNKNOWN"
        )
        true_label: int | str = (
            true_label_raw
            if isinstance(true_label_raw, (int, str))
            else str(true_label_raw)
            if true_label_raw is not None
            else "UNKNOWN"
        )
        record: PredictionRecord = PredictionRecord(
            sample_id=str(prediction_dict.get("sample_id", "")),
            predicted_label=predicted_label,
            true_label=true_label,
            confidence=prediction_dict.get("confidence"),
            response_text=str(prediction_dict.get("response_text", "")),
            processing_time=float(prediction_dict.get("processing_time", 0.0)),
            tokens_used=prediction_dict.get("tokens_used"),
        )
        return record

    def _describe_processing_times(
        self, predictions: list[PredictionRecord]
    ) -> dict[str, float | int]:
        """Compute descriptive statistics for processing times."""
        times: list[float] = [prediction.processing_time for prediction in predictions]
        if not times:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "variance": 0.0,
                "std": 0.0,
            }

        time_array: NDArray[np.float64] = np.array(times, dtype=float)
        summary: DescribeResult = stats.describe(time_array)

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
    ) -> dict[str, float | int] | None:
        """Compute descriptive statistics for token usage when available."""
        tokens: list[int] = [
            int(prediction.tokens_used)
            for prediction in predictions
            if prediction.tokens_used is not None
        ]
        if not tokens:
            return None

        token_array: NDArray[np.float64] = np.array(tokens, dtype=float)
        summary: DescribeResult = stats.describe(token_array)

        return {
            "count": int(summary.nobs),
            "min": float(summary.minmax[0]),
            "max": float(summary.minmax[1]),
            "mean": float(summary.mean),
            "variance": float(summary.variance),
            "std": float(np.sqrt(summary.variance)),
        }

    def _build_benchmark_info(
        self,
        total_time: float,
        total_samples: int,
        processing_stats: dict[str, float | int],
        token_stats: dict[str, float | int] | None,
    ) -> BenchmarkInfo:
        """Construct benchmark metadata for the report."""
        avg_time_per_sample: float = (
            total_time / total_samples if total_samples > 0 else 0.0
        )
        tokens_used_total: int | None = (
            int(token_stats["count"] * token_stats["mean"]) if token_stats else None
        )
        tokens_used_avg: float | None = (
            float(token_stats["mean"]) if token_stats else None
        )
        extra_metadata: dict[str, Any] = {}
        if hasattr(self.config, "vulnerability_type"):
            extra_metadata["vulnerability_type"] = getattr(
                self.config, "vulnerability_type"
            )

        return BenchmarkInfo(
            experiment_name=self.config.experiment_name,
            model_name=self.config.model_name,
            model_type=self.config.model_type.value,
            task_type=self.config.task_type.value,
            dataset_path=str(self.config.dataset_path),
            description=self.config.description,
            cwe_type=self.config.cwe_type,
            batch_size=int(self.config.batch_size),
            max_output_tokens=int(self.config.max_output_tokens),
            temperature=float(self.config.temperature),
            use_quantization=bool(self.config.use_quantization),
            is_thinking_enabled=bool(self.config.is_thinking_enabled),
            total_samples=int(total_samples),
            total_time_seconds=float(total_time),
            avg_time_per_sample=float(avg_time_per_sample),
            tokens_used_total=tokens_used_total,
            tokens_used_avg=tokens_used_avg,
            processing_time_stats=processing_stats,
            tokens_used_stats=token_stats,
            extra_metadata=extra_metadata,
            timestamp=datetime.now().isoformat(),
        )
