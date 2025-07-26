from llm.hugging_face import HuggingFaceLLM
from llm.llm import ILLMInference
from benchmark.IDatasetLoader import IDatasetLoader
from benchmark.config import BenchmarkConfig
from benchmark.metrics_calculator import MetricsCalculator
from benchmark.models import BenchmarkSample, PredictionResult
from benchmark.prompt_generator import IPromptGenerator
from benchmark.response_parser import IResponseParser


import numpy as np
import pandas as pd
from pydantic import BaseModel


import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


class BenchmarkRunner(BaseModel):
    """Main benchmark execution class."""

    dataset_loader: IDatasetLoader
    prompt_generator: IPromptGenerator
    response_parser: IResponseParser
    metrics_calculator: MetricsCalculator
    llm: ILLMInference
    config: BenchmarkConfig

    def model_post_init(self, context: Any) -> None:
        self._setup_logging()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.config.output_dir / "benchmark.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def run_benchmark(self) -> dict[str, Any]:
        """
        Execute the complete benchmark.

        Returns:
            dict[str, Any]: Benchmark results
        """
        logging.info("Starting benchmark execution")
        start_time = time.time()

        try:
            # Load dataset
            logging.info(f"Loading dataset from: {self.config.dataset_path}")
            samples = self.dataset_loader.load_dataset(self.config.dataset_path)
            logging.info(f"Loaded {len(samples)} samples")

            # Initialize model
            logging.info("Initializing model")
            self.llm = HuggingFaceLLM(self.config)

            # Run predictions
            predictions = self._run_predictions(samples)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate(predictions)

            # Generate report
            report = self._generate_report(
                samples, predictions, metrics, time.time() - start_time
            )

            # Save results
            self._save_results(report)

            logging.info("Benchmark completed successfully")
            return report

        except Exception as e:
            logging.exception(f"Benchmark failed: {e}")
            raise
        finally:
            if self.llm:
                self.llm.cleanup()

    def _run_predictions(
        self, samples: list[BenchmarkSample]
    ) -> list[PredictionResult]:
        """Run model predictions on all samples using batch processing."""
        predictions: list[PredictionResult] = []

        system_prompt = self.prompt_generator.get_system_prompt(
            self.config.task_type, self.config.cwe_type
        )

        # Prepare all prompts in advance for batch processing
        logging.info("Preparing prompts for batch processing...")
        formatted_prompts = []
        for sample in samples:
            user_prompt = self.prompt_generator.get_user_prompt(
                self.config.task_type, sample.code, self.config.cwe_type
            )
            # Cast to HuggingFaceLLM to access the _format_prompt method
            if isinstance(self.llm, HuggingFaceLLM):
                formatted_prompt = self.llm._format_prompt(system_prompt, user_prompt)
            else:
                # Fallback for other LLM types
                formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            formatted_prompts.append(formatted_prompt)

        # Process in batches
        logging.info(
            f"Processing {len(samples)} samples in batches of {self.config.batch_size}"
        )
        start_time = time.time()

        try:
            batch_responses = self.llm.generate_batch_responses(formatted_prompts)
            total_processing_time = time.time() - start_time
            avg_processing_time = total_processing_time / len(samples)

            # Process results
            for i, (sample, (response_text, tokens_used)) in enumerate(
                zip(samples, batch_responses)
            ):
                # Parse response
                predicted_label = self.response_parser.parse_response(response_text)

                prediction = PredictionResult(
                    sample_id=sample.id,
                    predicted_label=predicted_label,
                    true_label=sample.label
                    if isinstance(sample.label, int)
                    else self.response_parser.parse_response(sample.label),
                    confidence=None,  # Could be enhanced to extract confidence
                    response_text=response_text,
                    processing_time=avg_processing_time,  # Average time per sample
                    tokens_used=tokens_used,
                )

                predictions.append(prediction)

                # Log progress
                if (i + 1) % 50 == 0:
                    logging.info(f"Processed {i + 1}/{len(samples)} predictions")

        except Exception as e:
            logging.warning(
                f"Batch processing failed, falling back to sequential processing: {e}"
            )
            # Fallback to sequential processing if batch processing fails
            predictions = self._run_predictions_sequential(samples)

        logging.info(f"Completed all {len(samples)} predictions")
        return predictions

    def _run_predictions_sequential(
        self, samples: list[BenchmarkSample]
    ) -> list[PredictionResult]:
        """Fallback method for sequential prediction processing."""
        predictions: list[PredictionResult] = []

        system_prompt = self.prompt_generator.get_system_prompt(
            self.config.task_type, self.config.cwe_type
        )

        for i, sample in enumerate(samples):
            logging.info(f"Processing sample {i + 1}/{len(samples)}: {sample.id}")

            user_prompt = self.prompt_generator.get_user_prompt(
                self.config.task_type, sample.code, self.config.cwe_type
            )

            # Generate response
            start_time = time.time()
            response_text, tokens_used = self.llm.generate_response(
                system_prompt, user_prompt
            )
            processing_time = time.time() - start_time

            # Parse response
            predicted_label = self.response_parser.parse_response(response_text)

            prediction = PredictionResult(
                sample_id=sample.id,
                predicted_label=predicted_label,
                true_label=sample.label
                if isinstance(sample.label, int)
                else self.response_parser.parse_response(sample.label),
                confidence=None,  # Could be enhanced to extract confidence
                response_text=response_text,
                processing_time=processing_time,
                tokens_used=tokens_used,
            )

            predictions.append(prediction)

            # Log progress
            if (i + 1) % 10 == 0:
                logging.info(f"Completed {i + 1}/{len(samples)} predictions")

        return predictions

    def _generate_report(
        self,
        samples: list[BenchmarkSample],
        predictions: list[PredictionResult],
        metrics: dict[str, Any],
        total_time: float,
    ) -> dict[str, Any]:
        """Generate comprehensive benchmark report."""

        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.config.model_name,
                "model_type": self.config.model_type.value,
                "task_type": self.config.task_type.value,
                "dataset_path": self.config.dataset_path,
                "cwe_type": self.config.cwe_type,
                "total_samples": len(samples),
                "total_time_seconds": total_time,
            },
            "configuration": asdict(self.config),
            "metrics": metrics,
            "predictions": [asdict(pred) for pred in predictions],
            "sample_analysis": {
                "avg_processing_time": np.mean(
                    [p.processing_time for p in predictions]
                ),
                "total_tokens_used": sum(
                    p.tokens_used for p in predictions if p.tokens_used
                ),
                "error_count": len(
                    [p for p in predictions if "ERROR" in p.response_text]
                ),
            },
        }

        return report

    def _save_results(self, report: dict[str, Any]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full report
        report_file = (
            Path(self.config.output_dir) / f"benchmark_report_{timestamp}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Save metrics summary
        metrics_file = (
            Path(self.config.output_dir) / f"metrics_summary_{timestamp}.json"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report["metrics"], f, indent=2, ensure_ascii=False, default=str)

        # Save predictions as CSV
        predictions_df = pd.DataFrame(
            [
                asdict(pred)
                for pred in [
                    PredictionResult(**pred_dict) for pred_dict in report["predictions"]
                ]
            ]
        )
        predictions_csv = Path(self.config.output_dir) / f"predictions_{timestamp}.csv"
        predictions_df.to_csv(predictions_csv, index=False)

        logging.info(f"Results saved to: {report_file}")

    @staticmethod
    def process_samples_with_batch_optimization(
        samples: list[BenchmarkSample],
        llm: ILLMInference,
        system_prompt: str,
        prompt_generator: IPromptGenerator,
        response_parser: IResponseParser,
        config: BenchmarkConfig,
    ) -> list[PredictionResult]:
        """
        Process samples with batch optimization for better GPU utilization.

        This method can be used by benchmark runners to replace their sequential
        processing loops with efficient batch processing.

        Args:
            samples: List of benchmark samples to process
            llm: LLM interface instance
            system_prompt: System prompt to use
            prompt_generator: Prompt generator instance
            response_parser: Response parser instance
            config: Benchmark configuration

        Returns:
            List of prediction results
        """
        logging.info(f"Processing {len(samples)} samples with batch optimization")

        # Prepare all prompts for batch processing
        user_prompts: list[str] = []
        system_prompts: list[str] = []

        for sample in samples:
            # Handle custom user prompt template if provided
            system_prompt, user_prompt = (
                prompt_generator.get_system_prompt(),
                prompt_generator.get_user_prompt(),
            )
            user_prompts.append(user_prompt)
            system_prompts.append(system_prompt)

        # Process in batches
        try:
            # Use batch processing
            batch_responses: list[tuple[str, int, float]] = (
                llm.generate_responses_batch_optimized(system_prompts, user_prompts)
            )

        except Exception as e:
            logging.warning(f"Batch processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            batch_responses: list[tuple[str, int, float]] = []
            for i, sample in enumerate(samples):
                # Handle custom user prompt template if provided
                if config.user_prompt_template:
                    user_prompt = config.user_prompt_template.format(
                        code=sample.code, cwe_type=config.cwe_type
                    )
                else:
                    user_prompt = prompt_generator.get_user_prompt(
                        config.task_type, sample.code, config.cwe_type
                    )

                # Handle custom system prompt template if provided
                current_system_prompt = (
                    config.system_prompt_template
                    if config.system_prompt_template
                    else system_prompt
                )

                response_text, tokens_used, processing_duration = llm.generate_response(
                    current_system_prompt, user_prompt
                )

                batch_responses.append(
                    (response_text, tokens_used, processing_duration)
                )

                if (i + 1) % 10 == 0:
                    logging.info(
                        f"Sequential processing: {i + 1}/{len(samples)} completed"
                    )

        # Process results
        predictions = []
        for i, (sample, (response_text, tokens_used, processing_duration)) in enumerate(
            zip(samples, batch_responses)
        ):
            # Parse response
            predicted_label = response_parser.parse_response(response_text)

            # Handle true label - might be int or string depending on dataset
            if isinstance(sample.label, int):
                true_label = sample.label
            else:
                true_label = response_parser.parse_response(str(sample.label))

            prediction = PredictionResult(
                sample_id=sample.id,
                predicted_label=predicted_label,
                true_label=true_label,
                confidence=None,
                response_text=response_text,
                processing_time=processing_duration,
                tokens_used=tokens_used,
            )

            predictions.append(prediction)

            # Log progress
            if (i + 1) % 50 == 0:
                logging.info(f"Processed {i + 1}/{len(samples)} predictions")

        logging.info(f"Completed processing all {len(samples)} samples")
        return predictions