import random
import time

from pydantic import BaseModel, ConfigDict

from benchmark.config import ExperimentConfig
from benchmark.metrics_calculator import MetricsCalculatorFactory
from benchmark.models import PredictionResult, SampleCollection
from benchmark.prompt_generator import IPromptGenerator, get_prompt_generator
from benchmark.response_parser import IResponseParser, ResponseParserFactory
from benchmark.results import BenchmarkRunResult
from datasets.loaders.base import JsonDatasetLoader
from llm.factory import create_llm_inference
from llm.llm import ILLMInference
from logging_tools import get_logger

_LOGGER = get_logger(__name__)

_ERRORS_THRESHOLD = 0.2  # If more than 20% of samples fail, fail the experiment


class BenchmarkRunner(BaseModel):
    """Main benchmark execution class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ExperimentConfig
    dataset_loader: JsonDatasetLoader

    def run(self) -> BenchmarkRunResult:
        _LOGGER.info(f"Starting {self.config.experiment_name} benchmark execution")

        _LOGGER.info(
            f"Loading {self.config.dataset_name} dataset from: {self.config.dataset_path}"
        )
        samples = self.dataset_loader.load_dataset(self.config.dataset_path, None)
        _LOGGER.info(f"Loaded {len(samples)} samples")

        llm = create_llm_inference(self.config)
        try:
            # samples = self._filter_samples_by_token_limit(samples, llm)
            samples = self._apply_sample_limit(samples)

            if len(samples) == 0:
                raise RuntimeError(
                    "No samples available after token filtering and sample limit application"
                )

            prompt_generator = get_prompt_generator(
                self.config,
                template_values={"cwe": self.config.cwe_type}
                if self.config.cwe_type
                else {},
            )

            response_parser = ResponseParserFactory.create_parser(
                self.config.task_type,
            )

            start_time = time.time()
            predictions = self._process_samples_with_batch_optimization(
                samples, llm, prompt_generator, response_parser
            )
            total_time: float = time.time() - start_time
        finally:
            llm.cleanup()

        metrics_calculator = MetricsCalculatorFactory.create_calculator(
            self.config.task_type
        )
        metrics = metrics_calculator.calculate(predictions)

        # Generate results
        return BenchmarkRunResult(
            metrics=metrics,
            total_samples=len(samples),
            total_time=total_time,
            predictions=predictions,
        )

    def _filter_samples_by_token_limit(
        self,
        samples: SampleCollection,
        llm: ILLMInference,
    ) -> SampleCollection:
        """Filter samples using code token count threshold from config.max_tokens."""
        token_limit: int = self.config.max_tokens
        filtered_samples = [
            sample
            for sample in samples
            if llm.count_input_tokens(sample.code) <= token_limit
        ]

        removed_samples: int = len(samples) - len(filtered_samples)
        if removed_samples > 0:
            _LOGGER.info(
                "Filtered out %d samples with input tokens above max_tokens=%d",
                removed_samples,
                token_limit,
            )
        _LOGGER.info("Samples after token filtering: %d", len(filtered_samples))

        return SampleCollection(filtered_samples)

    def _apply_sample_limit(self, samples: SampleCollection) -> SampleCollection:
        """Apply randomized sample limit after token filtering."""
        sample_limit = self.config.sample_limit
        if sample_limit and sample_limit < len(samples):
            shuffled_samples = list(samples)
            random.shuffle(shuffled_samples)
            limited_samples = shuffled_samples[:sample_limit]
            _LOGGER.info(f"Limited to {sample_limit} samples")
            return SampleCollection(limited_samples)

        return samples

    def _process_samples_with_batch_optimization(
        self,
        samples: SampleCollection,
        llm: ILLMInference,
        prompt_generator: IPromptGenerator,
        response_parser: IResponseParser,
    ) -> list[PredictionResult]:
        """
        Process samples with batch optimization for better GPU utilization.

        This method can be used by benchmark runners to replace their sequential
        processing loops with efficient batch processing.

        Args:
            samples: List of benchmark samples to process

        Returns:
            List of prediction results
        """
        _LOGGER.info(f"Processing {len(samples)} samples with batch optimization")

        user_prompts: list[str] = []
        system_prompts: list[str] = []

        errors_count = 0

        for sample in samples:
            system_prompt, user_prompt = (
                prompt_generator.get_system_prompt(),
                prompt_generator.get_user_prompt({"code": sample.code}),
            )
            user_prompts.append(user_prompt)
            system_prompts.append(system_prompt)

        batch_responses: list[tuple[str, int, float]] = (
            llm.generate_responses_batch_optimized(system_prompts, user_prompts)
        )

        # Process results
        predictions: list[PredictionResult] = []
        for i, (sample, (response_text, tokens_used, processing_duration)) in enumerate(
            zip(samples, batch_responses)
        ):
            try:
                # Parse response
                predicted_label = response_parser.parse_response(response_text)

                # Handle true label - might be int or string depending on dataset
                true_label: int | str
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
                    is_success=True,
                    error_message=None,
                )

                # Log progress
                if (i + 1) % 50 == 0:
                    _LOGGER.info(f"Processed {i + 1}/{len(samples)} predictions")
            except Exception as e:
                _LOGGER.error(
                    f"Error processing sample ID {sample.id}: {e}\nResponse text: {response_text}"
                )
                prediction = PredictionResult(
                    sample_id=sample.id,
                    predicted_label="",
                    true_label=sample.label,
                    confidence=None,
                    response_text=response_text,
                    processing_time=processing_duration,
                    tokens_used=tokens_used,
                    error_message=str(e),
                    is_success=False,
                )
                errors_count += 1
                if errors_count / len(samples) > _ERRORS_THRESHOLD:
                    _LOGGER.error(
                        f"Error rate exceeded threshold of {_ERRORS_THRESHOLD:.0%} - aborting benchmark"
                    )
                    raise RuntimeError(
                        "Too many errors during benchmark execution - aborting"
                    ) from e

            predictions.append(prediction)

        _LOGGER.info(f"Completed processing all {len(samples)} samples")
        return predictions
