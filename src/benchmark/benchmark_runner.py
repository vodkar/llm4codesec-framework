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
_CHAT_TEMPLATE_TOKEN_OVERHEAD = 200  # Safety margin for chat-template special tokens


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
            prompt_generator = get_prompt_generator(
                self.config,
                template_values={"cwe": self.config.cwe_type}
                if self.config.cwe_type
                else {},
            )

            samples = self._filter_samples_by_token_limit(samples, llm, prompt_generator)
            samples = self._apply_sample_limit(samples)

            if len(samples) == 0:
                raise RuntimeError(
                    "No samples available after token filtering and sample limit application"
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
        prompt_generator: IPromptGenerator,
    ) -> SampleCollection:
        """Filter samples whose full formatted prompt exceeds the model's input budget.

        Counts tokens on the complete prompt text (system + user with code substituted)
        and compares against context_length - max_output_tokens - chat template overhead.
        """
        context_len = self.config.model_context_length_tokens
        output_budget = self.config.max_output_tokens
        input_budget = context_len - output_budget - _CHAT_TEMPLATE_TOKEN_OVERHEAD

        if input_budget <= 0:
            _LOGGER.warning(
                "Input budget is %d (context_length=%d, max_output_tokens=%d, overhead=%d). "
                "Token filtering skipped — set context_length separately in model config.",
                input_budget,
                context_len,
                output_budget,
                _CHAT_TEMPLATE_TOKEN_OVERHEAD,
            )
            return samples

        filtered_samples = []
        for sample in samples:
            system_prompt = prompt_generator.get_system_prompt()
            user_prompt = prompt_generator.get_user_prompt({"code": sample.code})
            full_text = system_prompt + "\n" + user_prompt
            if llm.count_input_tokens(full_text) <= input_budget:
                filtered_samples.append(sample)

        removed_samples = len(samples) - len(filtered_samples)
        if removed_samples > 0:
            _LOGGER.info(
                "Filtered out %d samples exceeding input budget of %d tokens "
                "(context=%d, output_budget=%d, overhead=%d)",
                removed_samples,
                input_budget,
                context_len,
                output_budget,
                _CHAT_TEMPLATE_TOKEN_OVERHEAD,
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
