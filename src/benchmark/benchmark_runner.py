import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from benchmark.config import BenchmarkConfig
from benchmark.models import BenchmarkSample, PredictionResult
from benchmark.prompt_generator import IPromptGenerator
from benchmark.response_parser import IResponseParser
from llm.llm import ILLMInference


class BenchmarkRunner(BaseModel):
    """Main benchmark execution class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_generator: IPromptGenerator
    response_parser: IResponseParser
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

    def process_samples_with_batch_optimization(
        self, samples: list[BenchmarkSample]
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
        logging.info(f"Processing {len(samples)} samples with batch optimization")

        user_prompts: list[str] = []
        system_prompts: list[str] = []

        for sample in samples:
            system_prompt, user_prompt = (
                self.prompt_generator.get_system_prompt(),
                self.prompt_generator.get_user_prompt({"code": sample.code}),
            )
            user_prompts.append(user_prompt)
            system_prompts.append(system_prompt)

        batch_responses: list[tuple[str, int, float]] = (
            self.llm.generate_responses_batch_optimized(system_prompts, user_prompts)
        )

        # Process results
        predictions: list[PredictionResult] = []
        for i, (sample, (response_text, tokens_used, processing_duration)) in enumerate(
            zip(samples, batch_responses)
        ):
            # Parse response
            predicted_label = self.response_parser.parse_response(response_text)

            # Handle true label - might be int or string depending on dataset
            true_label: int | str
            if isinstance(sample.label, int):
                true_label = sample.label
            else:
                true_label = self.response_parser.parse_response(str(sample.label))

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
