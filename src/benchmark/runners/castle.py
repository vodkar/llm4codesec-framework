import logging
import time

from pydantic import BaseModel, ConfigDict

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.config import ExperimentConfig
from benchmark.metrics_calculator import MetricsCalculatorFactory
from benchmark.prompt_generator import DefaultPromptGenerator
from benchmark.response_parser import ResponseParserFactory
from benchmark.results import BenchmarkRunResult
from datasets.loaders.base import JsonDatasetLoader
from llm.factory import create_llm_inference


class CastleBenchmarkRunner(BaseModel):
    """Custom benchmark runner for CASTLE datasets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ExperimentConfig
    dataset_loader: JsonDatasetLoader

    def run_benchmark(self, sample_limit: int | None = None) -> BenchmarkRunResult:
        """Run benchmark with CASTLE-specific dataset loading."""

        logging.info("Starting CASTLE benchmark execution")
        start_time = time.time()

        try:
            logging.info(f"Loading CASTLE dataset from: {self.config.dataset_path}")
            samples = self.dataset_loader.load_dataset(
                self.config.dataset_path, sample_limit
            )
            logging.info(f"Loaded {len(samples)} samples")

            # Initialize components
            llm = create_llm_inference(self.config)
            prompt_generator = DefaultPromptGenerator(
                system_prompt_template=self.config.system_prompt_template,
                user_prompt_template=self.config.user_prompt_template,
                template_values={"cwe": self.config.cwe_type}
                if self.config.cwe_type
                else {},
            )
            response_parser = ResponseParserFactory.create_parser(self.config.task_type)
            metrics_calculator = MetricsCalculatorFactory.create_calculator(
                self.config.task_type
            )
            runner = BenchmarkRunner(
                prompt_generator=prompt_generator,
                response_parser=response_parser,
                llm=llm,
                config=self.config,
            )

            predictions = runner._process_samples_with_batch_optimization(
                samples=samples
            )

            metrics = metrics_calculator.calculate(predictions)

            # Generate results
            total_time: float = time.time() - start_time
            results: BenchmarkRunResult = BenchmarkRunResult(
                metrics=metrics,
                total_samples=len(samples),
                total_time=total_time,
                predictions=predictions,
            )

            # Clean up
            llm.cleanup()

            logging.info("CASTLE benchmark completed successfully")
            return results

        except Exception:
            logging.exception("CASTLE benchmark failed")
            raise
