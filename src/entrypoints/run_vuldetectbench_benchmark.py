#!/usr/bin/env python3
"""
VulDetectBench Benchmark Runner

A specialized script for running LLM benchmarks on the VulDetectBench dataset with
flexible configuration options for 5 different vulnerability detection tasks.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from benchmark.config import ExperimentConfig
from benchmark.enums import ModelType, TaskType
from benchmark.metrics_calculator import (
    BinaryMetricsCalculator,
    CodeAnalysisMetricsCalculator,
    MulticlassMetricsCalculator,
)
from benchmark.models import BenchmarkSample, PredictionResult
from benchmark.prompt_generator import DefaultPromptGenerator
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import (
    BenchmarkReport,
    BenchmarkRunResult,
    MetricsResult,
    ResultArtifacts,
)
from datasets.loaders.vuldetectbench_dataset_loader import (
    VulDetectBenchDatasetLoaderFramework,
)
from llm.factory import create_llm_inference


class VulDetectBenchBenchmarkRunner:
    """Custom benchmark runner for VulDetectBench datasets."""

    def __init__(self, config: ExperimentConfig):
        self.config: ExperimentConfig = config
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Initialize dataset loader
        self.dataset_loader: VulDetectBenchDatasetLoaderFramework = (
            VulDetectBenchDatasetLoaderFramework()
        )

    def run_benchmark(self, sample_limit: int | None = None) -> BenchmarkRunResult:
        """Run benchmark with VulDetectBench-specific dataset loading."""

        logging.info("Starting VulDetectBench benchmark execution")
        start_time = time.time()

        try:
            # Load dataset using VulDetectBench loader
            logging.info(
                f"Loading VulDetectBench dataset from: {self.config.dataset_path}"
            )
            samples = self.dataset_loader.load_processed_dataset(
                self.config.dataset_path
            )

            # Apply sample limit if specified
            if sample_limit and sample_limit < len(samples):
                samples = samples[:sample_limit]
                logging.info(f"Limited to {sample_limit} samples")

            logging.info(f"Loaded {len(samples)} samples")

            # Initialize components
            llm = create_llm_inference(self.config)
            prompt_generator = DefaultPromptGenerator(
                system_prompt_template=self.config.system_prompt_template,
                user_prompt_template=self.config.user_prompt_template,
                template_values={},
            )

            # Determine task type from first sample
            task_type = (
                samples[0].metadata.get("task_type", "task1") if samples else "task1"
            )

            # Create task-specific response parser
            response_parser = VulDetectBenchResponseParser(task_type)

            # Create output directory
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            # VulDetectBench has task-specific prompt formatting that requires special handling
            # We need to handle task2 selection choices before batch processing
            processed_samples: list[BenchmarkSample] = []
            for sample in samples:
                # Create modified sample with enhanced code for task2
                enhanced_code = sample.code

                # Handle task-specific prompt formatting for task2
                if (
                    sample.metadata.get("task_type") == "task2"
                    and "selection_choices" in sample.metadata
                ):
                    # For Task 2, prepend selection choices to the code
                    enhanced_code = (
                        f"{sample.metadata['selection_choices']}\n\n{sample.code}"
                    )

                processed_sample = BenchmarkSample(
                    id=sample.id,
                    code=enhanced_code,
                    label=sample.label,
                    metadata=sample.metadata,
                    cwe_types=sample.cwe_types,
                    severity=sample.severity,
                )
                processed_samples.append(processed_sample)

            # Run predictions using batch optimization with processed samples
            from benchmark.benchmark_runner import BenchmarkRunner

            runner = BenchmarkRunner(
                prompt_generator=prompt_generator,
                response_parser=response_parser,
                llm=llm,
                config=self.config,
            )
            predictions = runner._process_samples_with_batch_optimization(
                processed_samples
            )

            # Calculate metrics based on task type
            task_type = (
                samples[0].metadata.get("task_type", "task1") if samples else "task1"
            )
            metrics = self._calculate_task_specific_metrics(predictions, task_type)

            # Generate results
            total_time = time.time() - start_time
            results: BenchmarkRunResult = BenchmarkRunResult(
                metrics=metrics,
                total_samples=len(samples),
                total_time=total_time,
                predictions=predictions,
            )

            # Clean up
            llm.cleanup()

            logging.info("VulDetectBench benchmark completed successfully")
            return results

        except Exception:
            logging.exception("VulDetectBench benchmark failed")
            raise

    def _calculate_task_specific_metrics(
        self, predictions: list[PredictionResult], task_type: str
    ) -> MetricsResult:
        """Calculate metrics specific to VulDetectBench tasks."""

        calculator: (
            BinaryMetricsCalculator
            | MulticlassMetricsCalculator
            | CodeAnalysisMetricsCalculator
        )

        if task_type == "task1":
            # Binary classification metrics
            calculator = BinaryMetricsCalculator()
            return calculator.calculate(predictions)
        elif task_type == "task2":
            # Multi-class classification metrics
            calculator = MulticlassMetricsCalculator()
            return calculator.calculate(predictions)
        else:
            # For tasks 3-5, use code analysis metrics
            calculator = CodeAnalysisMetricsCalculator(task_type)
            return calculator.calculate(predictions)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_vuldetectbench_config(config_path: str) -> dict[str, Any]:
    """Load VulDetectBench experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_benchmark_config(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    prompt_config: dict[str, Any],
    output_dir: str,
) -> ExperimentConfig:
    """
    Create a BenchmarkConfig from experiment configuration.

    Args:
        model_config: Model configuration dict
        dataset_config: Dataset configuration dict
        prompt_config: Prompt configuration dict
        output_dir: Output directory path

    Returns:
        BenchmarkConfig: Complete benchmark configuration
    """
    # Map string model types to enum
    model_type_map = {
        "LLAMA": ModelType.LLAMA_2,
        "QWEN": ModelType.QWEN_2_5,
        "DEEPSEEK": ModelType.DEEPSEEK_CODER,
        "CODEBERT": ModelType.CODEBERT,
        "WIZARDCODER": ModelType.CUSTOM,
        "GEMMA": ModelType.CUSTOM,
    }

    # Map string task types to enum
    task_type_map = {
        "binary_vulnerability": TaskType.BINARY_VULNERABILITY,
        "multiclass_vulnerability": TaskType.MULTICLASS_VULNERABILITY,
        "code_analysis": TaskType.BINARY_VULNERABILITY,  # Default for code analysis tasks
    }

    return ExperimentConfig(
        model_name=model_config["model_name"],
        model_type=model_type_map[model_config["model_type"]],
        task_type=task_type_map[dataset_config["task_type"]],
        description=f"{prompt_config['name']} - {dataset_config['description']}",
        dataset_path=Path(dataset_config["dataset_path"]),
        output_dir=Path(output_dir),
        backend=model_config.get("backend", "hf"),
        batch_size=model_config.get("batch_size", 1),
        max_tokens=model_config.get("max_tokens", 512),
        temperature=model_config.get("temperature", 0.1),
        use_quantization=model_config.get("use_quantization", True),
        cwe_type=dataset_config.get("cwe_type"),
        system_prompt_template=prompt_config.get("system_prompt") or "",
        user_prompt_template=prompt_config["user_prompt"],
        is_thinking_enabled=prompt_config.get("is_thinking_enabled", False),
    )


def run_single_experiment(
    model_key: str,
    dataset_key: str,
    prompt_key: str,
    vuldetectbench_config: dict[str, Any],
    sample_limit: int | None = None,
    output_base_dir: str = "results/vuldetectbench_experiments",
) -> dict[str, Any]:
    """
    Run a single benchmark experiment.

    Args:
        model_key: Model configuration key
        dataset_key: Dataset configuration key
        prompt_key: Prompt configuration key
        vuldetectbench_config: VulDetectBench experiment configuration
        sample_limit: Limit number of samples (for testing)
        output_base_dir: Base output directory

    Returns:
        dict containing experiment results
    """
    logger = logging.getLogger(__name__)

    # Get configurations
    model_config = vuldetectbench_config["model_configurations"][model_key]
    dataset_config = vuldetectbench_config["dataset_configurations"][dataset_key]
    prompt_config = vuldetectbench_config["prompt_strategies"][prompt_key]

    # Create output directory
    experiment_name = f"{model_key}_{dataset_key}_{prompt_key}"
    output_dir = Path(output_base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Dataset: {dataset_config['description']}")
    logger.info(f"Prompt: {prompt_config['name']}")

    # Create benchmark configuration
    config = create_benchmark_config(
        model_config, dataset_config, prompt_config, str(output_dir)
    )

    # Initialize and run benchmark
    runner = VulDetectBenchBenchmarkRunner(config)

    try:
        run_data: BenchmarkRunResult = runner.run_benchmark(sample_limit=sample_limit)

        result_processor = BenchmarkResultProcessor(
            config=config, experiment_name=experiment_name
        )
        report: BenchmarkReport
        artifacts: ResultArtifacts
        report, artifacts = result_processor.build_and_save(
            metrics=run_data.metrics,
            predictions=run_data.predictions,
            total_time=run_data.total_time,
            total_samples=run_data.total_samples,
        )

        logger.info(
            "Experiment completed successfully. Results saved to: %s",
            artifacts.report_json,
        )
        return {
            "experiment_name": experiment_name,
            "status": "success",
            "report": report,
            "artifacts": artifacts,
            "accuracy": report.metrics.accuracy,
            "total_samples": run_data.total_samples,
            "total_time": run_data.total_time,
            "output_dir": str(output_dir),
        }

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        return {
            "error": str(e),
            "experiment_name": experiment_name,
            "status": "failed",
            "output_dir": str(output_dir),
        }


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run VulDetectBench benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python run_vuldetectbench_benchmark.py --plan quick_test
  
  # Run specific task evaluation
  python run_vuldetectbench_benchmark.py --plan task1_evaluation
  
  # Run single experiment
  python run_vuldetectbench_benchmark.py --model qwen3-4b --dataset task1_vulnerability --prompt basic_security
  
  # List available configurations
  python run_vuldetectbench_benchmark.py --list-configs
        """,
    )

    parser.add_argument(
        "--config",
        default="configs/vuldetectbench_experiments.json",
        help="Path to VulDetectBench experiments configuration file",
    )

    parser.add_argument("--model", help="Model to use for benchmark")

    parser.add_argument("--dataset", help="Dataset configuration to use")

    parser.add_argument("--prompt", help="Prompt strategy to use")

    parser.add_argument("--plan", help="Experiment plan to run")

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit",
    )

    parser.add_argument(
        "--sample-limit", type=int, help="Limit number of samples for testing"
    )

    parser.add_argument(
        "--output-dir",
        default="results/vuldetectbench_experiments",
        help="Base output directory for results",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_vuldetectbench_config(args.config)

        if args.list_configs:
            print("Available Models:")
            for model_key in config["model_configurations"]:
                model_info = config["model_configurations"][model_key]
                print(f"  {model_key}: {model_info['model_name']}")

            print("\nAvailable Datasets:")
            for dataset_key in config["dataset_configurations"]:
                dataset_info = config["dataset_configurations"][dataset_key]
                print(f"  {dataset_key}: {dataset_info['description']}")

            print("\nAvailable Prompts:")
            for prompt_key in config["prompt_strategies"]:
                prompt_info = config["prompt_strategies"][prompt_key]
                print(f"  {prompt_key}: {prompt_info['name']}")

            print("\nAvailable Plans:")
            for plan_key in config["experiment_plans"]:
                plan_info = config["experiment_plans"][plan_key]
                print(f"  {plan_key}: {plan_info['description']}")

            return 0

        if args.plan:
            # Run experiment plan
            from entrypoints.run_unified_benchmark import run_experiment_plan

            results = run_experiment_plan(
                "vuldetectbench", args.plan, config, args.output_dir, args.sample_limit
            )
            logger.info(f"Experiment plan completed: {results['summary']}")

        elif args.model and args.dataset and args.prompt:
            # Run single experiment
            results = run_single_experiment(
                args.model,
                args.dataset,
                args.prompt,
                config,
                args.sample_limit,
                args.output_dir,
            )

            if "error" not in results:
                logger.info("Single experiment completed successfully")
                logger.info(f"Accuracy: {results.get('accuracy', 'N/A')}")
            else:
                logger.error(f"Single experiment failed: {results['error']}")
        else:
            parser.print_help()

    except Exception:
        logger.exception("Application failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
