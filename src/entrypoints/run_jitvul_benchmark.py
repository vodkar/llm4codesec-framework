#!/usr/bin/env python3
"""
JitVul Benchmark Runner

A specialized script for running LLM benchmarks on the JitVul dataset with
flexible configuration options for different vulnerability detection tasks.
"""

import argparse
import dataclasses
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.config import BenchmarkConfig
from benchmark.enums import ModelType, TaskType
from benchmark.prompt_generator import PromptGenerator
from benchmark.response_parser import ResponseParserFactory
from datasets.loaders.jitvul_dataset_loader import JitVulDatasetLoaderFramework
from llm.hugging_face import HuggingFaceLLM


class JitVulBenchmarkRunner:
    """Custom benchmark runner for JitVul datasets."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = JitVulDatasetLoaderFramework()

    def run_benchmark(self, sample_limit: int | None = None) -> dict[str, Any]:
        """Run benchmark with JitVul-specific dataset loading."""
        from benchmark.flash_attention import MetricsCalculator

        logging.info("Starting JitVul benchmark execution")
        start_time = time.time()

        try:
            # Load dataset using JitVul loader
            logging.info(f"Loading JitVul dataset from: {self.config.dataset_path}")
            samples = self.dataset_loader.load_dataset(self.config.dataset_path)

            # Apply sample limit if specified
            if sample_limit and sample_limit < len(samples):
                random.shuffle(samples)
                samples = samples[:sample_limit]
                logging.info(f"Limited to {sample_limit} samples")

            logging.info(f"Loaded {len(samples)} samples")

            # Initialize components
            llm = HuggingFaceLLM(self.config)
            prompt_generator = PromptGenerator()
            response_parser = ResponseParserFactory.create_parser(self.config.task_type)
            metrics_calculator = MetricsCalculator()

            # Create output directory
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

            # Run predictions
            predictions = []
            system_prompt = (
                self.config.system_prompt_template
                or prompt_generator.get_system_prompt(
                    self.config.task_type, self.config.cwe_type
                )
            )

            # JitVul has special handling for call graph context, so we need a custom approach
            # Prepare samples with augmented code for batch processing
            augmented_samples = []
            for sample in samples:
                # Create a copy of the sample with augmented code
                from benchmark.models import BenchmarkSample

                augmented_code = self._augment_code_with_context(sample)
                augmented_sample = BenchmarkSample(
                    id=sample.id,
                    code=augmented_code,
                    label=sample.label,
                    metadata=sample.metadata,
                    cwe_types=sample.cwe_types,
                    severity=sample.severity,
                )
                augmented_samples.append(augmented_sample)

            # Run predictions using batch optimization with augmented samples
            from benchmark.benchmark_runner import BenchmarkRunner

            predictions = BenchmarkRunner.process_samples_with_batch_optimization(
                samples=augmented_samples,
                llm=llm,
                system_prompt=system_prompt,
                prompt_generator=prompt_generator,
                response_parser=response_parser,
                config=self.config,
            )

            # Calculate metrics
            if self.config.task_type in [
                TaskType.BINARY_VULNERABILITY,
                TaskType.BINARY_CWE_SPECIFIC,
            ]:
                metrics = metrics_calculator.calculate_binary_metrics(predictions)
            else:
                metrics = metrics_calculator.calculate_multiclass_metrics(predictions)

            # Generate results
            total_time = time.time() - start_time
            results = {
                "accuracy": metrics.get("accuracy", 0.0),
                "metrics": metrics,
                "total_samples": len(samples),
                "total_time": total_time,
                "predictions": [
                    dataclasses.asdict(prediction) for prediction in predictions
                ],
            }

            # Clean up
            llm.cleanup()

            logging.info("JitVul benchmark completed successfully")
            return results

        except Exception as e:
            logging.exception(f"JitVul benchmark failed: {e}")
            raise

    def _augment_code_with_context(self, sample: Any) -> str:
        """
        Augment code with call graph context if available.

        Args:
            sample: BenchmarkSample with potential call graph metadata

        Returns:
            str: Code with optional call graph context
        """
        code = sample.code

        # Check if call graph information is available in metadata
        if sample.metadata and "call_graph" in sample.metadata:
            call_graph = sample.metadata["call_graph"]
            if call_graph:
                code = f"// Call graph context:\n// {call_graph}\n\n{code}"

        return code


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_jitvul_config(config_path: str) -> dict[str, Any]:
    """Load JitVul experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_benchmark_config(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    prompt_config: dict[str, Any],
    output_dir: str,
) -> BenchmarkConfig:
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
        "LLAMA": ModelType.LLAMA,
        "QWEN": ModelType.QWEN,
        "DEEPSEEK": ModelType.DEEPSEEK,
        "CODEBERT": ModelType.CODEBERT,
        "WIZARD": ModelType.CUSTOM,
        "GEMMA": ModelType.CUSTOM,
    }

    # Map string task types to enum
    task_type_map = {
        "binary_vulnerability": TaskType.BINARY_VULNERABILITY,
        "binary_cwe_specific": TaskType.BINARY_CWE_SPECIFIC,
        "multiclass_vulnerability": TaskType.MULTICLASS_VULNERABILITY,
    }

    return BenchmarkConfig(
        model_name=model_config["model_name"],
        model_type=model_type_map[model_config["model_type"]],
        task_type=task_type_map[dataset_config["task_type"]],
        description=f"{prompt_config['name']} - {dataset_config['description']}",
        dataset_path=dataset_config["dataset_path"],
        output_dir=output_dir,
        batch_size=model_config.get("batch_size", 1),
        max_tokens=model_config.get("max_tokens", 512),
        temperature=model_config.get("temperature", 0.1),
        use_quantization=model_config.get("use_quantization", True),
        cwe_type=dataset_config.get("cwe_type"),
        system_prompt_template=prompt_config.get("system_prompt"),
        user_prompt_template=prompt_config["user_prompt"],
        is_thinking_enabled=prompt_config.get("enable_thinking", False),
    )


def run_single_experiment(
    model_key: str,
    dataset_key: str,
    prompt_key: str,
    jitvul_config: dict[str, Any],
    sample_limit: int | None = None,
    output_base_dir: str = "results/jitvul_experiments",
) -> dict[str, Any]:
    """
    Run a single benchmark experiment.

    Args:
        model_key: Model configuration key
        dataset_key: Dataset configuration key
        prompt_key: Prompt configuration key
        jitvul_config: JitVul experiment configuration
        sample_limit: Limit number of samples (for testing)
        output_base_dir: Base output directory

    Returns:
        dict containing experiment results
    """
    logger = logging.getLogger(__name__)

    # Get configurations
    model_config = jitvul_config["model_configurations"][model_key]
    dataset_config = jitvul_config["dataset_configurations"][dataset_key]
    prompt_config = jitvul_config["prompt_strategies"][prompt_key]

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
    runner = JitVulBenchmarkRunner(config)

    try:
        results = runner.run_benchmark(sample_limit=sample_limit)

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Experiment completed: {experiment_name}")
        logger.info(f"Accuracy: {results['accuracy']:.3f}")
        logger.info(f"Results saved to: {results_file}")

        return {
            "experiment_name": experiment_name,
            "status": "success",
            "accuracy": results["accuracy"],
            "total_samples": results["total_samples"],
            "total_time": results["total_time"],
            "output_dir": str(output_dir),
        }

    except Exception as e:
        logger.exception(f"Experiment failed: {experiment_name}")
        return {
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "output_dir": str(output_dir),
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run JitVul benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="jitvul_experiments.json",
        help="Path to JitVul experiments configuration file",
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
        default="results/jitvul_experiments",
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
        config_path = Path(args.config)
        if not config_path.exists():
            # Try relative to configs directory
            config_path = Path("configs") / args.config
        if not config_path.exists():
            # Try relative to parent directory
            config_path = Path("../configs") / args.config
        if not config_path.exists():
            # Try absolute path from project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "src" / "configs" / args.config

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

        jitvul_config = load_jitvul_config(str(config_path))

        # Handle list-configs option
        if args.list_configs:
            print("Available JitVul Configurations:")
            print("\nDatasets:")
            for key, dataset in jitvul_config["dataset_configurations"].items():
                print(f"  {key}: {dataset['description']}")

            print("\nModels:")
            for key, model in jitvul_config["model_configurations"].items():
                print(f"  {key}: {model['model_name']}")

            print("\nPrompts:")
            for key, prompt in jitvul_config["prompt_strategies"].items():
                print(f"  {key}: {prompt['name']}")

            print("\nExperiment Plans:")
            for key, plan in jitvul_config["experiment_plans"].items():
                print(f"  {key}: {plan['description']}")

            return

        if args.plan:
            # Run experiment plan
            if args.plan not in jitvul_config["experiment_plans"]:
                raise ValueError(f"Experiment plan '{args.plan}' not found")

            plan = jitvul_config["experiment_plans"][args.plan]
            sample_limit = plan.get("sample_limit", args.sample_limit)

            logger.info(f"Running experiment plan: {args.plan}")
            logger.info(f"Description: {plan['description']}")

            results = []
            for dataset_key in plan["datasets"]:
                for model_key in plan["models"]:
                    for prompt_key in plan["prompts"]:
                        result = run_single_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            jitvul_config,
                            sample_limit,
                            args.output_dir,
                        )
                        results.append(result)

            # Save plan results
            plan_results_file = Path(args.output_dir) / f"plan_{args.plan}_results.json"
            with open(plan_results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Plan results saved to: {plan_results_file}")

        elif args.model and args.dataset and args.prompt:
            # Run single experiment
            result = run_single_experiment(
                args.model,
                args.dataset,
                args.prompt,
                jitvul_config,
                args.sample_limit,
                args.output_dir,
            )

            if result["status"] == "success":
                print(f"Experiment completed successfully: {result['experiment_name']}")
                print(f"Accuracy: {result['accuracy']:.3f}")
            else:
                print(f"Experiment failed: {result['experiment_name']}")
                print(f"Error: {result['error']}")
                sys.exit(1)

        else:
            parser.print_help()
            print("\nAvailable configurations:")
            print("Models:", list(jitvul_config["model_configurations"].keys()))
            print("Datasets:", list(jitvul_config["dataset_configurations"].keys()))
            print("Prompts:", list(jitvul_config["prompt_strategies"].keys()))
            print("Plans:", list(jitvul_config["experiment_plans"].keys()))

    except Exception as e:
        logger.exception(f"JitVul benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
