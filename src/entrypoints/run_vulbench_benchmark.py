#!/usr/bin/env python3
"""
VulBench Benchmark Runner (New Configuration-Based)

A specialized script for running LLM benchmarks on the VulBench dataset with
unified configuration approach matching CASTLE, JitVul, and CVEFixes patterns.
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.benchmark_framework import BenchmarkConfig
from benchmark.enums import ModelType, TaskType
from datasets.loaders.vulbench_dataset_loader import VulBenchDatasetLoaderFramework


class VulBenchBenchmarkRunner:
    """Custom benchmark runner for VulBench datasets."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize dataset loader
        self.dataset_loader = VulBenchDatasetLoaderFramework()

    def run_benchmark(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run benchmark with VulBench-specific dataset loading."""
        import dataclasses
        import time
        from pathlib import Path

        from benchmark.benchmark_framework import (
            HuggingFaceLLM,
            MetricsCalculator,
            PromptGenerator,
            VulBenchResponseParser,
        )

        logging.info("Starting VulBench benchmark execution")
        start_time = time.time()

        try:
            # Load dataset using VulBench loader
            logging.info(f"Loading VulBench dataset from: {self.config.dataset_path}")
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
            # Use VulBench-specific response parser for better VulBench pattern matching
            response_parser = VulBenchResponseParser(self.config.task_type)
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

            # Run predictions using batch optimization
            from benchmark.benchmark_framework import BenchmarkRunner

            predictions = BenchmarkRunner.process_samples_with_batch_optimization(
                samples=samples,
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

            logging.info("VulBench benchmark completed successfully")
            return results

        except Exception as e:
            logging.exception(f"VulBench benchmark failed: {e}")
            raise


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_vulbench_config(config_path: str) -> Dict[str, Any]:
    """Load VulBench experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_benchmark_config(
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    prompt_config: Dict[str, Any],
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
        "binary_vulnerability_specific": TaskType.BINARY_VULNERABILITY_SPECIFIC,
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
        cwe_type=dataset_config.get("cwe_type")
        or dataset_config.get("vulnerability_type"),
        system_prompt_template=prompt_config.get("system_prompt"),
        user_prompt_template=prompt_config["user_prompt"],
        enable_thinking=prompt_config.get("enable_thinking", False),
    )


def run_single_experiment(
    model_key: str,
    dataset_key: str,
    prompt_key: str,
    vulbench_config: Dict[str, Any],
    sample_limit: Optional[int] = None,
    output_base_dir: str = "results/vulbench_experiments",
) -> Dict[str, Any]:
    """
    Run a single benchmark experiment.

    Args:
        model_key: Model configuration key
        dataset_key: Dataset configuration key
        prompt_key: Prompt configuration key
        vulbench_config: VulBench experiment configuration
        sample_limit: Limit number of samples (for testing)
        output_base_dir: Base output directory

    Returns:
        Dict containing experiment results
    """
    logger = logging.getLogger(__name__)

    # Get configurations
    model_config = vulbench_config["model_configurations"][model_key]
    dataset_config = vulbench_config["dataset_configurations"][dataset_key]
    prompt_config = vulbench_config["prompt_strategies"][prompt_key]

    # Create output directory
    experiment_name = f"{model_key}_{dataset_key}_{prompt_key}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Dataset: {dataset_config['description']}")
    logger.info(f"Prompt: {prompt_config['name']}")
    logger.info(f"Output: {output_dir}")

    try:
        # Create benchmark configuration
        config = create_benchmark_config(
            model_config, dataset_config, prompt_config, str(output_dir)
        )

        # Run benchmark
        runner = VulBenchBenchmarkRunner(config)
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
            "results_file": str(results_file),
        }

    except Exception as e:
        logger.exception(f"Experiment failed: {experiment_name} - {e}")
        return {
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "output_dir": str(output_dir),
        }


def run_experiment_plan(
    plan_name: str,
    vulbench_config: Dict[str, Any],
    output_base_dir: str,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a complete experiment plan with multiple configurations.

    Args:
        plan_name: Name of the experiment plan to run
        vulbench_config: VulBench experiment configuration
        output_base_dir: Base output directory
        sample_limit: Limit samples for testing

    Returns:
        Dict containing all experiment results
    """
    logger = logging.getLogger(__name__)

    if plan_name not in vulbench_config["experiment_plans"]:
        raise ValueError(f"Unknown experiment plan: {plan_name}")

    plan = vulbench_config["experiment_plans"][plan_name]
    logger.info(f"Starting experiment plan: {plan_name}")
    logger.info(f"Description: {plan['description']}")

    # Override sample limit if specified in plan
    plan_sample_limit = plan.get("sample_limit", sample_limit)

    # Create plan-specific output directory
    plan_output_dir = (
        Path(output_base_dir)
        / f"vulbench_plan_{plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset_type": "vulbench",
        "plan_name": plan_name,
        "description": plan["description"],
        "start_time": datetime.now().isoformat(),
        "experiments": [],
        "summary": {},
        "output_dir": str(plan_output_dir),
    }

    # Calculate total experiments
    total_experiments = (
        len(plan["datasets"]) * len(plan["models"]) * len(plan["prompts"])
    )
    logger.info(f"Total experiments to run: {total_experiments}")

    experiment_count = 0
    successful_experiments = 0
    failed_experiments = 0

    # Run all combinations
    for dataset_key in plan["datasets"]:
        for model_key in plan["models"]:
            for prompt_key in plan["prompts"]:
                experiment_count += 1
                logger.info(
                    f"Running experiment {experiment_count}/{total_experiments}: "
                    f"{model_key} + {dataset_key} + {prompt_key}"
                )

                try:
                    result = run_single_experiment(
                        model_key=model_key,
                        dataset_key=dataset_key,
                        prompt_key=prompt_key,
                        vulbench_config=vulbench_config,
                        sample_limit=plan_sample_limit,
                        output_base_dir=str(plan_output_dir),
                    )

                    results["experiments"].append(result)

                    if result["status"] == "success":
                        successful_experiments += 1
                    else:
                        failed_experiments += 1

                except Exception as e:
                    logger.exception(f"Experiment failed: {e}")
                    failed_experiments += 1
                    results["experiments"].append(
                        {
                            "experiment_name": f"{model_key}_{dataset_key}_{prompt_key}",
                            "status": "failed",
                            "error": str(e),
                        }
                    )

    # Complete results
    results["end_time"] = datetime.now().isoformat()
    results["summary"] = {
        "total_experiments": total_experiments,
        "successful": successful_experiments,
        "failed": failed_experiments,
        "success_rate": successful_experiments / total_experiments
        if total_experiments > 0
        else 0,
    }

    # Save plan results
    plan_results_file = plan_output_dir / "experiment_plan_results.json"
    with open(plan_results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(
        f"Experiment plan completed: {successful_experiments}/{total_experiments} successful"
    )
    logger.info(f"Plan results saved to: {plan_results_file}")

    return results


def list_available_configs(vulbench_config: Dict[str, Any]) -> None:
    """List all available configurations."""
    print("=== VulBench Benchmark Configurations ===\n")

    print("📊 Available Datasets:")
    for key, config in vulbench_config["dataset_configurations"].items():
        print(f"  {key}: {config['description']}")

    print(
        f"\n🤖 Available Models ({len(vulbench_config['model_configurations'])} total):"
    )
    for key, config in vulbench_config["model_configurations"].items():
        print(f"  {key}: {config['model_name']}")

    print(
        f"\n💬 Available Prompts ({len(vulbench_config['prompt_strategies'])} total):"
    )
    for key, config in vulbench_config["prompt_strategies"].items():
        print(f"  {key}: {config['name']}")

    print(
        f"\n📋 Available Experiment Plans ({len(vulbench_config['experiment_plans'])} total):"
    )
    for key, config in vulbench_config["experiment_plans"].items():
        print(f"  {key}: {config['description']}")


def main():
    parser = argparse.ArgumentParser(
        description="VulBench Benchmark Runner (Configuration-Based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python run_vulbench_benchmark_new.py --plan quick_test
  
  # Run single experiment  
  python run_vulbench_benchmark_new.py --model qwen2.5-7b --dataset binary_d2a --prompt basic_security
  
  # List available configurations
  python run_vulbench_benchmark_new.py --list-configs
        """,
    )

    parser.add_argument(
        "--config",
        default="vulbench_experiments.json",
        help="Path to experiment configuration file",
    )

    parser.add_argument("--model", help="Model configuration to use")

    parser.add_argument("--dataset", help="Dataset configuration to use")

    parser.add_argument("--prompt", help="Prompt strategy to use")

    parser.add_argument("--plan", help="Experiment plan to run")

    parser.add_argument(
        "--sample-limit", type=int, help="Limit number of samples for testing"
    )

    parser.add_argument(
        "--output-dir",
        default="results/vulbench_experiments",
        help="Base output directory for results",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit",
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
    setup_logging(args.verbose or args.log_level == "DEBUG")
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            # Try relative to src/configs
            config_path = Path("configs") / args.config
        if not config_path.exists():
            # Try relative to project root
            config_path = Path("../configs") / args.config
        if not config_path.exists():
            # Try absolute path from project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "src" / "configs" / args.config

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

        vulbench_config = load_vulbench_config(str(config_path))

        # List configurations if requested
        if args.list_configs:
            list_available_configs(vulbench_config)
            return

        # Validate arguments
        if args.plan:
            # Run experiment plan
            if args.plan not in vulbench_config["experiment_plans"]:
                logger.error(f"Unknown experiment plan: {args.plan}")
                logger.info(
                    "Available plans: "
                    + ", ".join(vulbench_config["experiment_plans"].keys())
                )
                return

            logger.info(f"Running experiment plan: {args.plan}")
            run_experiment_plan(
                plan_name=args.plan,
                vulbench_config=vulbench_config,
                output_base_dir=args.output_dir,
                sample_limit=args.sample_limit,
            )

        elif args.model and args.dataset and args.prompt:
            # Run single experiment
            if args.model not in vulbench_config["model_configurations"]:
                logger.error(f"Unknown model: {args.model}")
                logger.info(
                    "Available models: "
                    + ", ".join(vulbench_config["model_configurations"].keys())
                )
                return

            if args.dataset not in vulbench_config["dataset_configurations"]:
                logger.error(f"Unknown dataset: {args.dataset}")
                logger.info(
                    "Available datasets: "
                    + ", ".join(vulbench_config["dataset_configurations"].keys())
                )
                return

            if args.prompt not in vulbench_config["prompt_strategies"]:
                logger.error(f"Unknown prompt: {args.prompt}")
                logger.info(
                    "Available prompts: "
                    + ", ".join(vulbench_config["prompt_strategies"].keys())
                )
                return

            logger.info(
                f"Running single experiment: {args.model} + {args.dataset} + {args.prompt}"
            )
            run_single_experiment(
                model_key=args.model,
                dataset_key=args.dataset,
                prompt_key=args.prompt,
                vulbench_config=vulbench_config,
                sample_limit=args.sample_limit,
                output_base_dir=args.output_dir,
            )

        else:
            parser.print_help()
            print(
                "\nError: Either specify --plan or provide --model, --dataset, and --prompt"
            )
            return

        logger.info("Benchmark completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
