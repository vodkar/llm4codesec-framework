#!/usr/bin/env python3
"""
VulBench Benchmark Runner (New Configuration-Based)

A specialized script for running LLM benchmarks on the VulBench dataset with
unified configuration approach matching CASTLE, JitVul, and CVEFixes patterns.
"""

import argparse
import logging
import sys
from pathlib import Path

from benchmark.config import ExperimentConfig
from benchmark.run_experiment import (
    create_experiment_summary,
    run_experiment_plan,
    run_single_experiment,
)
from entrypoints.utils import list_plans
from logging_tools import setup_logging


def main() -> None:
    """Main entry point for VulBench benchmark runner."""
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
        "--list-plans",
        action="store_true",
        help="List available experiment plans and exit",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)

    # List configurations if requested
    if args.list_plans:
        list_plans(config_path)
        return

    # Validate arguments
    if args.plan:
        results = run_experiment_plan(
            plan_name=args.plan,
            config=config_path,
            output_base_dir=args.output_dir,
        )
        summary = create_experiment_summary(results)
        print("\n" + "=" * 80)
        print(summary)
        print("=" * 80)

        if results.summary.failed_experiments > 0:
            logger.warning("Some experiments failed. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("All experiments completed successfully!")

    elif args.model and args.dataset and args.prompt:
        # Run single experiment
        config = ExperimentConfig.from_file(
            config=config_path,
            model_key=args.model,
            dataset_key=args.dataset,
            prompt_key=args.prompt,
            experiment_name="manual",
            sample_limit=args.sample_limit,
        )

        result = run_single_experiment(config=config)

        if result.is_success:
            print(
                f"Experiment completed successfully: {result.benchmark_info.experiment_name}"
            )
            print(f"Accuracy: {result.metrics.accuracy:.3f}")
        else:
            print(f"Experiment failed: {result.benchmark_info.experiment_name}")
            print(
                f"Error: {result.metrics.details.get('error', 'Unknown error') if result.metrics else 'Unknown error'}"
            )
            sys.exit(1)

        logger.info(
            f"Running single experiment: {args.model} + {args.dataset} + {args.prompt}"
        )
    else:
        parser.print_help()
        print(
            "\nError: Either specify --plan or provide --model, --dataset, and --prompt"
        )
        return

    logger.info("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
