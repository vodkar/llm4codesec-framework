#!/usr/bin/env python3
"""
CVEFixes Benchmark Runner

A specialized script for running LLM benchmarks on the CVEFixes dataset with
flexible configuration options for different vulnerability detection tasks.
"""

import argparse
import logging
import sys

from benchmark.config import ExperimentConfig
from benchmark.run_experiment import (
    create_experiment_summary,
    run_experiment_plan,
    run_single_experiment,
)
from entrypoints.utils import (
    list_plans,
    log_available_configurations,
    resolve_config_path,
)
from logging_tools import setup_logging


def main() -> None:
    """Main entry point for CVEFixes benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run CVEFixes benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="cvefixes_experiments.json",
        help="Path to CVEFixes experiments configuration file",
    )

    parser.add_argument("--model", help="Model to use for benchmark")

    parser.add_argument("--dataset", help="Dataset configuration to use")

    parser.add_argument("--prompt", help="Prompt strategy to use")

    parser.add_argument("--plan", help="Experiment plan to run")

    parser.add_argument(
        "--list-plans",
        action="store_true",
        help="List available experiment plans and exit",
    )

    parser.add_argument(
        "--sample-limit", type=int, help="Limit number of samples for testing"
    )

    parser.add_argument(
        "--output-dir",
        default="results/cvefixes_experiments",
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

    # Load configuration
    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", args.config)
        sys.exit(1)

    # Handle list-plans option
    if args.list_plans:
        list_plans(config_path)
        return

    if args.plan:
        results = run_experiment_plan(
            plan_name=args.plan,
            config=config_path,
            output_base_dir=args.output_dir,
        )
        summary = create_experiment_summary(results)
        logger.info("%s", "=" * 80)
        for line in summary.splitlines():
            logger.info(line)
        logger.info("%s", "=" * 80)

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
            logger.info(
                "Experiment completed successfully: %s",
                result.benchmark_info.experiment_name,
            )
            logger.info("Accuracy: %.3f", result.metrics.accuracy)
        else:
            logger.error("Experiment failed: %s", result.benchmark_info.experiment_name)
            logger.error(
                "Error: %s",
                result.metrics.details.get("error", "Unknown error")
                if result.metrics
                else "Unknown error",
            )
            sys.exit(1)

    else:
        parser.print_help()
        logger.info("")
        log_available_configurations(config_path, logger=logger)


if __name__ == "__main__":
    main()
