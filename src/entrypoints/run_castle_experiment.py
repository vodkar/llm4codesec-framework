"""
CASTLE Benchmark Runner

A specialized script for running LLM benchmarks on the CASTLE dataset with
flexible configuration options for different vulnerability detection tasks.
"""

import argparse
import logging
import sys
from pathlib import Path

from benchmark.config import ExperimentConfig
from benchmark.run_experiment import run_single_experiment
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for CASTLE benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run CASTLE benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="castle_experiments_config.json",
        help="Path to CASTLE experiments configuration file",
    )

    parser.add_argument(
        "--model",
        help="Model to use for benchmark",
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "binary_all",
            "multiclass_all",
            "cwe_125",
            "cwe_190",
            "cwe_476",
            "cwe_787",
        ],
        help="Dataset configuration to use",
    )

    parser.add_argument(
        "--prompt",
        choices=[
            "basic_security",
            "detailed_analysis",
            "cwe_focused",
            "context_aware",
            "step_by_step",
        ],
        help="Prompt strategy to use",
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit number of samples for testing (default: use all samples)",
    )

    parser.add_argument(
        "--output-dir",
        default="results/castle_experiments",
        help="Base output directory for results",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate required arguments
    if not all([args.model, args.dataset, args.prompt]):
        _LOGGER.error("Model, dataset, and prompt must be specified")
        _LOGGER.error("Use --help for usage information")
        sys.exit(1)

    config = ExperimentConfig.from_file(
        config=Path(args.config),
        model_key=args.model,
        dataset_key=args.dataset,
        prompt_key=args.prompt,
        experiment_name="manual",
        sample_limit=args.sample_limit,
    )

    run_single_experiment(config)


if __name__ == "__main__":
    main()
