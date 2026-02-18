"""
CASTLE Experiments Runner

Runs batch experiments on CASTLE dataset with predefined experiment plans
for comprehensive evaluation of LLMs on vulnerability detection tasks.
"""

import argparse
import logging
import sys
from pathlib import Path

from benchmark.run_experiment import (
    create_experiment_summary,
    run_experiment_plan,
)
from consts import CONFIG_DIRECTORY
from entrypoints.run_castle_experiment import (
    setup_logging,
)
from entrypoints.utils import list_plans


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run CASTLE experiment plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default=(CONFIG_DIRECTORY / "castle_experiments.json").absolute(),
        help="Path to CASTLE experiments configuration file",
    )

    parser.add_argument(
        "--plan",
        help="Experiment plan to run",
    )

    parser.add_argument(
        "--list-plans", action="store_true", help="List available experiment plans"
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
    _LOGGER = logging.getLogger(__name__)

    # Load configuration
    if not Path(args.config).exists():
        _LOGGER.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    castle_config = Path(args.config)
    _LOGGER.info(f"Loaded configuration from {args.config}")

    # List plans if requested
    if args.list_plans:
        list_plans(castle_config)
        return

    # Validate required arguments
    if not args.plan:
        _LOGGER.error("Experiment plan must be specified")
        _LOGGER.error("Use --list-plans to see available plans")
        sys.exit(1)

    # Run experiment plan
    _LOGGER.info(f"Starting experiment plan: {args.plan}")

    results = run_experiment_plan(
        plan_name=args.plan,
        config=castle_config,
        output_base_dir=args.output_dir,
    )

    # Print summary
    summary = create_experiment_summary(results)
    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)

    if results.summary.failed_experiments > 0:
        _LOGGER.warning("Some experiments failed. Check logs for details.")
        sys.exit(1)
    else:
        _LOGGER.info("All experiments completed successfully!")


if __name__ == "__main__":
    main()
