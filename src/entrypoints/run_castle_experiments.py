"""
CASTLE Experiments Runner

Runs batch experiments on CASTLE dataset with predefined experiment plans
for comprehensive evaluation of LLMs on vulnerability detection tasks.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from consts import CONFIG_DIRECTORY
from entrypoints.run_castle_benchmark import (
    load_castle_config,
    run_single_experiment,
    setup_logging,
)


def run_experiment_plan(
    plan_name: str,
    castle_config: dict[str, Any],
    output_base_dir: str = "results/castle_experiments",
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """
    Run a complete experiment plan with multiple configurations.

    Args:
        plan_name: Name of the experiment plan to run
        castle_config: CASTLE experiment configuration
        output_base_dir: Base output directory
        sample_limit: Limit samples for testing

    Returns:
        dict containing all experiment results
    """
    logger = logging.getLogger(__name__)

    if plan_name not in castle_config["experiment_plans"]:
        raise ValueError(f"Unknown experiment plan: {plan_name}")

    plan = castle_config["experiment_plans"][plan_name]
    logger.info(f"Starting experiment plan: {plan_name}")
    logger.info(f"Description: {plan['description']}")

    # Override sample limit if specified in plan
    plan_sample_limit = plan.get("sample_limit", sample_limit)

    # Create plan-specific output directory
    plan_output_dir = (
        Path(output_base_dir)
        / f"plan_{plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
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
                    f"Experiment {experiment_count}/{total_experiments}: {model_key} + {dataset_key} + {prompt_key}"
                )

                try:
                    # Run single experiment
                    experiment_result = run_single_experiment(
                        model_key=model_key,
                        dataset_key=dataset_key,
                        prompt_key=prompt_key,
                        castle_config=castle_config,
                        sample_limit=plan_sample_limit,
                        output_base_dir=str(plan_output_dir),
                    )

                    results["experiments"].append(experiment_result)

                    if experiment_result["status"] == "success":
                        successful_experiments += 1
                        logger.info("✓ Experiment completed successfully")
                    else:
                        failed_experiments += 1
                        logger.error(
                            f"✗ Experiment failed: {experiment_result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    failed_experiments += 1
                    error_result = {
                        "experiment_name": f"{model_key}_{dataset_key}_{prompt_key}",
                        "status": "failed",
                        "error": str(e),
                    }
                    results["experiments"].append(error_result)
                    logger.exception(f"✗ Experiment exception: {e}")

                # Brief pause between experiments
                time.sleep(2)

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
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Experiment plan completed: {successful_experiments}/{total_experiments} successful"
    )
    logger.info(f"Plan results saved to: {plan_results_file}")

    return results


def create_experiment_summary(results: dict[str, Any]) -> str:
    """
    Create a human-readable summary of experiment results.

    Args:
        results: Experiment results dictionary

    Returns:
        str: Formatted summary
    """
    summary_lines = [
        f"CASTLE Experiment Plan: {results['plan_name']}",
        f"Description: {results['description']}",
        f"Start Time: {results['start_time']}",
        f"End Time: {results['end_time']}",
        "",
        "Summary:",
        f"  Total Experiments: {results['summary']['total_experiments']}",
        f"  Successful: {results['summary']['successful']}",
        f"  Failed: {results['summary']['failed']}",
        f"  Success Rate: {results['summary']['success_rate']:.1%}",
        "",
        "Individual Experiments:",
    ]

    for exp in results["experiments"]:
        status_icon = "✓" if exp["status"] == "success" else "✗"
        line = f"  {status_icon} {exp['experiment_name']}: {exp['status']}"

        if exp["status"] == "success" and "results" in exp:
            accuracy = exp["results"].get("accuracy", "N/A")
            if accuracy != "N/A":
                line += f" (Accuracy: {accuracy:.3f})"
        elif exp["status"] == "failed":
            line += f" ({exp.get('error', 'Unknown error')})"

        summary_lines.append(line)

    return "\n".join(summary_lines)


def validate_datasets_exist(castle_config: dict[str, Any]) -> bool:
    """
    Validate that all required dataset files exist.

    Args:
        castle_config: CASTLE configuration

    Returns:
        bool: True if all datasets exist
    """
    logger = logging.getLogger(__name__)
    missing_datasets: list[Path] = []

    for _, dataset_config in castle_config["dataset_configurations"].items():
        dataset_path = Path(dataset_config["dataset_path"])
        if not dataset_path.exists():
            missing_datasets.append(dataset_path)

    if missing_datasets:
        logger.error("Missing dataset files:")
        for path in missing_datasets:
            logger.error(f"  - {path}")
        logger.error(
            "Run run_setup_castle_dataset.py first to create processed datasets"
        )
        return False

    return True


def main():
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
        "--sample-limit",
        type=int,
        help="Limit number of samples for testing (overrides plan setting)",
    )

    parser.add_argument(
        "--output-dir",
        default="results/castle_experiments",
        help="Base output directory for results",
    )

    parser.add_argument(
        "--validate-datasets",
        action="store_true",
        help="Validate that all required datasets exist",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

        castle_config = load_castle_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # List plans if requested
        if args.list_plans:
            print("\nAvailable experiment plans:")
            for plan_name, plan_config in castle_config["experiment_plans"].items():
                print(f"  {plan_name}: {plan_config['description']}")
                print(f"    Datasets: {', '.join(plan_config['datasets'])}")
                print(f"    Models: {', '.join(plan_config['models'])}")
                print(f"    Prompts: {', '.join(plan_config['prompts'])}")
                total_exp = (
                    len(plan_config["datasets"])
                    * len(plan_config["models"])
                    * len(plan_config["prompts"])
                )
                print(f"    Total experiments: {total_exp}")
                print()
            return

        # Validate datasets if requested
        if args.validate_datasets:
            if validate_datasets_exist(castle_config):
                logger.info("All required datasets found")
            else:
                sys.exit(1)
            return

        # Validate required arguments
        if not args.plan:
            logger.error("Experiment plan must be specified")
            logger.error("Use --list-plans to see available plans")
            sys.exit(1)

        # Validate datasets exist
        if not validate_datasets_exist(castle_config):
            sys.exit(1)

        # Run experiment plan
        logger.info(f"Starting experiment plan: {args.plan}")

        results = run_experiment_plan(
            plan_name=args.plan,
            castle_config=castle_config,
            output_base_dir=args.output_dir,
            sample_limit=args.sample_limit,
        )

        # Print summary
        summary = create_experiment_summary(results)
        print("\n" + "=" * 80)
        print(summary)
        print("=" * 80)

        if results["summary"]["failed"] > 0:
            logger.warning("Some experiments failed. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("All experiments completed successfully!")

    except Exception as e:
        logger.error(f"Experiment execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
