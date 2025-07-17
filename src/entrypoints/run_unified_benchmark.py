#!/usr/bin/env python3
"""
Unified Benchmark Runner

A unified script for running LLM benchmarks on CASTLE, JitVul, CVEFixes, and VulBench datasets
with consistent CLI interface and experiment configuration patterns.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from entrypoints.run_castle_benchmark import load_castle_config
from entrypoints.run_castle_benchmark import (
    run_single_experiment as run_castle_experiment,
)
from entrypoints.run_cvefixes_benchmark import load_cvefixes_config
from entrypoints.run_cvefixes_benchmark import (
    run_single_experiment as run_cvefixes_experiment,
)
from entrypoints.run_jitvul_benchmark import load_jitvul_config
from entrypoints.run_jitvul_benchmark import (
    run_single_experiment as run_jitvul_experiment,
)
from entrypoints.run_vulbench_benchmark import load_vulbench_config
from entrypoints.run_vulbench_benchmark import (
    run_single_experiment as run_vulbench_experiment,
)
from entrypoints.run_vuldetectbench_benchmark import load_vuldetectbench_config
from entrypoints.run_vuldetectbench_benchmark import (
    run_single_experiment as run_vuldetectbench_experiment,
)
from entrypoints.run_vulnerabilitydetection_benchmark import (
    load_vulnerabilitydetection_config,
)
from entrypoints.run_vulnerabilitydetection_benchmark import (
    run_single_experiment as run_vulnerabilitydetection_experiment,
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def run_experiment_plan(
    dataset_type: str,
    plan_name: str,
    config: Dict[str, Any],
    output_base_dir: str,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a complete experiment plan with multiple configurations.

    Args:
        dataset_type: Type of dataset ('castle', 'jitvul', 'cvefixes', 'vulbench')
        plan_name: Name of the experiment plan to run
        config: Experiment configuration
        output_base_dir: Base output directory
        sample_limit: Limit samples for testing

    Returns:
        Dict containing all experiment results
    """
    logger = logging.getLogger(__name__)

    if plan_name not in config["experiment_plans"]:
        raise ValueError(
            f"Experiment plan '{plan_name}' not found in {dataset_type} config"
        )

    plan = config["experiment_plans"][plan_name]
    logger.info(f"Starting experiment plan: {plan_name} for {dataset_type}")
    logger.info(f"Description: {plan['description']}")

    # Override sample limit if specified in plan
    plan_sample_limit = plan.get("sample_limit", sample_limit)

    # Create plan-specific output directory
    plan_output_dir = (
        Path(output_base_dir)
        / f"{dataset_type}_plan_{plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset_type": dataset_type,
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
                    f"Running experiment {experiment_count}/{total_experiments}"
                )

                try:
                    # Choose the appropriate runner function
                    if dataset_type == "castle":
                        result = run_castle_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            config,
                            plan_sample_limit,
                            str(plan_output_dir),
                        )
                    elif dataset_type == "jitvul":
                        result = run_jitvul_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            config,
                            plan_sample_limit,
                            str(plan_output_dir),
                        )
                    elif dataset_type == "cvefixes":
                        result = run_cvefixes_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            config,
                            plan_sample_limit,
                            str(plan_output_dir),
                        )
                    elif dataset_type == "vulbench":
                        result = run_vulbench_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            config,
                            plan_sample_limit,
                            str(plan_output_dir),
                        )
                    elif dataset_type == "vuldetectbench":
                        result = run_vuldetectbench_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            config,
                            plan_sample_limit,
                            str(plan_output_dir),
                        )
                    elif dataset_type == "vulnerabilitydetection":
                        result = run_vulnerabilitydetection_experiment(
                            model_key,
                            dataset_key,
                            prompt_key,
                            config,
                            plan_sample_limit,
                            str(plan_output_dir),
                        )
                    else:
                        raise ValueError(f"Unknown dataset type: {dataset_type}")

                    if result["status"] == "success":
                        successful_experiments += 1
                    else:
                        failed_experiments += 1

                    results["experiments"].append(result)

                except Exception as e:
                    logger.exception(
                        f"Experiment failed: {model_key}_{dataset_key}_{prompt_key}"
                    )
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


def load_config(dataset_type: str, config_path: str) -> Dict[str, Any]:
    """Load configuration based on dataset type."""
    if dataset_type == "castle":
        return load_castle_config(config_path)
    elif dataset_type == "jitvul":
        return load_jitvul_config(config_path)
    elif dataset_type == "cvefixes":
        return load_cvefixes_config(config_path)
    elif dataset_type == "vulbench":
        return load_vulbench_config(config_path)
    elif dataset_type == "vuldetectbench":
        return load_vuldetectbench_config(config_path)
    elif dataset_type == "vulnerabilitydetection":
        return load_vulnerabilitydetection_config(config_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def run_single_experiment_unified(
    dataset_type: str,
    model_key: str,
    dataset_key: str,
    prompt_key: str,
    config: Dict[str, Any],
    sample_limit: Optional[int] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Run a single experiment for any dataset type."""
    if dataset_type == "castle":
        return run_castle_experiment(
            model_key, dataset_key, prompt_key, config, sample_limit, output_dir
        )
    elif dataset_type == "jitvul":
        return run_jitvul_experiment(
            model_key, dataset_key, prompt_key, config, sample_limit, output_dir
        )
    elif dataset_type == "cvefixes":
        return run_cvefixes_experiment(
            model_key, dataset_key, prompt_key, config, sample_limit, output_dir
        )
    elif dataset_type == "vulbench":
        return run_vulbench_experiment(
            model_key, dataset_key, prompt_key, config, sample_limit, output_dir
        )
    elif dataset_type == "vuldetectbench":
        return run_vuldetectbench_experiment(
            model_key, dataset_key, prompt_key, config, sample_limit, output_dir
        )
    elif dataset_type == "vulnerabilitydetection":
        return run_vulnerabilitydetection_experiment(
            model_key, dataset_key, prompt_key, config, sample_limit, output_dir
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for CASTLE, JitVul, CVEFixes, VulBench, VulDetectBench, and VulnerabilityDetection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment plan for CASTLE
  python run_unified_benchmark.py castle --plan quick_test
  
  # Run single experiment for JitVul
  python run_unified_benchmark.py jitvul --model qwen3-4b --dataset binary_all --prompt basic_security
  
  # Run experiment plan for CVEFixes with sample limit
  python run_unified_benchmark.py cvefixes --plan model_comparison --sample-limit 100
  
  # Run experiment plan for VulBench
  python run_unified_benchmark.py vulbench --plan quick_test
  
  # Run experiment plan for VulDetectBench
  python run_unified_benchmark.py vuldetectbench --plan task1_evaluation
  
  # Run experiment plan for VulnerabilityDetection
  python run_unified_benchmark.py vulnerabilitydetection --plan quick_test
  
  # Run single experiment for VulnerabilityDetection
  python run_unified_benchmark.py vulnerabilitydetection --model qwen2.5-7b --dataset vulnerabilitydetection_binary --prompt basic_security
  
  # List available configurations
  python run_unified_benchmark.py castle --list-configs
        """,
    )

    parser.add_argument(
        "dataset_type",
        choices=[
            "castle",
            "jitvul",
            "cvefixes",
            "vulbench",
            "vuldetectbench",
            "vulnerabilitydetection",
        ],
        help="Type of dataset to run benchmark on",
    )

    parser.add_argument(
        "--config",
        help="Path to experiment configuration file (auto-detected if not provided)",
    )

    parser.add_argument("--model", help="Model configuration to use")

    parser.add_argument("--dataset", help="Dataset configuration to use")

    parser.add_argument("--prompt", help="Prompt strategy to use")

    parser.add_argument("--plan", help="Experiment plan to run")

    parser.add_argument(
        "--sample-limit", type=int, help="Limit number of samples for testing"
    )

    parser.add_argument(
        "--output-dir", default="results", help="Base output directory for results"
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
        # Auto-detect config file if not provided
        if not args.config:
            config_files = {
                "castle": "castle_experiments.json",
                "jitvul": "jitvul_experiments.json",
                "cvefixes": "cvefixes_experiments.json",
                "vulbench": "vulbench_experiments.json",
                "vuldetectbench": "vuldetectbench_experiments.json",
                "vulnerabilitydetection": "vulnerabilitydetection_experiments.json",
            }
            args.config = config_files[args.dataset_type]

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

        config = load_config(args.dataset_type, str(config_path))

        # List configurations if requested
        if args.list_configs:
            print(f"\nAvailable configurations for {args.dataset_type}:")
            print("=" * 50)
            print("Models:", list(config["model_configurations"].keys()))
            print("Datasets:", list(config["dataset_configurations"].keys()))
            print("Prompts:", list(config["prompt_strategies"].keys()))
            print("Plans:", list(config["experiment_plans"].keys()))
            return

        # Run experiment plan
        if args.plan:
            result = run_experiment_plan(
                args.dataset_type, args.plan, config, args.output_dir, args.sample_limit
            )

            print(f"\nExperiment plan '{args.plan}' completed for {args.dataset_type}")
            print(f"Success rate: {result['summary']['success_rate']:.1%}")
            print(f"Results saved to: {result['output_dir']}")

        # Run single experiment
        elif args.model and args.dataset and args.prompt:
            result = run_single_experiment_unified(
                args.dataset_type,
                args.model,
                args.dataset,
                args.prompt,
                config,
                args.sample_limit,
                args.output_dir,
            )

            if result["status"] == "success":
                print(f"Experiment completed successfully: {result['experiment_name']}")
                print(f"Accuracy: {result['accuracy']:.3f}")
                print(f"Results saved to: {result['output_dir']}")
            else:
                print(f"Experiment failed: {result['experiment_name']}")
                print(f"Error: {result['error']}")
                sys.exit(1)

        else:
            parser.print_help()
            print(
                f"\nUse --list-configs to see available configurations for {args.dataset_type}"
            )

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
