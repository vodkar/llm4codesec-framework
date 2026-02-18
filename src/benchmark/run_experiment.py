import json
import logging
from datetime import datetime
from pathlib import Path

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.config import ExperimentConfig, ExperimentsPlanConfig
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import (
    BenchmarkReport,
    ExperimentPlanResult,
    ExperimentPlanSummary,
    ShortExperimentReport,
)
from datasets.loaders.base import JsonDatasetLoader

_LOGGER = logging.getLogger(__name__)


def run_single_experiment(config: ExperimentConfig) -> BenchmarkReport:
    """
    Run a single benchmark experiment.
    """

    # Initialize and run benchmark
    runner = BenchmarkRunner(
        config=config,
        dataset_loader=JsonDatasetLoader(),
    )

    result = runner.run()

    result_processor = BenchmarkResultProcessor(config=config)
    report, artifacts = result_processor.build_and_save(
        metrics=result.metrics,
        predictions=result.predictions,
        total_time=result.total_time,
        total_samples=result.total_samples,
    )

    _LOGGER.info("Experiment completed successfully")
    _LOGGER.info(f"Accuracy: {report.metrics.accuracy:.3f}")
    _LOGGER.info(f"Results saved to: {artifacts.report_json}")

    return report


def create_experiment_summary(results: ExperimentPlanResult) -> str:
    """
    Create a human-readable summary of experiment results.

    Args:
        results: Experiment results object

    Returns:
        str: Formatted summary
    """
    summary_lines = [
        f"CASTLE Experiment Plan: {results.plan_name}",
        f"Description: {results.description}",
        f"Start Time: {results.start_time}",
        f"End Time: {results.end_time}",
        "",
        "Summary:",
        f"  Total Experiments: {results.summary.total_experiments}",
        f"  Successful: {results.summary.successful_experiments}",
        f"  Failed: {results.summary.failed_experiments}",
        f"  Success Rate: {results.summary.success_rate:.1f}%",
        "",
        "Individual Experiments:",
    ]

    for exp in results.experiments:
        status_icon = "✓" if exp.is_success else "✗"
        line = f"  {status_icon} {exp.benchmark_info.experiment_name}"

        if exp.is_success:
            accuracy = exp.metrics.accuracy if exp.metrics else "N/A"
            if accuracy != "N/A":
                line += f" (Accuracy: {accuracy:.3f})"
        elif not exp.is_success:
            line += f" ({exp.metrics.details.get('error', 'Unknown error') if exp.metrics else 'Unknown error'})"

        summary_lines.append(line)

    return "\n".join(summary_lines)


def run_experiment_plan(
    plan_name: str,
    config: Path,
    output_base_dir: str,
) -> ExperimentPlanResult:
    """
    Run a complete experiment plan with multiple configurations.

    Args:
        plan_name: Name of the experiment plan to run
        config: Path to the experiment configuration file
        output_base_dir: Base output directory
        sample_limit: Limit samples for testing

    Returns:
        dict containing all experiment results
    """

    plan = ExperimentsPlanConfig.from_file(config, plan_name)
    _LOGGER.info(f"Starting experiment plan: {plan_name}")
    _LOGGER.info(f"Description: {plan.description}")

    # Create plan-specific output directory
    plan_output_dir = (
        Path(output_base_dir)
        / f"plan_{plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    # Calculate total experiments
    total_experiments = len(plan.experiments)
    _LOGGER.info(f"Total experiments to run: {total_experiments}")

    experiment_count = 0
    successful_experiments = 0
    failed_experiments = 0

    # Run all experiments
    experiments: list[ShortExperimentReport] = []
    for experiment_config in plan.experiments:
        experiment_count += 1

        _LOGGER.info(
            f"Experiment {experiment_count}/{total_experiments}: {experiment_config.model_name} + "
            f"{experiment_config.dataset_name} + {experiment_config.system_prompt_template}"
        )

        result = run_single_experiment(experiment_config)

        experiments.append(result.short_summary)

        if result.is_success:
            successful_experiments += 1
            _LOGGER.info("✓ Experiment completed successfully")
        else:
            failed_experiments += 1
            _LOGGER.error("✗ Experiment failed, check logs for details")

    # Complete results
    plan_result = ExperimentPlanResult(
        plan_name=plan_name,
        description=plan.description,
        start_time=start_time,
        end_time=datetime.now(),
        experiments=experiments,
        summary=ExperimentPlanSummary(
            total_experiments=total_experiments,
            successful_experiments=successful_experiments,
            failed_experiments=failed_experiments,
        ),
        output_dir=str(plan_output_dir),
    )

    # Save plan results
    plan_results_file = plan_output_dir / "experiment_plan_results.json"
    with open(plan_results_file, "w", encoding="utf-8") as f:
        json.dump(plan_result.model_dump(), f, indent=2, ensure_ascii=False)

    _LOGGER.info(
        f"Experiment plan completed: {successful_experiments}/{total_experiments} successful"
    )
    _LOGGER.info(f"Plan results saved to: {plan_results_file}")

    return plan_result
