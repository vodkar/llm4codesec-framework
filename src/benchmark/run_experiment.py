import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _select_latest_reports_by_experiment_dir(
    report_paths: list[Path],
) -> list[Path]:
    """Pick the latest benchmark report file from each experiment directory."""
    reports_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for report_path in report_paths:
        reports_by_dir[report_path.parent].append(report_path)

    latest_reports: list[Path] = []
    for directory_path in sorted(reports_by_dir.keys()):
        files_in_directory: list[Path] = reports_by_dir[directory_path]
        latest_report: Path = max(
            files_in_directory, key=lambda file_path: file_path.name
        )
        latest_reports.append(latest_report)

    return latest_reports


def _read_benchmark_report(report_file_path: Path) -> BenchmarkReport:
    """Load a benchmark report JSON file into a typed report model."""
    with open(report_file_path, "r", encoding="utf-8") as report_file:
        payload: dict[str, Any] = json.load(report_file)
    return BenchmarkReport.model_validate(payload)


def _extract_report_timestamp(
    report: BenchmarkReport, report_file_path: Path
) -> datetime:
    """Extract timestamp from report metadata with fallback to file mtime."""
    raw_timestamp: str = report.benchmark_info.timestamp
    try:
        return datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return datetime.fromtimestamp(report_file_path.stat().st_mtime)


def rebuild_experiment_plan_results(
    input_path: Path | str,
    plan_name: str | None = None,
    description: str | None = None,
    output_file: Path | str | None = None,
) -> ExperimentPlanResult:
    """Rebuild experiment plan result JSON from existing benchmark report artifacts.

    Args:
        input_path: Directory containing experiment subfolders and benchmark report files.
        plan_name: Optional explicit plan name for the rebuilt result.
        description: Optional explicit plan description for the rebuilt result.
        output_file: Optional explicit output file path.

    Returns:
        Reconstructed experiment plan result object.

    Raises:
        FileNotFoundError: If input path does not exist.
        RuntimeError: If no benchmark report files are found.
    """
    base_dir: Path = Path(input_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {base_dir}")

    all_report_paths: list[Path] = sorted(base_dir.rglob("benchmark_report_*.json"))
    if len(all_report_paths) == 0:
        raise RuntimeError(
            f"No benchmark report files found under {base_dir}. Expected benchmark_report_*.json files."
        )

    selected_report_paths: list[Path] = _select_latest_reports_by_experiment_dir(
        all_report_paths
    )
    _LOGGER.info(
        "Found %d report files across %d experiment directories",
        len(all_report_paths),
        len(selected_report_paths),
    )

    report_entries: list[tuple[Path, BenchmarkReport, datetime]] = []
    for report_path in selected_report_paths:
        report: BenchmarkReport = _read_benchmark_report(report_path)
        report_timestamp: datetime = _extract_report_timestamp(report, report_path)
        report_entries.append((report_path, report, report_timestamp))

    report_entries.sort(
        key=lambda entry: str(entry[0].relative_to(base_dir).parent).lower()
    )

    short_reports: list[ShortExperimentReport] = [
        report.short_summary for _, report, _ in report_entries
    ]

    successful_experiments: int = sum(
        1 for short_report in short_reports if short_report.is_success
    )
    failed_experiments: int = len(short_reports) - successful_experiments

    inferred_names: set[str] = {
        report.benchmark_info.experiment_name
        for _, report, _ in report_entries
        if report.benchmark_info.experiment_name is not None
    }
    inferred_descriptions: set[str] = {
        report.benchmark_info.description for _, report, _ in report_entries
    }

    resolved_plan_name: str = (
        plan_name
        if plan_name is not None
        else next(iter(inferred_names))
        if len(inferred_names) == 1
        else base_dir.name
    )
    if len(inferred_names) > 1 and plan_name is None:
        _LOGGER.warning(
            "Multiple experiment names found in reports (%s). Using directory name '%s' as plan name.",
            ", ".join(sorted(inferred_names)),
            resolved_plan_name,
        )

    resolved_description: str = (
        description
        if description is not None
        else next(iter(inferred_descriptions))
        if len(inferred_descriptions) == 1
        else f"Rebuilt from benchmark reports in {base_dir}"
    )

    timestamps: list[datetime] = [timestamp for _, _, timestamp in report_entries]
    start_time: datetime = min(timestamps)
    end_time: datetime = max(timestamps)

    plan_result: ExperimentPlanResult = ExperimentPlanResult(
        plan_name=resolved_plan_name,
        description=resolved_description,
        start_time=start_time,
        end_time=end_time,
        experiments=short_reports,
        summary=ExperimentPlanSummary(
            total_experiments=len(short_reports),
            successful_experiments=successful_experiments,
            failed_experiments=failed_experiments,
        ),
        output_dir=str(base_dir),
    )

    output_file_path: Path = (
        Path(output_file)
        if output_file is not None
        else base_dir / "experiment_plan_results.json"
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(
        plan_result.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _LOGGER.info("Rebuilt plan results saved to: %s", output_file_path)
    _LOGGER.info(
        "Rebuilt summary: %d total, %d successful, %d failed",
        plan_result.summary.total_experiments,
        plan_result.summary.successful_experiments,
        plan_result.summary.failed_experiments,
    )

    return plan_result


def run_single_experiment(
    config: ExperimentConfig, runner: BenchmarkRunner | None = None
) -> BenchmarkReport:
    """
    Run a single benchmark experiment.
    """

    # Initialize and run benchmark
    if not runner:
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
        f"Experiment Plan: {results.plan_name}",
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
    config: Path | str | dict[str, Any],
    output_base_dir: Path,
    skip_existing: bool = False,
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
    plan_output_dir = output_base_dir / plan_name
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    # Calculate total experiments
    total_experiments = len(plan.experiments)
    _LOGGER.info(f"Total experiments to run: {total_experiments}")

    experiment_count = 0
    successful_experiments = 0
    failed_experiments = 0

    dataset_loader = JsonDatasetLoader()

    # Run all experiments
    experiments: list[ShortExperimentReport] = []
    for experiment_config in plan.experiments:
        experiment_count += 1

        _LOGGER.info(
            f"Experiment {experiment_count}/{total_experiments}: {experiment_config.model_name} + "
            f"{experiment_config.dataset_name} + {experiment_config.prompt_identifier}"
        )

        if skip_existing and any(experiment_config.output_dir.glob("benchmark_report_*.json")):
            _LOGGER.info(
                "Skipping experiment %d/%d (results already exist): %s",
                experiment_count,
                total_experiments,
                experiment_config.output_dir,
            )
            # Count as successful since a report exists
            successful_experiments += 1
            continue

        try:
            result = run_single_experiment(
                experiment_config,
                BenchmarkRunner(config=experiment_config, dataset_loader=dataset_loader),
            )
        except Exception as exc:
            failed_experiments += 1
            _LOGGER.error(
                "✗ Experiment %d/%d failed with unhandled exception: %s",
                experiment_count,
                total_experiments,
                exc,
                exc_info=True,
            )
            continue

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
    plan_results_file.write_text(
        plan_result.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _LOGGER.info(
        f"Experiment plan completed: {successful_experiments}/{total_experiments} successful"
    )
    _LOGGER.info(f"Plan results saved to: {plan_results_file}")

    return plan_result
