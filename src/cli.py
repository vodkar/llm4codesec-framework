"""Unified Typer CLI for benchmark experiments."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import typer

from analysis.reference_context.acceptance import CheckResult, run_all
from analysis.reference_context.inputs import (
    AnalysisInputs,
    Condition,
    ItemRecord,
    load_inputs,
)
from analysis.reference_context.metrics import (
    ClusterBootstrap,
    ConditionTable,
    condition_table,
    decomposition,
    length_stats,
    stratum_tables_with_redraws,
)
from analysis.reference_context.report import (
    BootstrapRedraws,
    write_invalid_report,
    write_report,
)
from analysis.reference_context.run_config import ReferenceRunConfig
from benchmark.config import ExperimentConfig
from benchmark.run_experiment import (
    create_experiment_summary,
    rebuild_experiment_plan_results,
    run_experiment_plan,
    run_single_experiment,
)
from consts import CONFIG_DIRECTORY
from entrypoints.utils import compose_benchmark_config, log_available_configurations
from entrypoints.utils import list_plans as log_plans
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)
BASE_RESULTS_DIR = Path("results")
_FRAMEWORK_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_FRAMEWORK_DIRTY_SCOPE: Final[tuple[str, ...]] = ("src", "config", "scripts")


@dataclass(frozen=True)
class BenchmarkCliConfig:
    """Configuration metadata for benchmark CLI routing."""

    name: str
    config_file: Path
    output_dir: Path


BENCHMARKS: dict[str, BenchmarkCliConfig] = {
    "castle": BenchmarkCliConfig(
        name="castle",
        config_file=CONFIG_DIRECTORY / "castle_experiments.json",
        output_dir=BASE_RESULTS_DIR / "castle_experiments",
    ),
    "cvefixes": BenchmarkCliConfig(
        name="cvefixes",
        config_file=CONFIG_DIRECTORY / "cvefixes_experiments.json",
        output_dir=BASE_RESULTS_DIR / "cvefixes_experiments",
    ),
    "jitvul": BenchmarkCliConfig(
        name="jitvul",
        config_file=CONFIG_DIRECTORY / "jitvul_experiments.json",
        output_dir=BASE_RESULTS_DIR / "jitvul_experiments",
    ),
    "vulbench": BenchmarkCliConfig(
        name="vulbench",
        config_file=CONFIG_DIRECTORY / "vulbench_experiments.json",
        output_dir=BASE_RESULTS_DIR / "vulbench_experiments",
    ),
    "context_assembler": BenchmarkCliConfig(
        name="context_assembler",
        config_file=CONFIG_DIRECTORY / "context_assembler_experiments.json",
        output_dir=BASE_RESULTS_DIR / "context_assembler_experiments",
    ),
    "primevul": BenchmarkCliConfig(
        name="primevul",
        config_file=CONFIG_DIRECTORY / "primevul_experiments.json",
        output_dir=BASE_RESULTS_DIR / "primevul_experiments",
    ),
    "cleanvul": BenchmarkCliConfig(
        name="cleanvul",
        config_file=CONFIG_DIRECTORY / "cleanvul_experiments.json",
        output_dir=BASE_RESULTS_DIR / "cleanvul_experiments",
    ),
}

app = typer.Typer(
    help="Unified benchmark CLI for experiments and plans.",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


def _configure_logging(verbose: bool, log_level: str) -> None:
    """Configure application logging once per command execution."""
    setup_logging(verbose=verbose)

    level_name: str = log_level.upper()
    if level_name not in logging.getLevelNamesMapping():
        raise typer.BadParameter(
            f"Invalid log level: {log_level}. Use DEBUG, INFO, WARNING, ERROR or CRITICAL."
        )

    logging.getLogger().setLevel(logging.getLevelNamesMapping()[level_name])


def _get_benchmark(name: str) -> BenchmarkCliConfig:
    """Get benchmark metadata by benchmark name."""
    benchmark: BenchmarkCliConfig | None = BENCHMARKS.get(name)
    if benchmark is None:
        raise typer.BadParameter(
            f"Unknown benchmark '{name}'. Available: {', '.join(sorted(BENCHMARKS.keys()))}"
        )
    return benchmark


def _resolve_scanner_out(config: ReferenceRunConfig) -> Path:
    """Resolve llm_scanner's ``output_dir`` against the pinned llm_scanner checkout.

    Args:
        config: The loaded reference-context run configuration.

    Returns:
        ``llm_scanner.output_dir`` as-is when absolute, otherwise resolved
        against ``pins.llm_scanner_path``.

    Raises:
        ValueError: If ``llm_scanner.output_dir`` is missing or not a string.
    """

    raw_output_dir: object = config.llm_scanner.get("output_dir")
    if not isinstance(raw_output_dir, str):
        raise ValueError(
            "config.llm_scanner.output_dir must be a string path, "
            f"got {raw_output_dir!r}"
        )
    output_dir = Path(raw_output_dir)
    if output_dir.is_absolute():
        return output_dir
    return config.pins.llm_scanner_path / output_dir


def _items_by_condition(inputs: AnalysisInputs) -> dict[Condition, list[ItemRecord]]:
    """Group aligned items by condition, dropping conditions with no items."""

    grouped: dict[Condition, list[ItemRecord]] = {
        condition: [item for item in inputs.items if item.condition is condition]
        for condition in Condition
    }
    return {condition: items for condition, items in grouped.items() if items}


def _framework_git_state() -> tuple[str, bool]:
    """Read the framework repository's HEAD SHA and working-tree cleanliness.

    Always targets :data:`_FRAMEWORK_REPO_ROOT` (derived from this module's
    own path) via ``git -C``, so the result does not depend on the caller's
    current working directory. Never mutates git state: runs only
    ``git rev-parse HEAD`` and ``git status --porcelain --untracked-files=no
    -- src config scripts``. The dirty scope matches the evaluation script's
    preflight: untracked files (e.g. run results) and files outside
    :data:`_FRAMEWORK_DIRTY_SCOPE` never make the state dirty.

    Returns:
        ``(framework_sha, framework_dirty)``.

    Raises:
        RuntimeError: If either git command fails or ``git`` is unavailable.
    """

    repo_root: str = str(_FRAMEWORK_REPO_ROOT)
    try:
        sha_result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                repo_root,
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                *_FRAMEWORK_DIRTY_SCOPE,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to read framework git state") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found") from exc
    return sha_result.stdout.strip(), bool(status_result.stdout.strip())


def _resolve_benchmark_config(
    benchmark_name: str,
    benchmark: BenchmarkCliConfig,
    config: str | None,
    config_dir: str | None,
    experiments_config: str | None,
    datasets_config: str | None,
) -> dict[str, Any]:
    """Resolve benchmark config data from monolithic and split config sources."""
    base_config: str | Path | None = (
        config if config is not None else benchmark.config_file
    )
    return compose_benchmark_config(
        benchmark_name=benchmark_name,
        base_config=base_config,
        config_directory=config_dir,
        experiments_config=experiments_config,
        datasets_config=datasets_config,
    )


@app.command("run")
def run(
    benchmark: str = typer.Argument(
        ..., help="Benchmark name (castle/cvefixes/jitvul/vulbench)."
    ),
    model: str = typer.Option(..., "--model", help="Model key in experiment config."),
    dataset: str = typer.Option(
        ..., "--dataset", help="Dataset key in experiment config."
    ),
    prompt: str = typer.Option(
        ..., "--prompt", help="Prompt key in experiment config."
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to experiment config file."
    ),
    config_dir: str | None = typer.Option(
        "src/configs/shared",
        "--config-dir",
        help="Directory containing split config files (models.json/prompts.json/etc).",
    ),
    experiments_config: str | None = typer.Option(
        None,
        "--experiments-config",
        help="Path to experiments config file (overrides config-dir discovery).",
    ),
    datasets_config: str | None = typer.Option(
        None,
        "--datasets-config",
        help="Path to datasets config file (overrides config-dir discovery).",
    ),
    sample_limit: int | None = typer.Option(
        None,
        "--sample-limit",
        help="Limit number of samples for testing.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """Run a single benchmark experiment."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_data: dict[str, Any] = _resolve_benchmark_config(
        benchmark_name=benchmark,
        benchmark=benchmark_config,
        config=config,
        config_dir=config_dir,
        experiments_config=experiments_config,
        datasets_config=datasets_config,
    )

    _LOGGER.info("Running benchmark '%s'", benchmark)
    experiment_config: ExperimentConfig = ExperimentConfig.from_file(
        config=config_data,
        model_key=model,
        dataset_key=dataset,
        prompt_key=prompt,
        experiment_name="manual",
        sample_limit=sample_limit,
    )

    result = run_single_experiment(config=experiment_config)
    if not result.is_success:
        _LOGGER.error("Experiment failed: %s", result.benchmark_info.experiment_name)
        raise typer.Exit(1)

    _LOGGER.info("Experiment completed: %s", result.benchmark_info.experiment_name)


@app.command("run-plan")
def run_plan(
    benchmark: str = typer.Argument(
        ..., help="Benchmark name (castle/cvefixes/jitvul/vulbench)."
    ),
    plan: str = typer.Argument(..., help="Experiment plan name."),
    config: str | None = typer.Option(
        None, "--config", help="Path to experiment config file."
    ),
    config_dir: str | None = typer.Option(
        "src/configs/shared",
        "--config-dir",
        help="Directory containing split config files (models.json/prompts.json/etc).",
    ),
    experiments_config: str | None = typer.Option(
        None,
        "--experiments-config",
        help="Path to experiments config file (overrides config-dir discovery).",
    ),
    datasets_config: str | None = typer.Option(
        None,
        "--datasets-config",
        help="Path to datasets config file (overrides config-dir discovery).",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory base for plan results.",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip experiments whose output dir already has a benchmark report.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """Run an experiment plan for a benchmark."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_data: dict[str, Any] = _resolve_benchmark_config(
        benchmark_name=benchmark,
        benchmark=benchmark_config,
        config=config,
        config_dir=config_dir,
        experiments_config=experiments_config,
        datasets_config=datasets_config,
    )
    selected_output_dir: Path | None = Path(output_dir) if output_dir else None

    _LOGGER.info("Running plan '%s' for benchmark '%s'", plan, benchmark)
    results = run_experiment_plan(
        plan_name=plan,
        config=config_data,
        output_base_dir=selected_output_dir,
        skip_existing=skip_existing,
    )

    summary: str = create_experiment_summary(results)
    for line in summary.splitlines():
        _LOGGER.info(line)

    if results.summary.failed_experiments > 0:
        _LOGGER.warning("Some experiments failed. Check logs for details.")
        raise typer.Exit(1)

    _LOGGER.info("All experiments completed successfully")


@app.command("list-plans")
def list_available_plans(
    benchmark: str = typer.Argument(
        ..., help="Benchmark name (castle/cvefixes/jitvul/vulbench)."
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to experiment config file."
    ),
    config_dir: str | None = typer.Option(
        None,
        "--config-dir",
        help="Directory containing split config files (models.json/prompts.json/etc).",
    ),
    experiments_config: str | None = typer.Option(
        None,
        "--experiments-config",
        help="Path to experiments config file (overrides config-dir discovery).",
    ),
    datasets_config: str | None = typer.Option(
        None,
        "--datasets-config",
        help="Path to datasets config file (overrides config-dir discovery).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """List available experiment plans."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_data: dict[str, Any] = _resolve_benchmark_config(
        benchmark_name=benchmark,
        benchmark=benchmark_config,
        config=config,
        config_dir=config_dir,
        experiments_config=experiments_config,
        datasets_config=datasets_config,
    )
    log_plans(config_data, logger=_LOGGER)


@app.command("list-configs")
def list_available_configs(
    benchmark: str = typer.Argument(
        ..., help="Benchmark name (castle/cvefixes/jitvul/vulbench)."
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to experiment config file."
    ),
    config_dir: str | None = typer.Option(
        None,
        "--config-dir",
        help="Directory containing split config files (models.json/prompts.json/etc).",
    ),
    experiments_config: str | None = typer.Option(
        None,
        "--experiments-config",
        help="Path to experiments config file (overrides config-dir discovery).",
    ),
    datasets_config: str | None = typer.Option(
        None,
        "--datasets-config",
        help="Path to datasets config file (overrides config-dir discovery).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """List available models, datasets, prompts, and plans."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_data: dict[str, Any] = _resolve_benchmark_config(
        benchmark_name=benchmark,
        benchmark=benchmark_config,
        config=config,
        config_dir=config_dir,
        experiments_config=experiments_config,
        datasets_config=datasets_config,
    )
    log_available_configurations(config_data, logger=_LOGGER)


@app.command("rebuild-plan-results")
def rebuild_plan_results(
    input_path: str = typer.Argument(
        ..., help="Path to plan output directory containing experiment subfolders."
    ),
    plan_name: str | None = typer.Option(
        None, "--plan-name", help="Override rebuilt plan name."
    ),
    description: str | None = typer.Option(
        None, "--description", help="Override rebuilt plan description."
    ),
    output_file: str | None = typer.Option(
        None,
        "--output-file",
        help="Output JSON file path (default: <input_path>/experiment_plan_results.json).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """Rebuild experiment_plan_results.json from existing benchmark report files."""
    _configure_logging(verbose=verbose, log_level=log_level)

    rebuilt_result = rebuild_experiment_plan_results(
        input_path=input_path,
        plan_name=plan_name,
        description=description,
        output_file=output_file,
    )

    summary: str = create_experiment_summary(rebuilt_result)
    for line in summary.splitlines():
        _LOGGER.info(line)


@app.command("analyze-reference-context")
def analyze_reference_context(
    config: str = typer.Option(
        ..., "--config", help="Path to the reference-context run config YAML."
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        help="Directory to write report.md/report.json/acceptance.json into.",
    ),
    plan: str | None = typer.Option(
        None, "--plan", help="Override framework.plan from the config."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """Run the reference-context oracle analysis and write its report.

    Loads the pinned run config, aligns llm_scanner's per-item artifacts
    with the framework's five-condition benchmark reports, builds the
    repository cluster bootstrap and metrics tables, runs every acceptance
    check (design spec §11), and writes ``report.md``/``report.json``/
    ``acceptance.json`` into ``--output-dir``. Exits with code 1 when the
    run is invalid, after every artifact has been written; a run with no
    surviving pair gets a **RUN INVALID** report and exits 1 as well.
    Mixed or stale condition reports raise ``ValueError`` before analysis.
    """
    _configure_logging(verbose=verbose, log_level=log_level)

    config_path = Path(config)
    run_config: ReferenceRunConfig = ReferenceRunConfig.from_yaml(config_path)
    plan_name: str = plan if plan is not None else run_config.framework.plan

    results_dir: Path = run_config.framework.results_dir / plan_name
    scanner_out: Path = _resolve_scanner_out(run_config)

    _LOGGER.info(
        "Loading reference-context analysis inputs for plan '%s' from %s",
        plan_name,
        results_dir,
    )
    inputs: AnalysisInputs = load_inputs(results_dir, scanner_out, _FRAMEWORK_REPO_ROOT)

    config_sha256: str = ReferenceRunConfig.sha256(config_path)
    framework_sha, framework_dirty = _framework_git_state()
    manifest_extra: dict[str, object] = {
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "framework_sha": framework_sha,
        "framework_dirty": framework_dirty,
        "plan": plan_name,
        "bootstrap_resamples": run_config.framework.bootstrap_resamples,
        "bootstrap_seed": run_config.framework.bootstrap_seed,
        "probe_seed": run_config.framework.probe_seed,
        "mismatch_tolerance": run_config.framework.mismatch_tolerance,
    }

    if not inputs.pair_ids:
        note: str = (
            "No pair survived into the analysis (see the coverage funnel and "
            "inference-time exclusions); nothing can be evaluated."
        )
        report_path_invalid: Path = write_invalid_report(
            Path(output_dir), inputs, note, manifest_extra
        )
        _LOGGER.error(
            "Reference-context run is INVALID: %s See %s", note, report_path_invalid
        )
        raise typer.Exit(1)

    bootstrap = ClusterBootstrap(
        _items_by_condition(inputs),
        run_config.framework.bootstrap_resamples,
        run_config.framework.bootstrap_seed,
    )
    tables: ConditionTable = condition_table(inputs, bootstrap)
    strata, strata_redraws = stratum_tables_with_redraws(
        inputs,
        run_config.framework.bootstrap_resamples,
        run_config.framework.bootstrap_seed,
    )
    lengths = length_stats(inputs)
    decomp = decomposition(bootstrap)
    redraws = BootstrapRedraws(overall=bootstrap.redraws, strata=strata_redraws)
    manifest_extra["bootstrap_redraws"] = redraws.to_json()

    checks: list[CheckResult] = run_all(
        inputs, bootstrap, run_config, config_sha256, framework_sha, framework_dirty
    )

    report_path: Path = write_report(
        Path(output_dir),
        inputs,
        tables,
        strata,
        lengths,
        decomp,
        checks,
        manifest_extra,
        bootstrap_redraws=redraws,
    )
    _LOGGER.info("Wrote reference-context report to %s", report_path)

    if not all(check.passed for check in checks):
        failed: list[str] = [check.name for check in checks if not check.passed]
        _LOGGER.error(
            "Reference-context run is INVALID (failed: %s); see %s",
            ", ".join(failed),
            report_path,
        )
        raise typer.Exit(1)

    _LOGGER.info("Reference-context run is valid")


if __name__ == "__main__":
    app()
