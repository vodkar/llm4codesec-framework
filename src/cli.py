"""Unified Typer CLI for benchmark experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import typer

from benchmark.config import ExperimentConfig
from benchmark.run_experiment import (
    create_experiment_summary,
    run_experiment_plan,
    run_single_experiment,
)
from consts import CONFIG_DIRECTORY
from entrypoints.utils import list_plans as log_plans
from entrypoints.utils import (
    log_available_configurations,
    resolve_config_path,
)
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkCliConfig:
    """Configuration metadata for benchmark CLI routing."""

    name: str
    config_file: Path
    output_dir: str


BENCHMARKS: dict[str, BenchmarkCliConfig] = {
    "castle": BenchmarkCliConfig(
        name="castle",
        config_file=CONFIG_DIRECTORY / "castle_experiments.json",
        output_dir="results/castle_experiments",
    ),
    "cvefixes": BenchmarkCliConfig(
        name="cvefixes",
        config_file=CONFIG_DIRECTORY / "cvefixes_experiments.json",
        output_dir="results/cvefixes_experiments",
    ),
    "jitvul": BenchmarkCliConfig(
        name="jitvul",
        config_file=CONFIG_DIRECTORY / "jitvul_experiments.json",
        output_dir="results/jitvul_experiments",
    ),
    "vulbench": BenchmarkCliConfig(
        name="vulbench",
        config_file=CONFIG_DIRECTORY / "vulbench_experiments.json",
        output_dir="results/vulbench_experiments",
    ),
}

app = typer.Typer(help="Unified benchmark CLI for experiments and plans.")


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


def _resolve_benchmark_config(
    benchmark: BenchmarkCliConfig, config: str | None
) -> Path:
    """Resolve config path from explicit value or benchmark defaults."""
    selected: Path = benchmark.config_file if config is None else Path(config)
    resolved: Path = resolve_config_path(selected)
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {selected}")
    return resolved


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
    config_path: Path = _resolve_benchmark_config(benchmark_config, config)

    _LOGGER.info("Running benchmark '%s' from config %s", benchmark, config_path)
    experiment_config: ExperimentConfig = ExperimentConfig.from_file(
        config=config_path,
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
    plan: str = typer.Option(..., "--plan", help="Experiment plan name."),
    config: str | None = typer.Option(
        None, "--config", help="Path to experiment config file."
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory base for plan results.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """Run an experiment plan for a benchmark."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_path: Path = _resolve_benchmark_config(benchmark_config, config)
    selected_output_dir: str = output_dir or benchmark_config.output_dir

    _LOGGER.info("Running plan '%s' for benchmark '%s'", plan, benchmark)
    results = run_experiment_plan(
        plan_name=plan,
        config=config_path,
        output_base_dir=selected_output_dir,
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """List available experiment plans."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_path: Path = _resolve_benchmark_config(benchmark_config, config)
    log_plans(config_path, logger=_LOGGER)


@app.command("list-configs")
def list_available_configs(
    benchmark: str = typer.Argument(
        ..., help="Benchmark name (castle/cvefixes/jitvul/vulbench)."
    ),
    config: str | None = typer.Option(
        None, "--config", help="Path to experiment config file."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """List available models, datasets, prompts, and plans."""
    _configure_logging(verbose=verbose, log_level=log_level)

    benchmark_config: BenchmarkCliConfig = _get_benchmark(benchmark)
    config_path: Path = _resolve_benchmark_config(benchmark_config, config)
    log_available_configurations(config_path, logger=_LOGGER)


if __name__ == "__main__":
    app()
