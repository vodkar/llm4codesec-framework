import json
import logging
from pathlib import Path
from typing import Any

from consts import CONFIG_DIRECTORY

_SHARED_MODELS_FILENAME: str = "models.json"
_SHARED_PROMPTS_FILENAME: str = "prompts.json"
_DEFAULT_EXPERIMENT_FILENAMES: tuple[str, ...] = (
    "experiments.json",
    "experiment_plans.json",
)
_DEFAULT_DATASET_FILENAMES: tuple[str, ...] = ("datasets.json",)


def resolve_config_path(config: str | Path) -> Path:
    """Resolve experiment config path with backward-compatible fallbacks."""
    candidate = Path(config)
    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        return candidate

    fallback_paths: list[Path] = [
        CONFIG_DIRECTORY / candidate.name,
        Path("configs") / candidate.name,
        Path("../configs") / candidate.name,
        Path(__file__).resolve().parents[2] / "src" / "configs" / candidate.name,
    ]

    for path in fallback_paths:
        if path.exists():
            return path

    return candidate


def resolve_optional_config_path(config: str | Path | None) -> Path | None:
    """Resolve an optional config path.

    Args:
        config: Optional path string or Path.

    Returns:
        Resolved path if provided, otherwise None.
    """
    if config is None:
        return None
    return resolve_config_path(config)


def load_config_dict(config: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Load raw JSON experiment config from file or return provided mapping.

    Args:
        config: Config file path or config dictionary.

    Returns:
        Loaded configuration dictionary.
    """
    if isinstance(config, dict):
        return dict(config)

    config_path: Path = resolve_config_path(config)
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def normalize_config_schema(config_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy config schema keys into canonical benchmark schema."""
    normalized: dict[str, Any] = dict(config_data)

    models: dict[str, Any] = dict(
        config_data.get("models", config_data.get("model_configurations", {}))
    )
    for model_key, model_config in models.items():
        if (
            isinstance(model_config, dict)
            and "model_identifier" not in model_config
            and "model_name" in model_config
        ):
            models[model_key] = {
                **model_config,
                "model_identifier": model_config["model_name"],
            }
    normalized["models"] = models

    datasets: dict[str, Any] = dict(
        config_data.get("datasets", config_data.get("dataset_configurations", {}))
    )
    for dataset_key, dataset_config in datasets.items():
        if (
            isinstance(dataset_config, dict)
            and "path" not in dataset_config
            and "dataset_path" in dataset_config
        ):
            datasets[dataset_key] = {
                **dataset_config,
                "path": dataset_config["dataset_path"],
            }
    normalized["datasets"] = datasets

    prompts: dict[str, Any] = dict(
        config_data.get("prompts", config_data.get("prompt_strategies", {}))
    )
    normalized["prompts"] = prompts

    return normalized


def _discover_in_directory(
    directory: Path,
    candidate_filenames: tuple[str, ...],
) -> Path | None:
    """Find the first existing config file in directory from candidate names."""
    for filename in candidate_filenames:
        candidate: Path = directory / filename
        if candidate.exists():
            return candidate
    return None


def _extract_section(
    source_data: dict[str, Any],
    canonical_key: str,
    legacy_key: str,
) -> dict[str, Any] | None:
    """Extract a canonical section from config data.

    Supports fully wrapped files ({"models": {...}}) and section-only files
    ({"model_a": {...}, "model_b": {...}}).
    """
    if canonical_key in source_data and isinstance(source_data[canonical_key], dict):
        return dict(source_data[canonical_key])

    if legacy_key in source_data and isinstance(source_data[legacy_key], dict):
        return dict(source_data[legacy_key])

    if source_data and all(isinstance(value, dict) for value in source_data.values()):
        known_root_keys: set[str] = {
            "experiment_metadata",
            "output_settings",
            "evaluation_settings",
            "experiment_plans",
            "models",
            "model_configurations",
            "datasets",
            "dataset_configurations",
            "prompts",
            "prompt_strategies",
        }
        if not any(key in known_root_keys for key in source_data):
            return dict(source_data)

    return None


def compose_benchmark_config(
    benchmark_name: str,
    base_config: Path | str | dict[str, Any] | None,
    config_directory: Path | str | None,
    experiments_config: Path | str | None,
    datasets_config: Path | str | None,
) -> dict[str, Any]:
    """Compose benchmark configuration from monolithic and split config sources.

    Precedence for split sections: explicit file path > discovered file in config
    directory > section from base config.

    Args:
        benchmark_name: Benchmark identifier.
        base_config: Legacy monolithic config source.
        config_directory: Directory containing split config files.
        experiments_config: Optional explicit experiment plans config file.
        datasets_config: Optional explicit datasets config file.

    Returns:
        A canonical config dictionary compatible with benchmark loaders.
    """
    base_data: dict[str, Any] = (
        load_config_dict(base_config) if base_config is not None else {}
    )
    composed: dict[str, Any] = dict(base_data)

    resolved_dir: Path | None = None
    if config_directory is not None:
        resolved_dir = resolve_config_path(config_directory)
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {config_directory}")

    models_path: Path | None = (
        _discover_in_directory(resolved_dir, (_SHARED_MODELS_FILENAME,))
        if resolved_dir is not None
        else None
    )
    prompts_path: Path | None = (
        _discover_in_directory(resolved_dir, (_SHARED_PROMPTS_FILENAME,))
        if resolved_dir is not None
        else None
    )

    discovered_experiment_filenames: tuple[str, ...] = (
        f"{benchmark_name}_experiments.json",
        *_DEFAULT_EXPERIMENT_FILENAMES,
    )
    discovered_dataset_filenames: tuple[str, ...] = (
        f"{benchmark_name}_datasets.json",
        *_DEFAULT_DATASET_FILENAMES,
    )

    experiments_path: Path | None = resolve_optional_config_path(experiments_config)
    if experiments_path is None and resolved_dir is not None:
        experiments_path = _discover_in_directory(
            resolved_dir,
            discovered_experiment_filenames,
        )

    datasets_path: Path | None = resolve_optional_config_path(datasets_config)
    if datasets_path is None and resolved_dir is not None:
        datasets_path = _discover_in_directory(
            resolved_dir,
            discovered_dataset_filenames,
        )

    if models_path is not None:
        models_data: dict[str, Any] = load_config_dict(models_path)
        models_section: dict[str, Any] | None = _extract_section(
            source_data=models_data,
            canonical_key="models",
            legacy_key="model_configurations",
        )
        if models_section is None:
            raise ValueError(f"No models section found in {models_path}")
        composed["models"] = models_section

    if prompts_path is not None:
        prompts_data: dict[str, Any] = load_config_dict(prompts_path)
        prompts_section: dict[str, Any] | None = _extract_section(
            source_data=prompts_data,
            canonical_key="prompts",
            legacy_key="prompt_strategies",
        )
        if prompts_section is None:
            raise ValueError(f"No prompts section found in {prompts_path}")
        composed["prompts"] = prompts_section

    if datasets_path is not None:
        datasets_data: dict[str, Any] = load_config_dict(datasets_path)
        datasets_section: dict[str, Any] | None = _extract_section(
            source_data=datasets_data,
            canonical_key="datasets",
            legacy_key="dataset_configurations",
        )
        if datasets_section is None:
            raise ValueError(f"No datasets section found in {datasets_path}")
        composed["datasets"] = datasets_section

    if experiments_path is not None:
        experiments_data: dict[str, Any] = load_config_dict(experiments_path)
        for root_key in (
            "experiment_metadata",
            "experiment_plans",
            "output_settings",
            "evaluation_settings",
        ):
            if root_key in experiments_data:
                composed[root_key] = experiments_data[root_key]

    return normalize_config_schema(composed)


def list_plans(
    config: Path | str | dict[str, Any], logger: logging.Logger | None = None
) -> None:
    """Log all available experiment plans from configuration."""
    active_logger: logging.Logger = logger or logging.getLogger(__name__)
    config_data: dict[str, Any] = normalize_config_schema(load_config_dict(config))
    plans: dict[str, Any] = config_data.get("experiment_plans", {})

    active_logger.info("Available experiment plans:")
    for plan_name, plan_config in plans.items():
        models: list[str] = plan_config.get("models", [])
        datasets: list[str] = plan_config.get("datasets", [])
        prompts: list[str] = plan_config.get("prompts", [])

        active_logger.info("  %s: %s", plan_name, plan_config.get("description", ""))
        active_logger.info("    Models: %s", ", ".join(models) if models else "-")
        active_logger.info("    Datasets: %s", ", ".join(datasets) if datasets else "-")
        active_logger.info("    Prompts: %s", ", ".join(prompts) if prompts else "-")
        active_logger.info(
            "    Total experiments: %d",
            len(models) * len(datasets) * len(prompts),
        )


def log_available_configurations(
    config: Path | str | dict[str, Any], logger: logging.Logger | None = None
) -> None:
    """Log available models, datasets, prompts, and plans from configuration."""
    active_logger: logging.Logger = logger or logging.getLogger(__name__)
    config_data: dict[str, Any] = normalize_config_schema(load_config_dict(config))

    models: dict[str, Any] = config_data.get("models", {})
    datasets: dict[str, Any] = config_data.get("datasets", {})
    prompts: dict[str, Any] = config_data.get("prompts", {})
    plans: dict[str, Any] = config_data.get("experiment_plans", {})

    active_logger.info("Available configurations:")
    active_logger.info("  Models (%d): %s", len(models), ", ".join(models.keys()))
    active_logger.info("  Datasets (%d): %s", len(datasets), ", ".join(datasets.keys()))
    active_logger.info("  Prompts (%d): %s", len(prompts), ", ".join(prompts.keys()))
    active_logger.info("  Plans (%d): %s", len(plans), ", ".join(plans.keys()))
