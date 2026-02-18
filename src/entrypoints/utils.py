import json
import logging
from pathlib import Path
from typing import Any

from consts import CONFIG_DIRECTORY


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


def load_config_dict(config: Path | str) -> dict[str, Any]:
    """Load raw JSON experiment config from file."""
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


def list_plans(config: Path | str, logger: logging.Logger | None = None) -> None:
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
    config: Path | str, logger: logging.Logger | None = None
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
