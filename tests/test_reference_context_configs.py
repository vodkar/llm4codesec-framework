"""Plain-python checks: the sweep holds everything but the dataset constant.

Run: PYTHONPATH=src uv run python tests/test_reference_context_configs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "configs"
CONDITIONS = ["none", "random", "cpg", "reference", "mismatched"]

# Add src to path for importing benchmark config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_sweep_plan_parity() -> None:
    experiments = json.loads((ROOT / "reference_context" / "experiments.json").read_text())
    plan = experiments["experiment_plans"]["reference_context_sweep"]
    assert plan["datasets"] == [f"reference_context_{c}" for c in CONDITIONS]
    assert plan["models"] == ["gemma4-12b-it-thinking-sc7-logprobs-seeded"]
    assert plan["prompts"] == ["strict_exploitable_security"]
    assert "sample_limit" not in plan


def test_seeded_model_differs_only_by_seed() -> None:
    models = json.loads((ROOT / "shared" / "models.json").read_text())
    models = models.get("models", models)
    base = dict(models["gemma4-12b-it-thinking-sc7-logprobs"])
    seeded = dict(models["gemma4-12b-it-thinking-sc7-logprobs-seeded"])
    assert seeded.pop("sampling_seed") == 20260925
    assert seeded == base


def test_dataset_paths() -> None:
    datasets = json.loads((ROOT / "reference_context" / "datasets.json").read_text())["datasets"]
    for condition in CONDITIONS:
        assert datasets[f"reference_context_{condition}"]["dataset_path"].endswith(
            f"reference_context/cleanvul_cond_{condition}.json"
        )


def test_framework_config_loading() -> None:
    """Verify that the framework's ModelConfig class has sampling_seed field.

    JSON-level check: ModelConfig is defined in src/benchmark/config.py and
    already includes 'sampling_seed: int | None = None' field (confirmed in Task 1),
    so it will accept sampling_seed when loading models.json.
    """
    from benchmark.config import ModelConfig

    # Verify that ModelConfig has sampling_seed in its fields
    fields = ModelConfig.model_fields
    assert "sampling_seed" in fields, "ModelConfig missing sampling_seed field"

    # Verify the seeded model entry has sampling_seed in the config
    models = json.loads((ROOT / "shared" / "models.json").read_text())
    models_dict = models.get("models", models)
    seeded_model = models_dict.get("gemma4-12b-it-thinking-sc7-logprobs-seeded")
    assert seeded_model is not None, "Seeded model not found in models.json"
    assert "sampling_seed" in seeded_model, "Seeded model missing sampling_seed field"


if __name__ == "__main__":
    test_sweep_plan_parity()
    test_seeded_model_differs_only_by_seed()
    test_dataset_paths()
    test_framework_config_loading()
    print("ALL PASSED")
