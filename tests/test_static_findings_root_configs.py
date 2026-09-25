"""Plain-python checks: the root-findings sweep varies only dataset and findings flag.

Run: PYTHONPATH=src uv run python tests/test_static_findings_root_configs.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmark.config import ExperimentsPlanConfig
from entrypoints.utils import compose_benchmark_config

CONFIGS = REPO / "src" / "configs"
DATASETS = [
    "cleanvul_python_matched_root_findings",
    "cpg_structural_root_findings_off",
    "cpg_structural_root_findings_on",
    "mult_amp_root_findings_off",
    "mult_amp_root_findings_on",
]


def _composed() -> dict:
    return compose_benchmark_config(
        benchmark_name="context_assembler",
        base_config=None,
        config_directory=CONFIGS / "shared",
        experiments_config=CONFIGS / "static_findings_root" / "experiments.json",
        datasets_config=CONFIGS / "static_findings_root" / "datasets.json",
    )


def test_prompt_differs_from_base_only_by_placeholder() -> None:
    prompts = json.loads((CONFIGS / "shared" / "prompts.json").read_text())["prompts"]
    base = prompts["strict_exploitable_security"]
    new = prompts["strict_exploitable_security_root_findings"]
    assert new["system_prompt"] == base["system_prompt"]
    assert new["user_prompt"] == base["user_prompt"] + "{root_static_findings}"


def test_plans() -> None:
    plans = json.loads((CONFIGS / "static_findings_root" / "experiments.json").read_text())[
        "experiment_plans"
    ]
    for name, limit in (("static_findings_root_sweep", None), ("static_findings_root_smoke", 40)):
        plan = plans[name]
        assert plan["datasets"] == DATASETS
        assert plan["models"] == ["gemma4-12b-it-thinking-sc7-logprobs-seeded"]
        assert plan["prompts"] == ["strict_exploitable_security_root_findings"]
        assert plan.get("sample_limit") == limit


def test_dataset_paths_and_flags() -> None:
    datasets = json.loads((CONFIGS / "static_findings_root" / "datasets.json").read_text())[
        "datasets"
    ]
    base = "benchmarks/context-assembler-dataset/"
    assert datasets[DATASETS[0]]["dataset_path"] == (
        "datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json"
    )
    for key in DATASETS[1:3]:
        assert datasets[key]["dataset_path"] == base + "context_assembler_cpg_structural.json"
    for key in DATASETS[3:]:
        assert datasets[key]["dataset_path"] == (
            base + "context_assembler_multiplicative_amplification.json"
        )
    assert [datasets[k].get("render_root_findings", False) for k in DATASETS] == [
        False, False, True, False, True,
    ]


def test_plan_resolves_to_five_experiments() -> None:
    plan = ExperimentsPlanConfig.from_file(_composed(), "static_findings_root_sweep")
    assert [e.dataset_name for e in plan.experiments] == DATASETS
    assert [e.render_root_findings for e in plan.experiments] == [False, False, True, False, True]
    assert all(e.sampling_seed is not None for e in plan.experiments)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
