"""Plain-python checks: verification prompt, exclusion lists and verify plans.

Run: PYTHONPATH=src python tests/test_finding_verification_configs.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmark.config import ExperimentsPlanConfig
from entrypoints.utils import compose_benchmark_config

CONFIGS = REPO / "src" / "configs"
VERIFY_PARAGRAPH = (
    "\n\nStatic analyzer findings:\nThe code may come with static-analyzer findings. Treat each finding "
    "as a lead, not as evidence. For every finding, decide CONFIRMED or REFUTED. REFUTED means the "
    "flagged data is not attacker-controlled, the line is unreachable, or validation, escaping, an "
    "allow-list, or a safe API in the shown code prevents exploitation. Most analyzer findings are "
    "false positives. Report a flaw the analyzers did not flag only if you can trace a concrete "
    "source-to-sink path. If every finding is refuted and you found no other concrete flaw, the code "
    "is NOT vulnerable. If no findings are given, analyze the code on its own merits."
)
VERIFY_DATASETS = ["cpg_structural_root_findings_verify", "mult_amp_root_findings_verify"]


def _datasets() -> dict:
    return json.loads((CONFIGS / "static_findings_root" / "datasets.json").read_text())["datasets"]


def test_prompt_is_base_plus_verification_paragraph() -> None:
    prompts = json.loads((CONFIGS / "shared" / "prompts.json").read_text())["prompts"]
    base = prompts["strict_exploitable_security"]
    new = prompts["finding_verification_root_findings"]
    assert new["system_prompt"] == base["system_prompt"] + VERIFY_PARAGRAPH
    assert new["user_prompt"] == base["user_prompt"] + "{root_static_findings}"


def test_all_seven_entries_share_the_18_exclusions() -> None:
    datasets = _datasets()
    assert len(datasets) == 7
    lists = [entry["exclude_samples"] for entry in datasets.values()]
    assert all(entry_list == lists[0] for entry_list in lists)
    assert len(lists[0]) == 18
    assert len({(tuple(e["source_row_ids"]), e["label"]) for e in lists[0]}) == 18


def test_verify_entries() -> None:
    datasets = _datasets()
    base = "benchmarks/context-assembler-dataset/"
    assert datasets[VERIFY_DATASETS[0]]["dataset_path"] == base + "context_assembler_cpg_structural.json"
    assert datasets[VERIFY_DATASETS[1]]["dataset_path"] == (
        base + "context_assembler_multiplicative_amplification.json")
    for key in VERIFY_DATASETS:
        entry = datasets[key]
        assert entry["render_root_findings"] is True
        assert entry["omit_empty_root_findings"] is True
        assert entry["exclude_finding_rules"] == ["B101", "B113"]
    for key in set(datasets) - set(VERIFY_DATASETS):
        assert "exclude_finding_rules" not in datasets[key]
        assert "omit_empty_root_findings" not in datasets[key]


def test_verify_plans_resolve() -> None:
    composed = compose_benchmark_config(
        benchmark_name="context_assembler", base_config=None,
        config_directory=CONFIGS / "shared",
        experiments_config=CONFIGS / "static_findings_root" / "experiments.json",
        datasets_config=CONFIGS / "static_findings_root" / "datasets.json",
    )
    for name, limit in (("static_findings_verify_sweep", None), ("static_findings_verify_smoke", 40)):
        plan = ExperimentsPlanConfig.from_file(composed, name)
        assert [e.dataset_name for e in plan.experiments] == VERIFY_DATASETS
        assert {e.prompt_identifier for e in plan.experiments} == {"finding_verification_root_findings"}
        assert {e.model_name for e in plan.experiments} == {"gemma4-12b-it-thinking-sc7-logprobs-seeded"}
        assert all(e.coverage_levels == [0.25, 0.5, 0.75, 1.0] for e in plan.experiments)
        assert all(e.sample_limit == limit for e in plan.experiments)
        assert all(len(e.exclude_samples) == 18 and e.omit_empty_root_findings for e in plan.experiments)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
