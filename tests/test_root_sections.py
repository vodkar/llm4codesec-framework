"""Plain-python checks: v2 samples render roots and reference context in separate sections,
the function-only baseline becomes a single root, and the v2 plans resolve.

Run: PYTHONPATH=src uv run python tests/test_root_sections.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmark.benchmark_runner import user_template_values
from benchmark.config import ExperimentsPlanConfig
from benchmark.models import BenchmarkSample
from benchmark.root_sections import RootSection, render_root_sections
from entrypoints.attach_root_findings import attach_root_findings
from entrypoints.utils import compose_benchmark_config

CONFIGS = REPO / "src" / "configs"
DATASETS = [
    "cleanvul_python_matched_v2_root",
    "cpg_structural_v2_findings_off",
    "cpg_structural_v2_findings_on",
    "mult_amp_v2_findings_off",
    "mult_amp_v2_findings_on",
]
# Flat v2 `code`: marker lines are snippet lines that static findings index into.
V2_CODE = (
    "# ===== ROOT 1/2: app/views.py:10-12 | code under analysis =====\n"
    "def view(request):\n"
    "    os.system(request.GET['cmd'])\n"
    "# ----- CONTEXT for ROOT 1 | reference only: related callers, callees and definitions -----\n"
    "# file: app/urls.py\n"
    "urls = [view]\n"
    "# ===== ROOT 2/2: app/util.py:3-4 | code under analysis =====\n"
    "def helper(x):\n"
    "    return x"
)
ROOTS = [
    {
        "file_path": "app/views.py", "line_start": 10, "line_end": 12,
        "code": "def view(request):\n    os.system(request.GET['cmd'])",
        "context": "# file: app/urls.py\nurls = [view]",
    },
    {
        "file_path": "app/util.py", "line_start": 3, "line_end": 4,
        "code": "def helper(x):\n    return x", "context": "",
    },
]
FINDING = {
    "tool": "bandit", "rule_id": "B605", "cwe_id": 78, "severity": "HIGH",
    "message": "Starting a process with a shell", "file_path": "app/views.py",
    "repo_line": 11, "snippet_line": 3, "is_root": True,
}


def _v2_sample(findings: list[dict] | None = None) -> BenchmarkSample:
    return BenchmarkSample.model_validate({
        "id": "s1", "code": V2_CODE, "roots": ROOTS, "label": 1,
        "metadata": {"source_row_ids": [7]}, "static_findings": findings or [],
    })


def test_roots_render_before_context_in_separate_sections() -> None:
    code = user_template_values(_v2_sample(), render_root_findings=False)["code"]
    assert "=====" not in code and "CONTEXT for ROOT" not in code
    assert code.index("## CODE UNDER ANALYSIS") < code.index("### ROOT 1: app/views.py, lines 10-12")
    assert code.index("### ROOT 2: app/util.py, lines 3-4") < code.index(
        "## REFERENCE CONTEXT (read-only)"
    )
    assert "2 functions changed together in one commit" in code
    assert "### Context for ROOT 1\n```python\n# file: app/urls.py\nurls = [view]\n```" in code
    # A root without context gets no context subsection.
    assert "Context for ROOT 2" not in code


def test_no_context_section_without_context() -> None:
    code = render_root_sections([RootSection(code="def f():\n    pass\n")])
    assert code == (
        "## CODE UNDER ANALYSIS\n\nThe function under analysis:\n\n"
        "### ROOT 1\n```python\ndef f():\n    pass\n```"
    )


def test_findings_still_index_the_flat_code() -> None:
    sample = _v2_sample([FINDING])
    assert [f.flagged_line for f in sample.root_static_findings] == [
        "os.system(request.GET['cmd'])"
    ]
    assert "B605" in user_template_values(sample, render_root_findings=True)["root_static_findings"]


def test_empty_roots_rejected() -> None:
    try:
        render_root_sections([])
    except ValueError:
        return
    raise AssertionError("empty roots must raise")


def test_attach_as_root_wraps_function_only_code() -> None:
    target = {"samples": [{
        "id": "f1", "code": "def view(request):\n    # note\n    os.system(request.GET['cmd'])",
        "label": 1, "metadata": {"source_row_ids": [7]},
    }]}
    source = {"samples": [{
        "id": "c1", "code": V2_CODE, "roots": ROOTS + [dict(ROOTS[0], line_start=20)],
        "label": 1, "metadata": {"source_row_ids": [7]}, "static_findings": [FINDING],
    }]}
    sample = attach_root_findings(target, source, as_root=True)["samples"][0]
    assert sample["roots"] == [
        {"file_path": "app/views.py, app/util.py", "code": target["samples"][0]["code"]}
    ]
    assert sample["root_static_findings"][0]["rule_id"] == "B605"
    assert "roots" not in attach_root_findings(target, source)["samples"][0]
    code = user_template_values(BenchmarkSample.model_validate(sample), False)["code"]
    assert "### ROOT 1: app/views.py, app/util.py\n```python\ndef view(request):\n    # note" in code


def test_prompt_is_v2_plus_layout_and_reminder() -> None:
    prompts = json.loads((CONFIGS / "shared" / "prompts.json").read_text())["prompts"]
    base = prompts["strict_exploitable_security_v2_root_findings"]["system_prompt"]
    new = prompts["root_context_security_v2_root_findings"]
    first, rest = base.split("\n\n", 1)
    assert new["system_prompt"].startswith(first + "\n\nInput layout: CODE UNDER ANALYSIS")
    assert new["system_prompt"].endswith("\n\n" + rest)
    assert "{code}{root_static_findings}\n\nReminder: judge only the ROOT code" in new["user_prompt"]


def test_v3_prompt_drops_path_examples_and_keeps_placeholders() -> None:
    prompt = json.loads((CONFIGS / "shared" / "prompts.json").read_text())["prompts"][
        "root_context_security_v3_root_findings"
    ]
    assert "'..'" not in prompt["system_prompt"]
    assert "Step 5 - Challenge" in prompt["system_prompt"]
    assert "{code}{root_static_findings}\n\nReminder: judge only the ROOT code" in prompt["user_prompt"]


def test_every_dataset_variant_has_five_conditions() -> None:
    datasets = json.loads((CONFIGS / "root_context_v2" / "datasets.json").read_text())["datasets"]
    for variant in ("v2", "v2_2k", "v2_score34", "v2_2k_score34"):
        names = [name.replace("v2", variant, 1) for name in DATASETS]
        assert all(name in datasets for name in names), variant
        assert [datasets[name].get("render_root_findings", False) for name in names] == [
            False, False, True, False, True,
        ]
        for name in names[1:]:
            assert f"context-assembler-dataset/{variant}/" in datasets[name]["dataset_path"]
        assert datasets[names[0]]["dataset_path"] == (
            f"datasets_processed/context_assembler/{variant}/cleanvul_python_matched_root_findings.json"
        )


def test_score34_gemma_nothinking_plan() -> None:
    composed = compose_benchmark_config(
        benchmark_name="context_assembler",
        base_config=None,
        config_directory=CONFIGS / "shared",
        experiments_config=CONFIGS / "root_context_v2" / "experiments.json",
        datasets_config=CONFIGS / "root_context_v2" / "datasets.json",
    )
    plan = ExperimentsPlanConfig.from_file(composed, "root_context_v2_score34_gemma_nothinking_sweep")
    assert [e.dataset_name for e in plan.experiments] == [
        name.replace("v2", "v2_score34", 1) for name in DATASETS
    ]
    assert {e.model_name for e in plan.experiments} == {"gemma4-12b-it-nothinking-sc7-logprobs-seeded"}
    assert {e.is_thinking_enabled for e in plan.experiments} == {False}
    assert {e.self_consistency_samples for e in plan.experiments} == {7}


def test_gemma_sampling_sweep_varies_one_decoding_param() -> None:
    models = json.loads((CONFIGS / "shared" / "models.json").read_text())["models"]
    base = models["gemma4-12b-nt-sc3-base"]
    plans = json.loads((CONFIGS / "root_context_v2" / "experiments.json").read_text())[
        "experiment_plans"
    ]
    plan = plans["root_context_v2_gemma_sampling_sweep"]
    assert plan["datasets"] == ["cpg_structural_v2_findings_off"]
    assert len(plan["models"]) == 28 and plan["models"][0] == "gemma4-12b-nt-sc3-base"
    decoding = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
    for name in plan["models"]:
        model = models[name]
        changed = {key for key in set(model) | set(base) if model.get(key) != base.get(key)}
        assert changed - {"self_consistency_samples"} <= decoding, name
        assert len(changed & decoding) == (0 if name.endswith("-base") else 1), name
        expected_draws = 1 if model["temperature"] == 0.0 else 3
        assert model["self_consistency_samples"] == expected_draws, name
        assert model["is_thinking_enabled"] is False


def test_gemma_sampling_combos_only_combine_sweep_values() -> None:
    models = json.loads((CONFIGS / "shared" / "models.json").read_text())["models"]
    base = models["gemma4-12b-nt-sc3-base"]
    plan = json.loads((CONFIGS / "root_context_v2" / "experiments.json").read_text())[
        "experiment_plans"
    ]["root_context_v2_gemma_sampling_combos"]
    assert plan["datasets"] == ["cpg_structural_v2_findings_off"]
    assert 6 <= len(plan["models"]) <= 9
    winners = {"temperature": 1.2, "repetition_penalty": 1.1, "presence_penalty": 1.5, "top_k": 20}
    for name in plan["models"]:
        model = models[name]
        changed = {key for key in set(model) | set(base) if model.get(key) != base.get(key)}
        assert len(changed) >= 2 and all(model[key] == winners[key] for key in changed), name


def test_gemma_temperature_sc7_plan_varies_only_temperature() -> None:
    models = json.loads((CONFIGS / "shared" / "models.json").read_text())["models"]
    base = models["gemma4-12b-nt-sc3-base"]
    plan = json.loads((CONFIGS / "root_context_v2" / "experiments.json").read_text())[
        "experiment_plans"
    ]["root_context_v2_gemma_temperature_sc7"]
    assert plan["datasets"] == ["cpg_structural_v2_findings_off"]
    assert [models[name]["temperature"] for name in plan["models"]] == [1.0, 1.2, 1.5, 1.8]
    for name in plan["models"]:
        model = models[name]
        changed = {key for key in set(model) | set(base) if model.get(key) != base.get(key)}
        assert changed <= {"temperature", "self_consistency_samples"}, name
        assert model["self_consistency_samples"] == 7, name


def test_gemma_compare_rankings_plan() -> None:
    composed = compose_benchmark_config(
        benchmark_name="context_assembler",
        base_config=None,
        config_directory=CONFIGS / "shared",
        experiments_config=CONFIGS / "root_context_v2" / "experiments.json",
        datasets_config=CONFIGS / "root_context_v2" / "datasets.json",
    )
    plan = ExperimentsPlanConfig.from_file(composed, "root_context_v2_gemma_compare_rankings")
    datasets = [e.dataset_name for e in plan.experiments]
    assert datasets[0] == "cleanvul_python_matched_v2_root" and len(datasets) == 14
    assert all(name.startswith("compare_rankings_v2_") for name in datasets[1:])
    assert {e.render_root_findings for e in plan.experiments} == {False}
    model = json.loads((CONFIGS / "shared" / "models.json").read_text())["models"][
        "gemma4-12b-it-nothinking-sc7-logprobs-seeded"
    ]
    assert (model["temperature"], model["repetition_penalty"]) == (1.2, 1.1)
    assert model["self_consistency_samples"] == 7 and model["is_thinking_enabled"] is False
    datasets_config = json.loads((CONFIGS / "root_context_v2" / "datasets.json").read_text())
    for name in datasets[1:]:
        assert (REPO / datasets_config["datasets"][name]["dataset_path"]).exists(), name


def test_ctx_4k_vs_6k_plan_pairs_budgets() -> None:
    composed = compose_benchmark_config(
        benchmark_name="context_assembler",
        base_config=None,
        config_directory=CONFIGS / "shared",
        experiments_config=CONFIGS / "root_context_v2" / "experiments.json",
        datasets_config=CONFIGS / "root_context_v2" / "datasets.json",
    )
    plan = ExperimentsPlanConfig.from_file(composed, "root_context_v2_gemma_ctx_4k_vs_6k")
    assert [e.dataset_name for e in plan.experiments] == [
        f"{strategy}_v2_{size}_changed"
        for strategy in ("cpg_structural", "mult_amp")
        for size in ("4k", "6k")
    ]
    assert {e.model_name for e in plan.experiments} == {"gemma4-12b-it-nothinking-sc7-logprobs-seeded"}
    assert {e.render_root_findings for e in plan.experiments} == {False}


def test_plans_resolve() -> None:
    composed = compose_benchmark_config(
        benchmark_name="context_assembler",
        base_config=None,
        config_directory=CONFIGS / "shared",
        experiments_config=CONFIGS / "root_context_v2" / "experiments.json",
        datasets_config=CONFIGS / "root_context_v2" / "datasets.json",
    )
    for name, limit, model in (
        ("root_context_v2_lfm_sweep", None, "lfm2.5-8b-a1b-thinking-sc7-logprobs-seeded"),
        ("root_context_v2_lfm_smoke", 20, "lfm2.5-8b-a1b-thinking-sc7-logprobs-seeded"),
        ("root_context_v2_gemma_sweep", None, "gemma4-12b-it-thinking-sc3-logprobs-seeded-16k"),
        ("root_context_v2_gemma_smoke", 20, "gemma4-12b-it-thinking-sc3-logprobs-seeded-16k"),
        ("root_context_v2_qwen35_sweep", None, "qwen3.5-9b-thinking-sc7-logprobs-seeded-16k"),
        ("root_context_v2_qwen3_8b_sweep", None, "qwen3-8b-thinking-sc7-logprobs-seeded"),
        ("root_context_v3_lfm_sweep", None, "lfm2.5-8b-a1b-thinking-sc7-logprobs-seeded"),
        ("root_context_v3_lfm_smoke", 20, "lfm2.5-8b-a1b-thinking-sc7-logprobs-seeded"),
    ):
        plan = ExperimentsPlanConfig.from_file(composed, name)
        expected = DATASETS
        assert [e.dataset_name for e in plan.experiments] == expected
        assert [e.render_root_findings for e in plan.experiments] == [
            "_on" in dataset for dataset in expected
        ]
        assert {e.model_name for e in plan.experiments} == {model}
        prompt = "v3" if name.startswith("root_context_v3") else "v2"
        assert {e.prompt_identifier for e in plan.experiments} == {
            f"root_context_security_{prompt}_root_findings"
        }
        assert {e.sample_limit for e in plan.experiments} == {limit}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
