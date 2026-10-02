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
        ("root_context_v2_gemma_sweep", None, "gemma4-12b-it-thinking-sc7-logprobs-seeded"),
        ("root_context_v2_gemma_smoke", 20, "gemma4-12b-it-thinking-sc7-logprobs-seeded"),
        ("root_context_v3_lfm_sweep", None, "lfm2.5-8b-a1b-thinking-sc7-logprobs-seeded"),
        ("root_context_v3_lfm_smoke", 20, "lfm2.5-8b-a1b-thinking-sc7-logprobs-seeded"),
    ):
        plan = ExperimentsPlanConfig.from_file(composed, name)
        assert [e.dataset_name for e in plan.experiments] == DATASETS
        assert [e.render_root_findings for e in plan.experiments] == [
            False, False, True, False, True,
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
