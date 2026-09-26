"""Plain-python checks: analyzer rules are filtered and empty findings sections can be omitted.

Run: PYTHONPATH=src python tests/test_finding_filters.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import BenchmarkRunner, sample_provenance, user_template_values
from benchmark.config import ExperimentConfig
from benchmark.models import BenchmarkSample, SampleCollection
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import MetricsResult
from benchmark.static_findings import (
    RootStaticFinding,
    filter_root_findings,
    render_root_findings_block,
)
from datasets.loaders.base import JsonDatasetLoader


def _finding(rule_id: str) -> RootStaticFinding:
    return RootStaticFinding(tool="bandit", rule_id=rule_id, message="m", file_path="a.py",
                             repo_line=1, flagged_line="x")


def _config(render: bool, omit_empty: bool, rules: list[str]) -> dict:
    dataset: dict = {"path": __file__, "task_type": "binary_vulnerability", "description": "d",
                     "render_root_findings": render, "omit_empty_root_findings": omit_empty,
                     "exclude_finding_rules": rules}
    return {
        "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                         "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                         "use_quantization": False}},
        "datasets": {"d": dataset},
        "prompts": {"p": {"name": "p", "system_prompt": "sys",
                          "user_prompt": "Analyze:\n\n{code}{root_static_findings}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {"plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}},
    }


def _experiment(render: bool = True, omit_empty: bool = True, rules: list[str] | None = None) -> ExperimentConfig:
    return ExperimentConfig.from_file(
        config=_config(render, omit_empty, rules if rules is not None else ["B101", "B113"]),
        model_key="m", dataset_key="d", prompt_key="p", experiment_name="plan",
    )


def _sample(findings: list[RootStaticFinding] | None) -> BenchmarkSample:
    return BenchmarkSample(id="s", code="def f(): pass", label=1,
                           metadata={"source_row_ids": [1]}, root_static_findings=findings)


def test_filter_exact_and_wildcard() -> None:
    findings = [_finding("B101"), _finding("B602"), _finding("python.x.request-with-http.y")]
    assert [f.rule_id for f in filter_root_findings(findings, ["B101"])] == [
        "B602", "python.x.request-with-http.y"]
    assert [f.rule_id for f in filter_root_findings(findings, ["*request-with-http*", "B6*"])] == ["B101"]


def test_filter_none_and_no_patterns_pass_through() -> None:
    assert filter_root_findings(None, ["B101"]) is None
    findings = [_finding("B101")]
    assert filter_root_findings(findings, []) == findings


def test_render_omit_empty() -> None:
    assert render_root_findings_block([], omit_empty=True) == ""
    assert render_root_findings_block([]) == (
        "\n\nStatic analyzer findings in the function under analysis: none reported."
    )
    assert render_root_findings_block([_finding("B602")], omit_empty=True) == render_root_findings_block(
        [_finding("B602")])


def test_template_values_omit_empty() -> None:
    assert user_template_values(_sample([]), True, True)["root_static_findings"] == ""
    assert user_template_values(_sample([]), True)["root_static_findings"].endswith("none reported.")
    # Omit flag with rendering off still renders nothing.
    assert user_template_values(_sample([_finding("B602")]), False, True)["root_static_findings"] == ""


def test_config_carries_filter_flags() -> None:
    config = _experiment()
    assert config.exclude_finding_rules == ["B101", "B113"]
    assert config.omit_empty_root_findings is True
    default = ExperimentConfig.from_file(
        config={**_config(True, False, []), "datasets": {"d": {"path": __file__,
                "task_type": "binary_vulnerability", "description": "d"}}},
        model_key="m", dataset_key="d", prompt_key="p", experiment_name="plan")
    assert default.exclude_finding_rules == [] and default.omit_empty_root_findings is False


def test_prepare_samples_filters_rules_for_prompt_and_storage() -> None:
    runner = BenchmarkRunner(config=_experiment(), dataset_loader=JsonDatasetLoader())
    kept, _ = runner._prepare_samples(SampleCollection([_sample([_finding("B101"), _finding("B602")])]))
    sample = kept[0]
    assert [f.rule_id for f in sample.root_static_findings] == ["B602"]
    assert "B101" not in user_template_values(sample, True, True)["root_static_findings"]
    assert [f.rule_id for f in sample_provenance(sample, True)["root_static_findings"]] == ["B602"]


def test_prepare_samples_all_filtered_renders_nothing() -> None:
    runner = BenchmarkRunner(config=_experiment(), dataset_loader=JsonDatasetLoader())
    kept, _ = runner._prepare_samples(SampleCollection([_sample([_finding("B101")])]))
    assert kept[0].root_static_findings == []
    assert user_template_values(kept[0], True, True)["root_static_findings"] == ""


def test_report_records_filter_flags() -> None:
    processor = BenchmarkResultProcessor(config=_experiment())
    metrics = MetricsResult.model_validate({"task_type": "binary", "accuracy": 0.0, "summary": {}, "details": {}})
    info = processor.build_report(metrics=metrics, predictions=[], total_time=0.0, total_samples=0).benchmark_info
    assert info.extra_metadata["exclude_finding_rules"] == ["B101", "B113"]
    assert info.extra_metadata["omit_empty_root_findings"] is True



def test_provenance_flag_false_when_nothing_rendered() -> None:
    assert sample_provenance(_sample([]), True, True)["root_findings_in_prompt"] is False
    assert sample_provenance(_sample([]), True, False)["root_findings_in_prompt"] is True
    assert sample_provenance(_sample([_finding("B602")]), True, True)["root_findings_in_prompt"] is True
    assert sample_provenance(_sample([_finding("B602")]), False, True)["root_findings_in_prompt"] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
