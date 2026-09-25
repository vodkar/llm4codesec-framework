"""Plain-python checks: root findings reach the prompt only when the dataset flag is on,
and every prediction/report record stores them with a stable join key.

Run: PYTHONPATH=src uv run python tests/test_root_findings_runner.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import BenchmarkRunner, sample_provenance, user_template_values
from benchmark.config import ExperimentConfig
from benchmark.models import BenchmarkSample, PredictionResult, SampleCollection
from benchmark.prompt_generator import get_prompt_generator
from benchmark.response_parser import ResponseParserFactory
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import MetricsResult, PredictionRecord
from benchmark.static_findings import RootStaticFinding
from datasets.loaders.base import JsonDatasetLoader
from llm.llm import ILLMInference, InferenceResult

FINDING = RootStaticFinding(
    tool="bandit", rule_id="B602", cwe_id=78, severity="HIGH", message="shell=True",
    file_path="a.py", repo_line=3, flagged_line="subprocess.call(cmd, shell=True)",
)
PLACEHOLDER_TEMPLATE = "Analyze:\n\n{code}{root_static_findings}"


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _config(user_prompt: str = PLACEHOLDER_TEMPLATE, render: bool = False) -> dict:
    dataset: dict = {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}
    if render:
        dataset["render_root_findings"] = True
    return {
        "models": {
            "m": {
                "model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                "use_quantization": False,
                # Positive input budget, so _filter_samples_by_token_limit really
                # formats and counts every prompt instead of skipping filtering.
                "context_length": 100000,
            }
        },
        "datasets": {"d": dataset},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": user_prompt}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {"plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}},
    }


def _experiment(user_prompt: str = PLACEHOLDER_TEMPLATE, render: bool = False) -> ExperimentConfig:
    return ExperimentConfig.from_file(
        config=_config(user_prompt, render), model_key="m", dataset_key="d",
        prompt_key="p", experiment_name="plan",
    )


def _sample(findings: list[RootStaticFinding] | None) -> BenchmarkSample:
    return BenchmarkSample(
        id="s1", code="def f(): pass", label=1, metadata={"source_row_ids": [4236]},
        root_static_findings=findings,
    )


class _CapturingLLM(ILLMInference):
    """Fake backend recording user prompts and always answering vulnerable."""

    def __init__(self) -> None:
        self.user_prompts: list[str] = []
        self.counted_texts: list[str] = []

    def generate_response(self, system_prompt: str, user_prompt: str) -> InferenceResult:
        raise NotImplementedError

    def generate_batch_responses(self, prompts: list[str]) -> list[InferenceResult]:
        raise NotImplementedError

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str], seeds: list[int] | None = None,
    ) -> list[InferenceResult]:
        self.user_prompts.extend(user_prompts)
        return [
            InferenceResult(response_text='{"is_vulnerable": true}', tokens_used=1, duration=0.0)
            for _ in user_prompts
        ]

    def count_input_tokens(self, text: str) -> int:
        self.counted_texts.append(text)
        return 0

    def cleanup(self) -> None:
        pass


def _run(experiment: ExperimentConfig, sample: BenchmarkSample) -> tuple[_CapturingLLM, PredictionResult]:
    runner = BenchmarkRunner(config=experiment, dataset_loader=JsonDatasetLoader())
    llm = _CapturingLLM()
    prompt_generator = get_prompt_generator(experiment, template_values={})
    parser = ResponseParserFactory.create_parser(experiment.task_type)
    samples, _ = runner._filter_samples_by_token_limit(SampleCollection([sample]), llm, prompt_generator)
    predictions = runner._process_samples_with_batch_optimization(samples, llm, prompt_generator, parser)
    return llm, predictions[0]


def test_flag_copied_from_dataset_config() -> None:
    assert _experiment().render_root_findings is False
    assert _experiment(render=True).render_root_findings is True


def test_flag_on_requires_placeholder_in_template() -> None:
    _raises(ValueError, lambda: _experiment(user_prompt="Analyze:\n\n{code}", render=True))


def test_template_values_flag_off() -> None:
    assert user_template_values(_sample([FINDING]), False) == {
        "code": "def f(): pass", "root_static_findings": "",
    }


def test_template_values_flag_on_requires_findings_info() -> None:
    _raises(ValueError, lambda: user_template_values(_sample(None), True))


def test_flag_on_renders_block_after_code() -> None:
    llm, prediction = _run(_experiment(render=True), _sample([FINDING]))
    assert llm.user_prompts[0].startswith(
        "Analyze:\n\ndef f(): pass\n\nStatic analyzer findings in the function under analysis "
        "(automated tools; may be false positives):\n1. [bandit B602 | CWE-78 | HIGH] shell=True\n"
        "   Flagged line: subprocess.call(cmd, shell=True)"
    )
    assert prediction.root_findings_in_prompt is True
    assert prediction.root_static_findings == [FINDING]
    assert prediction.source_row_ids == [4236]


def test_flag_on_with_no_findings_says_none_reported() -> None:
    llm, _ = _run(_experiment(render=True), _sample([]))
    assert llm.user_prompts[0].startswith(
        "Analyze:\n\ndef f(): pass\n\nStatic analyzer findings in the function under analysis: none reported."
    )


def test_flag_off_prompt_identical_to_template_without_placeholder() -> None:
    with_placeholder, prediction = _run(_experiment(), _sample([FINDING]))
    without_placeholder, _ = _run(_experiment(user_prompt="Analyze:\n\n{code}"), _sample([FINDING]))
    assert with_placeholder.user_prompts == without_placeholder.user_prompts
    assert with_placeholder.counted_texts, "token filter did not run"
    assert with_placeholder.counted_texts == without_placeholder.counted_texts
    # Findings are stored even when not shown.
    assert prediction.root_findings_in_prompt is False
    assert prediction.root_static_findings == [FINDING]


def test_legacy_dataset_and_prompt_unchanged() -> None:
    llm, prediction = _run(_experiment(user_prompt="{code}"), BenchmarkSample(
        id="s1", code="int x;", label=0, metadata={},
    ))
    assert llm.user_prompts[0].startswith("int x;")
    assert prediction.root_static_findings is None
    assert prediction.source_row_ids is None
    assert prediction.root_findings_in_prompt is False


def test_sample_provenance_keys() -> None:
    assert sample_provenance(_sample([FINDING]), True) == {
        "source_row_ids": [4236], "root_static_findings": [FINDING], "root_findings_in_prompt": True,
    }


def test_record_serializes_new_fields_and_old_records_still_load() -> None:
    _, prediction = _run(_experiment(render=True), _sample([FINDING]))
    record = BenchmarkResultProcessor._to_prediction_record(
        BenchmarkResultProcessor.model_construct(), prediction
    )
    dumped = record.model_dump(mode="json")
    assert dumped["source_row_ids"] == [4236]
    assert dumped["root_findings_in_prompt"] is True
    assert dumped["root_static_findings"][0]["flagged_line"] == "subprocess.call(cmd, shell=True)"
    legacy = {k: v for k, v in dumped.items()
              if k not in {"source_row_ids", "root_static_findings", "root_findings_in_prompt"}}
    old = PredictionRecord.model_validate(legacy)
    assert old.root_static_findings is None and old.root_findings_in_prompt is False


def test_report_records_render_flag() -> None:
    processor = BenchmarkResultProcessor(config=_experiment(render=True))
    metrics = MetricsResult.model_validate(
        {"task_type": "binary", "accuracy": 0.0, "summary": {}, "details": {}}
    )
    report = processor.build_report(metrics=metrics, predictions=[], total_time=0.0, total_samples=0)
    assert report.benchmark_info.extra_metadata["render_root_findings"] is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
