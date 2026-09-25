# Root Static-Findings Condition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experiment condition that appends `is_root` static-analyzer findings (Bandit/Dlint/Semgrep) to the LLM prompt, and store root findings plus a stable join key on every prediction so external notebooks can compare function-only, context, context+findings and analyzer-only results.

**Architecture:** `BenchmarkSample` extracts `is_root` findings from the raw `static_findings` list at validation time. A per-dataset flag `render_root_findings` makes the runner fill a `{root_static_findings}` prompt placeholder with a rendered findings block (or `""` when off). Predictions and report records carry `source_row_ids`, `root_static_findings` and `root_findings_in_prompt`. A small entrypoint attaches the same findings to the function-only dataset (stored, never shown). A new split config (`configs/static_findings_root/`) defines the 5-dataset plan.

**Tech Stack:** Python 3.13, pydantic v2, uv, Docker (vLLM). Tests are plain-python scripts (pytest is not installed): `PYTHONPATH=src uv run python tests/<file>.py`.

**Spec:** `docs/superpowers/specs/2026-09-26-root-static-findings-condition-design.md`

## Global Constraints

- **Never `git commit` unless the user explicitly asks in that message.** "Checkpoint" steps only stage files (`git add`).
- Rebuild the Docker image after any change under `src/`: `./build_docker.sh --no-gpu-test`.
- Placeholder name is exactly `root_static_findings` (template text `{root_static_findings}`).
- `is_root` is used exactly as llm_scanner wrote it; do not re-derive or filter by line location.
- Flag-off prompts must be byte-identical to the same template without the placeholder.
- Findings block text (verbatim):
  - header: `Static analyzer findings in the function under analysis (automated tools; may be false positives):`
  - item: `{n}. [{tool} {rule_id}[ | CWE-{cwe_id}][ | {severity}]] {message}` then `   Flagged line: {flagged_line}`
  - none: `Static analyzer findings in the function under analysis: none reported.`
  - the block value always starts with `\n\n`.
- Input datasets (never the stale `datasets_processed/context_assembler/cleanvul_python_matched.json`):
  - `benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json`
  - `benchmarks/context-assembler-dataset/context_assembler_multiplicative_amplification.json`
  - `benchmarks/context-assembler-dataset/cleanvul_python_matched.json`
- Join key: `(tuple(metadata["source_row_ids"]), int(label))`.
- Model `gemma4-12b-it-thinking-sc7-logprobs-seeded`; prompt `strict_exploitable_security_root_findings`; plans `static_findings_root_sweep`, `static_findings_root_smoke` (`sample_limit: 40`).
- No analysis/metrics code (done in external notebooks).

## Review Focus

1. Semgrep messages containing newlines or long whitespace runs → rendered on one line (whitespace collapsed), so the numbered list stays readable. Test in Task 1.
2. Existing datasets with no `static_findings` key (reference_context, cleanvul, …) running with existing `{code}`-only prompts → prompts unchanged, `root_static_findings` is `None`, no KeyError. Test in Task 3.
3. Token-filter pass (`_filter_samples_by_token_limit`) with a template that contains `{root_static_findings}` and the flag off → no KeyError, placeholder renders `""`. Test in Task 3.
4. Report JSON with findings serializes and an old report record without the new fields still validates (defaults). Test in Task 3.
5. A raw finding with no `is_root` key or `is_root: false`, and `static_findings: null` → excluded / `None`, never a crash. Test in Task 1.

---

## File Structure

- Create `src/benchmark/static_findings.py` — `RootStaticFinding` model, `root_findings_from_raw()`, `render_root_findings_block()`, `ROOT_FINDINGS_PLACEHOLDER`. Pure, depends only on pydantic (so `models.py` and `config.py` can import it without cycles).
- Modify `src/benchmark/models.py` — `BenchmarkSample.root_static_findings` + before-validator; `PredictionResult` gains 3 fields.
- Modify `src/benchmark/config.py` — `DatasetConfig.render_root_findings`, `ExperimentConfig.render_root_findings` + placeholder validation.
- Modify `src/benchmark/benchmark_runner.py` — template-value and provenance helpers; both `get_user_prompt` call sites; both `PredictionResult(...)` constructions.
- Modify `src/benchmark/results.py` — `PredictionRecord` gains 3 fields.
- Modify `src/benchmark/result_processor.py` — `_to_prediction_record` copies fields; `extra_metadata["render_root_findings"]`.
- Create `src/entrypoints/attach_root_findings.py` — join + write function-only dataset with findings.
- Modify `src/configs/shared/prompts.json` — new prompt.
- Create `src/configs/static_findings_root/datasets.json`, `src/configs/static_findings_root/experiments.json`.
- Modify `CLAUDE.md` — setup + run section.
- Tests: `tests/test_root_static_findings.py` (Tasks 1–2), `tests/test_root_findings_runner.py` (Task 3), `tests/test_attach_root_findings.py` (Task 4), `tests/test_static_findings_root_configs.py` (Task 5).

---

### Task 1: Static-findings module (model, extraction, rendering)

**Files:**
- Create: `src/benchmark/static_findings.py`
- Test: `tests/test_root_static_findings.py`

**Interfaces:**
- Produces:
  - `ROOT_FINDINGS_PLACEHOLDER: str = "{root_static_findings}"`
  - `class RootStaticFinding(BaseModel)`: `tool: str`, `rule_id: str`, `cwe_id: int | None = None`, `severity: str | None = None`, `message: str`, `file_path: str`, `repo_line: int`, `flagged_line: str`
  - `root_findings_from_raw(code: str, raw_findings: list[dict[str, Any]]) -> list[RootStaticFinding]`
  - `render_root_findings_block(findings: list[RootStaticFinding]) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_root_static_findings.py`:

```python
"""Plain-python checks: root static findings extraction and prompt rendering.

Run: PYTHONPATH=src uv run python tests/test_root_static_findings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.static_findings import (
    ROOT_FINDINGS_PLACEHOLDER,
    RootStaticFinding,
    render_root_findings_block,
    root_findings_from_raw,
)

CODE = "def f(cmd):\n    x = 1\n    subprocess.call(cmd, shell=True)\n    assert x"


def _raw(**overrides) -> dict:
    finding = {
        "tool": "bandit",
        "rule_id": "B602",
        "cwe_id": 78,
        "severity": "HIGH",
        "message": "subprocess call with shell=True identified, security issue.",
        "file_path": "pkg/mod.py",
        "repo_line": 42,
        "snippet_line": 3,
        "is_root": True,
    }
    finding.update(overrides)
    return finding


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_placeholder_constant() -> None:
    assert ROOT_FINDINGS_PLACEHOLDER == "{root_static_findings}"


def test_keeps_only_root_findings_sorted_with_flagged_line() -> None:
    raw = [
        _raw(tool="dlint", rule_id="DUO116", snippet_line=3),
        _raw(snippet_line=4, rule_id="B101", severity="LOW", cwe_id=703),
        _raw(is_root=False),
        {k: v for k, v in _raw().items() if k != "is_root"},
        _raw(),
    ]
    findings = root_findings_from_raw(CODE, raw)
    assert [(f.tool, f.rule_id) for f in findings] == [
        ("bandit", "B602"),
        ("dlint", "DUO116"),
        ("bandit", "B101"),
    ]
    assert findings[0].flagged_line == "subprocess.call(cmd, shell=True)"
    assert findings[2].flagged_line == "assert x"
    assert findings[0].repo_line == 42
    assert findings[0].file_path == "pkg/mod.py"


def test_snippet_line_out_of_range_raises() -> None:
    _raises(ValueError, lambda: root_findings_from_raw(CODE, [_raw(snippet_line=5)]))
    _raises(ValueError, lambda: root_findings_from_raw(CODE, [_raw(snippet_line=0)]))


def test_empty_input_gives_empty_list() -> None:
    assert root_findings_from_raw(CODE, []) == []


def test_render_with_findings() -> None:
    findings = [
        RootStaticFinding(
            tool="bandit", rule_id="B602", cwe_id=78, severity="HIGH",
            message="subprocess call with shell=True identified, security issue.",
            file_path="a.py", repo_line=1, flagged_line="subprocess.call(cmd, shell=True)",
        ),
        RootStaticFinding(
            tool="dlint", rule_id="DUO116", message="use of \"shell=True\" is insecure",
            file_path="a.py", repo_line=1, flagged_line="subprocess.call(cmd, shell=True)",
        ),
    ]
    assert render_root_findings_block(findings) == (
        "\n\nStatic analyzer findings in the function under analysis "
        "(automated tools; may be false positives):\n"
        "1. [bandit B602 | CWE-78 | HIGH] subprocess call with shell=True identified, security issue.\n"
        "   Flagged line: subprocess.call(cmd, shell=True)\n"
        "2. [dlint DUO116] use of \"shell=True\" is insecure\n"
        "   Flagged line: subprocess.call(cmd, shell=True)"
    )


def test_render_without_findings() -> None:
    assert render_root_findings_block([]) == (
        "\n\nStatic analyzer findings in the function under analysis: none reported."
    )


def test_render_collapses_multiline_messages() -> None:
    finding = RootStaticFinding(
        tool="semgrep", rule_id="python.x", message="Line one.\n   Line   two.\n",
        file_path="a.py", repo_line=1, flagged_line="x = 1",
    )
    assert "1. [semgrep python.x] Line one. Line two.\n" in render_root_findings_block([finding])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run python tests/test_root_static_findings.py`
Expected: `ModuleNotFoundError: No module named 'benchmark.static_findings'`

- [ ] **Step 3: Implement `src/benchmark/static_findings.py`**

```python
"""Root static-analysis findings: extraction from raw dataset samples and prompt rendering.

Datasets built by llm_scanner with ``--include-static-findings`` carry a
``static_findings`` list per sample. Findings with ``is_root`` lie inside the
function under analysis; only those are used here.
"""

from typing import Any

from pydantic import BaseModel

ROOT_FINDINGS_PLACEHOLDER = "{root_static_findings}"

_FINDINGS_HEADER = (
    "Static analyzer findings in the function under analysis "
    "(automated tools; may be false positives):"
)
_NO_FINDINGS_TEXT = "Static analyzer findings in the function under analysis: none reported."
_BLOCK_SEPARATOR = "\n\n"


class RootStaticFinding(BaseModel):
    """Analyzer finding located inside the function under analysis."""

    tool: str
    rule_id: str
    cwe_id: int | None = None
    severity: str | None = None
    message: str
    file_path: str
    repo_line: int
    flagged_line: str
    """Stripped text of the flagged line in the snippet the finding was resolved against."""


def root_findings_from_raw(
    code: str, raw_findings: list[dict[str, Any]]
) -> list[RootStaticFinding]:
    """Keep ``is_root`` findings, sorted by snippet position, with their flagged line text.

    Args:
        code: Snippet the findings' ``snippet_line`` values index into.
        raw_findings: Raw ``static_findings`` entries from a dataset sample.

    Returns:
        Root findings sorted by ``(snippet_line, tool, rule_id)``.

    Raises:
        ValueError: If a root finding's ``snippet_line`` lies outside ``code``.
    """
    code_lines: list[str] = code.split("\n")
    root_findings: list[dict[str, Any]] = sorted(
        (finding for finding in raw_findings if finding.get("is_root") is True),
        key=lambda finding: (
            int(finding["snippet_line"]),
            str(finding["tool"]),
            str(finding["rule_id"]),
        ),
    )
    findings: list[RootStaticFinding] = []
    for raw in root_findings:
        snippet_line: int = int(raw["snippet_line"])
        if not 1 <= snippet_line <= len(code_lines):
            raise ValueError(
                f"snippet_line {snippet_line} is outside a snippet of {len(code_lines)} lines"
            )
        findings.append(
            RootStaticFinding(
                tool=str(raw["tool"]),
                rule_id=str(raw["rule_id"]),
                cwe_id=raw.get("cwe_id"),
                severity=raw.get("severity"),
                message=str(raw["message"]),
                file_path=str(raw["file_path"]),
                repo_line=int(raw["repo_line"]),
                flagged_line=code_lines[snippet_line - 1].strip(),
            )
        )
    return findings


def render_root_findings_block(findings: list[RootStaticFinding]) -> str:
    """Render findings as the value of the ``{root_static_findings}`` placeholder.

    The value starts with a blank-line separator so it lays out cleanly whether
    the placeholder sits after or before ``{code}``.
    """
    if not findings:
        return _BLOCK_SEPARATOR + _NO_FINDINGS_TEXT
    lines: list[str] = [_FINDINGS_HEADER]
    for index, finding in enumerate(findings, start=1):
        tags: list[str] = [f"{finding.tool} {finding.rule_id}"]
        if finding.cwe_id is not None:
            tags.append(f"CWE-{finding.cwe_id}")
        if finding.severity:
            tags.append(finding.severity)
        message: str = " ".join(finding.message.split())
        lines.append(f"{index}. [{' | '.join(tags)}] {message}")
        lines.append(f"   Flagged line: {finding.flagged_line}")
    return _BLOCK_SEPARATOR + "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run python tests/test_root_static_findings.py`
Expected: `ALL PASSED`

- [ ] **Step 5: Checkpoint (stage only)**

```bash
git add src/benchmark/static_findings.py tests/test_root_static_findings.py
```

---

### Task 2: `BenchmarkSample.root_static_findings`

**Files:**
- Modify: `src/benchmark/models.py:1-28` (imports, `BenchmarkSample`)
- Test: `tests/test_root_static_findings.py` (append)

**Interfaces:**
- Consumes: `RootStaticFinding`, `root_findings_from_raw` (Task 1).
- Produces: `BenchmarkSample.root_static_findings: list[RootStaticFinding] | None` (default `None`). Raw input key `static_findings` is converted; a raw `root_static_findings` key (written by Task 4's entrypoint) is loaded as-is and wins.

- [ ] **Step 1: Append failing tests** (above the `if __name__ == "__main__":` block)

```python
from benchmark.models import BenchmarkSample


def _sample_dict(**extra) -> dict:
    return {"id": "s1", "code": CODE, "label": 1, "metadata": {"source_row_ids": [7]}, **extra}


def test_sample_extracts_root_findings_from_static_findings() -> None:
    sample = BenchmarkSample.model_validate(
        _sample_dict(static_findings=[_raw(), _raw(is_root=False)], source_map=[])
    )
    assert sample.root_static_findings is not None
    assert [f.rule_id for f in sample.root_static_findings] == ["B602"]


def test_sample_without_static_findings_has_none() -> None:
    assert BenchmarkSample.model_validate(_sample_dict()).root_static_findings is None
    assert (
        BenchmarkSample.model_validate(_sample_dict(static_findings=None)).root_static_findings
        is None
    )


def test_sample_with_no_root_findings_has_empty_list() -> None:
    sample = BenchmarkSample.model_validate(_sample_dict(static_findings=[_raw(is_root=False)]))
    assert sample.root_static_findings == []


def test_sample_loads_precomputed_root_findings() -> None:
    precomputed = RootStaticFinding(
        tool="bandit", rule_id="B101", message="m", file_path="a.py",
        repo_line=3, flagged_line="assert x",
    ).model_dump(mode="json")
    sample = BenchmarkSample.model_validate(_sample_dict(root_static_findings=[precomputed]))
    assert sample.root_static_findings == [RootStaticFinding.model_validate(precomputed)]


def test_real_context_dataset_root_findings() -> None:
    import json

    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json"
    )
    if not path.exists():
        print(f"SKIP: {path} missing")
        return
    samples = [
        BenchmarkSample.model_validate(raw)
        for raw in json.loads(path.read_text())["samples"]
    ]
    assert len(samples) == 732
    assert sum(len(s.root_static_findings or []) for s in samples) == 626
    assert sum(bool(s.root_static_findings) for s in samples) == 183
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run python tests/test_root_static_findings.py`
Expected: `AssertionError` (or `AttributeError`) in `test_sample_extracts_root_findings_from_static_findings`.

- [ ] **Step 3: Implement in `src/benchmark/models.py`**

Change the pydantic import and add the static-findings import:

```python
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

from benchmark.enums import TaskType
from benchmark.static_findings import RootStaticFinding, root_findings_from_raw
```

In `BenchmarkSample`, after `severity: str | None = None`, add:

```python
    root_static_findings: list[RootStaticFinding] | None = None
    """Findings inside the function under analysis; None when the dataset has no findings."""

    @model_validator(mode="before")
    @classmethod
    def extract_root_static_findings(cls, data: Any) -> Any:
        """Derive ``root_static_findings`` from a raw llm_scanner ``static_findings`` list."""
        if not isinstance(data, dict) or "root_static_findings" in data:
            return data
        raw_findings: list[dict[str, Any]] | None = data.get("static_findings")
        if raw_findings is None:
            return data
        return {
            **data,
            "root_static_findings": root_findings_from_raw(data["code"], raw_findings),
        }
```

(`static_findings` and `source_map` stay unknown keys, which pydantic ignores by default.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run python tests/test_root_static_findings.py`
Expected: `ALL PASSED` (the real-dataset test prints nothing extra when the file exists).

Also run the existing suite that constructs samples to confirm no regression:
Run: `PYTHONPATH=src uv run python tests/test_prompt_retention.py`
Expected: `ALL PASSED`

- [ ] **Step 5: Checkpoint (stage only)**

```bash
git add src/benchmark/models.py tests/test_root_static_findings.py
```

---

### Task 3: Dataset flag, prompt rendering, and stored per-sample provenance

**Files:**
- Modify: `src/benchmark/config.py` (`DatasetConfig` ~line 75; `ExperimentConfig` fields ~line 114–170, `model_post_init` ~line 177, `from_configs` ~line 283–338)
- Modify: `src/benchmark/models.py` (`PredictionResult`, after `vote_counts`)
- Modify: `src/benchmark/benchmark_runner.py` (imports; new module-level helpers; line ~177 and ~264 `get_user_prompt` calls; both `PredictionResult(...)` at ~357 and ~390)
- Modify: `src/benchmark/results.py` (`PredictionRecord` ~line 56)
- Modify: `src/benchmark/result_processor.py` (`_to_prediction_record` ~line 159; `_build_benchmark_info` `extra_metadata` ~line 281)
- Test: `tests/test_root_findings_runner.py`

**Interfaces:**
- Consumes: `ROOT_FINDINGS_PLACEHOLDER`, `RootStaticFinding`, `render_root_findings_block` (Task 1); `BenchmarkSample.root_static_findings` (Task 2).
- Produces:
  - `DatasetConfig.render_root_findings: bool = False`; `ExperimentConfig.render_root_findings: bool = False` (copied from the dataset in `from_configs`).
  - `ExperimentConfig.model_post_init` raises `ValueError` when `render_root_findings` is true and `"{root_static_findings}"` is not in `user_prompt_template`.
  - `benchmark_runner.user_template_values(sample: BenchmarkSample, render_root_findings: bool) -> dict[str, str]` — keys `code`, `root_static_findings`; raises `ValueError` when rendering is on and `sample.root_static_findings is None`.
  - `benchmark_runner.sample_provenance(sample: BenchmarkSample, render_root_findings: bool) -> dict[str, Any]` — keys `source_row_ids`, `root_static_findings`, `root_findings_in_prompt`.
  - `PredictionResult` and `PredictionRecord` fields: `source_row_ids: list[int] | None = None`, `root_static_findings: list[RootStaticFinding] | None = None`, `root_findings_in_prompt: bool = False`.
  - `BenchmarkInfo.extra_metadata["render_root_findings"]: bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_root_findings_runner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run python tests/test_root_findings_runner.py`
Expected: `ImportError: cannot import name 'sample_provenance' from 'benchmark.benchmark_runner'`

- [ ] **Step 3: Config flag and validation (`src/benchmark/config.py`)**

Add the import near the other `benchmark.` imports:

```python
from benchmark.static_findings import ROOT_FINDINGS_PLACEHOLDER
```

`DatasetConfig`, after `cwe_type: str | None = None`:

```python
    render_root_findings: bool = False
    """Fill the {root_static_findings} prompt placeholder with the sample's root findings."""
```

`ExperimentConfig`, after `sampling_seed: int | None = None` and its docstring:

```python
    render_root_findings: bool = False
    """Copied from the dataset config; see DatasetConfig.render_root_findings."""
```

In `ExperimentConfig.model_post_init`, directly before `super().model_post_init(context)`:

```python
        if (
            self.render_root_findings
            and ROOT_FINDINGS_PLACEHOLDER not in self.user_prompt_template
        ):
            raise ValueError(
                "render_root_findings requires the user prompt template to contain "
                f"{ROOT_FINDINGS_PLACEHOLDER}"
            )
```

In `from_configs`, in the `cls(...)` call after `cwe_type=dataset_config.cwe_type,`:

```python
            render_root_findings=dataset_config.render_root_findings,
```

- [ ] **Step 4: Prediction fields (`src/benchmark/models.py`, `src/benchmark/results.py`)**

`PredictionResult`, after `vote_counts` and its docstring:

```python
    source_row_ids: list[int] | None = None
    """CleanVul source row ids; with true_label, a join key that survives dataset rebuilds."""
    root_static_findings: list[RootStaticFinding] | None = None
    """Root findings of the sample, stored whether or not they were shown to the model."""
    root_findings_in_prompt: bool = False
    """True when the findings block was rendered into the prompt."""
```

`src/benchmark/results.py`: add `from benchmark.static_findings import RootStaticFinding` to the imports, and in `PredictionRecord` after `error_message: str | None`:

```python
    source_row_ids: list[int] | None = None
    """CleanVul source row ids; with true_label, a join key that survives dataset rebuilds."""
    root_static_findings: list[RootStaticFinding] | None = None
    """Root findings of the sample, stored whether or not they were shown to the model."""
    root_findings_in_prompt: bool = False
    """True when the findings block was rendered into the prompt."""
```

- [ ] **Step 5: Runner helpers and call sites (`src/benchmark/benchmark_runner.py`)**

Imports: add `from typing import Any`, change the models import to `from benchmark.models import BenchmarkSample, PredictionResult, SampleCollection`, and add `from benchmark.static_findings import render_root_findings_block`.

Add module-level helpers after `_aggregate_draw_confidences`:

```python
def user_template_values(
    sample: BenchmarkSample, render_root_findings: bool
) -> dict[str, str]:
    """Per-sample user-prompt template values.

    Raises:
        ValueError: If rendering is on but the sample carries no findings information.
    """
    if not render_root_findings:
        return {"code": sample.code, "root_static_findings": ""}
    if sample.root_static_findings is None:
        raise ValueError(
            f"Sample {sample.id} has no static findings but its dataset sets render_root_findings"
        )
    return {
        "code": sample.code,
        "root_static_findings": render_root_findings_block(sample.root_static_findings),
    }


def sample_provenance(
    sample: BenchmarkSample, render_root_findings: bool
) -> dict[str, Any]:
    """Per-sample fields stored on every prediction for cross-report analysis."""
    raw_row_ids: Any = sample.metadata.get("source_row_ids")
    return {
        "source_row_ids": [int(row_id) for row_id in raw_row_ids]
        if raw_row_ids is not None
        else None,
        "root_static_findings": sample.root_static_findings,
        "root_findings_in_prompt": render_root_findings,
    }
```

Replace line ~177:

```python
            user_prompt = prompt_generator.get_user_prompt(
                user_template_values(sample, self.config.render_root_findings)
            )
```

Replace line ~264:

```python
            usr_p = prompt_generator.get_user_prompt(
                user_template_values(sample, self.config.render_root_findings)
            )
```

In `_process_samples_with_batch_optimization`, at the top of the `for i, sample in enumerate(samples):` body add:

```python
            provenance: dict[str, Any] = sample_provenance(
                sample, self.config.render_root_findings
            )
```

and add `**provenance,` as the last argument of **both** `PredictionResult(...)` constructions (success branch ~357 and error branch ~390).

- [ ] **Step 6: Report record and metadata (`src/benchmark/result_processor.py`)**

In `_to_prediction_record`, extend the `PredictionRecord(...)` call:

```python
            source_row_ids=prediction.source_row_ids,
            root_static_findings=prediction.root_static_findings,
            root_findings_in_prompt=prediction.root_findings_in_prompt,
```

In `_build_benchmark_info`, after the `vulnerability_type` block:

```python
        extra_metadata["render_root_findings"] = self.config.render_root_findings
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run python tests/test_root_findings_runner.py`
Expected: `ALL PASSED`

Regression:
Run: `for t in tests/test_prompt_retention.py tests/test_confidence_methods.py tests/test_root_static_findings.py; do PYTHONPATH=src uv run python $t || break; done`
Expected: `ALL PASSED` for each (if a file has no `__main__` runner, run it with `uv run --with pytest pytest <file> -q` instead and expect all passed).

- [ ] **Step 8: Checkpoint (stage only)**

```bash
git add src/benchmark/config.py src/benchmark/models.py src/benchmark/benchmark_runner.py \
    src/benchmark/results.py src/benchmark/result_processor.py tests/test_root_findings_runner.py
```

---

### Task 4: Attach root findings to the function-only dataset

**Files:**
- Create: `src/entrypoints/attach_root_findings.py`
- Test: `tests/test_attach_root_findings.py`

**Interfaces:**
- Consumes: `root_findings_from_raw`, `RootStaticFinding` (Task 1); Task 2's validator loads the written `root_static_findings` key as-is.
- Produces: `attach_root_findings(target_payload: dict[str, Any], source_payload: dict[str, Any]) -> dict[str, Any]`; CLI `--target/--findings-source/--output`; file `datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_attach_root_findings.py`:

```python
"""Plain-python checks: function-only samples get the context dataset's root findings.

Run: PYTHONPATH=src uv run python tests/test_attach_root_findings.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.models import BenchmarkSample
from entrypoints.attach_root_findings import attach_root_findings, main

SOURCE_CODE = "ctx = 1\ndef f(cmd):\n    subprocess.call(cmd, shell=True)"


def _finding(is_root: bool = True) -> dict:
    return {
        "tool": "bandit", "rule_id": "B602", "cwe_id": 78, "severity": "HIGH",
        "message": "shell=True", "file_path": "a.py", "repo_line": 10,
        "snippet_line": 3, "is_root": is_root,
    }


def _source(row_ids: list[int], label: int, findings: list[dict]) -> dict:
    return {"id": f"src-{row_ids}-{label}", "code": SOURCE_CODE, "label": label,
            "metadata": {"source_row_ids": row_ids}, "static_findings": findings}


def _target(row_ids: list[int], label: int) -> dict:
    return {"id": f"fo-{row_ids}-{label}", "code": "def f(cmd): ...", "label": label,
            "metadata": {"source_row_ids": row_ids, "commit_url": "u"}}


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_attaches_root_findings_by_row_ids_and_label() -> None:
    source = {"metadata": {}, "samples": [
        _source([1], 1, [_finding(), _finding(is_root=False)]),
        _source([1], 0, []),
    ]}
    target = {"metadata": {"name": "fo"}, "samples": [_target([1], 1), _target([1], 0)]}
    result = attach_root_findings(target, source)
    assert result["metadata"] == {"name": "fo"}
    vuln, fixed = result["samples"]
    assert vuln["code"] == "def f(cmd): ..."
    assert vuln["metadata"]["commit_url"] == "u"
    assert [f["rule_id"] for f in vuln["root_static_findings"]] == ["B602"]
    assert vuln["root_static_findings"][0]["flagged_line"] == "subprocess.call(cmd, shell=True)"
    assert fixed["root_static_findings"] == []
    loaded = BenchmarkSample.model_validate(vuln)
    assert loaded.root_static_findings is not None and loaded.root_static_findings[0].repo_line == 10


def test_unmatched_target_raises() -> None:
    source = {"samples": [_source([1], 1, [])]}
    _raises(ValueError, lambda: attach_root_findings({"samples": [_target([2], 1)]}, source))


def test_duplicate_source_key_raises() -> None:
    source = {"samples": [_source([1], 1, []), _source([1], 1, [])]}
    _raises(ValueError, lambda: attach_root_findings({"samples": [_target([1], 1)]}, source))


def test_source_without_static_findings_raises() -> None:
    bare = _source([1], 1, [])
    del bare["static_findings"]
    _raises(ValueError, lambda: attach_root_findings({"samples": [_target([1], 1)]}, {"samples": [bare]}))


def test_cli_writes_nothing_on_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "t.json").write_text(json.dumps({"samples": [_target([2], 1)]}))
        (tmp_path / "s.json").write_text(json.dumps({"samples": [_source([1], 1, [])]}))
        output = tmp_path / "out" / "o.json"
        _raises(ValueError, lambda: main([
            "--target", str(tmp_path / "t.json"), "--findings-source", str(tmp_path / "s.json"),
            "--output", str(output),
        ]))
        assert not output.exists()


def test_cli_writes_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "t.json").write_text(json.dumps({"samples": [_target([1], 1)]}))
        (tmp_path / "s.json").write_text(json.dumps({"samples": [_source([1], 1, [_finding()])]}))
        output = tmp_path / "out" / "o.json"
        main(["--target", str(tmp_path / "t.json"), "--findings-source", str(tmp_path / "s.json"),
              "--output", str(output)])
        written = json.loads(output.read_text())
        assert len(written["samples"][0]["root_static_findings"]) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run python tests/test_attach_root_findings.py`
Expected: `ModuleNotFoundError: No module named 'entrypoints.attach_root_findings'`

- [ ] **Step 3: Implement `src/entrypoints/attach_root_findings.py`**

```python
#!/usr/bin/env python3
"""
Attach root static findings to a function-only dataset.

Copies the ``is_root`` static-analysis findings of a context dataset (built by
llm_scanner with static findings) onto the matching samples of a function-only
dataset, as ``root_static_findings``. The findings are stored for analysis only;
they reach a prompt only when the dataset config sets ``render_root_findings``.

Samples are matched on ``(metadata.source_row_ids, label)``. Every target sample
must match exactly one source sample, otherwise nothing is written.

Usage (host):
    PYTHONPATH=src uv run python src/entrypoints/attach_root_findings.py \\
        --target benchmarks/context-assembler-dataset/cleanvul_python_matched.json \\
        --findings-source benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json \\
        --output datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from benchmark.static_findings import root_findings_from_raw
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

SampleKey = tuple[tuple[int, ...], int]


def _sample_key(sample: dict[str, Any]) -> SampleKey:
    """Join key that survives dataset rebuilds (sample ids do not)."""
    row_ids: list[int] | None = sample.get("metadata", {}).get("source_row_ids")
    if not row_ids:
        raise ValueError(f"Sample {sample.get('id')} has no metadata.source_row_ids")
    return tuple(int(row_id) for row_id in row_ids), int(sample["label"])


def attach_root_findings(
    target_payload: dict[str, Any], source_payload: dict[str, Any]
) -> dict[str, Any]:
    """Return ``target_payload`` with ``root_static_findings`` added to every sample.

    Raises:
        ValueError: On a duplicate source key, a source sample without
            ``static_findings``, or a target sample with no matching source.
    """
    source_by_key: dict[SampleKey, dict[str, Any]] = {}
    for source in source_payload["samples"]:
        key: SampleKey = _sample_key(source)
        if key in source_by_key:
            raise ValueError(f"Duplicate findings-source key {key}")
        if source.get("static_findings") is None:
            raise ValueError(f"Findings-source sample {source.get('id')} has no static_findings")
        source_by_key[key] = source

    samples: list[dict[str, Any]] = []
    for target in target_payload["samples"]:
        key = _sample_key(target)
        source = source_by_key.get(key)
        if source is None:
            raise ValueError(f"No findings-source sample for key {key} (target {target.get('id')})")
        findings = root_findings_from_raw(source["code"], source["static_findings"])
        samples.append(
            {**target, "root_static_findings": [f.model_dump(mode="json") for f in findings]}
        )
    return {**target_payload, "samples": samples}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", type=Path, required=True, help="Function-only dataset JSON")
    parser.add_argument(
        "--findings-source", type=Path, required=True,
        help="Context dataset JSON carrying static_findings",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output dataset JSON")
    args = parser.parse_args(argv)

    target_payload: dict[str, Any] = json.loads(args.target.read_text(encoding="utf-8"))
    source_payload: dict[str, Any] = json.loads(args.findings_source.read_text(encoding="utf-8"))
    result: dict[str, Any] = attach_root_findings(target_payload, source_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with_findings: int = sum(bool(s["root_static_findings"]) for s in result["samples"])
    _LOGGER.info(
        "Wrote %d samples (%d with root findings) to %s",
        len(result["samples"]), with_findings, args.output,
    )


if __name__ == "__main__":
    setup_logging()
    main()
```

(`setup_logging(verbose: bool = False)` in `src/logging_tools.py` needs no arguments.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run python tests/test_attach_root_findings.py`
Expected: `ALL PASSED`

- [ ] **Step 5: Generate the real file and verify counts**

Run:
```bash
PYTHONPATH=src uv run python src/entrypoints/attach_root_findings.py \
    --target benchmarks/context-assembler-dataset/cleanvul_python_matched.json \
    --findings-source benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json \
    --output datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json
python3 -c "
import json; s=json.load(open('datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json'))['samples']
print(len(s), sum(len(x['root_static_findings']) for x in s), sum(bool(x['root_static_findings']) for x in s))"
```
Expected: log line `Wrote 732 samples (183 with root findings)`, then `732 626 183`.

- [ ] **Step 6: Checkpoint (stage only)**

```bash
git add src/entrypoints/attach_root_findings.py tests/test_attach_root_findings.py
```
(`datasets_processed/` is git-ignored; the generated file is not staged.)

---

### Task 5: Prompt, configs, docs, and Docker smoke run

**Files:**
- Modify: `src/configs/shared/prompts.json` (add entry after `strict_exploitable_security`)
- Create: `src/configs/static_findings_root/datasets.json`
- Create: `src/configs/static_findings_root/experiments.json`
- Modify: `CLAUDE.md` (new section after "Reference-context evaluation")
- Test: `tests/test_static_findings_root_configs.py`

**Interfaces:**
- Consumes: `DatasetConfig.render_root_findings`, `ExperimentConfig` validation (Task 3); dataset file from Task 4.
- Produces: plans `static_findings_root_sweep`, `static_findings_root_smoke`; prompt `strict_exploitable_security_root_findings`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_static_findings_root_configs.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run python tests/test_static_findings_root_configs.py`
Expected: `KeyError: 'strict_exploitable_security_root_findings'`

- [ ] **Step 3: Add the prompt**

In `src/configs/shared/prompts.json`, directly after the closing `},` of `"strict_exploitable_security"`, insert an entry `"strict_exploitable_security_root_findings"` with:
- `"name": "Strict Exploitable Security Analysis + root static findings"`
- `"system_prompt"`: copied **verbatim** from `strict_exploitable_security` (copy the JSON string exactly; the test compares them).
- `"user_prompt": "Analyze this code for a real, exploitable security vulnerability:\n\n{code}{root_static_findings}"`

Validate the file still parses: `python3 -m json.tool src/configs/shared/prompts.json > /dev/null && echo OK` → `OK`.

- [ ] **Step 4: Create `src/configs/static_findings_root/datasets.json`**

```json
{
    "datasets": {
        "cleanvul_python_matched_root_findings": {
            "dataset_path": "datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json",
            "task_type": "binary_vulnerability",
            "description": "CleanVul function-only baseline; root static findings stored, not shown (1=vulnerable, 0=safe)"
        },
        "cpg_structural_root_findings_off": {
            "dataset_path": "benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json",
            "task_type": "binary_vulnerability",
            "description": "cpg_structural context; root static findings stored, not shown"
        },
        "cpg_structural_root_findings_on": {
            "dataset_path": "benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json",
            "task_type": "binary_vulnerability",
            "description": "cpg_structural context with root static findings in the prompt",
            "render_root_findings": true
        },
        "mult_amp_root_findings_off": {
            "dataset_path": "benchmarks/context-assembler-dataset/context_assembler_multiplicative_amplification.json",
            "task_type": "binary_vulnerability",
            "description": "multiplicative_amplification context; root static findings stored, not shown"
        },
        "mult_amp_root_findings_on": {
            "dataset_path": "benchmarks/context-assembler-dataset/context_assembler_multiplicative_amplification.json",
            "task_type": "binary_vulnerability",
            "description": "multiplicative_amplification context with root static findings in the prompt",
            "render_root_findings": true
        }
    }
}
```

- [ ] **Step 5: Create `src/configs/static_findings_root/experiments.json`**

```json
{
    "experiment_metadata": {
        "name": "Root Static Findings Condition",
        "description": "Function-only vs cpg/multiplicative context, with and without is_root static-analyzer findings in the prompt",
        "version": "1.0",
        "dataset": "CleanVul score-4 Python with static findings",
        "created_date": "2026-09-26"
    },
    "experiment_plans": {
        "static_findings_root_sweep": {
            "description": "Five-condition root static findings sweep",
            "datasets": [
                "cleanvul_python_matched_root_findings",
                "cpg_structural_root_findings_off",
                "cpg_structural_root_findings_on",
                "mult_amp_root_findings_off",
                "mult_amp_root_findings_on"
            ],
            "models": ["gemma4-12b-it-thinking-sc7-logprobs-seeded"],
            "prompts": ["strict_exploitable_security_root_findings"]
        },
        "static_findings_root_smoke": {
            "description": "Smoke test of the root static findings sweep",
            "datasets": [
                "cleanvul_python_matched_root_findings",
                "cpg_structural_root_findings_off",
                "cpg_structural_root_findings_on",
                "mult_amp_root_findings_off",
                "mult_amp_root_findings_on"
            ],
            "models": ["gemma4-12b-it-thinking-sc7-logprobs-seeded"],
            "prompts": ["strict_exploitable_security_root_findings"],
            "sample_limit": 40
        }
    },
    "output_settings": {
        "base_output_dir": "results/static_findings_root",
        "save_predictions": true,
        "save_metrics": true,
        "save_detailed_report": true,
        "include_timestamp": true
    }
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run python tests/test_static_findings_root_configs.py`
Expected: `ALL PASSED`

- [ ] **Step 7: Document in `CLAUDE.md`**

Add after the "Reference-context evaluation" section:

````markdown
## Root static-findings condition

Five conditions: function-only (`cleanvul_python_matched_root_findings`),
cpg_structural and multiplicative_amplification context, each with root
(`is_root`) Bandit/Dlint/Semgrep findings off/on in the prompt. Datasets with
`"render_root_findings": true` fill the `{root_static_findings}` placeholder of
prompt `strict_exploitable_security_root_findings`; otherwise it renders `""`.
Every prediction record stores `source_row_ids` (join key with `true_label`),
`root_static_findings` and `root_findings_in_prompt`, whether or not the
findings were shown. Analysis happens in external notebooks.

One-time setup (attach findings to the function-only dataset):
```bash
PYTHONPATH=src uv run python src/entrypoints/attach_root_findings.py \
    --target benchmarks/context-assembler-dataset/cleanvul_python_matched.json \
    --findings-source benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json \
    --output datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json
```

Run (`static_findings_root_smoke` for 40 samples per condition):
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler static_findings_root_sweep \
    --config-dir configs/shared \
    --experiments-config configs/static_findings_root/experiments.json \
    --datasets-config configs/static_findings_root/datasets.json
```
````

- [ ] **Step 8: Rebuild the image and run the smoke plan**

Run: `./build_docker.sh --no-gpu-test`
Expected: build succeeds.

Run:
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler static_findings_root_smoke \
    --config-dir configs/shared \
    --experiments-config configs/static_findings_root/experiments.json \
    --datasets-config configs/static_findings_root/datasets.json
```
Expected: 5 experiments complete, reports under `results/static_findings_root/static_findings_root_smoke/`.

Verify the reports:
```bash
python3 - <<'EOF'
import glob, json
for path in sorted(glob.glob("results/static_findings_root/static_findings_root_smoke/*/*/*/benchmark_report_*.json")):
    report = json.load(open(path))
    preds = report["predictions"]
    shown = {p["root_findings_in_prompt"] for p in preds}
    with_findings = sum(bool(p["root_static_findings"]) for p in preds)
    has_block = sum("Static analyzer findings" in (p["inference_data"]["prompt_text"] or "") for p in preds)
    print(path.split("/")[3], len(preds), shown, with_findings, has_block,
          all(p["source_row_ids"] for p in preds), report["benchmark_info"]["extra_metadata"])
EOF
```
Expected: `_on` datasets → `{True}` and `has_block == len(preds)`; `_off` and function-only → `{False}` and `has_block == 0`; `source_row_ids` present for all; `extra_metadata.render_root_findings` matches. If `prompt_text` is `None` (backend didn't retain it), check `has_block` via the runner test instead and note it in the report.

- [ ] **Step 9: Checkpoint (stage only)**

```bash
git add src/configs/shared/prompts.json src/configs/static_findings_root/ \
    tests/test_static_findings_root_configs.py CLAUDE.md
```
