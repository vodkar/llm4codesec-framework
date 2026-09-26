# Finding Verification, Exclusions and Coverage Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude audited wrong-label samples and noisy analyzer rules via dataset config, add a finding-verification prompt, report full metrics at configurable coverage levels, and recompute/merge old and new sweep results without re-running inference.

**Architecture:** Pure helpers (`sample_exclusions.py`, `coverage.py`, additions to `static_findings.py`) are wired through `DatasetConfig` → `ExperimentConfig` → `BenchmarkRunner._prepare_samples` / `user_template_values` / `BinaryMetricsCalculator(coverage_levels)`. A new `merge_results.py` re-reads saved reports, applies exclusions and an analysis-time rule filter, recomputes metrics with the same calculator, and writes merged results plus a bolded `summary.md`; `cli.py merge-plan-results` wraps it.

**Tech Stack:** Python 3.13, pydantic v2, typer, scikit-learn, Docker + vLLM. Tests are plain-python scripts with a `__main__` runner.

**Spec:** `docs/superpowers/specs/2026-09-26-finding-verification-design.md`

## Global Constraints

- Work only in the worktree `~/worktrees/llm4codesec-fv` (branch `feature/finding-verification`). **Never edit `/var/opt/llm4codesec-framework` source files** — another session has uncommitted edits there. Commits go to `feature/finding-verification` only, never `main`.
- Python for tests (reuses the main checkout's venv, no sync): `PY=/var/opt/llm4codesec-framework/.venv/bin/python`; single test file: `PYTHONPATH=src $PY tests/<file>.py`.
- Whole suite: `PYTHONPATH=src UV_PROJECT_ENVIRONMENT=/var/opt/llm4codesec-framework/.venv uv run --no-sync --with pytest pytest tests -q -p no:cacheprovider > /tmp/claude-1000/-var-opt-llm4codesec-framework/52f0f71a-515d-4449-ad71-de2fb78391f3/scratchpad/suite.log 2>&1; tail -3 …/suite.log`.
- Exclusion key: `(tuple(source_row_ids), int(label))`. Exactly the 18 entries listed in Task 5; identical list on all 7 entries of `src/configs/static_findings_root/datasets.json`.
- Rule filter patterns: `["B101", "B113"]`, `fnmatch.fnmatchcase` on `rule_id`; only on the 2 new dataset entries; merge uses `--exclude-finding-rules B101,B113` (analysis only).
- Default coverage levels: `(0.25, 0.5, 0.75, 1.0)`; each value must be in (0, 1].
- Coverage metric names per level: `accuracy`, `precision`, `recall`, `f1_score`, `fpr`, `fnr`; summary key `{prefix}{name}_at_coverage_{round(level*100)}`; `None` for zero denominators.
- New prompt key `finding_verification_root_findings`; new dataset keys `cpg_structural_root_findings_verify`, `mult_amp_root_findings_verify`; new plans `static_findings_verify_sweep`, `static_findings_verify_smoke` (`sample_limit: 40`); model `gemma4-12b-it-thinking-sc7-logprobs-seeded`.
- Docker: build this branch as `llm4codesec-benchmark:fv` (never overwrite `:latest`, the other session uses it); run with the compose override from Task 6. One GPU job at a time — check `nvidia-smi` and `docker ps` first.

## Review Focus

1. A saved report whose predictions lack `source_row_ids` merged with exclusions configured → clear `ValueError` naming the report, not a `TypeError`. Test in Task 4.
2. A condition directory holding several reports (smoke re-runs, restarts) → merge uses only the newest by file name. Test in Task 4.
3. Coverage level selecting samples of a single class (e.g. top 25% all label 1) → `fpr`/`precision` are `None` where undefined, never `ZeroDivisionError`. Test in Task 3.
4. `omit_empty_root_findings` true but `render_root_findings` false → placeholder still renders `""` and prompts stay byte-identical to the base prompt. Test in Task 2.
5. A plan directory holding several models' reports for the same dataset/prompt → merge refuses (`ValueError`) unless `--model` picks one, never silently overwrites a row. Test in Task 4.
6. `summary.md` columns where some rows are `None` (e.g. no `pr_auc`) or tie → no crash, `—` shown, all tied best values bolded. Test in Task 4.

---

## File Structure

- Create `src/benchmark/sample_exclusions.py` — `SampleExclusion`, `sample_key`, `ExclusionResult`, `apply_sample_exclusions`.
- Create `src/benchmark/coverage.py` — `DEFAULT_COVERAGE_LEVELS`, `validate_coverage_levels`.
- Create `src/benchmark/merge_results.py` — record→prediction conversion, report recompute, summary rendering, `merge_plan_results`.
- Modify `src/benchmark/static_findings.py` — `filter_root_findings`; `render_root_findings_block(..., omit_empty)`.
- Modify `src/benchmark/config.py` — dataset/experiment fields, coverage plumbing.
- Modify `src/benchmark/benchmark_runner.py` — `_prepare_samples`, `user_template_values(..., omit_empty)`, calculator coverage levels, `run_metadata`.
- Modify `src/benchmark/results.py` — `BenchmarkRunResult.run_metadata`.
- Modify `src/benchmark/result_processor.py` — `run_metadata` into `extra_metadata`; config fields into `extra_metadata`.
- Modify `src/benchmark/run_experiment.py` — pass `run_metadata`.
- Modify `src/benchmark/metrics_calculator.py` — per-instance coverage levels and full per-level metrics.
- Modify `src/cli.py` — `merge-plan-results` command.
- Modify `src/configs/shared/prompts.json`, `src/configs/static_findings_root/datasets.json`, `src/configs/static_findings_root/experiments.json`, `CLAUDE.md`.
- Tests: `tests/test_sample_exclusions.py` (T1), `tests/test_finding_filters.py` (T2), `tests/test_coverage_metrics.py` (T3), `tests/test_merge_results.py` (T4), `tests/test_finding_verification_configs.py` (T5); modify `tests/test_metrics_calculator.py` (T3).

---

### Task 1: Sample exclusions

**Files:**
- Create: `src/benchmark/sample_exclusions.py`
- Modify: `src/benchmark/config.py` (`DatasetConfig` fields; `ExperimentConfig` fields; `from_configs`)
- Modify: `src/benchmark/benchmark_runner.py` (`run`, new `_prepare_samples`)
- Modify: `src/benchmark/results.py` (`BenchmarkRunResult`)
- Modify: `src/benchmark/result_processor.py` (`build_report`, `build_and_save`)
- Modify: `src/benchmark/run_experiment.py` (`build_and_save` call ~line 249)
- Test: `tests/test_sample_exclusions.py`

**Interfaces:**
- Produces:
  - `SampleKey = tuple[tuple[int, ...], int]`
  - `class SampleExclusion(BaseModel)`: `source_row_ids: list[int]`, `label: int`, `reason: str = ""`, property `key -> SampleKey`
  - `sample_key(sample: BenchmarkSample) -> SampleKey` (raises `ValueError` without `source_row_ids`)
  - `class ExclusionResult(BaseModel)`: `kept: list[BenchmarkSample]`, `excluded: list[SampleExclusion]`, `unmatched: list[SampleExclusion]`
  - `apply_sample_exclusions(samples: list[BenchmarkSample], exclusions: list[SampleExclusion]) -> ExclusionResult`
  - `DatasetConfig.exclude_samples` / `ExperimentConfig.exclude_samples: list[SampleExclusion] = []`
  - `BenchmarkRunner._prepare_samples(samples: SampleCollection) -> tuple[SampleCollection, dict[str, Any]]`
  - `BenchmarkRunResult.run_metadata: dict[str, Any]`; `build_report(..., run_metadata: dict[str, Any] | None = None)` merges it into `benchmark_info.extra_metadata` (keys `excluded_samples`, `unmatched_exclusions`).

- [ ] **Step 1: Write the failing tests** — create `tests/test_sample_exclusions.py`:

```python
"""Plain-python checks: audited samples are excluded by (source_row_ids, label).

Run: PYTHONPATH=src python tests/test_sample_exclusions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.config import ExperimentConfig
from benchmark.models import BenchmarkSample, SampleCollection
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import MetricsResult
from benchmark.sample_exclusions import SampleExclusion, apply_sample_exclusions, sample_key
from datasets.loaders.base import JsonDatasetLoader


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _sample(sample_id: str, row_ids: list[int] | None, label: int) -> BenchmarkSample:
    metadata = {} if row_ids is None else {"source_row_ids": row_ids}
    return BenchmarkSample(id=sample_id, code="x", label=label, metadata=metadata)


def _config(exclusions: list[dict] | None = None) -> dict:
    dataset: dict = {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}
    if exclusions is not None:
        dataset["exclude_samples"] = exclusions
    return {
        "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                         "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                         "use_quantization": False}},
        "datasets": {"d": dataset},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {"plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}},
    }


def _experiment(exclusions: list[dict] | None = None) -> ExperimentConfig:
    return ExperimentConfig.from_file(
        config=_config(exclusions), model_key="m", dataset_key="d", prompt_key="p",
        experiment_name="plan",
    )


def test_key_uses_row_ids_and_label() -> None:
    assert sample_key(_sample("a", [3, 1], 1)) == ((3, 1), 1)
    assert SampleExclusion(source_row_ids=[3, 1], label=1).key == ((3, 1), 1)


def test_excludes_matching_samples_and_reports_unmatched() -> None:
    samples = [_sample("a", [1], 1), _sample("b", [1], 0), _sample("c", [2, 3], 1)]
    result = apply_sample_exclusions(samples, [
        SampleExclusion(source_row_ids=[1], label=1, reason="wrong"),
        SampleExclusion(source_row_ids=[9], label=0),
    ])
    assert [s.id for s in result.kept] == ["b", "c"]
    assert [e.key for e in result.excluded] == [((1,), 1)]
    assert [e.key for e in result.unmatched] == [((9,), 0)]


def test_missing_row_ids_raises_only_when_exclusions_configured() -> None:
    samples = [_sample("a", None, 1)]
    assert [s.id for s in apply_sample_exclusions(samples, []).kept] == ["a"]
    _raises(ValueError, lambda: apply_sample_exclusions(samples, [SampleExclusion(source_row_ids=[1], label=1)]))


def test_config_carries_exclusions() -> None:
    assert _experiment().exclude_samples == []
    config = _experiment([{"source_row_ids": [789, 2754], "label": 1, "reason": "not_a_security_fix"}])
    assert [e.key for e in config.exclude_samples] == [((789, 2754), 1)]


def test_runner_prepare_samples_applies_exclusions() -> None:
    runner = BenchmarkRunner(
        config=_experiment([{"source_row_ids": [1], "label": 1}, {"source_row_ids": [5], "label": 0}]),
        dataset_loader=JsonDatasetLoader(),
    )
    kept, metadata = runner._prepare_samples(
        SampleCollection([_sample("a", [1], 1), _sample("b", [1], 0)])
    )
    assert [s.id for s in kept] == ["b"]
    assert metadata["excluded_samples"] == [{"source_row_ids": [1], "label": 1, "reason": ""}]
    assert metadata["unmatched_exclusions"] == [{"source_row_ids": [5], "label": 0, "reason": ""}]


def test_report_records_run_metadata() -> None:
    processor = BenchmarkResultProcessor(config=_experiment())
    metrics = MetricsResult.model_validate({"task_type": "binary", "accuracy": 0.0, "summary": {}, "details": {}})
    report = processor.build_report(
        metrics=metrics, predictions=[], total_time=0.0, total_samples=0,
        run_metadata={"excluded_samples": [{"source_row_ids": [1], "label": 1, "reason": ""}]},
    )
    assert report.benchmark_info.extra_metadata["excluded_samples"][0]["source_row_ids"] == [1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY tests/test_sample_exclusions.py`
Expected: `ModuleNotFoundError: No module named 'benchmark.sample_exclusions'`

- [ ] **Step 3: Create `src/benchmark/sample_exclusions.py`**

```python
"""Exclude audited samples (for example wrong labels) from a run by a stable key.

Sample ids are regenerated whenever a dataset is rebuilt, so samples are matched
on ``(metadata.source_row_ids, label)`` instead.
"""

from pydantic import BaseModel

from benchmark.models import BenchmarkSample

SampleKey = tuple[tuple[int, ...], int]


class SampleExclusion(BaseModel):
    """One sample to drop, as written in a dataset config's ``exclude_samples``."""

    source_row_ids: list[int]
    label: int
    reason: str = ""

    @property
    def key(self) -> SampleKey:
        return tuple(self.source_row_ids), self.label


class ExclusionResult(BaseModel):
    """Samples kept, exclusions that matched, and exclusions that matched nothing."""

    kept: list[BenchmarkSample]
    excluded: list[SampleExclusion]
    unmatched: list[SampleExclusion]


def sample_key(sample: BenchmarkSample) -> SampleKey:
    """Return the ``(source_row_ids, label)`` key of a sample.

    Raises:
        ValueError: If the sample has no ``metadata.source_row_ids``.
    """
    row_ids: list[int] | None = sample.metadata.get("source_row_ids")
    if not row_ids:
        raise ValueError(
            f"Sample {sample.id} has no metadata.source_row_ids; sample exclusions need it"
        )
    return tuple(int(row_id) for row_id in row_ids), int(sample.label)


def apply_sample_exclusions(
    samples: list[BenchmarkSample], exclusions: list[SampleExclusion]
) -> ExclusionResult:
    """Drop samples whose key matches an exclusion; report unmatched exclusions."""
    if not exclusions:
        return ExclusionResult(kept=list(samples), excluded=[], unmatched=[])
    excluded_keys: set[SampleKey] = {exclusion.key for exclusion in exclusions}
    matched: set[SampleKey] = set()
    kept: list[BenchmarkSample] = []
    for sample in samples:
        key: SampleKey = sample_key(sample)
        if key in excluded_keys:
            matched.add(key)
        else:
            kept.append(sample)
    return ExclusionResult(
        kept=kept,
        excluded=[exclusion for exclusion in exclusions if exclusion.key in matched],
        unmatched=[exclusion for exclusion in exclusions if exclusion.key not in matched],
    )
```

- [ ] **Step 4: Config fields (`src/benchmark/config.py`)**

Add import next to the other `benchmark.` imports:

```python
from benchmark.sample_exclusions import SampleExclusion
```

In `DatasetConfig`, after the `render_root_findings` field and its docstring:

```python
    exclude_samples: list[SampleExclusion] = []
    """Samples to drop by (source_row_ids, label), e.g. audited wrong labels."""
```

In `ExperimentConfig`, after the `render_root_findings` field and its docstring:

```python
    exclude_samples: list[SampleExclusion] = []
    """Copied from the dataset config; see DatasetConfig.exclude_samples."""
```

In `from_configs`, in the `cls(...)` call after `render_root_findings=dataset_config.render_root_findings,`:

```python
            exclude_samples=dataset_config.exclude_samples,
```

- [ ] **Step 5: Run metadata plumbing**

`src/benchmark/results.py`, in `BenchmarkRunResult` after `filtered_sample_ids` and its docstring:

```python
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    """Run-time facts merged into benchmark_info.extra_metadata (e.g. excluded samples)."""
```

`src/benchmark/result_processor.py`: add parameter `run_metadata: dict[str, Any] | None = None` (after `filtered_sample_ids`) to both `build_report` and `build_and_save`; document it in both docstrings as "Run-time facts merged into benchmark_info.extra_metadata."; in `build_and_save` pass `run_metadata=run_metadata` to `self.build_report(...)`; in `build_report`, directly after `benchmark_info: BenchmarkInfo = self._build_benchmark_info(...)`:

```python
        benchmark_info.extra_metadata.update(run_metadata or {})
```

`src/benchmark/run_experiment.py`, in the `result_processor.build_and_save(...)` call, after `filtered_sample_ids=result.filtered_sample_ids,`:

```python
        run_metadata=result.run_metadata,
```

- [ ] **Step 6: Runner (`src/benchmark/benchmark_runner.py`)**

Add import: `from benchmark.sample_exclusions import apply_sample_exclusions`.

Add method to `BenchmarkRunner` (before `_filter_samples_by_token_limit`):

```python
    def _prepare_samples(
        self, samples: SampleCollection
    ) -> tuple[SampleCollection, dict[str, Any]]:
        """Drop excluded samples (e.g. audited wrong labels) before any filtering.

        Returns:
            The kept samples and run metadata recording what was excluded.
        """
        exclusion = apply_sample_exclusions(list(samples), self.config.exclude_samples)
        if exclusion.unmatched:
            _LOGGER.warning(
                "%d sample exclusions matched no sample: %s",
                len(exclusion.unmatched),
                [entry.key for entry in exclusion.unmatched],
            )
        if exclusion.excluded:
            _LOGGER.info("Excluded %d samples", len(exclusion.excluded))
        run_metadata: dict[str, Any] = {
            "excluded_samples": [entry.model_dump() for entry in exclusion.excluded],
            "unmatched_exclusions": [entry.model_dump() for entry in exclusion.unmatched],
        }
        return SampleCollection(exclusion.kept), run_metadata
```

In `run()`, directly after `_LOGGER.info(f"Loaded {len(samples)} samples")`:

```python
        samples, run_metadata = self._prepare_samples(samples)
```

and add `run_metadata=run_metadata,` to the returned `BenchmarkRunResult(...)`.

- [ ] **Step 7: Run to verify pass**

Run: `PYTHONPATH=src $PY tests/test_sample_exclusions.py` → `ALL PASSED`
Run: `PYTHONPATH=src $PY tests/test_root_findings_runner.py` → `ALL PASSED`

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/sample_exclusions.py src/benchmark/config.py src/benchmark/benchmark_runner.py \
    src/benchmark/results.py src/benchmark/result_processor.py src/benchmark/run_experiment.py \
    tests/test_sample_exclusions.py
git commit -m "Exclude audited samples by source row ids and label via dataset config"
```

---

### Task 2: Rule filter and omitting the empty findings section

**Files:**
- Modify: `src/benchmark/static_findings.py`
- Modify: `src/benchmark/config.py` (`DatasetConfig`, `ExperimentConfig`, `from_configs`)
- Modify: `src/benchmark/benchmark_runner.py` (`user_template_values`, `_prepare_samples`, both call sites)
- Modify: `src/benchmark/result_processor.py` (`_build_benchmark_info`)
- Test: `tests/test_finding_filters.py`

**Interfaces:**
- Consumes: `RootStaticFinding`, `render_root_findings_block` (existing); `_prepare_samples` (Task 1).
- Produces:
  - `filter_root_findings(findings: list[RootStaticFinding] | None, patterns: list[str]) -> list[RootStaticFinding] | None`
  - `render_root_findings_block(findings: list[RootStaticFinding], omit_empty: bool = False) -> str`
  - `DatasetConfig` / `ExperimentConfig`: `exclude_finding_rules: list[str] = []`, `omit_empty_root_findings: bool = False`
  - `user_template_values(sample, render_root_findings: bool, omit_empty_root_findings: bool = False) -> dict[str, str]`
  - `extra_metadata["exclude_finding_rules"]`, `extra_metadata["omit_empty_root_findings"]`

- [ ] **Step 1: Write the failing tests** — create `tests/test_finding_filters.py`:

```python
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY tests/test_finding_filters.py`
Expected: `ImportError: cannot import name 'filter_root_findings'`

- [ ] **Step 3: `src/benchmark/static_findings.py`**

Add `import fnmatch` at the top (before `from typing import Any`). Add after `root_findings_from_raw`:

```python
def filter_root_findings(
    findings: list[RootStaticFinding] | None, patterns: list[str]
) -> list[RootStaticFinding] | None:
    """Drop findings whose ``rule_id`` matches any ``fnmatch`` pattern; None stays None."""
    if findings is None or not patterns:
        return findings
    return [
        finding
        for finding in findings
        if not any(fnmatch.fnmatchcase(finding.rule_id, pattern) for pattern in patterns)
    ]
```

Change `render_root_findings_block`'s signature and empty branch:

```python
def render_root_findings_block(
    findings: list[RootStaticFinding], omit_empty: bool = False
) -> str:
    """Render findings as the value of the ``{root_static_findings}`` placeholder.

    The value starts with a blank-line separator so it lays out cleanly whether
    the placeholder sits after or before ``{code}``. With ``omit_empty``, an
    empty list renders nothing instead of the "none reported" line.
    """
    if not findings:
        return "" if omit_empty else _BLOCK_SEPARATOR + _NO_FINDINGS_TEXT
```

(the rest of the function is unchanged).

- [ ] **Step 4: Config fields (`src/benchmark/config.py`)**

`DatasetConfig`, after `exclude_samples`:

```python
    exclude_finding_rules: list[str] = []
    """fnmatch patterns of analyzer rule ids dropped from root findings before rendering."""
    omit_empty_root_findings: bool = False
    """Render nothing instead of "none reported" when no root findings remain."""
```

`ExperimentConfig`, after `exclude_samples`:

```python
    exclude_finding_rules: list[str] = []
    """Copied from the dataset config; see DatasetConfig.exclude_finding_rules."""
    omit_empty_root_findings: bool = False
    """Copied from the dataset config; see DatasetConfig.omit_empty_root_findings."""
```

`from_configs`, after `exclude_samples=dataset_config.exclude_samples,`:

```python
            exclude_finding_rules=dataset_config.exclude_finding_rules,
            omit_empty_root_findings=dataset_config.omit_empty_root_findings,
```

- [ ] **Step 5: Runner (`src/benchmark/benchmark_runner.py`)**

Import `filter_root_findings` alongside `render_root_findings_block`. Replace `user_template_values`:

```python
def user_template_values(
    sample: BenchmarkSample,
    render_root_findings: bool,
    omit_empty_root_findings: bool = False,
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
        "root_static_findings": render_root_findings_block(
            sample.root_static_findings, omit_empty=omit_empty_root_findings
        ),
    }
```

At both `user_template_values(sample, self.config.render_root_findings)` call sites, add the third argument:

```python
                user_template_values(
                    sample,
                    self.config.render_root_findings,
                    self.config.omit_empty_root_findings,
                )
```

In `_prepare_samples`, replace `return SampleCollection(exclusion.kept), run_metadata` with:

```python
        kept: list[BenchmarkSample] = [
            sample.model_copy(
                update={
                    "root_static_findings": filter_root_findings(
                        sample.root_static_findings, self.config.exclude_finding_rules
                    )
                }
            )
            for sample in exclusion.kept
        ]
        return SampleCollection(kept), run_metadata
```

and update its docstring first line to "Drop excluded samples and filtered analyzer rules before any filtering."

- [ ] **Step 6: Report metadata (`src/benchmark/result_processor.py`)**

In `_build_benchmark_info`, after `extra_metadata["render_root_findings"] = ...`:

```python
        extra_metadata["exclude_finding_rules"] = list(self.config.exclude_finding_rules)
        extra_metadata["omit_empty_root_findings"] = self.config.omit_empty_root_findings
```

- [ ] **Step 7: Run to verify pass**

Run: `PYTHONPATH=src $PY tests/test_finding_filters.py` → `ALL PASSED`
Run: `for t in tests/test_root_static_findings.py tests/test_root_findings_runner.py tests/test_sample_exclusions.py; do PYTHONPATH=src $PY $t || break; done` → `ALL PASSED` ×3

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/static_findings.py src/benchmark/config.py src/benchmark/benchmark_runner.py \
    src/benchmark/result_processor.py tests/test_finding_filters.py
git commit -m "Filter analyzer rules and optionally omit the empty findings section"
```

---

### Task 3: Configurable coverage levels with full per-level metrics

**Files:**
- Create: `src/benchmark/coverage.py`
- Modify: `src/benchmark/metrics_calculator.py` (`_COVERAGE_LEVELS` removal, `BinaryMetricsCalculator.__init__`, `_calculate_coverage_metrics`, `MetricsCalculatorFactory.create_calculator`)
- Modify: `src/benchmark/config.py` (`ExperimentConfig.coverage_levels`, `model_post_init`, `from_file`, `from_configs`, plan loop)
- Modify: `src/benchmark/benchmark_runner.py` (`create_calculator` call ~line 166)
- Modify: `src/benchmark/result_processor.py` (`extra_metadata["coverage_levels"]`)
- Modify: `tests/test_metrics_calculator.py` (`test_coverage_selection_rounds_up`)
- Test: `tests/test_coverage_metrics.py`

**Interfaces:**
- Produces:
  - `DEFAULT_COVERAGE_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)`
  - `validate_coverage_levels(levels: Sequence[float]) -> tuple[float, ...]` (raises `ValueError` if empty or any value outside (0, 1])
  - `COVERAGE_METRIC_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "f1_score", "fpr", "fnr")` (in `coverage.py`)
  - `BinaryMetricsCalculator(coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS)`
  - `MetricsCalculatorFactory.create_calculator(task_type, task_specific_type=None, coverage_levels: Sequence[float] | None = None)`
  - `ExperimentConfig.coverage_levels: list[float]`; `ExperimentConfig.from_file(..., coverage_levels: list[float] | None = None)`; plan key `"coverage_levels"`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_coverage_metrics.py`:

```python
"""Plain-python checks: full metrics at configurable coverage levels.

Run: PYTHONPATH=src python tests/test_coverage_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.config import ExperimentConfig, ExperimentsPlanConfig
from benchmark.coverage import DEFAULT_COVERAGE_LEVELS, validate_coverage_levels
from benchmark.enums import TaskType
from benchmark.metrics_calculator import BinaryMetricsCalculator, MetricsCalculatorFactory
from benchmark.models import PredictionResult


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _pred(index: int, predicted: int, true: int, probability: float) -> PredictionResult:
    return PredictionResult(sample_id=f"s{index}", predicted_label=predicted, true_label=true,
                            confidence=None, answer_probability=probability, response_text="",
                            processing_time=0.0, is_success=True, error_message=None)


# Ranked by answer probability: TP, FP, TN, FN.
PREDICTIONS = [_pred(0, 1, 1, 0.9), _pred(1, 1, 0, 0.8), _pred(2, 0, 0, 0.7), _pred(3, 0, 1, 0.6)]


def _close(a, b) -> bool:
    return a is not None and abs(a - b) < 1e-9


def test_validation() -> None:
    assert validate_coverage_levels([0.5, 1]) == (0.5, 1.0)
    _raises(ValueError, lambda: validate_coverage_levels([]))
    _raises(ValueError, lambda: validate_coverage_levels([0.0]))
    _raises(ValueError, lambda: validate_coverage_levels([1.5]))


def test_default_levels_include_full_coverage() -> None:
    assert DEFAULT_COVERAGE_LEVELS == (0.25, 0.5, 0.75, 1.0)
    summary = BinaryMetricsCalculator().calculate(PREDICTIONS).summary
    for pct in (25, 50, 75, 100):
        for name in ("accuracy", "precision", "recall", "f1_score", "fpr", "fnr"):
            assert f"{name}_at_coverage_{pct}" in summary, (name, pct)


def test_full_metrics_at_half_and_full_coverage() -> None:
    summary = BinaryMetricsCalculator().calculate(PREDICTIONS).summary
    # Top 2 = TP, FP.
    assert _close(summary["accuracy_at_coverage_50"], 0.5)
    assert _close(summary["precision_at_coverage_50"], 0.5)
    assert _close(summary["recall_at_coverage_50"], 1.0)
    assert _close(summary["f1_score_at_coverage_50"], 2 / 3)
    assert _close(summary["fpr_at_coverage_50"], 1.0)
    assert _close(summary["fnr_at_coverage_50"], 0.0)
    # All 4 = TP, FP, TN, FN.
    for name in ("accuracy", "precision", "recall", "f1_score", "fpr", "fnr"):
        assert _close(summary[f"{name}_at_coverage_100"], 0.5), name


def test_undefined_ratios_are_none() -> None:
    # Top 25% = one TP only: no negatives -> FPR undefined.
    result = BinaryMetricsCalculator().calculate(PREDICTIONS)
    assert result.summary["fpr_at_coverage_25"] is None
    assert _close(result.summary["precision_at_coverage_25"], 1.0)
    level = result.details["selective_prediction"]["levels"][0]
    assert level["selected_samples"] == 1 and level["fpr"] is None


def test_single_class_selection_never_divides_by_zero() -> None:
    only_negatives = [_pred(0, 0, 0, 0.9), _pred(1, 0, 0, 0.8)]
    summary = BinaryMetricsCalculator().calculate(only_negatives).summary
    assert summary["precision_at_coverage_100"] is None
    assert summary["recall_at_coverage_100"] is None
    assert summary["f1_score_at_coverage_100"] is None
    assert _close(summary["fpr_at_coverage_100"], 0.0)


def test_custom_levels_change_keys() -> None:
    summary = BinaryMetricsCalculator([0.5]).calculate(PREDICTIONS).summary
    assert "accuracy_at_coverage_50" in summary
    assert "accuracy_at_coverage_25" not in summary and "accuracy_at_coverage_100" not in summary


def test_factory_passes_levels() -> None:
    calculator = MetricsCalculatorFactory.create_calculator(
        TaskType.BINARY_VULNERABILITY, coverage_levels=[0.3])
    assert calculator.coverage_levels == (0.3,)


def _config(plan_levels: list[float] | None) -> dict:
    plan: dict = {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}
    if plan_levels is not None:
        plan["coverage_levels"] = plan_levels
    return {
        "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                         "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                         "use_quantization": False}},
        "datasets": {"d": {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {"plan": plan},
    }


def test_plan_coverage_levels_flow_into_experiments() -> None:
    assert ExperimentsPlanConfig.from_file(_config(None), "plan").experiments[0].coverage_levels == [
        0.25, 0.5, 0.75, 1.0]
    assert ExperimentsPlanConfig.from_file(_config([0.1, 0.9]), "plan").experiments[0].coverage_levels == [
        0.1, 0.9]
    _raises(ValueError, lambda: ExperimentsPlanConfig.from_file(_config([2.0]), "plan"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY tests/test_coverage_metrics.py`
Expected: `ModuleNotFoundError: No module named 'benchmark.coverage'`

- [ ] **Step 3: Create `src/benchmark/coverage.py`**

```python
"""Coverage levels for selective prediction (metrics on the most confident samples)."""

from collections.abc import Sequence

DEFAULT_COVERAGE_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

COVERAGE_METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "fpr",
    "fnr",
)


def validate_coverage_levels(levels: Sequence[float]) -> tuple[float, ...]:
    """Return levels as floats.

    Raises:
        ValueError: If no level is given or a level lies outside (0, 1].
    """
    values: tuple[float, ...] = tuple(float(level) for level in levels)
    if not values:
        raise ValueError("coverage_levels must not be empty")
    invalid: list[float] = [value for value in values if not 0.0 < value <= 1.0]
    if invalid:
        raise ValueError(f"coverage_levels must lie in (0, 1]: {invalid}")
    return values
```

- [ ] **Step 4: Calculator (`src/benchmark/metrics_calculator.py`)**

Add `from collections.abc import Sequence` to the imports and
`from benchmark.coverage import COVERAGE_METRIC_NAMES, DEFAULT_COVERAGE_LEVELS, validate_coverage_levels`.
Delete the line `_COVERAGE_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75)`.

Add a module-level helper after `_PAIR_ID_SUFFIXES`:

```python
def _selection_metrics(predictions: list[PredictionResult]) -> dict[str, float | None]:
    """Accuracy, precision, recall, F1, FPR and FNR of a selection; None where undefined."""
    tp: int = sum(p.predicted_label == 1 and p.true_label == 1 for p in predictions)
    fp: int = sum(p.predicted_label == 1 and p.true_label == 0 for p in predictions)
    tn: int = sum(p.predicted_label == 0 and p.true_label == 0 for p in predictions)
    fn: int = sum(p.predicted_label == 0 and p.true_label == 1 for p in predictions)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision: float | None = ratio(tp, tp + fp)
    recall: float | None = ratio(tp, tp + fn)
    f1_score: float | None = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "accuracy": sum(p.predicted_label == p.true_label for p in predictions)
        / len(predictions),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
    }
```

Add to `BinaryMetricsCalculator` (first method in the class body):

```python
    def __init__(
        self, coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS
    ) -> None:
        self.coverage_levels: tuple[float, ...] = validate_coverage_levels(
            coverage_levels
        )
```

Replace the whole `_calculate_coverage_metrics` method with:

```python
    def _calculate_coverage_metrics(
        self, predictions: list[PredictionResult], score_field: str, key_prefix: str
    ) -> tuple[dict[str, float | int | str | None], dict[str, Any]]:
        """Calculate selective-prediction metrics for one per-sample confidence score.

        At each coverage level, keeps the most confident fraction of samples and
        reports accuracy, precision, recall, F1, FPR and FNR on it, plus the AUROC
        of the score for separating correct from incorrect predictions.
        """
        auroc_key: str = f"{key_prefix or 'answer_probability_'}correctness_auroc"
        summary: dict[str, float | int | str | None] = {
            f"{key_prefix}{name}_at_coverage_{round(level * 100)}": None
            for level in self.coverage_levels
            for name in COVERAGE_METRIC_NAMES
        }
        summary[auroc_key] = None
        scored: list[tuple[float, PredictionResult]] = sorted(
            (
                (score, pred)
                for pred in predictions
                if (score := getattr(pred, score_field)) is not None
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        details: dict[str, Any] = {
            "score_source": score_field,
            "scored_samples": len(scored),
            "total_samples": len(predictions),
            "correctness_auroc": None,
            "levels": [],
        }
        if not scored:
            details["skipped_reason"] = f"no per-sample {score_field} scores available"
            return summary, details

        is_correct: list[bool] = [
            pred.predicted_label == pred.true_label for _, pred in scored
        ]
        if len(set(is_correct)) == 2:
            auroc: float = float(
                roc_auc_score(is_correct, [score for score, _ in scored])
            )
            summary[auroc_key] = auroc
            details["correctness_auroc"] = auroc

        for level in self.coverage_levels:
            selected: list[tuple[float, PredictionResult]] = scored[
                : math.ceil(level * len(scored))
            ]
            values: dict[str, float | None] = _selection_metrics(
                [pred for _, pred in selected]
            )
            for name, value in values.items():
                summary[f"{key_prefix}{name}_at_coverage_{round(level * 100)}"] = value
            details["levels"].append(
                {
                    "coverage": level,
                    "selected_samples": len(selected),
                    "min_answer_probability": selected[-1][0],
                    **values,
                }
            )
        return summary, details
```

In `MetricsCalculatorFactory.create_calculator`, add parameter
`coverage_levels: Sequence[float] | None = None` (after `task_specific_type`), document it as "Coverage levels for binary selective-prediction metrics; defaults to DEFAULT_COVERAGE_LEVELS.", and replace `return BinaryMetricsCalculator()` with:

```python
            return BinaryMetricsCalculator(
                coverage_levels
                if coverage_levels is not None
                else DEFAULT_COVERAGE_LEVELS
            )
```

- [ ] **Step 5: Config (`src/benchmark/config.py`)**

Import: `from benchmark.coverage import DEFAULT_COVERAGE_LEVELS, validate_coverage_levels`.

`ExperimentConfig`, after `omit_empty_root_findings`:

```python
    coverage_levels: list[float] = list(DEFAULT_COVERAGE_LEVELS)
    """Coverage levels for selective-prediction metrics, set per experiment plan."""
```

In `ExperimentConfig.model_post_init`, directly before `super().model_post_init(context)`:

```python
        self.coverage_levels = list(validate_coverage_levels(self.coverage_levels))
```

Add parameter `coverage_levels: list[float] | None = None` (after `confidence_methods`) to both `ExperimentConfig.from_file` and `ExperimentConfig.from_configs`; `from_file` passes `coverage_levels=coverage_levels` to `cls.from_configs(...)`; `from_configs` adds to the `cls(...)` call:

```python
            coverage_levels=coverage_levels
            if coverage_levels is not None
            else list(DEFAULT_COVERAGE_LEVELS),
```

In `ExperimentsPlanConfig.from_file`'s loop, after `confidence_methods=plan_config.get("confidence_methods"),`:

```python
                            coverage_levels=plan_config.get("coverage_levels"),
```

- [ ] **Step 6: Runner and report**

`src/benchmark/benchmark_runner.py`: in the `MetricsCalculatorFactory.create_calculator(` call inside `run()`, add the keyword argument `coverage_levels=self.config.coverage_levels`.

`src/benchmark/result_processor.py` `_build_benchmark_info`, after the `omit_empty_root_findings` line:

```python
        extra_metadata["coverage_levels"] = list(self.config.coverage_levels)
```

- [ ] **Step 7: Update the existing coverage test**

In `tests/test_metrics_calculator.py::test_coverage_selection_rounds_up`, the default levels now include 1.0, so replace

```python
    assert [level["selected_samples"] for level in levels] == [2, 3, 4], levels
```

with

```python
    # The default levels include full coverage (100% -> all 5).
    assert [level["selected_samples"] for level in levels] == [2, 3, 4, 5], levels
```

- [ ] **Step 8: Run to verify pass**

Run: `PYTHONPATH=src $PY tests/test_coverage_metrics.py` → `ALL PASSED`
Run: `PYTHONPATH=src $PY tests/test_metrics_calculator.py` → all `PASSED` lines, no traceback
Run the whole suite (Global Constraints command) → `passed`, 0 failed.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/coverage.py src/benchmark/metrics_calculator.py src/benchmark/config.py \
    src/benchmark/benchmark_runner.py src/benchmark/result_processor.py \
    tests/test_coverage_metrics.py tests/test_metrics_calculator.py
git commit -m "Report full metrics at configurable coverage levels"
```

---

### Task 4: Recompute and merge saved reports

**Files:**
- Create: `src/benchmark/merge_results.py`
- Modify: `src/cli.py` (new command after `rebuild-plan-results`)
- Test: `tests/test_merge_results.py`

**Interfaces:**
- Consumes: `SampleExclusion` (T1), `filter_root_findings` (T2), `BinaryMetricsCalculator(coverage_levels)`, `validate_coverage_levels` (T3), `BenchmarkResultProcessor._to_prediction_record`, `rebuild_experiment_plan_results`.
- Produces:
  - `record_to_prediction(record: PredictionRecord) -> PredictionResult`
  - `latest_condition_reports(plan_dir: Path) -> dict[Path, Path]`
  - `recompute_report(report: BenchmarkReport, exclusions: list[SampleExclusion], analysis_exclude_rules: list[str], coverage_levels: Sequence[float], source_path: Path) -> BenchmarkReport`
  - `summary_row(summary: dict[str, Any], coverage_levels: Sequence[float]) -> dict[str, float | None]`
  - `render_summary_markdown(rows: dict[str, dict[str, float | None]], coverage_levels: Sequence[float]) -> str`
  - `merge_plan_results(plan_dirs: list[Path], datasets_config: Path, coverage_levels: Sequence[float], analysis_exclude_rules: list[str], output_dir: Path, models: list[str] | None = None) -> Path` (returns the `summary.md` path; `models` keeps only conditions whose model directory is listed; duplicate row names raise `ValueError`)
  - CLI `merge-plan-results --plan-dir … (repeatable) --datasets-config … --output-dir … [--model …] (repeatable) [--coverage-levels 0.25,0.5,0.75,1.0] [--exclude-finding-rules B101,B113]`

- [ ] **Step 1: Write the failing tests** — create `tests/test_merge_results.py`:

```python
"""Plain-python checks: saved reports are recomputed under exclusions and merged.

Run: PYTHONPATH=src python tests/test_merge_results.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.config import ExperimentConfig
from benchmark.merge_results import (
    latest_condition_reports,
    merge_plan_results,
    recompute_report,
    record_to_prediction,
    render_summary_markdown,
)
from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult
from benchmark.result_processor import BenchmarkResultProcessor
from benchmark.results import BenchmarkReport
from benchmark.sample_exclusions import SampleExclusion
from benchmark.static_findings import RootStaticFinding

FINDINGS = [
    RootStaticFinding(tool="bandit", rule_id="B101", message="m", file_path="a.py", repo_line=1, flagged_line="x"),
    RootStaticFinding(tool="bandit", rule_id="B602", message="m", file_path="a.py", repo_line=2, flagged_line="y"),
]


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _prediction(index: int, predicted: int, true: int, probability: float,
                row_ids: list[int] | None) -> PredictionResult:
    return PredictionResult(
        sample_id=f"s{index}", predicted_label=predicted, true_label=true, confidence=None,
        binary_label_confidence=probability if predicted == 1 else 1 - probability,
        answer_probability=probability, response_text="r", processing_time=0.1, tokens_used=3,
        is_success=True, error_message=None, all_responses=["r"], vote_counts={str(predicted): 1},
        source_row_ids=row_ids, root_static_findings=list(FINDINGS), root_findings_in_prompt=True,
    )


PREDICTIONS = [
    _prediction(0, 1, 1, 0.9, [10]), _prediction(1, 1, 0, 0.8, [10]),
    _prediction(2, 0, 0, 0.7, [11]), _prediction(3, 0, 1, 0.6, [11]),
]


def _record(prediction: PredictionResult):
    return BenchmarkResultProcessor._to_prediction_record(BenchmarkResultProcessor.model_construct(), prediction)


def _report(predictions: list[PredictionResult]) -> BenchmarkReport:
    """Build a report with the real processor so it matches the saved-report schema."""
    config = ExperimentConfig.from_file(
        config={
            "models": {"m": {"model_identifier": "org/model", "model_type": "qwen-3", "backend": "vllm",
                             "max_output_tokens": 16, "batch_size": 1, "temperature": 0.0,
                             "use_quantization": False}},
            "datasets": {"d": {"path": __file__, "task_type": "binary_vulnerability", "description": "d"}},
            "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
            "output_settings": {"base_output_dir": "results/x"},
            "experiment_plans": {"plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"]}},
        },
        model_key="m", dataset_key="d", prompt_key="p", experiment_name="plan",
    )
    metrics = BinaryMetricsCalculator().calculate(predictions)
    return BenchmarkResultProcessor(config=config).build_report(
        metrics=metrics, predictions=predictions, total_time=1.0, total_samples=len(predictions))


def test_record_round_trip() -> None:
    record = _record(PREDICTIONS[0])
    assert _record(record_to_prediction(record)) == record


def test_recompute_without_changes_matches_direct_metrics() -> None:
    report = _report(PREDICTIONS)
    recomputed = recompute_report(report, [], [], (0.5, 1.0), Path("src.json"))
    direct = BinaryMetricsCalculator((0.5, 1.0)).calculate(PREDICTIONS)
    assert recomputed.metrics.summary == direct.summary
    assert recomputed.benchmark_info.extra_metadata["recomputed_from"] == "src.json"


def test_recompute_applies_exclusions_and_rule_filter() -> None:
    report = _report(PREDICTIONS)
    recomputed = recompute_report(
        report, [SampleExclusion(source_row_ids=[10], label=0)], ["B101"], (1.0,), Path("x"))
    assert [p.sample_id for p in recomputed.predictions] == ["s0", "s2", "s3"]
    assert all([f.rule_id for f in p.root_static_findings] == ["B602"] for p in recomputed.predictions)
    assert recomputed.benchmark_info.stats.total_samples == 3
    meta = recomputed.benchmark_info.extra_metadata
    assert meta["excluded_samples"] == [{"source_row_ids": [10], "label": 0}]
    assert meta["analysis_exclude_finding_rules"] == ["B101"]
    expected = BinaryMetricsCalculator((1.0,)).calculate([PREDICTIONS[0], PREDICTIONS[2], PREDICTIONS[3]])
    assert recomputed.metrics.summary == expected.summary


def test_recompute_without_row_ids_raises_when_excluding() -> None:
    report = _report([_prediction(0, 1, 1, 0.9, None)])
    recompute_report(report, [], [], (1.0,), Path("x"))  # fine without exclusions
    _raises(ValueError, lambda: recompute_report(
        report, [SampleExclusion(source_row_ids=[1], label=1)], [], (1.0,), Path("x")))


def test_latest_condition_reports_picks_newest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan = Path(tmp) / "plan_a"
        condition = plan / "ds_a" / "m" / "p"
        condition.mkdir(parents=True)
        for stamp in ("20260925_100000", "20260926_090000"):
            (condition / f"benchmark_report_{stamp}.json").write_text("{}")
        latest = latest_condition_reports(plan)
        assert latest == {Path("ds_a/m/p"): condition / "benchmark_report_20260926_090000.json"}
        _raises(RuntimeError, lambda: latest_condition_reports(Path(tmp) / "empty_missing"))


def test_summary_bolds_best_and_handles_none() -> None:
    rows = {
        "a": {"accuracy": 0.6, "fpr": 0.3, "pr_auc": None},
        "b": {"accuracy": 0.6, "fpr": 0.4, "pr_auc": 0.7},
    }
    text = render_summary_markdown(rows, ())
    line_a = next(line for line in text.splitlines() if line.startswith("| a "))
    line_b = next(line for line in text.splitlines() if line.startswith("| b "))
    assert "**0.600**" in line_a and "**0.600**" in line_b  # tie -> both bold
    assert "**0.300**" in line_a and "0.400" in line_b and "**0.400**" not in line_b  # lower FPR wins
    assert "—" in line_a and "**0.700**" in line_b


def test_merge_plan_results_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for plan, dataset in (("plan_a", "ds_a"), ("plan_b", "ds_b")):
            condition = root / plan / dataset / "m" / "p"
            condition.mkdir(parents=True)
            (condition / "benchmark_report_20260926_000000.json").write_text(
                json.dumps(_report(PREDICTIONS).model_dump(), default=str))
        exclusion = {"source_row_ids": [10], "label": 0, "reason": "wrong"}
        datasets = root / "datasets.json"
        datasets.write_text(json.dumps({"datasets": {
            "ds_a": {"dataset_path": "x", "task_type": "binary_vulnerability", "description": "a",
                     "exclude_samples": [exclusion]},
            "ds_b": {"dataset_path": "y", "task_type": "binary_vulnerability", "description": "b",
                     "exclude_samples": [exclusion]},
        }}))
        out = root / "merged"
        summary_path = merge_plan_results([root / "plan_a", root / "plan_b"], datasets, (0.5, 1.0), ["B101"], out)
        assert summary_path == out / "summary.md"
        text = summary_path.read_text()
        assert "ds_a · p" in text and "ds_b · p" in text
        assert (out / "experiment_plan_results.json").exists()
        merged = BenchmarkReport.model_validate(json.loads(
            (out / "plan_a" / "ds_a" / "m" / "p" / "benchmark_report_20260926_000000.json").read_text()))
        assert len(merged.predictions) == 3


def test_merge_unknown_dataset_key_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        condition = root / "plan_a" / "ds_unknown" / "m" / "p"
        condition.mkdir(parents=True)
        (condition / "benchmark_report_20260926_000000.json").write_text(
            json.dumps(_report(PREDICTIONS).model_dump(), default=str))
        datasets = root / "datasets.json"
        datasets.write_text(json.dumps({"datasets": {}}))
        _raises(KeyError, lambda: merge_plan_results([root / "plan_a"], datasets, (1.0,), [], root / "out"))


def test_merge_model_filter_and_duplicate_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for model in ("m1", "m2"):
            condition = root / "plan_a" / "ds_a" / model / "p"
            condition.mkdir(parents=True)
            (condition / "benchmark_report_20260926_000000.json").write_text(
                json.dumps(_report(PREDICTIONS).model_dump(), default=str))
        datasets = root / "datasets.json"
        datasets.write_text(json.dumps({"datasets": {
            "ds_a": {"dataset_path": "x", "task_type": "binary_vulnerability", "description": "a"}}}))
        # Two models give the same "ds_a · p" row: refuse instead of overwriting.
        _raises(ValueError, lambda: merge_plan_results([root / "plan_a"], datasets, (1.0,), [], root / "o1"))
        summary = merge_plan_results([root / "plan_a"], datasets, (1.0,), [], root / "o2", models=["m2"])
        assert (root / "o2" / "plan_a" / "ds_a" / "m2").exists()
        assert not (root / "o2" / "plan_a" / "ds_a" / "m1").exists()
        assert summary.read_text().count("ds_a · p") == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY tests/test_merge_results.py`
Expected: `ModuleNotFoundError: No module named 'benchmark.merge_results'`

- [ ] **Step 3: Create `src/benchmark/merge_results.py`**

```python
"""Recompute metrics of saved benchmark reports and merge several plans into one result set.

No inference is run: predictions stored in each report are re-scored after
dropping excluded samples (dataset config ``exclude_samples``) and filtering the
stored root findings with analysis-time rule patterns.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult
from benchmark.results import BenchmarkReport, PredictionRecord
from benchmark.run_experiment import rebuild_experiment_plan_results
from benchmark.sample_exclusions import SampleExclusion, SampleKey
from benchmark.static_findings import filter_root_findings
from entrypoints.utils import load_config_dict, normalize_config_schema

# (summary key, column label, higher is better)
_BASE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "Acc", True),
    ("precision", "P", True),
    ("recall", "R", True),
    ("f1_score", "F1", True),
    ("fpr", "FPR", False),
    ("fnr", "FNR", False),
    ("pr_auc", "AP", True),
)
_COVERAGE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "Acc", True),
    ("f1_score", "F1", True),
    ("fpr", "FPR", False),
)


def record_to_prediction(record: PredictionRecord) -> PredictionResult:
    """Rebuild the runner's PredictionResult from a saved PredictionRecord."""
    data = record.inference_data
    return PredictionResult(
        sample_id=record.sample_id,
        predicted_label=record.predicted_label,
        true_label=record.true_label,
        confidence=data.confidence,
        binary_label_confidence=data.binary_label_confidence,
        answer_probability=data.answer_probability,
        stated_confidence=data.stated_confidence,
        self_validation_probability=data.self_validation_probability,
        prompt_text=data.prompt_text,
        prompt_tokens=data.prompt_tokens,
        p_vulnerable_per_draw=data.p_vulnerable_per_draw,
        response_text=data.responses[0] if data.responses else "",
        processing_time=data.processing_time,
        tokens_used=data.tokens_used,
        is_success=record.is_success,
        error_message=record.error_message,
        all_responses=data.responses,
        answer_probabilities=data.answer_probabilities,
        stated_confidences=data.stated_confidences,
        self_validation_probabilities=data.self_validation_probabilities,
        vote_counts=data.vote_counts,
        source_row_ids=record.source_row_ids,
        root_static_findings=record.root_static_findings,
        root_findings_in_prompt=record.root_findings_in_prompt,
    )


def latest_condition_reports(plan_dir: Path) -> dict[Path, Path]:
    """Map each condition directory (relative to ``plan_dir``) to its newest report.

    Raises:
        RuntimeError: If ``plan_dir`` holds no ``benchmark_report_*.json``.
    """
    latest: dict[Path, Path] = {}
    # Report names embed a sortable timestamp, so the last one per directory is the newest.
    for path in sorted(plan_dir.rglob("benchmark_report_*.json")):
        latest[path.parent.relative_to(plan_dir)] = path
    if not latest:
        raise RuntimeError(f"No benchmark_report_*.json under {plan_dir}")
    return latest


def recompute_report(
    report: BenchmarkReport,
    exclusions: list[SampleExclusion],
    analysis_exclude_rules: list[str],
    coverage_levels: Sequence[float],
    source_path: Path,
) -> BenchmarkReport:
    """Drop excluded samples, filter stored findings and recompute binary metrics.

    Raises:
        ValueError: If exclusions are configured but a prediction has no source_row_ids.
    """
    excluded_keys: set[SampleKey] = {exclusion.key for exclusion in exclusions}
    kept: list[PredictionRecord] = []
    dropped: list[SampleKey] = []
    for record in report.predictions:
        if exclusions and record.source_row_ids is None:
            raise ValueError(
                f"{source_path}: prediction {record.sample_id} has no source_row_ids; "
                "re-run it or drop exclude_samples for this condition"
            )
        if record.source_row_ids is not None:
            key: SampleKey = (tuple(record.source_row_ids), int(record.true_label))
            if key in excluded_keys:
                dropped.append(key)
                continue
        kept.append(
            record.model_copy(
                update={
                    "root_static_findings": filter_root_findings(
                        record.root_static_findings, analysis_exclude_rules
                    )
                }
            )
        )

    metrics = BinaryMetricsCalculator(coverage_levels).calculate(
        [record_to_prediction(record) for record in kept]
    )
    info = report.benchmark_info
    benchmark_info = info.model_copy(
        update={
            "stats": info.stats.model_copy(update={"total_samples": len(kept)}),
            "extra_metadata": {
                **info.extra_metadata,
                "recomputed_from": str(source_path),
                "excluded_samples": [
                    {"source_row_ids": list(row_ids), "label": label}
                    for row_ids, label in dropped
                ],
                "analysis_exclude_finding_rules": list(analysis_exclude_rules),
                "coverage_levels": list(coverage_levels),
            },
        }
    )
    return report.model_copy(
        update={"benchmark_info": benchmark_info, "metrics": metrics, "predictions": kept}
    )


def summary_row(
    summary: dict[str, Any], coverage_levels: Sequence[float]
) -> dict[str, float | None]:
    """Pick the merged-table metrics from a metrics summary."""
    row: dict[str, float | None] = {
        key: summary.get(key)
        for key in ("accuracy", "precision", "recall", "f1_score", "pr_auc")
    }
    specificity: float | None = summary.get("specificity")
    row["fpr"] = None if specificity is None else 1.0 - specificity
    recall: float | None = row["recall"]
    row["fnr"] = None if recall is None else 1.0 - recall
    for level in coverage_levels:
        pct: int = round(level * 100)
        for name, _, _ in _COVERAGE_COLUMNS:
            row[f"{name}_at_coverage_{pct}"] = summary.get(f"{name}_at_coverage_{pct}")
    return row


def render_summary_markdown(
    rows: dict[str, dict[str, float | None]], coverage_levels: Sequence[float]
) -> str:
    """Render the merged table; the best value per column is bold (ties all bold)."""
    columns: list[tuple[str, str, bool]] = list(_BASE_COLUMNS)
    for level in coverage_levels:
        pct = round(level * 100)
        columns.extend(
            (f"{name}_at_coverage_{pct}", f"{label}@{pct}%", higher)
            for name, label, higher in _COVERAGE_COLUMNS
        )
    best: dict[str, float | None] = {}
    for key, _, higher in columns:
        values: list[float] = [
            round(value, 3) for row in rows.values() if (value := row.get(key)) is not None
        ]
        best[key] = (max(values) if higher else min(values)) if values else None

    lines: list[str] = [
        "| Condition | " + " | ".join(label for _, label, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name, row in rows.items():
        cells: list[str] = []
        for key, _, _ in columns:
            value: float | None = row.get(key)
            if value is None:
                cells.append("—")
            elif round(value, 3) == best[key]:
                cells.append(f"**{value:.3f}**")
            else:
                cells.append(f"{value:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def merge_plan_results(
    plan_dirs: list[Path],
    datasets_config: Path,
    coverage_levels: Sequence[float],
    analysis_exclude_rules: list[str],
    output_dir: Path,
    models: list[str] | None = None,
) -> Path:
    """Recompute the newest report of every condition and merge them.

    Writes recomputed reports under ``output_dir/<plan>/<condition>/``,
    ``output_dir/experiment_plan_results.json`` and ``output_dir/summary.md``.

    Args:
        models: If given, only conditions whose model directory is listed are merged.

    Raises:
        KeyError: If a condition's dataset key is missing from the datasets config.
        ValueError: If two merged conditions share a summary row name.
    """
    entries: dict[str, dict[str, Any]] = normalize_config_schema(
        load_config_dict(datasets_config)
    )["datasets"]
    rows: dict[str, dict[str, float | None]] = {}
    for plan_dir in plan_dirs:
        for condition, report_path in latest_condition_reports(plan_dir).items():
            # Condition directories are <dataset_key>/<model>/<prompt>.
            if models is not None and condition.parts[1] not in models:
                continue
            dataset_key: str = condition.parts[0]
            if dataset_key not in entries:
                raise KeyError(f"Dataset key {dataset_key!r} not in {datasets_config}")
            exclusions: list[SampleExclusion] = [
                SampleExclusion.model_validate(entry)
                for entry in entries[dataset_key].get("exclude_samples", [])
            ]
            report = BenchmarkReport.model_validate(
                json.loads(report_path.read_text(encoding="utf-8"))
            )
            recomputed: BenchmarkReport = recompute_report(
                report, exclusions, analysis_exclude_rules, coverage_levels, report_path
            )
            target: Path = output_dir / plan_dir.name / condition / report_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(recomputed.model_dump(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            row_name: str = f"{dataset_key} · {condition.parts[-1]}"
            if row_name in rows:
                raise ValueError(
                    f"Two conditions map to summary row {row_name!r}; pass --model to pick one"
                )
            rows[row_name] = summary_row(recomputed.metrics.summary, coverage_levels)

    rebuild_experiment_plan_results(
        input_path=output_dir,
        plan_name="merged",
        description="Merged recomputed results of: " + ", ".join(str(p) for p in plan_dirs),
    )
    summary_path: Path = output_dir / "summary.md"
    summary_path.write_text(render_summary_markdown(rows, coverage_levels), encoding="utf-8")
    return summary_path
```

- [ ] **Step 4: CLI command (`src/cli.py`)**

Add imports near the other `benchmark.` imports:

```python
from benchmark.coverage import validate_coverage_levels
from benchmark.merge_results import merge_plan_results
```

Add after the `rebuild_plan_results` command function:

```python
@app.command("merge-plan-results")
def merge_plan_results_command(
    plan_dir: list[str] = typer.Option(
        ..., "--plan-dir", help="Plan output directory; repeat to merge several plans."
    ),
    datasets_config: str = typer.Option(
        ..., "--datasets-config", help="Datasets config with each condition's exclude_samples."
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", help="Directory for recomputed reports and summary.md."
    ),
    model: list[str] | None = typer.Option(
        None, "--model", help="Only merge conditions of this model directory; repeatable."
    ),
    coverage_levels: str = typer.Option(
        "0.25,0.5,0.75,1.0", "--coverage-levels", help="Comma-separated levels in (0, 1]."
    ),
    exclude_finding_rules: str = typer.Option(
        "", "--exclude-finding-rules",
        help="Comma-separated fnmatch rule patterns dropped from stored findings (analysis only).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level."),
) -> None:
    """Recompute saved reports under sample exclusions and merge plans into one summary."""
    _configure_logging(verbose=verbose, log_level=log_level)
    levels: tuple[float, ...] = validate_coverage_levels(
        [float(level) for level in coverage_levels.split(",") if level.strip()]
    )
    rules: list[str] = [rule.strip() for rule in exclude_finding_rules.split(",") if rule.strip()]
    summary_path: Path = merge_plan_results(
        [Path(path) for path in plan_dir], Path(datasets_config), levels, rules, Path(output_dir),
        models=model or None,
    )
    typer.echo(summary_path.read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run to verify pass**

Run: `PYTHONPATH=src $PY tests/test_merge_results.py` → `ALL PASSED`
Run: `PYTHONPATH=src $PY src/cli.py merge-plan-results --help | head -5` → usage text listing `--plan-dir`.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/merge_results.py src/cli.py tests/test_merge_results.py
git commit -m "Add merge-plan-results to recompute saved reports under exclusions"
```

---

### Task 5: Prompt, configs and docs

**Files:**
- Modify: `src/configs/shared/prompts.json`, `src/configs/static_findings_root/datasets.json`, `src/configs/static_findings_root/experiments.json`, `CLAUDE.md`
- Test: `tests/test_finding_verification_configs.py`

**Interfaces:**
- Consumes: all config fields from Tasks 1–3.
- Produces: prompt `finding_verification_root_findings`; dataset keys `cpg_structural_root_findings_verify`, `mult_amp_root_findings_verify`; plans `static_findings_verify_sweep`, `static_findings_verify_smoke`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_finding_verification_configs.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY tests/test_finding_verification_configs.py`
Expected: `KeyError: 'finding_verification_root_findings'`

- [ ] **Step 3: Add the prompt** (inserts after the `strict_exploitable_security_root_findings` entry; the system prompt is built from the base string so it is byte-exact):

```bash
python3 - <<'EOF'
import json
p = "src/configs/shared/prompts.json"
text = open(p).read()
prompts = json.loads(text)["prompts"]
base = prompts["strict_exploitable_security"]
paragraph = (
    "\n\nStatic analyzer findings:\nThe code may come with static-analyzer findings. Treat each finding "
    "as a lead, not as evidence. For every finding, decide CONFIRMED or REFUTED. REFUTED means the "
    "flagged data is not attacker-controlled, the line is unreachable, or validation, escaping, an "
    "allow-list, or a safe API in the shown code prevents exploitation. Most analyzer findings are "
    "false positives. Report a flaw the analyzers did not flag only if you can trace a concrete "
    "source-to-sink path. If every finding is refuted and you found no other concrete flaw, the code "
    "is NOT vulnerable. If no findings are given, analyze the code on its own merits."
)
entry = [
    '        "finding_verification_root_findings": {',
    '            "name": "Finding verification: root static findings as leads to confirm or refute",',
    '            "system_prompt": ' + json.dumps(base["system_prompt"] + paragraph, ensure_ascii=False) + ",",
    '            "user_prompt": ' + json.dumps(base["user_prompt"] + "{root_static_findings}", ensure_ascii=False),
    "        },",
]
lines = text.split("\n")
start = next(i for i, l in enumerate(lines) if l.strip() == '"strict_exploitable_security_root_findings": {')
end = next(i for i in range(start, len(lines)) if lines[i] == "        },")
lines[end + 1:end + 1] = entry
open(p, "w").write("\n".join(lines))
json.loads(open(p).read())
print("OK")
EOF
```

Expected: `OK`.

- [ ] **Step 4: Rewrite `src/configs/static_findings_root/datasets.json`** (existing 5 entries keep their fields; all 7 get the same inline exclusion list, one entry per line):

```bash
python3 - <<'EOF'
import json
p = "src/configs/static_findings_root/datasets.json"
datasets = json.load(open(p))["datasets"]
EXCLUSIONS = [
    ([789, 2754], 1, "not_a_security_fix"),
    ([3607], 1, "collateral_change"),
    ([3875], 1, "mismatched_pair_upstream"),
    ([1054, 1679, 1721, 3767], 0, "incomplete_fix"),
    ([5192], 1, "collateral_change"),
    ([3304, 3716, 4838], 0, "incomplete_fix"),
    ([2813], 0, "incomplete_fix"),
    ([1530, 2253, 2678, 3270, 3719, 4771, 4937, 5155, 5533, 5732, 5800], 0, "incomplete_fix"),
    ([1310, 4752], 0, "other_vulnerability_remains"),
    ([1190, 5922], 0, "incomplete_fix"),
    ([4300, 5858], 1, "collateral_change"),
    ([3982, 4580, 5040], 0, "incomplete_fix"),
    ([4728], 0, "incomplete_fix"),
    ([1693], 1, "mismatched_pair_upstream"),
    ([4415], 0, "other_vulnerability_remains"),
    ([50, 1125, 2048, 3812, 4114, 4276, 4972, 5439], 0, "incomplete_fix"),
    ([5882], 0, "incomplete_fix"),
    ([1869], 0, "other_vulnerability_remains"),
]
base = "benchmarks/context-assembler-dataset/"
datasets["cpg_structural_root_findings_verify"] = {
    "dataset_path": base + "context_assembler_cpg_structural.json",
    "task_type": "binary_vulnerability",
    "description": "cpg_structural context; root findings (B101/B113 dropped) as leads to verify; empty section omitted",
    "render_root_findings": True,
    "omit_empty_root_findings": True,
    "exclude_finding_rules": ["B101", "B113"],
}
datasets["mult_amp_root_findings_verify"] = {
    "dataset_path": base + "context_assembler_multiplicative_amplification.json",
    "task_type": "binary_vulnerability",
    "description": "multiplicative_amplification context; root findings (B101/B113 dropped) as leads to verify; empty section omitted",
    "render_root_findings": True,
    "omit_empty_root_findings": True,
    "exclude_finding_rules": ["B101", "B113"],
}
exclusion_lines = ",\n".join(
    "                " + json.dumps({"source_row_ids": ids, "label": label, "reason": reason})
    for ids, label, reason in EXCLUSIONS
)
blocks = []
for key, entry in datasets.items():
    fields = [f'            "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in entry.items() if k != "exclude_samples"]
    fields.append('            "exclude_samples": [\n' + exclusion_lines + "\n            ]")
    blocks.append(f'        "{key}": {{\n' + ",\n".join(fields) + "\n        }")
open(p, "w").write('{\n    "datasets": {\n' + ",\n".join(blocks) + "\n    }\n}\n")
json.load(open(p))
print("OK", len(datasets))
EOF
```

Expected: `OK 7`.

- [ ] **Step 5: Add plans to `src/configs/static_findings_root/experiments.json`**

```bash
python3 - <<'EOF'
import json
p = "src/configs/static_findings_root/experiments.json"
d = json.load(open(p))
common = {
    "datasets": ["cpg_structural_root_findings_verify", "mult_amp_root_findings_verify"],
    "models": ["gemma4-12b-it-thinking-sc7-logprobs-seeded"],
    "prompts": ["finding_verification_root_findings"],
    "coverage_levels": [0.25, 0.5, 0.75, 1.0],
}
d["experiment_plans"]["static_findings_verify_sweep"] = {
    "description": "Finding-verification prompt, B101/B113 filtered, empty section omitted, wrong labels excluded",
    **common,
}
d["experiment_plans"]["static_findings_verify_smoke"] = {
    "description": "Smoke test of the finding-verification sweep",
    **common,
    "sample_limit": 40,
}
open(p, "w").write(json.dumps(d, indent=4) + "\n")
print("OK")
EOF
```

- [ ] **Step 6: Run to verify pass**

Run: `PYTHONPATH=src $PY tests/test_finding_verification_configs.py` → `ALL PASSED`
Run: `PYTHONPATH=src $PY tests/test_static_findings_root_configs.py` → `ALL PASSED` (existing plans and 5 datasets untouched except the added exclusions)
Run the whole suite (Global Constraints command) → 0 failed.

- [ ] **Step 7: Document in `CLAUDE.md`** — append to the end of the "Root static-findings condition" section (before `## File layout notes`):

````markdown
Follow-up (finding verification): dataset entries may set `exclude_samples`
(`[{source_row_ids, label, reason}]`, audited wrong labels; all 7
`static_findings_root` entries carry the same 18), `exclude_finding_rules`
(fnmatch on `rule_id`, e.g. `["B101", "B113"]`) and `omit_empty_root_findings`
(render nothing instead of "none reported"). Plans may set `coverage_levels`
(default `[0.25, 0.5, 0.75, 1.0]`); binary reports carry
`{accuracy,precision,recall,f1_score,fpr,fnr}_at_coverage_<pct>`. Plans
`static_findings_verify_sweep` / `_smoke` run prompt
`finding_verification_root_findings` on the two `*_verify` datasets.
Recompute and merge saved reports without inference (container, `results/` is
root-owned):
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py merge-plan-results \
    --plan-dir results/static_findings_root/static_findings_root_sweep \
    --plan-dir results/static_findings_root/static_findings_verify_sweep \
    --datasets-config configs/static_findings_root/datasets.json \
    --exclude-finding-rules B101,B113 \
    --output-dir results/static_findings_root/merged_verify
```
````

- [ ] **Step 8: Commit**

```bash
git add src/configs/shared/prompts.json src/configs/static_findings_root/ \
    tests/test_finding_verification_configs.py CLAUDE.md
git commit -m "Add finding-verification prompt, exclusions and verify plans"
```

---

### Task 6: Build, smoke gate, sweep and merge

**Files:**
- Create (scratchpad, not committed): `/tmp/claude-1000/-var-opt-llm4codesec-framework/52f0f71a-515d-4449-ad71-de2fb78391f3/scratchpad/compose.fv.yml`

**Interfaces:**
- Consumes: everything above; image `llm4codesec-benchmark:fv`.
- Produces: `results/static_findings_root/static_findings_verify_{smoke,sweep}/…` and `results/static_findings_root/merged_verify/summary.md` in the main checkout.

- [ ] **Step 1: Check the GPU and image are free**

Run: `nvidia-smi --query-gpu=memory.used --format=csv,noheader; docker ps --format '{{.Names}} {{.Image}} {{.Status}}'; tmux ls`
Expected: memory used < 500 MiB and no running `llm4codesec` container. If another job is running, wait for it — never start a second GPU job.

- [ ] **Step 2: Build this branch under its own tag**

Run (from the worktree): `./build_docker.sh --no-gpu-test -t fv > ~/logs/build_fv.log 2>&1; echo exit=$?; tail -2 ~/logs/build_fv.log`
Expected: `exit=0`; `docker images llm4codesec-benchmark` lists tag `fv`.

- [ ] **Step 3: Write the compose override**

```yaml
# compose.fv.yml — run this branch's image against the main checkout's data and results.
services:
  llm4codesec-benchmark:
    image: llm4codesec-benchmark:fv
    volumes:
      - /var/opt/llm4codesec-framework/results:/app/results
      - /var/opt/llm4codesec-framework/datasets_processed:/app/datasets_processed
      - /var/opt/llm4codesec-framework/benchmarks:/app/benchmarks
```

Define (used below): `DC="docker-compose -f $HOME/worktrees/llm4codesec-fv/docker-compose.yml -f /tmp/claude-1000/-var-opt-llm4codesec-framework/52f0f71a-515d-4449-ad71-de2fb78391f3/scratchpad/compose.fv.yml"`
Check: `$DC config | grep -E "image:|/app/results"` shows `llm4codesec-benchmark:fv` and the absolute results path.

- [ ] **Step 4: Smoke run (tmux)**

```bash
tmux new-session -d -s fv_smoke -c "$HOME/worktrees/llm4codesec-fv" \
  "$DC run --rm llm4codesec-benchmark python cli.py run-plan context_assembler static_findings_verify_smoke --config-dir configs/shared --experiments-config configs/static_findings_root/experiments.json --datasets-config configs/static_findings_root/datasets.json 2>&1 | tee $HOME/logs/fv_smoke.log; echo EXIT=\${PIPESTATUS[0]} | tee -a $HOME/logs/fv_smoke.log; exec bash"
```

Wait until `EXIT=` appears in `~/logs/fv_smoke.log`. Expected: `EXIT=0`, two reports under `/var/opt/llm4codesec-framework/results/static_findings_root/static_findings_verify_smoke/`.

- [ ] **Step 5: Smoke gate (negative case)**

```bash
python3 - <<'EOF'
import glob, json
base = "/var/opt/llm4codesec-framework/results/static_findings_root/static_findings_verify_smoke"
for path in sorted(glob.glob(base + "/*/*/*/benchmark_report_*.json")):
    report = json.load(open(path))
    preds = report["predictions"]
    meta = report["benchmark_info"]["extra_metadata"]
    neg = [p for p in preds if p["true_label"] == 0]
    with_f = [p for p in neg if p["root_static_findings"]]
    without_f = [p for p in neg if not p["root_static_findings"]]
    fpr = lambda s: sum(p["predicted_label"] == 1 for p in s) / len(s) if s else None
    shown_none = sum("none reported" in (p["inference_data"]["prompt_text"] or "") for p in preds)
    b101 = sum(any(f["rule_id"] in ("B101", "B113") for f in p["root_static_findings"] or []) for p in preds)
    print(path.split("/")[-4], "n", len(preds), "excluded", len(meta["excluded_samples"]),
          "FPR with", fpr(with_f), f"(n={len(with_f)})", "FPR without", fpr(without_f), f"(n={len(without_f)})",
          "none-reported prompts", shown_none, "B101/B113 kept", b101,
          "f1@50", report["metrics"]["summary"].get("f1_score_at_coverage_50"))
EOF
```

Expected: per report `none-reported prompts 0`, `B101/B113 kept 0`, coverage keys present, and **gate**: FPR with findings ≤ 0.443 and FPR without findings ≤ 0.314 (a `None` FPR, i.e. no label-0 samples in that subset, passes). If a gate value fails, stop and report the numbers and sample responses to the user before the full sweep (prompt wording revision is a user decision).

- [ ] **Step 6: Full sweep (tmux)** — only after the gate passes:

```bash
tmux new-session -d -s fv_sweep -c "$HOME/worktrees/llm4codesec-fv" \
  "$DC run --rm llm4codesec-benchmark python cli.py run-plan context_assembler static_findings_verify_sweep --config-dir configs/shared --experiments-config configs/static_findings_root/experiments.json --datasets-config configs/static_findings_root/datasets.json 2>&1 | tee $HOME/logs/fv_sweep.log; echo EXIT=\${PIPESTATUS[0]} | tee -a $HOME/logs/fv_sweep.log; exec bash"
```

Expected after ~3.5 h: `EXIT=0`; two reports with 714 predictions each (732 − 18) under `…/static_findings_verify_sweep/`.

- [ ] **Step 7: Merge**

```bash
$DC run --rm llm4codesec-benchmark python cli.py merge-plan-results \
    --plan-dir results/static_findings_root/static_findings_root_sweep \
    --plan-dir results/static_findings_root/static_findings_verify_sweep \
    --datasets-config configs/static_findings_root/datasets.json \
    --exclude-finding-rules B101,B113 \
    --model gemma4-12b-it-thinking-sc7-logprobs-seeded \
    --output-dir results/static_findings_root/merged_verify
```

Expected: the printed `summary.md` has exactly 7 condition rows (5 old + 2 verify, gemma only), and every recomputed report under `merged_verify/` has `benchmark_info.stats.total_samples == 714`.

- [ ] **Step 8: Report**

Show the user `summary.md` and the gate numbers. No commit (results are git-ignored).
