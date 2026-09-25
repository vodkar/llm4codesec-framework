# Root Static-Analysis Findings Condition — Design

Date: 2026-09-26
Status: Approved design, awaiting spec review

## Goal

Add an experiment condition that gives the LLM the static-analyzer findings
(Bandit, Dlint, Semgrep) located in the function under analysis, alongside the
code. First iteration: only findings with `is_root == true`.

The results must support (in external notebooks, not in this framework):

- Function-only CleanVul baseline vs context (cpg_structural, multiplicative
  amplification) with and without root findings.
- Static analyzers alone vs static analyzers + LLM: how FPR, FNR, precision,
  recall, F1, MCC change.

Analysis code is **out of scope**. The requirement is that every report carries
enough per-sample data to compute those metrics and to join samples across
reports.

## Inputs (existing, from llm_scanner `feature-vulnerability-meta`)

| File (under `benchmarks/context-assembler-dataset/`) | Samples | Findings |
|---|---|---|
| `context_assembler_cpg_structural.json` | 732 | `static_findings` + `source_map` |
| `context_assembler_multiplicative_amplification.json` | 732 | `static_findings` + `source_map` |
| `cleanvul_python_matched.json` (function-only) | 732 | none |

- All three align 732/732 on the key `(metadata.source_row_ids, label)`.
- Root findings are identical per sample in both context datasets (626 findings,
  183 samples with ≥1: 104 vulnerable, 79 fixed).
- `is_root` is used exactly as llm_scanner defines it. Known issue, not fixed
  here: in 11 samples the llm_scanner root node is an enclosing function larger
  than CleanVul's (nested) function, so some root findings lie outside the
  function-only code. This only affects stored findings for the function-only
  baseline, which never shows them.
- `datasets_processed/context_assembler/cleanvul_python_matched.json` is a stale
  copy that differs from the `benchmarks/` one; it is not used.

Each finding has: `tool`, `rule_id`, `cwe_id`, `severity`, `message`,
`file_path`, `repo_line`, `snippet_line` (1-based line in `code`), `is_root`.

## Conditions

One plan, one model, one prompt, five datasets (plans cross-product
model × dataset × prompt):

| Dataset key | Code | Findings in prompt |
|---|---|---|
| `cleanvul_python_matched_root_findings` | function-only | no (stored only) |
| `cpg_structural_root_findings_off` | cpg_structural context | no |
| `cpg_structural_root_findings_on` | cpg_structural context | yes |
| `mult_amp_root_findings_off` | multiplicative context | no |
| `mult_amp_root_findings_on` | multiplicative context | yes |

The analyzer-only baseline (≥1 root finding ⇒ vulnerable) needs no LLM run; it
is computed in notebooks from the stored findings.

## Design

### 1. Sample model and loader

`src/benchmark/models.py`:

```python
class RootStaticFinding(BaseModel):
    tool: str                 # "bandit" | "dlint" | "semgrep"
    rule_id: str
    cwe_id: int | None = None
    severity: str | None = None
    message: str
    file_path: str            # repo-relative
    repo_line: int
    flagged_line: str         # stripped text of code line `snippet_line`

class BenchmarkSample(BaseModel):
    ...
    root_static_findings: list[RootStaticFinding] | None = None
```

`None` = dataset carries no findings information; `[]` = analyzed, none found.

Benchmarks load samples through `JsonDatasetLoader` →
`BenchmarkSample.model_validate(raw)` (`ContextAssemblerDatasetLoader` is only
used at dataset setup), so extraction happens in a `model_validator(mode="before")`
on `BenchmarkSample`, calling a pure function in `src/benchmark/static_findings.py`:

- If the raw sample has a `static_findings` key: keep entries with
  `is_root == true`, sorted by `(snippet_line, tool, rule_id)`, and build
  `RootStaticFinding` with `flagged_line = code.split("\n")[snippet_line - 1].strip()`.
  A `snippet_line` outside `code` raises `ValueError` (corrupt dataset).
- If the raw sample already has `root_static_findings` (written by the attach
  entrypoint, §4), load it as-is.
- Otherwise `root_static_findings = None`.

### 2. Dataset config flag

`DatasetConfig` gains `render_root_findings: bool = False`, read from the
dataset entry in the experiments JSON (`"render_root_findings": true`).

### 3. Prompt placeholder and rendering

New prompt `strict_exploitable_security_root_findings` in
`src/configs/shared/prompts.json`: system prompt identical to
`strict_exploitable_security`; user prompt

```
Analyze this code for a real, exploitable security vulnerability:\n\n{code}{root_static_findings}
```

The placeholder name is `root_static_findings` so future finding sets can use
their own placeholders. Its position in the template controls where the block
appears (e.g. `...vulnerability:{root_static_findings}\n\n{code}` puts it first).

A single helper, `build_user_template_values(sample, dataset_config) ->
dict[str, str]`, replaces the two `get_user_prompt({"code": sample.code})` calls
in `benchmark_runner.py` (lines ~177 and ~264). It returns `code` and
`root_static_findings`:

- flag off → `""` (prompt identical to the same template without the
  placeholder);
- flag on, ≥1 finding →

  ```
  \n\nStatic analyzer findings in the function under analysis (automated tools; may be false positives):
  1. [bandit B602 | CWE-78 | HIGH] <message>
     Flagged line: <flagged_line>
  2. [dlint DUO116] <message>
     Flagged line: <flagged_line>
  ```

  `CWE-<n>` and severity are omitted when `None`. No cap on count, no rule
  filtering (first iteration).
- flag on, no findings → `\n\nStatic analyzer findings in the function under analysis: none reported.`

The block carries its own leading `\n\n` so it lays out cleanly both after and
before `{code}`.

Templates without the `{root_static_findings}` placeholder keep working:
`DefaultPromptGenerator.get_user_prompt` uses `str.format`, which ignores unused
keyword arguments.

Validation (fail loudly, before inference):

- `render_root_findings` true and the prompt's user template lacks
  `{root_static_findings}` → `ValueError` at experiment config resolution.
- `render_root_findings` true and a sample has `root_static_findings is None`
  → `ValueError` when building the prompt.

### 4. Function-only findings (stored, not shown)

New entrypoint `src/entrypoints/attach_root_findings.py`:

```
python entrypoints/attach_root_findings.py \
    --target benchmarks/context-assembler-dataset/cleanvul_python_matched.json \
    --findings-source benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json \
    --output datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json
```

- Joins on `(tuple(metadata.source_row_ids), label)`; any target sample without
  a unique match raises and nothing is written.
- Writes the target samples unchanged plus `root_static_findings` (the
  `RootStaticFinding` list computed from the source sample, `flagged_line` taken
  from the source snippet).

### 5. Stored per-sample results

`PredictionResult` and `PredictionRecord` (top level, written by
`ResultProcessor._to_prediction_record`) gain:

- `source_row_ids: list[int] | None` — from `sample.metadata["source_row_ids"]`;
  with `true_label` it is the stable cross-report join key (sample ids are
  regenerated per dataset build and are not safe to join on).
- `root_static_findings: list[RootStaticFinding] | None` — always stored when the
  dataset has them, regardless of the flag.
- `root_findings_in_prompt: bool` — the dataset's `render_root_findings`.

`benchmark_info` records `render_root_findings` for the dataset. Prompt text is
already retained in `inference_data.prompt_text`.

### 6. Configuration

New `src/configs/static_findings_root/datasets.json` and
`src/configs/static_findings_root/experiments.json` (same split layout as
`configs/reference_context/`; `--experiments-config` only contributes plans and
output settings, datasets come from `--datasets-config`). Output base dir
`results/static_findings_root`. Run:
`python cli.py run-plan context_assembler <plan> --config-dir configs/shared --experiments-config configs/static_findings_root/experiments.json --datasets-config configs/static_findings_root/datasets.json`.

- Datasets: the five keys above. `cpg_*` point at
  `benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json`,
  `mult_*` at `.../context_assembler_multiplicative_amplification.json`; the
  `_on` entries set `"render_root_findings": true`.
- Plans:
  - `static_findings_root_sweep`: the five datasets, model
    `gemma4-12b-it-thinking-sc7-logprobs-seeded`, prompt
    `strict_exploitable_security_root_findings`.
  - `static_findings_root_smoke`: same with `sample_limit: 40`.

CLAUDE.md gets a short section: attach step + run command.

## Testing

- Loader: keeps only `is_root` findings, sorted; `flagged_line` correct;
  out-of-range `snippet_line` raises; no `static_findings` → `None`; preloaded
  `root_static_findings` passes through.
- Rendering: flag on with findings (CWE/severity present and absent), flag on
  with none, flag off. Flag-off prompt is byte-identical to
  `strict_exploitable_security` for the same sample.
- Validation: flag on + template without placeholder; flag on + `None` findings.
- Attach entrypoint: successful join; missing/duplicate key raises and writes
  nothing.
- `PredictionRecord` serializes `source_row_ids`, `root_static_findings`,
  `root_findings_in_prompt`.
- Smoke run of `static_findings_root_smoke` in Docker after rebuilding the image.

## Out of scope

- Metrics/analysis code (analyzer-only baseline, OR/AND fusion, stratified
  metrics) — done in external notebooks.
- Non-root findings, rule filtering (e.g. B101), finding caps, findings-before-code
  ablation.
- Fixing llm_scanner's root-node definition for nested functions.
- Function-only + findings condition.
