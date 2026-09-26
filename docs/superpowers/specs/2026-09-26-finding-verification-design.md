# Finding Verification, Sample Exclusions and Coverage Metrics — Design

Date: 2026-09-26
Status: Approved design, awaiting spec review
Builds on: `docs/superpowers/specs/2026-09-26-root-static-findings-condition-design.md`

## Goal

Follow-up to the root static-findings sweep (gemma4-12b, 5 conditions, 732
samples). That sweep showed findings shift the model toward "safe" (FPR down,
FNR up), help ranking for cpg (AP 0.648 → 0.662), and that "none reported"
costs recall on the 549 samples without findings. Part of the ~0.60 accuracy
ceiling is label noise (18 confirmed wrong labels in the audit).

This work adds:

1. **Sample exclusions** — drop audited wrong-label samples via the dataset config.
2. **Rule filter** — drop root findings of chosen analyzer rules via the dataset config.
3. **Finding-verification prompt** — findings are leads to CONFIRM or REFUTE;
   the findings section is omitted when a sample has none.
4. **Coverage metrics** — configurable coverage levels with the full metric set
   per level, instead of judging only the forced verdict at 0.5.
5. **Merge/recompute command** — recompute metrics of existing reports under the
   exclusions/rule filter and merge old and new conditions into one result set,
   without re-running inference.

Then run the two new conditions and merge them with the five existing ones.

## Decisions (agreed)

- Exclusions: only the 18 confirmed wrong labels (`samples` in
  `benchmarks/cleanvul_wrong_labels.json`); not `disputed`, `pairing_issues`,
  `not_judgeable`. Only the listed sample is excluded; its vuln/fixed partner stays.
- Rule filter default: `B101`, `B113` only (most frequent, no signal: vuln share
  0.53 / 0.50, 142 of 626 root findings). Other rules (B311, B603, B607, B110, …)
  can be real issues and stay for the model to verify.
- New prompt strategy: finding verification; empty findings section omitted.
- Coverage: configurable levels, full metrics per level.
- New inference only for the two changed conditions; old reports are recomputed.
- Work happens in worktree `~/worktrees/llm4codesec-fv`, branch
  `feature/finding-verification` (another session has uncommitted edits in the
  main checkout).

## Design

### 1. Sample exclusions

Dataset config entry gains:

```json
"exclude_samples": [
  {"source_row_ids": [789, 2754], "label": 1, "reason": "not_a_security_fix"},
  ...
]
```

- `DatasetConfig.exclude_samples: list[SampleExclusion] = []`, where
  `SampleExclusion(source_row_ids: list[int], label: int, reason: str = "")`.
  Key = `(tuple(source_row_ids), label)`, the same join key reports store.
- Applied in `BenchmarkRunner.run` right after loading, before token filtering
  and `sample_limit`, by a pure helper
  `apply_sample_exclusions(samples, exclusions) -> (kept, excluded_keys, unmatched)`.
- A sample without `metadata.source_row_ids` while exclusions are configured →
  `ValueError` (the key cannot be formed).
- Unmatched exclusion entries are logged as a warning and recorded (typo or
  changed dataset), not an error.
- `BenchmarkInfo.extra_metadata` records `excluded_samples` (list of
  `{"source_row_ids", "label"}`) and `unmatched_exclusions`.

The 18 entries (from the current `samples` list; all 18 verified present in the
cpg, multiplicative and function-only datasets):

| source_row_ids | label | reason |
|---|---|---|
| [789, 2754] | 1 | not_a_security_fix |
| [3607] | 1 | collateral_change |
| [3875] | 1 | mismatched_pair_upstream |
| [1054, 1679, 1721, 3767] | 0 | incomplete_fix |
| [5192] | 1 | collateral_change |
| [3304, 3716, 4838] | 0 | incomplete_fix |
| [2813] | 0 | incomplete_fix |
| [1530, 2253, 2678, 3270, 3719, 4771, 4937, 5155, 5533, 5732, 5800] | 0 | incomplete_fix |
| [1310, 4752] | 0 | other_vulnerability_remains |
| [1190, 5922] | 0 | incomplete_fix |
| [4300, 5858] | 1 | collateral_change |
| [3982, 4580, 5040] | 0 | incomplete_fix |
| [4728] | 0 | incomplete_fix |
| [1693] | 1 | mismatched_pair_upstream |
| [4415] | 0 | other_vulnerability_remains |
| [50, 1125, 2048, 3812, 4114, 4276, 4972, 5439] | 0 | incomplete_fix |
| [5882] | 0 | incomplete_fix |
| [1869] | 0 | other_vulnerability_remains |

The list is written inline into every dataset entry of
`src/configs/static_findings_root/datasets.json` (5 existing + 2 new). A config
test asserts all 7 lists are identical.

### 2. Rule filter

Dataset config entry gains `"exclude_finding_rules": ["B101", "B113"]`
(`DatasetConfig.exclude_finding_rules: list[str] = []`), matched with
`fnmatch.fnmatchcase` against `RootStaticFinding.rule_id`.

- Pure helper `filter_root_findings(findings, patterns) -> list[RootStaticFinding]`
  in `src/benchmark/static_findings.py`; `None` stays `None`.
- Applied in the runner once per sample, before rendering and before
  `sample_provenance`, so the prompt and the stored `root_static_findings` are
  the same filtered list (the analyzer-only baseline follows the filter).
- `extra_metadata["exclude_finding_rules"]` records the patterns.
- Set only on the 2 new dataset entries. The 5 existing entries keep describing
  how their reports were produced (their findings-on prompts showed unfiltered
  findings). For a uniform analysis-time definition of "has findings", the merge
  command filters stored findings with its own `--exclude-finding-rules` option
  (§5) without implying the prompts changed.

### 3. Finding-verification prompt

`DatasetConfig.omit_empty_root_findings: bool = False`. When rendering is on and
the (filtered) list is empty, the placeholder renders `""` instead of the
"none reported" line. `render_root_findings_block(findings, omit_empty)` gains
the flag. Findings-block format is otherwise unchanged.

New prompt `finding_verification_root_findings` in `src/configs/shared/prompts.json`:

- `system_prompt`: the `strict_exploitable_security` system prompt verbatim,
  followed by:

  ```
  \n\nStatic analyzer findings:\nThe code may come with static-analyzer findings. Treat each finding as a lead, not as evidence. For every finding, decide CONFIRMED or REFUTED. REFUTED means the flagged data is not attacker-controlled, the line is unreachable, or validation, escaping, an allow-list, or a safe API in the shown code prevents exploitation. Most analyzer findings are false positives. Report a flaw the analyzers did not flag only if you can trace a concrete source-to-sink path. If every finding is refuted and you found no other concrete flaw, the code is NOT vulnerable. If no findings are given, analyze the code on its own merits.
  ```

- `user_prompt`: `Analyze this code for a real, exploitable security vulnerability:\n\n{code}{root_static_findings}`

The response contract (JSON verdict on the last line) is unchanged and is
appended by the framework as today.

### 4. Coverage metrics

- Plan config gains optional `"coverage_levels": [0.25, 0.5, 0.75, 1.0]`
  (default when absent: `[0.25, 0.5, 0.75, 1.0]`). Values must be in (0, 1];
  otherwise `ValueError` at config load. `ExperimentConfig.coverage_levels`
  carries them; `MetricsCalculatorFactory.create_calculator(task_type,
  coverage_levels=...)` passes them to the binary calculator, which replaces
  the module constant `_COVERAGE_LEVELS` with the instance value.
- For each level (samples ranked by `answer_probability`, most confident
  first, as today) the details record `coverage`, `selected_samples`,
  `min_answer_probability`, `accuracy`, `precision`, `recall`, `f1_score`,
  `fpr`, `fnr`. Undefined ratios (zero denominator) are `None`.
- Summary keys: `{metric}_at_coverage_{pct}` for accuracy, precision, recall,
  f1_score, fpr, fnr (e.g. `f1_score_at_coverage_50`). Existing `accuracy_at_coverage_*` keys keep their
  names; the confidence-method variants get the same metric set with their prefix.

### 5. Merge / recompute command

`cli.py merge-plan-results` (runs in the container; `results/` is root-owned):

```
python cli.py merge-plan-results \
    --plan-dir results/static_findings_root/static_findings_root_sweep \
    --plan-dir results/static_findings_root/static_findings_verify_sweep \
    --datasets-config configs/static_findings_root/datasets.json \
    --coverage-levels 0.25,0.5,0.75,1.0 \
    --exclude-finding-rules B101,B113 \
    --output-dir results/static_findings_root/merged_verify
```

For every condition directory (`<plan-dir>/<dataset_key>/<model>/<prompt>/`),
take the latest `benchmark_report_*.json`, then:

1. Look up `<dataset_key>` in the datasets config (missing key → error).
2. Drop predictions matching that entry's `exclude_samples` (by stored
   `source_row_ids` + `true_label`); filter stored `root_static_findings` with
   `--exclude-finding-rules` (analysis only; prompts are not re-rendered).
3. Convert `PredictionRecord` → `PredictionResult` (pure helper
   `record_to_prediction`) and recompute metrics with the binary calculator and
   the given coverage levels.
4. Write the recomputed report to
   `<output-dir>/<plan>/<dataset_key>/<model>/<prompt>/benchmark_report_<ts>.json`
   with `extra_metadata.recomputed_from` (source path), `excluded_samples`,
   `analysis_exclude_finding_rules`.

Then write `<output-dir>/experiment_plan_results.json` (via the existing
`rebuild_experiment_plan_results`) and `<output-dir>/summary.md`: one row per
condition, columns accuracy, precision, recall, F1, FPR, FNR, AP (from ranking
metrics), and accuracy/F1/FPR at each coverage level; best value per column in
**bold** (lowest for FPR/FNR). Reports whose predictions lack
`source_row_ids` while exclusions are configured → error.

### 6. Runs

`src/configs/static_findings_root/datasets.json` gains:

- `cpg_structural_root_findings_verify` → cpg dataset file, `render_root_findings`,
  `omit_empty_root_findings`, `exclude_finding_rules`, `exclude_samples`.
- `mult_amp_root_findings_verify` → multiplicative dataset file, same flags.

`src/configs/static_findings_root/experiments.json` gains plans
`static_findings_verify_sweep` (2 datasets, model
`gemma4-12b-it-thinking-sc7-logprobs-seeded`, prompt
`finding_verification_root_findings`, `coverage_levels`) and
`static_findings_verify_smoke` (same, `sample_limit: 40`). The existing plans'
`models` stay the gemma entry on this branch.

**Smoke gate (negative case).** On the smoke run, compute FPR on label-0
samples: with ≥1 finding it must be ≤ 0.443 (cpg findings-on, hint prompt), and
without findings ≤ 0.314 (cpg, no findings). If either fails, revise the prompt
wording before the full sweep. With ~20 label-0 smoke samples this catches only
gross regressions; the full sweep is the real test.

After the sweep, one `merge-plan-results` call merges the 5 old conditions and
the 2 new ones.

## Error handling

- Exclusions configured but a sample lacks `source_row_ids` → `ValueError`.
- Unmatched exclusion entries → warning + recorded, not fatal.
- Invalid coverage level → `ValueError` at config load.
- Merge: dataset key not in datasets config, or no report in a condition dir → error.

## Testing

- `apply_sample_exclusions`: matching by row ids + label (same row ids, other
  label kept); unmatched reported; missing row ids raises.
- `filter_root_findings`: exact and wildcard patterns; `None` passthrough.
- Rendering: `omit_empty` on/off with empty list; non-empty unchanged.
- Runner: exclusions applied before `sample_limit`; filtered findings both in
  prompt and stored; `extra_metadata` records exclusions and rules.
- Coverage: per-level metrics on a hand-computed fixture; custom levels change
  summary keys; `None` for zero denominators; level validation.
- Merge: record→prediction round trip; recomputed metrics on a fixture equal
  metrics computed directly; exclusions and rule filter applied; summary bolding.
- Configs: 7 identical exclusion lists (18 entries); verify plans resolve; new
  prompt = base system prompt + verification paragraph.
- Existing suite stays green.

## Out of scope

- Re-running the 3 unchanged conditions (recomputed instead).
- Relabeling samples (exclusion only).
- Disputed / not-judgeable / pairing-issue entries.
- Changes to llm_scanner or the `is_root` definition.
- New models (handled in another session).
