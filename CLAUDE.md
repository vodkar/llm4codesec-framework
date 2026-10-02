To build image
```bash
./build_docker.sh --no-gpu-test
```

Always rebuild image after changes to `src/`.

## Running experiments

General pattern:
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan <benchmark> <plan> --config src/configs/<benchmark>_experiments.json
```

Available benchmarks: `castle`, `cvefixes`, `jitvul`, `vulbench`, `primevul`, `context_assembler`, `cleanvul`

Examples:
```bash
# context_assembler
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler vllm_big_models_comparison --config src/configs/context_assembler_experiments.json

# primevul quick sanity check
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan primevul quick_test --config src/configs/primevul_experiments.json
```

## PrimeVul dataset setup

Before running PrimeVul experiments, generate the processed JSON files:
```bash
# From a raw PrimeVul JSONL file (or the bundled sample):
docker-compose run --rm -e PYTHONPATH=/app llm4codesec-benchmark \
    python entrypoints/setup_primevul_dataset.py \
    --source benchmarks/PrimeVul/<your_file>.jsonl \
    --output-dir datasets_processed/primevul
```

The sample file `benchmarks/PrimeVul/primevul_sample.jsonl` can be used for testing.

## CleanVul dataset setup

CleanVul data lives in the `benchmarks/CleanVul` submodule. Initialise it once:
```bash
git submodule update --init benchmarks/CleanVul
```

Then generate the processed JSON files. The entrypoint takes a single `--source`
CSV per call, so run it once per score tier (the CSVs are pre-split by score):
```bash
docker-compose run --rm -e PYTHONPATH=/app llm4codesec-benchmark \
    python entrypoints/setup_cleanvul_dataset.py \
    --source benchmarks/CleanVul/vulnerability_score_4.csv \
    --output-dir datasets_processed/cleanvul/score4
docker-compose run --rm -e PYTHONPATH=/app llm4codesec-benchmark \
    python entrypoints/setup_cleanvul_dataset.py \
    --source benchmarks/CleanVul/vulnerability_score_3.csv \
    --output-dir datasets_processed/cleanvul/score3
```

Optional filters: `--languages py`, `--tasks {binary,multiclass,both}`,
`--min-score`/`--max-score`, and `--min-tokens`/`--max-tokens` (with
`--tokenizer`, `--name-suffix`) for building token-range context-size buckets.

Generated files: `datasets_processed/cleanvul/score4/cleanvul_{c,cpp,java,py,js,cs}_{binary,multiclass}.json`

Quick test after setup:
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan cleanvul quick_test --config src/configs/cleanvul_experiments.json
```

## Reference-context evaluation

Single-command, pinned-config run of the reference-context (oracle) evaluation
(S1-S5: llm_scanner entity-set precompute → condition-dataset build →
retrieval-overlap report → framework benchmark sweep → analysis/report):

```bash
scripts/run_reference_context_eval.sh [config/reference_context_eval.yaml] [--smoke]
scripts/run_reference_context_eval.sh --smoke   # same, using the default config
```

- The config is `config/reference_context_eval.yaml`, pinned so its
  `config_sha256` (and the llm_scanner cache) only changes when a value
  changes. It has two per-machine placeholders (`pins.llm_scanner_git_sha`,
  `llm_scanner.dataset_sha256`) that start with `<fill ...>`; the script
  refuses to run (exit 2) until both are filled in.
- `llm_scanner.dataset_path` points at
  `benchmarks/CleanVul/vulnerability_score_4.csv` — the only existing copy of
  that CSV on this machine (it lives in the framework's `benchmarks/CleanVul`
  submodule, not under the llm_scanner checkout). Fill
  `llm_scanner.dataset_sha256` with
  `sha256sum benchmarks/CleanVul/vulnerability_score_4.csv` (run from this
  repo root, or use the absolute path
  `/var/opt/llm4codesec-framework/benchmarks/CleanVul/vulnerability_score_4.csv`).
- Before the first full run, verify `llm_scanner.token_budget` and
  `llm_scanner.max_call_depth` in the config against the build of the
  `cleanvul_vs_cpg_structural` datasets they must match.
- Stages S1-S3 run against the pinned `llm_scanner` checkout
  (`pins.llm_scanner_path`) via `uv run llm-scanner precompute-entity-sets /
  build-condition-datasets / retrieval-overlap-report --config <this yaml>`;
  the datasets it writes are copied into
  `datasets_processed/reference_context/` for the container.
- The framework benchmark sweep runs as
  `docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler <plan> --config-dir configs/shared --experiments-config configs/reference_context/experiments.json --datasets-config configs/reference_context/datasets.json`.
  `reference_context` is **not** a registered `run-plan` benchmark name (the
  `BENCHMARKS` dict in `src/cli.py` is fixed to
  castle/cvefixes/jitvul/vulbench/primevul/context_assembler/cleanvul); the
  benchmark argument only drives config-file discovery and the default
  output dir, both overridden here by `--experiments-config`/
  `--datasets-config` and `output_settings.base_output_dir` in
  `configs/reference_context/experiments.json`, so `context_assembler` (the
  generic-JSON-dataset benchmark) is used instead — the plans themselves
  (`reference_context_sweep`, `reference_context_smoke`) come from that
  experiments config.
- `--smoke` runs plan `reference_context_smoke` instead of
  `framework.plan` and analyzes its own results. It limits **only
  inference** (the smoke plan's `sample_limit`): stages S1-S2 (and S3) still
  run over **all** eligible pairs, so a cold-cache smoke run is as slow as a
  full run up to the Docker step.
- After S2 the script reads `<llm_scanner.output_dir>/run_manifest.json` and
  aborts (exit 2) before Docker unless `llm_scanner_dirty` is `false`,
  `llm_scanner_git_sha` equals `pins.llm_scanner_git_sha`, and
  `config_sha256` equals the sha256 of the config file. At startup it also
  checks that `framework.experiments_config`/`framework.datasets_config`
  (container paths, i.e. `src/<path>` on the host) exist and that every
  `dataset_path` in the datasets config lies under `framework.datasets_dir`.
- Dirty checks never count untracked files: the analysis records
  `framework_dirty` from `git status --porcelain --untracked-files=no -- src
  config scripts`, and llm_scanner's manifest `llm_scanner_dirty` from
  `--untracked-files=no -- llm_scanner`. `results/reference_context/` is
  git-ignored.
- Analysis runs on the host:
  `PYTHONPATH=src uv run python src/cli.py analyze-reference-context --config <yaml> --output-dir <framework.results_dir>/<plan>/analysis --plan <plan>`,
  where `<framework.results_dir>` is read from the config (`results/reference_context`
  by default) rather than hard-coded, so a config that points `results_dir`
  elsewhere is honored end to end.
  It writes `report.md`/`report.json`/`acceptance.json` and exits 1 if the
  run fails any of the six acceptance checks, or (with a RUN INVALID report)
  if no pair survives into the analysis.
- Before analysing, it picks the latest `benchmark_report_*.json` per
  condition and refuses (ValueError) a mixed or stale set: all five must
  share `benchmark_info.model` (including a non-null `sampling_seed`) and
  prompt identity (`prompt_identifier` + `prompt_template_sha256`), each
  dataset file's sha256 must match the manifest's
  `condition_dataset_sha256`, and no report's `timestamp_utc` may predate the
  manifest's `created_utc`. Reports written before these fields existed must
  be re-run.

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
root-owned; `--output-dir` must be new or empty; `--model` is required when a
plan dir holds several models):
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py merge-plan-results \
    --plan-dir results/static_findings_root/static_findings_root_sweep \
    --plan-dir results/static_findings_root/static_findings_verify_sweep \
    --datasets-config configs/static_findings_root/datasets.json \
    --exclude-finding-rules B101,B113 \
    --model gemma4-12b-it-thinking-sc7-logprobs-seeded \
    --output-dir results/static_findings_root/merged_verify
```

Target/context split: context datasets are one flat, comment-stripped snippet with
nothing marking the function under test, so the model flags flaws in callers and
callees. `split_target_context.py` puts the commented function-only code in `code`
and the context (minus the target's lines, matched per top-level definition) in
`context`, which renders after the target as reference-only code. Function-only
prompts are unchanged, so plan `static_findings_target_sweep` runs only the four
`*_target_findings_{off,on}` conditions and reuses the function-only report:
```bash
for c in cpg_structural multiplicative_amplification; do
  PYTHONPATH=src uv run python src/entrypoints/split_target_context.py \
    --context-dataset benchmarks/context-assembler-dataset/context_assembler_$c.json \
    --function-dataset benchmarks/context-assembler-dataset/cleanvul_python_matched.json \
    --output datasets_processed/context_assembler/${c}_target_split.json
done
```

## Root/context split (dataset v2)

`benchmarks/context-assembler-dataset/v2/` (same 710 samples, audited wrong labels
already removed, new sample ids) adds a `roots` field: per changed function
(`file_path`, `line_start`, `line_end`, `code`) plus the context attributed to it.
Samples with `roots` render `{code}` via `src/benchmark/root_sections.py`: a
`## CODE UNDER ANALYSIS` section (one fenced `### ROOT n: file, lines a-b` block per
root), then `## REFERENCE CONTEXT (read-only)` with a `### Context for ROOT n` block
per non-empty context. The flat `code` field is kept only because static findings'
`snippet_line` indexes into it. Prompt `root_context_security_v2_root_findings` is
the v2 prompt plus an input-layout paragraph and a scope reminder after the code.
v2 root code is comment-stripped (the function-only baseline is not). Every prompt
fits LFM's 16k input budget (max 12.6k tokens), so none are filtered.

One-time setup (function-only baseline as a single root, findings attached):
```bash
PYTHONPATH=src uv run python src/entrypoints/attach_root_findings.py --as-root \
    --target benchmarks/context-assembler-dataset/v2/cleanvul_python_matched.json \
    --findings-source benchmarks/context-assembler-dataset/v2/context_assembler_cpg_structural.json \
    --output datasets_processed/context_assembler/v2/cleanvul_python_matched_root_findings.json
```

Run (`root_context_v2_lfm_smoke` for 20 samples per condition):
```bash
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler root_context_v2_lfm_sweep \
    --config-dir configs/shared \
    --experiments-config configs/root_context_v2/experiments.json \
    --datasets-config configs/root_context_v2/datasets.json
```

## File layout notes

- `src/` is copied to `/app/` inside Docker — paths inside the container have no `src/` prefix.
- Config files: `src/configs/` → `/app/configs/`
- Scripts needing `from benchmark import …` must be invoked with `-e PYTHONPATH=/app` (or called via `cli.py`).
