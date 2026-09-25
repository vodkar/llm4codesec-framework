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

## File layout notes

- `src/` is copied to `/app/` inside Docker — paths inside the container have no `src/` prefix.
- Config files: `src/configs/` → `/app/configs/`
- Scripts needing `from benchmark import …` must be invoked with `-e PYTHONPATH=/app` (or called via `cli.py`).
