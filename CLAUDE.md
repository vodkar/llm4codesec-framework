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

## File layout notes

- `src/` is copied to `/app/` inside Docker — paths inside the container have no `src/` prefix.
- Config files: `src/configs/` → `/app/configs/`
- Scripts needing `from benchmark import …` must be invoked with `-e PYTHONPATH=/app` (or called via `cli.py`).
