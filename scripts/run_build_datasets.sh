#!/bin/bash

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

run_benchmark="uv run python"

$run_benchmark src/entrypoints/loaders/run_setup_castle_dataset.py
$run_benchmark src/entrypoints/loaders/run_setup_context_assembler_compare_rankings.py
$run_benchmark src/entrypoints/loaders/run_setup_context_assembler_datasets.py
$run_benchmark src/entrypoints/loaders/run_setup_jitvul_dataset.py --data-file benchmarks/JitVul/data/final_benchmark.jsonl
# CleanVul source CSVs are pre-split by score; the setup entrypoint now takes a
# single --source, so reproduce the score4/score3 layouts with one call each.
$run_benchmark src/entrypoints/setup_cleanvul_dataset.py \
    --source benchmarks/CleanVul/vulnerability_score_4.csv \
    --output-dir datasets_processed/cleanvul/score4
$run_benchmark src/entrypoints/setup_cleanvul_dataset.py \
    --source benchmarks/CleanVul/vulnerability_score_3.csv \
    --output-dir datasets_processed/cleanvul/score3

mkdir -p benchmarks/PrimeVul
wget --no-check-certificate \
     "https://drive.usercontent.google.com/download?id=12b1QkCwW0SC6l9KvxSmMe4jHF7VhjwCa&confirm=t" \
     -O "benchmarks/PrimeVul/primevul.jsonl"
$run_benchmark src/entrypoints/setup_primevul_dataset.py
