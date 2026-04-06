#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

uv run python src/entrypoints/loaders/run_setup_context_assembler_compare_rankings.py "$@"

run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

$run_benchmark cli.py run-plan context_assembler vllm_compare_rankings \
    --config-dir configs/shared \
    --experiments-config configs/context_assembler_compare_rankings/experiments.json \
    --datasets-config configs/context_assembler_compare_rankings/datasets.json

# $run_benchmark cli.py run-plan context_assembler big_models_compare_rankings \
#     --config-dir configs/shared \
#     --experiments-config configs/context_assembler_compare_rankings/experiments.json \
#     --datasets-config configs/context_assembler_compare_rankings/datasets.json
