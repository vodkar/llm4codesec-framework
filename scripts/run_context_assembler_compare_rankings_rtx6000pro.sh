#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

uv run python src/entrypoints/loaders/run_setup_context_assembler_compare_rankings.py "$@"

run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

$run_benchmark cli.py run-plan context_assembler rtx_6000_pro_compare_rankings \
    --config-dir configs/context_assembler_compare_rankings/rtx_6000_pro \
    --experiments-config configs/context_assembler_compare_rankings/rtx_6000_pro/experiments.json \
    --datasets-config configs/context_assembler_compare_rankings/datasets.json
