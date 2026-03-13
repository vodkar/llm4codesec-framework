#!/bin/bash

set -euo pipefail

PLAN_NAME=sampling_sweep_binary
#PLAN_NAME=sampling_sweep_quick
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

cd "$REPO_ROOT"

benchmarks=(
 castle
 cvefixes
 jitvul
)

for benchmark in "${benchmarks[@]}"; do
  echo "Running ${benchmark} with plan ${PLAN_NAME}"
  $run_benchmark cli.py run-plan "$benchmark" "$PLAN_NAME" \
    --config-dir configs/parameter_sweeps \
    --datasets-config configs/"$benchmark"/datasets.json \
    --output-dir results_parameter_sweeps
done
