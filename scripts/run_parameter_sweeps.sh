#!/bin/bash

set -euo pipefail

# PLAN_SUFFIX=binary
PLAN_SUFFIX=combinations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

cd "$REPO_ROOT"

benchmarks=(
  castle
  cvefixes
  jitvul
  primevul
  cleanvul
)

for benchmark in "${benchmarks[@]}"; do
  for thinking in "_thinking" ""; do
    plan_name="sampling_sweep_${PLAN_SUFFIX}${thinking}"
    echo "Running ${benchmark} with plan ${plan_name}"
    $run_benchmark cli.py run-plan "$benchmark" "$plan_name" \
      --config-dir configs/parameter_sweeps \
      --datasets-config configs/"$benchmark"/datasets.json \
      --output-dir results_parameter_sweeps
  done
done
