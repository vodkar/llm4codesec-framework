#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

# Context sizes to process. Each lives in its own sub-directory under the
# compare-rankings context_sizes folder and is normalized into a matching
# processed output directory.
CONTEXT_SIZES=(1k 2k 4k 8k 16k)
SIZES_SOURCE_ROOT="benchmarks/context-assembler-dataset/cleanvul_compare_rankings/context_sizes"
SIZES_OUTPUT_ROOT="datasets_processed/context_assembler_compare_rankings/context_sizes"

# for size in "${CONTEXT_SIZES[@]}"; do
#     echo "=== Preparing context-size datasets: ${size} ==="
#     uv run python src/entrypoints/loaders/run_setup_context_assembler_compare_rankings.py \
#         --source-dir "${SIZES_SOURCE_ROOT}/${size}" \
#         --output-dir "${SIZES_OUTPUT_ROOT}/${size}"
# done

# Stop leftover benchmark containers (an interrupted docker-compose run can keep
# holding all GPU memory, which makes the next vLLM startup fail its
# gpu_memory_utilization check) and wait until the VRAM is actually released.
stale=$(docker ps -q --filter "name=llm4codesec")
if [ -n "$stale" ]; then
    echo "Stopping stale benchmark containers: $stale"
    docker stop $stale
fi
for _ in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 1500 ] && break
    echo "Waiting for GPU memory to be released (currently ${used} MiB used)..."
    sleep 5
done

run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

$run_benchmark cli.py run-plan context_assembler context_sizes \
    --config-dir configs/shared \
    --experiments-config configs/context_assembler_compare_rankings/experiments.json \
    --datasets-config configs/context_assembler_compare_rankings/datasets.json \
    --skip-existing
