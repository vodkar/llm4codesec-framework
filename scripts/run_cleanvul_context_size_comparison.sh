#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

# ---------------------------------------------------------------------------
# Path to the CleanVul source CSV. FILL THIS IN.
# Use a lower-score file (e.g. vulnerability_score_2.csv) to include the
# lower-quality score 1-2 samples; score/token filters are applied below.
# ---------------------------------------------------------------------------
SOURCE_CSV="datasets_processed/vulnerability_score_0.csv"

TOKENIZER="google/gemma-4-12B-it-qat-w4a16-ct"
OUTPUT_DIR="datasets_processed/cleanvul/context_sizes"

# label   min_tokens  max_tokens   (half-open [min, max))
BUCKETS=(
    "0k_1k 0 1000"
    "1k_2k 1000 2000"
    "2k_4k 2000 4000"
    "4k_8k 4000 8000"
    "8k_16k 8000 16000"
)

for bucket in "${BUCKETS[@]}"; do
    read -r label lo hi <<< "$bucket"
    echo "=== Building CleanVul Python binary bucket ${label} (${lo}-${hi} tokens) ==="
    uv run python src/entrypoints/setup_cleanvul_dataset.py \
        --source "$SOURCE_CSV" \
        --output-dir "$OUTPUT_DIR" \
        --languages py \
        --tasks binary \
        --tokenizer "$TOKENIZER" \
        --min-tokens "$lo" \
        --max-tokens "$hi" \
        --name-suffix "_${label}"
done

# GPU offload for the large GGUFs (27B: 48/64 layers, 26B-A4B: 24/30) is set
# per model via "n_gpu_layers" in configs/shared/models.json — full offload
# plus 22k-ctx KV/compute buffers does not fit the 16 GB GPU and fails with
# "Failed to create llama_context".
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan \
    cleanvul context_size_comparison \
    --config src/configs/cleanvul_experiments.json
