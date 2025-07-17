#!/bin/bash

# Exit on any error
set -e

# Define the benchmark runner alias
run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

echo "Starting comprehensive benchmark experiments..."

# CASTLE Dataset Experiments
echo "=== Running CASTLE Dataset Experiments ==="
echo "Setting up CASTLE dataset..."
$run_benchmark entrypoints/run_setup_castle_dataset.py

# Define CASTLE experiment plans
castle_plans=(
    "small_models_binary"
    "small_models_multiclass"
    "large_models_binary"
    "large_models_multiclass"
)

echo "Running CASTLE experiments..."
for plan in "${castle_plans[@]}"; do
    echo "Executing CASTLE plan: $plan"
    $run_benchmark entrypoints/run_castle_experiments.py --plan "$plan"
done

# CVEFixes Dataset Experiments
echo "=== Running CVEFixes Dataset Experiments ==="
echo "Preparing CVEFixes datasets..."
$run_benchmark entrypoints/prepare_cvefixes_datasets.py \
  --database-path datasets_processed/cvefixes/CVEfixes.db \
  --languages C Java Python

# Define CVEFixes experiment plans
cvefixes_plans=(
    "small_models_binary"
    "small_models_multiclass"
    "large_models_binary"
    "large_models_multiclass"
)

echo "Running CVEFixes experiments..."
for plan in "${cvefixes_plans[@]}"; do
    echo "Executing CVEFixes plan: $plan"
    $run_benchmark entrypoints/run_cvefixes_benchmark.py --plan "$plan"
done

echo "All benchmark experiments completed successfully!"
