#!/bin/bash

# Exit on any error
set -e

# Define the benchmark runner alias
run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

$run_benchmark cli.py run-plan context_assembler vllm_comparison --config src/configs/context_assembler_experiments.json