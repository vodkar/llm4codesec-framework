#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

python src/entrypoints/loaders/run_setup_context_assembler_compare_rankings.py "$@"