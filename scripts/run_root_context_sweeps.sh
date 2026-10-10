#!/usr/bin/env bash
# Run root_context_v2 plans one after another without smoke runs; stops at the first failure.
# Usage: scripts/run_root_context_sweeps.sh <plan> [<plan> ...]
cd /var/opt/llm4codesec-framework
for plan in "$@"; do
  echo "[$(date -u +%FT%TZ)] START $plan"
  docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler "$plan" \
    --config-dir configs/shared \
    --experiments-config configs/root_context_v2/experiments.json \
    --datasets-config configs/root_context_v2/datasets.json
  rc=$?
  echo "[$(date -u +%FT%TZ)] END $plan rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
