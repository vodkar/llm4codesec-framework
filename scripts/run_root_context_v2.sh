#!/usr/bin/env bash
# Run root_context_<VARIANT>_smoke and, if it passes, root_context_<VARIANT>_sweep
# (e.g. VARIANT=v2_lfm, v2_gemma, v3_lfm).
# With AFTER_LOG, first wait until that earlier run's log ends with EXIT= (the GPU
# is briefly free between a running plan's experiments, so free memory alone is not enough).
# Usage: scripts/run_root_context_v2.sh <variant> [AFTER_LOG]
cd /var/opt/llm4codesec-framework
VARIANT=${1:?variant, e.g. v3_lfm}
AFTER_LOG=$2
if [ -n "$AFTER_LOG" ]; then
  echo "[$(date -u +%FT%TZ)] waiting for $AFTER_LOG to finish"
  until grep -aq '^EXIT=' "$AFTER_LOG" 2>/dev/null; do sleep 60; done
fi
echo "[$(date -u +%FT%TZ)] waiting for a free GPU"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done
run() {
  echo "[$(date -u +%FT%TZ)] START $1"
  docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler "$1" \
    --config-dir configs/shared \
    --experiments-config configs/root_context_v2/experiments.json \
    --datasets-config configs/root_context_v2/datasets.json
  local rc=$?; echo "[$(date -u +%FT%TZ)] END $1 rc=$rc"; return $rc
}
run "root_context_${VARIANT}_smoke" && run "root_context_${VARIANT}_sweep"
