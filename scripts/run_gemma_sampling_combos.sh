#!/usr/bin/env bash
# Skip repeat-150 (repeat >=1.3 collapses recall and runs 1.5-5h), then run the combinations plan.
cd /var/opt/llm4codesec-framework
SWEEP_LOG=results/root_context_v2/root_context_v2_gemma_sampling_sweep.log
echo "[$(date -u +%FT%TZ)] waiting for repeat-140 to finish (Experiment 28/28 starts)"
until grep -aq 'Experiment 28/28' $SWEEP_LOG || grep -aq '^EXIT=' $SWEEP_LOG; do sleep 60; done
if ! grep -aq '^EXIT=' $SWEEP_LOG; then
  sleep 30
  echo "[$(date -u +%FT%TZ)] stopping the sweep before repeat-150"
  docker ps -q --filter name=llm4codesec-framework-llm4codesec-benchmark-run | xargs -r docker stop
  until grep -aq '^EXIT=' $SWEEP_LOG; do sleep 10; done
fi
DC="docker-compose run --rm -v ./src/configs:/app/configs:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 llm4codesec-benchmark"
echo "[$(date -u +%FT%TZ)] rebuilding sweep plan results"
$DC python cli.py rebuild-plan-results results/root_context_v2/root_context_v2_gemma_sampling_sweep
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1500 ]; do sleep 30; done
echo "[$(date -u +%FT%TZ)] START root_context_v2_gemma_sampling_combos"
$DC python cli.py run-plan context_assembler root_context_v2_gemma_sampling_combos \
  --config-dir configs/shared --experiments-config configs/root_context_v2/experiments.json \
  --datasets-config configs/root_context_v2/datasets.json
rc=$?; echo "[$(date -u +%FT%TZ)] END root_context_v2_gemma_sampling_combos rc=$rc"; exit $rc
