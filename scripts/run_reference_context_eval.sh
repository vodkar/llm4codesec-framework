#!/bin/bash
# Single command: pinned-config reference-context evaluation (S1 -> S5).
# Usage: scripts/run_reference_context_eval.sh [config/reference_context_eval.yaml] [--smoke]
#    or: scripts/run_reference_context_eval.sh --smoke   (uses the default config)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG="${1:-config/reference_context_eval.yaml}"
PLAN_OVERRIDE="${2:-}"
if [[ "$CONFIG" == "--smoke" ]]; then
  CONFIG="config/reference_context_eval.yaml"
  PLAN_OVERRIDE="--smoke"
fi

cfg() { uv run python -c "import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); print(eval('d'+sys.argv[2]))" "$CONFIG" "$1"; }

SCANNER_PATH="$(cfg "['pins']['llm_scanner_path']")"
SCANNER_SHA="$(cfg "['pins']['llm_scanner_git_sha']")"
DATASET="$(cfg "['llm_scanner']['dataset_path']")"
DATASET_SHA="$(cfg "['llm_scanner']['dataset_sha256']")"
SCANNER_OUT="$(cfg "['llm_scanner']['output_dir']")"
PLAN="$(cfg "['framework']['plan']")"
DATASETS_DIR="$(cfg "['framework']['datasets_dir']")"
RESULTS_DIR="$(cfg "['framework']['results_dir']")"
# Container-side config paths (src/ is /app in the image, so configs/... on the
# container is src/configs/... on the host).
EXPERIMENTS_CONFIG="$(cfg "['framework']['experiments_config']")"
DATASETS_CONFIG="$(cfg "['framework']['datasets_config']")"
[[ "$PLAN_OVERRIDE" == "--smoke" ]] && PLAN="reference_context_smoke"

for value in "$SCANNER_SHA" "$DATASET_SHA"; do
  [[ "$value" == \<* ]] && { echo "Unfilled pin in $CONFIG" >&2; exit 2; }
done

# 0. The framework configs named by the YAML must exist, and every dataset in
#    datasets.json must live under framework.datasets_dir (where step 3 stages
#    the condition datasets), or the sweep would run on other files.
for container_path in "$EXPERIMENTS_CONFIG" "$DATASETS_CONFIG"; do
  [[ -f "src/$container_path" ]] || { echo "Config src/$container_path (framework.*_config) not found" >&2; exit 2; }
done
uv run python - "src/$DATASETS_CONFIG" "$DATASETS_DIR" <<'PY' || exit 2
import json
import posixpath
import sys

datasets_config, datasets_dir = sys.argv[1:3]
prefix = posixpath.normpath(datasets_dir) + "/"
with open(datasets_config, encoding="utf-8") as handle:
    datasets = json.load(handle)["datasets"]
outside = [
    f"{key}: {entry['dataset_path']}"
    for key, entry in datasets.items()
    if not posixpath.normpath(entry["dataset_path"]).startswith(prefix)
]
if outside:
    print(
        f"{datasets_config}: dataset_path(s) not under framework.datasets_dir "
        f"({datasets_dir}): {'; '.join(outside)}",
        file=sys.stderr,
    )
    sys.exit(2)
PY

# 1. Verify pins
[[ "$(git -C "$SCANNER_PATH" rev-parse HEAD)" == "$SCANNER_SHA" ]] || { echo "llm_scanner HEAD != pinned SHA" >&2; exit 2; }
[[ -z "$(git -C "$SCANNER_PATH" status --porcelain -- llm_scanner)" ]] || { echo "llm_scanner package has uncommitted changes" >&2; exit 2; }
[[ -z "$(git status --porcelain -- src config scripts)" ]] || { echo "framework has uncommitted changes" >&2; exit 2; }
echo "$DATASET_SHA  $DATASET" | sha256sum --check --status || { echo "dataset sha256 mismatch" >&2; exit 2; }

# 2. llm_scanner stages S1-S3 (idempotent, cached)
CONFIG_ABS="$(realpath "$CONFIG")"
[[ "$SCANNER_OUT" == /* ]] || SCANNER_OUT="$SCANNER_PATH/$SCANNER_OUT"
for stage in precompute-entity-sets build-condition-datasets; do
  (cd "$SCANNER_PATH" && uv run llm-scanner "$stage" --config "$CONFIG_ABS")
done

# 2b. The S2 run manifest must come from the pinned, clean llm_scanner and this
#     exact config file; otherwise the datasets are not the pinned ones.
uv run python - "$SCANNER_OUT/run_manifest.json" "$SCANNER_SHA" "$CONFIG_ABS" <<'PY' || exit 2
import hashlib
import json
import sys

manifest_path, pinned_sha, config_path = sys.argv[1:4]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
with open(config_path, "rb") as handle:
    config_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
problems = []
if manifest.get("llm_scanner_dirty") is not False:
    problems.append(f"llm_scanner_dirty={manifest.get('llm_scanner_dirty')!r} (must be false)")
if manifest.get("llm_scanner_git_sha") != pinned_sha:
    problems.append(f"llm_scanner_git_sha={manifest.get('llm_scanner_git_sha')!r} != pin {pinned_sha!r}")
if manifest.get("config_sha256") != config_sha256:
    problems.append(f"config_sha256={manifest.get('config_sha256')!r} != sha256({config_path})={config_sha256!r}")
if problems:
    print(f"{manifest_path} does not match this run: {'; '.join(problems)}", file=sys.stderr)
    sys.exit(2)
PY

(cd "$SCANNER_PATH" && uv run llm-scanner retrieval-overlap-report --config "$CONFIG_ABS")

# 3. Stage datasets for the container
mkdir -p "$DATASETS_DIR"
cp "$SCANNER_OUT"/datasets/cleanvul_cond_*.json "$DATASETS_DIR"/

# 4. Rebuild image (configs are baked in) and run the sweep
./build_docker.sh --no-gpu-test
# NOTE: "reference_context" is not a registered run-plan benchmark (src/cli.py
# BENCHMARKS is a fixed dict: castle/cvefixes/jitvul/vulbench/primevul/
# context_assembler/cleanvul). run-plan only uses the benchmark argument for
# config-file discovery and the default output dir, both overridden below by
# --experiments-config/--datasets-config and experiments.json's
# output_settings.base_output_dir, so any registered name works; we reuse
# "context_assembler" because it is the generic-JSON-dataset benchmark and
# matches the task_type used in configs/reference_context/datasets.json.
docker-compose run --rm llm4codesec-benchmark python cli.py run-plan context_assembler "$PLAN" \
  --config-dir configs/shared \
  --experiments-config "$EXPERIMENTS_CONFIG" \
  --datasets-config "$DATASETS_CONFIG"

# 5. Analysis on the host
PYTHONPATH=src uv run python src/cli.py analyze-reference-context \
  --config "$CONFIG" --output-dir "$RESULTS_DIR/$PLAN/analysis" \
  --plan "$PLAN"
