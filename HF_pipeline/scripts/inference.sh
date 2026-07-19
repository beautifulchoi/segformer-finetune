#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/HF_pipeline/work_dirs/segformer_b0/best_model}"
CSV_PATH="${CSV_PATH:-$REPO_ROOT/data/coco_binary/manifest.csv}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/HF_pipeline/outputs/test}"
DEVICE="${DEVICE:-cuda:0}"
INSECURE_ARGS=()
if [[ "${ALLOW_INSECURE_DOWNLOAD:-0}" == "1" ]]; then
  INSECURE_ARGS+=(--allow-insecure-download)
fi

INPUT_ARGS=()
HAS_INPUT=false
for argument in "$@"; do
  case "$argument" in
    --csv|--csv=*|--image|--image=*) HAS_INPUT=true ;;
  esac
done
if [[ "$HAS_INPUT" == false ]]; then
  INPUT_ARGS+=(--csv "$CSV_PATH" --split "$SPLIT")
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m HF_pipeline.inference \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "${INSECURE_ARGS[@]}" \
  "${INPUT_ARGS[@]}" \
  "$@"
