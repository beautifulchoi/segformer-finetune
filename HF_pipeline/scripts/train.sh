#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CSV_PATH="${CSV_PATH:-$REPO_ROOT/data/coco_binary/manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/HF_pipeline/work_dirs/segformer_b0}"
MODEL_ID="${MODEL_ID:-nvidia/segformer-b0-finetuned-ade-512-512}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m HF_pipeline.train \
  --csv "$CSV_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-id "$MODEL_ID" \
  --device "$DEVICE" \
  "$@"
