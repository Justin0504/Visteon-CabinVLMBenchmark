#!/usr/bin/env bash
# End-to-end caption pipeline: Stage A (vision) -> Stage B (causal caption + QA + cross-check).
# Usage:  VULTR_KEYS=keys.env ./run_pipeline.sh INPUT_RAW.jsonl OUTDIR
# INPUT_RAW.jsonl rows: {image, camera, gt, vehicles, vrus}
set -euo pipefail

INP="${1:?usage: run_pipeline.sh INPUT_RAW.jsonl OUTDIR}"
OUT="${2:?usage: run_pipeline.sh INPUT_RAW.jsonl OUTDIR}"
mkdir -p "$OUT"
: "${VULTR_KEYS:?set VULTR_KEYS to your key file or comma-separated keys}"

# run as a module from the parent dir so package imports resolve
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

echo "[1/2] Stage A: vision descriptions"
python -m caption_pipeline.stage_a_vision --inp "$INP" --out "$OUT/vision.jsonl"

echo "[2/2] Stage B: causal caption + QA + cross-check"
python -m caption_pipeline.stage_b_fusion --inp "$OUT/vision.jsonl" --raw "$OUT/fusion_raw.jsonl" --out "$OUT/fusion_sharegpt.json"

echo "DONE -> $OUT/fusion_sharegpt.json"
