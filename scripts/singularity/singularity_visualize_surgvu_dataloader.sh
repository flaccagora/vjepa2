#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml}"
NUM_BATCHES="${NUM_BATCHES:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
FRAMES_PER_ROW="${FRAMES_PER_ROW:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-239}"
OUT_DIR="${OUT_DIR:-output/surgvu_dataloader_preview}"

exec scripts/singularity_exec.sh python scripts/visualize_surgvu_dataloader.py \
  --config "${CONFIG}" \
  --num-batches "${NUM_BATCHES}" \
  --max-samples "${MAX_SAMPLES}" \
  --frames-per-row "${FRAMES_PER_ROW}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  --disable-augment
