#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/matteo/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

CONFIG="${CONFIG:-configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm.yaml}"
ACCOUNT="${ACCOUNT:-IscrC_FLAC}"
PARTITION="${PARTITION:-boost_usr_prod}"
TIME_MIN="${TIME_MIN:-30}"
CACHE_DIR="${CACHE_DIR:-checkpoints}"
RESOLVED_CONFIG="${RESOLVED_CONFIG:-output/surgvu_slurm/resolved_config.yaml}"

python scripts/cache_vjepa_models.py \
  --config "${CONFIG}" \
  --cache-dir "${CACHE_DIR}" \
  --write-resolved-config "${RESOLVED_CONFIG}"

python -m app.main_distributed \
  --fname "${RESOLVED_CONFIG}" \
  --account "${ACCOUNT}" \
  --partition "${PARTITION}" \
  --time "${TIME_MIN}"
