#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml}"
DEVICES="${DEVICES:-cuda:0}"
DEBUGMODE="${DEBUGMODE:-true}"

exec scripts/singularity_exec.sh python -m app.main \
  --fname "${CONFIG}" \
  --devices ${DEVICES} \
  --debugmode "${DEBUGMODE}"
