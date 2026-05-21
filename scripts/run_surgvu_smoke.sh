#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_surgvu.py \
  --sample-zip data/SURGVU25_cat_2_sample_set_public.zip \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu_smoke \
  --limit 4 \
  --val-fraction 0.25

python scripts/check_surgvu_manifest.py data/surgvu_smoke/manifest_train.csv --limit 2

python -m app.main \
  --fname configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml \
  --devices cuda:0 \
  --debugmode true
