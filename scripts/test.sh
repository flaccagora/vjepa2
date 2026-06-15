#!/usr/bin/env bash
set -euo pipefail

#python scripts/prepare_surgvu.py \
#  --sample-zip  data/SURGVU25_cat_2_sample_set_public.zip \
#  --labels-zip data/surgvu24_labels_updated_v2.zip \
#  --out-dir data/surgvu \
#  --limit 100 \
#  --val-fraction 0.25

# python scripts/check_surgvu_manifest.py data/surgvu/manifest_train.csv --limit 0

python -m app.main \
  --fname configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml \
  --devices cuda:0 \
  --debugmode true
