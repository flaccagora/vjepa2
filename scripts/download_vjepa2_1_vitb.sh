#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-checkpoints}"
python scripts/cache_vjepa_models.py --cache-dir "$out_dir" --model vjepa2_1_vitb
