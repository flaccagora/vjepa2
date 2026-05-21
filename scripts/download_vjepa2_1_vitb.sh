#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-checkpoints}"
mkdir -p "$out_dir"

url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
out_file="$out_dir/vjepa2_1_vitb_dist_vitG_384.pt"

if [[ -s "$out_file" ]]; then
  echo "Checkpoint already exists: $out_file"
  exit 0
fi

curl -L "$url" -o "$out_file"
echo "Downloaded: $out_file"
