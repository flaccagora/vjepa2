# SurgVU V-JEPA2.1 Fine-tuning Setup

This repository is now set up to fine-tune V-JEPA2.1 on SurgVU surgical videos using the existing
`app.vjepa_2_1` self-supervised training loop and `VideoDataset` loader.

## Dataset Notes

Source documentation: `SurgVU_dataset.pdf`.

Relevant details from the dataset document:

- SurgVU contains 280 surgical video clips from 155 robotic-surgery training sessions.
- Total scale is over 840 hours of 60 FPS endoscopic video, about 18 million labeled images at 720p.
- Public data URLs listed in the PDF:
  - Videos: `https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip`
  - Labels: `https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip`
  - Public tool-detection validation set: `https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip`
  - Public VQA sample set: `https://storage.googleapis.com/isi-surgvu/SURGVU25_cat_2_sample_set_public.zip`
- The V-JEPA2.1 fine-tuning path uses videos only. SurgVU task labels are parsed only to create stable numeric
  manifest labels because the repo's `VideoDataset` expects a second manifest column.

## Environment

The README setup path was used:

```bash
conda create -y -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install -e .
```

Validated environment:

- Python: 3.12
- PyTorch: `2.12.0+cu130`
- GPU: NVIDIA RTX A4000
- `decord` import and video decoding work.

## Added Files

- `scripts/prepare_surgvu.py`
  - Extracts optional video/sample/label zip files.
  - Recursively discovers videos.
  - Produces V-JEPA-compatible manifests:
    - `manifest_all.csv`
    - `manifest_train.csv`
    - `manifest_val.csv`
  - Writes `metadata.csv` and `summary.json`.
- `scripts/check_surgvu_manifest.py`
  - Opens a few manifest videos with `decord` and reports frames, FPS, and duration.
- `scripts/cache_vjepa_models.py`
  - Caches public V-JEPA checkpoints on the submit/login node and can write a resolved config with absolute paths.
- `scripts/download_vjepa2_1_vitb.sh`
  - Thin wrapper that caches the smallest public V-JEPA2.1 checkpoint:
    `checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`.
- `scripts/run_surgvu_smoke.sh`
  - Prepares the public sample split, validates decoding, and runs the smoke training config.
- `configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml`
  - One-iteration, one-GPU smoke run on 4 public sample clips.
- `configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml`
  - Main SurgVU ViT-B/16 fine-tuning config using the public V-JEPA2.1 checkpoint.
- `configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm.yaml`
  - 1-node, 4-GPU SLURM config using the repo's normal distributed config fields.
- `scripts/submit_surgvu_finetune_distributed.sh`
  - Convenience wrapper around `python -m app.main_distributed`.

## Code Fixes Needed For ViT-B Fine-tuning

Two repo fixes were required:

- `app/vjepa_2_1/train.py`
  - Fixed encoder embedding dimension lookup for `vit_base`, the smallest V-JEPA2.1 model.
- `app/vjepa_2_1/utils.py`
  - Made checkpoint loading compatible with public pretrained checkpoints and resume checkpoints.
  - Public checkpoints can now initialize encoder, predictor, and target encoder while starting a fresh optimizer schedule.
- `app/vjepa_2_1/models/utils/masks_dist.py`
  - Fixed batch-size-1 distance weighting by preserving the batch dimension with `.squeeze(1)`.

## Data Preparation

For the local public sample smoke split:

```bash
conda activate vjepa2-312

python scripts/prepare_surgvu.py \
  --sample-zip data/SURGVU25_cat_2_sample_set_public.zip \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu_smoke \
  --limit 4 \
  --val-fraction 0.25 \
  --force
```

Smoke split produced:

- `data/surgvu_smoke/manifest_train.csv`: 3 clips
- `data/surgvu_smoke/manifest_val.csv`: 1 clip

For the full SurgVU videos, first place or extract `surgvu24_videos_only.zip`, then run:

```bash
conda activate vjepa2-312

python scripts/prepare_surgvu.py \
  --video-zip data/surgvu24_videos_only.zip \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu \
  --val-fraction 0.1
```

If the full videos are already extracted:

```bash
python scripts/prepare_surgvu.py \
  --videos-root /path/to/extracted/surgvu/videos \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu \
  --val-fraction 0.1
```

Validate a manifest:

```bash
python scripts/check_surgvu_manifest.py data/surgvu/manifest_train.csv --limit 4
```

## Checkpoint

The smallest V-JEPA2.1 public checkpoint is ViT-B/16:

```bash
conda activate vjepa2-312
python scripts/cache_vjepa_models.py --cache-dir checkpoints --model vjepa2_1_vitb
```

Downloaded file:

```text
checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

The cache script also supports the other public V-JEPA model filenames used by this repo:

```bash
python scripts/cache_vjepa_models.py --cache-dir checkpoints --model all
```

For training configs, prefer config-aware caching:

```bash
python scripts/cache_vjepa_models.py \
  --config configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm.yaml \
  --cache-dir checkpoints \
  --write-resolved-config output/surgvu_slurm/resolved_config.yaml
```

This runs before SLURM submission on the internet-enabled submit/login node. It downloads the checkpoint if missing and
writes a resolved config where `folder`, `data.datasets`, and `meta.read_checkpoint` are absolute filesystem paths.

## Training

Smoke test:

```bash
conda activate vjepa2-312
scripts/run_surgvu_smoke.sh
```

Equivalent training command:

```bash
python -m app.main \
  --fname configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml \
  --devices cuda:0 \
  --debugmode true
```

## W&B Logging

W&B logging is optional and rank-0 only. The smoke config uses:

```yaml
wandb:
  enable: false
  mode: offline
  strict: false
```

Keep `enable: false` for local smoke tests and no-internet compute nodes. If W&B is enabled, `strict: false` means
authentication, permission, or service errors are logged as warnings and training continues. Set `wandb.strict: true`
only when you want W&B failures to abort the run.

Full fine-tuning:

```bash
conda activate vjepa2-312

python -m app.main \
  --fname configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml \
  --devices cuda:0 \
  --debugmode true
```

For multiple local GPUs, remove `--debugmode true` and pass all devices:

```bash
python -m app.main \
  --fname configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml \
  --devices cuda:0 cuda:1
```

## SLURM Distributed Training

Distributed training uses the repo's existing submitit entry point, `app.main_distributed`.
Compute nodes do not need internet access. The wrapper first caches the checkpoint and writes a resolved config on the
submit/login node, then submits that resolved config through `app.main_distributed`.

Submit with:

```bash
scripts/submit_surgvu_finetune_distributed.sh
```

Equivalent explicit command:

```bash
python -m app.main_distributed \
  --fname output/surgvu_slurm/resolved_config.yaml \
  --account IscrC_FLAC \
  --partition boost_usr_prod \
  --time 30
```

The config requests:

- Account: `IscrC_FLAC`
- Partition: `boost_usr_prod`
- Nodes: 1
- Tasks per node: 4
- GPUs: 4
- CPUs per task: 8
- Memory: `120G` per GPU, matching about 480 GB on a 4-GPU node
- Time: 30 minutes
- Exclusive node allocation: true

These are represented through the existing config fields:

```yaml
nodes: 1
tasks_per_node: 4
cpus_per_task: 8
mem_per_gpu: 120G
exclusive: true
```

You can override the config at submission time:

```bash
CONFIG=configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml \
  scripts/submit_surgvu_finetune_distributed.sh
```

Training output goes to:

```text
output/surgvu_slurm/vjepa2_1_vitb_384px_16f
```

The distributed submitter snapshots code into the run folder. `app/main_distributed.py` now excludes local `data/`,
`checkpoints/`, and `output/` directories from that code snapshot; the resolved config uses absolute paths instead.

## Tested Smoke Run

The smoke split was prepared from `data/SURGVU25_cat_2_sample_set_public.zip`.

Manifest validation opened the first two train clips with `decord`:

- `case126.mp4`: 1800 frames, 60 FPS, 30 seconds
- `case127.mp4`: 1800 frames, 60 FPS, 30 seconds

Checkpoint-backed smoke training completed successfully:

- Config: `configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml`
- Checkpoint loaded: `checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`
- Model: V-JEPA2.1 `vit_base` / ViT-B/16
- Batch size: 1
- Frames per clip: 16
- Crop: 256
- Iterations: 1
- Final smoke loss: `0.45729`
- Peak logged GPU memory: about `3.59 GB`

Smoke log:

```text
output/surgvu_smoke/vjepa2_1_vitb_256px_16f/log_r0.csv
```

The large one-step smoke checkpoint was removed after validation because it is not useful for real training.

## Practical Defaults

- Start with the ViT-B/16 config before scaling to larger V-JEPA2.1 models.
- Keep `batch_size: 1` for 16 GB GPUs at 384px, then increase after checking memory.
- Use `fps: 4` and `dataset_fpcs: [16]` for the initial SurgVU run; this gives 4-second clips sampled from 60 FPS videos.
- The setup is self-supervised and ignores noisy SurgVU labels during representation learning.
- For longer temporal adaptation after the first run, make a second config with `dataset_fpcs: [64]` and a lower batch size.
