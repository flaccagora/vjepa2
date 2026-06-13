# PhysicalAI Open-H Full Fine-Tuning Notes

## Scope

This adds a full fine-tuning path for V-JEPA 2.1 on
`nvidia/PhysicalAI-Robotics-Open-H-Embodiment` using the existing V-JEPA 2.1
JEPA objective. The encoder and predictor remain trainable; this is not a
LoRA/adapters path.

The Hugging Face dataset card describes Open-H-Embodiment as LeRobot v2.1 data
with MP4 videos, Parquet kinematics, JSON/JSONL metadata, approximately 750
hours, approximately 120k trajectories, and approximately 4.5 TB total size.

## Files Changed

- `src/datasets/physicalai_openh.py`: Hugging Face Hub backed LeRobot v2.1 dataset.
- `src/datasets/data_manager.py`: dataset registry hook for `PhysicalAIOpenH` and
  `PhysicalAI-Robotics-Open-H-Embodiment`.
- `app/vjepa_2_1/train.py`: optional validation loop, shared JEPA loss helper,
  gradient clipping, and step checkpointing.
- `configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml`:
  full fine-tuning config.
- `scripts/prepare_physicalai_openh_cache.py`: login-node cache preparation for
  selected metadata and episodes.
- `scripts/verify_physicalai_openh_cache.py`: offline cache completeness check.
- `scripts/evaluate_physicalai_openh.py`: representation extraction and optional
  linear action readout.
- `tests/datasets/test_physicalai_openh.py`: synthetic metadata tests.
- `requirements.txt`: adds `pyarrow`.

## Assumptions

- Supported leaves are LeRobot v2.1 directories containing `data`, `videos`,
  and `meta` with `info.json`, `modality.json`, `tasks.jsonl`, and
  `episodes.jsonl`.
- Legacy layouts such as per-episode JSON metadata are intentionally rejected
  with a clear error.
- Video, action, state, and proprioception fields are derived from
  `meta/info.json` and `meta/modality.json`; field names are not hardcoded.
- JEPA fine-tuning uses video clips as the primary objective. Action/state fields
  are loaded and exposed when available, and the eval script can use actions for
  a linear readout.
- Validation is split by episode. Existing LeRobot split ranges are used when
  present; otherwise a deterministic episode-level fallback split is used only
  when a leaf has no split metadata at all. Leaves that declare `train` but not
  `val`/`test` are skipped for missing splits to avoid train/validation leakage.
- For embodiment generalization, set `dataset_kwargs.split_strategy:
  embodiment` and provide `split_embodiments`, for example `train: [dvrk,
  versius]`, `test: [panda]`. This holds out entire `robot_type` groups rather
  than only episodes.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download or place the V-JEPA 2.1 ViT-B checkpoint at:

```bash
checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

## Offline Dataset Workflow

Compute-node training defaults to `data.local_files_only: true` and
`data.cache_dir: ${PHYSICALAI_OPENH_CACHE}`. It will fail if required files are
missing rather than attempting network downloads.

Expected cache layout:

```text
$PHYSICALAI_OPENH_CACHE/
  Surgical/hamlyn/knot_tying/
    meta/info.json
    meta/modality.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.color/episode_000000.mp4
  physicalai_openh_cache_manifest.json
```

### Login-Node Preparation

Run this on a login node with internet access. For a small pilot cache:

```bash
export PHYSICALAI_OPENH_CACHE=/shared/datasets/physicalai-openh

python scripts/prepare_physicalai_openh_cache.py \
  --config configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml \
  --cache-dir "$PHYSICALAI_OPENH_CACHE" \
  --roots Surgical/hamlyn/knot_tying \
  --split train \
  --max-episodes 8 \
  --video-key observation.images.color
```

Prepare validation episodes separately:

```bash
python scripts/prepare_physicalai_openh_cache.py \
  --config configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml \
  --cache-dir "$PHYSICALAI_OPENH_CACHE" \
  --roots Surgical/hamlyn/knot_tying \
  --split val \
  --max-episodes 4 \
  --video-key observation.images.color
```

For metadata-only planning over a broader area:

```bash
python scripts/prepare_physicalai_openh_cache.py \
  --config configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml \
  --cache-dir "$PHYSICALAI_OPENH_CACHE" \
  --roots Surgical \
  --split train \
  --max-episodes 0 \
  --metadata-only
```

To expand later, rerun the preparation command with larger `--max-episodes`,
additional `--roots`, or filters such as:

```bash
--task-filter "Tie a surgical knot in the suture."
--embodiment-filter dvrk
```

### Cache Verification

Before submitting compute jobs, verify the exact planned split without internet:

```bash
export HF_HUB_OFFLINE=1
export PHYSICALAI_OPENH_CACHE=/shared/datasets/physicalai-openh

python scripts/verify_physicalai_openh_cache.py \
  --config configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml \
  --cache-dir "$PHYSICALAI_OPENH_CACHE" \
  --roots Surgical/hamlyn/knot_tying \
  --split train \
  --max-episodes 8 \
  --video-key observation.images.color \
  --check-samples 2
```

The verifier instantiates `PhysicalAIOpenHDataset` with `local_files_only=True`,
checks every required metadata/Parquet/MP4 path, and optionally decodes samples.

### Compute-Node Training

Run or submit training with the same cache variable. Compute nodes do not need
internet access:

```bash
export HF_HUB_OFFLINE=1
export PHYSICALAI_OPENH_CACHE=/shared/datasets/physicalai-openh

python scripts/run_training_config.py \
  --fname configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml
```

For a smoke run, copy the config and set:

```yaml
data:
  datasets:
  - Surgical/hamlyn/knot_tying
  max_episodes: 2
  cache_dir: ${PHYSICALAI_OPENH_CACHE}
  local_files_only: true
  dataset_kwargs:
    discover_roots: false
    video_key: observation.images.color
validation:
  ipe: 1
  dataset_kwargs:
    max_episodes: 1
optimization:
  ipe: 1
  epochs: 1
wandb:
  enable: false
```

Extract representations:

```bash
python scripts/evaluate_physicalai_openh.py \
  --config configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml \
  --checkpoint output/physicalai_openh/vjepa2_1_vitb_384px_16f_full_ft/latest.pth.tar \
  --split val \
  --max-samples 16 \
  --output output/physicalai_openh/physicalai_val_reps.npz
```

Run an action readout if sampled episodes expose action tensors:

```bash
python scripts/evaluate_physicalai_openh.py \
  --config configs/train_2_1/vitb16/physicalai-openh-full-finetune-384px-16f.yaml \
  --checkpoint output/physicalai_openh/vjepa2_1_vitb_384px_16f_full_ft/latest.pth.tar \
  --split val \
  --max-samples 32 \
  --action-readout
```

## Expected GPU Requirements

- ViT-B/16 at 384 px and 16 frames with full fine-tuning should be treated as a
  multi-GPU or high-memory single-GPU job. Start with batch size 1-2 per GPU.
- The provided config requests 48 GB per GPU and enables activation
  checkpointing, bfloat16, conservative learning rate, 2 warmup epochs, weight
  decay, gradient clipping, periodic validation, gradient accumulation support,
  and step/epoch checkpoints.
- Larger V-JEPA 2.1 variants or larger batches will need substantially more
  memory.

## Known Risks

- The dataset is heterogeneous. Some leaves may have legacy layouts or missing
  action/state fields; those are rejected or omitted explicitly rather than
  silently fabricated.
- Recursive metadata discovery over the full HF repo can take time, although it
  does not download media.
- Compute-node training no longer downloads missing files. Cache preparation
  must be rerun whenever roots, splits, filters, `max_episodes`, or `video_key`
  change. The full corpus is approximately 4.5 TB.
- Validation uses the same video transform stack unless the config narrows eval
  resize ranges, so validation loss is still a representation-learning proxy.
- Online action readout is intentionally not part of the main JEPA training loop;
  it remains a separate evaluation because action dimensionality varies across
  embodiments.
