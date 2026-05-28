# SurgVU VJEPA-2.1 YAML Config Guide

This file explains the YAML entries used by the SurgVU VJEPA-2.1 training configs, especially:

- `configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml`
- `configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml`
- `configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm.yaml`

It also includes values suggested for the final SurgVU training recipe. The recommendations are based on the current repository implementation and on hints from `VJEPA-2.1.pdf`.

## Paper Hints Used

The VJEPA-2.1 paper uses a two-phase recipe:

- Primary phase: 16 video frames, 4 FPS, 256x256 crops, warmup-constant learning rate schedule.
- Cooldown phase: 64 video frames, 4 FPS, higher resolution, learning rate decay to `1e-6`.
- Video masking uses spatial scales `[0.15, 0.7]`, temporal scale `[1.0, 1.0]`, aspect ratio `[0.75, 1.5]`, patch size 16, and tubelet size 2.
- The video context loss weight is `lambda_value_vid: 0.5`.
- Dense/context prediction is useful, but a fixed dense-loss weight can hurt some classification settings; the VJEPA-2.1 recipe uses distance-weighted context loss and progressive weighting.
- Public distilled ViT-B VJEPA-2.1 checkpoints use the smaller ViT-B-compatible predictor shape in this repo: `pred_depth: 12`, `pred_embed_dim: 384`, `pred_num_heads: 12`.

For SurgVU, we are not training VJEPA-2.1 from scratch on web-scale video. We are doing domain adaptation from a public pretrained checkpoint onto surgical videos. That means the final recipe should keep the paper's masking, temporal sampling, EMA, and cooldown ideas, but use a much smaller learning rate than the paper pretraining LR.

## Recommended Final Recipe

Use two stages if budget allows.

### Stage 1: stable surgical domain adaptation

Start from the public VJEPA-2.1 ViT-B checkpoint:

- `read_checkpoint: checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`
- `crop_size: 384`
- `dataset_fpcs: [16]`
- `fps: 4`
- `batch_size: 1` per GPU on the 4 GPU SLURM node
- `epochs: 20`
- `lr: 1.0e-5`
- `start_lr: 1.0e-6`
- `final_lr: 1.0e-6`
- `warmup: 2`

This is the current recommended first production run:

- `configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm.yaml`

### Stage 2: optional cooldown for stronger temporal features

After Stage 1 finishes, run a cooldown from the Stage 1 `latest.pth.tar` checkpoint:

- `crop_size: 384`
- `dataset_fpcs: [64]`
- `fps: 4`
- `batch_size: 1`
- `epochs: 5` to `10`
- `lr: 5.0e-6` to `1.0e-5`
- `start_lr: 1.0e-6`
- `final_lr: 1.0e-6`
- `warmup: 0` or `1`
- `read_checkpoint: output/surgvu_slurm/vjepa2_1_vitb_384px_16f/latest.pth.tar`

The paper's cooldown uses longer clips and higher resolution to improve downstream transfer. For surgical videos, this should help represent tool motion and workflow context, but it is more memory intensive. If 64 frames is unstable, use 32 frames before falling back to 16.

## Top-Level Entries

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `app` | Selects the training app loaded by `app/scaffold.py`. | `vjepa_2_1` |
| `nodes` | Number of SLURM nodes requested by `main_distributed`. | `1` for current cluster recipe |
| `tasks_per_node` | Number of distributed processes per node. Usually one process per GPU. | `4` for `--gres=gpu:4` |
| `cpus_per_task` | CPU cores assigned to each distributed process. | `8` |
| `mem_per_gpu` | Memory hint passed into the generated SLURM job. | `120G` for 480 GB total node memory |
| `exclusive` | Custom flag added for this setup. When `true`, the generated SLURM job requests the node exclusively. | `true` |
| `folder` | Output directory for checkpoints, logs, resolved configs, and W&B files if `wandb.dir` is unset. | `output/surgvu_slurm/vjepa2_1_vitb_384px_16f` for Stage 1 |

## `data`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `dataset_type` | Dataset implementation name. The current video manifests are compatible with `VideoDataset`. | `VideoDataset` |
| `datasets` | List of manifest CSV files. Each row should point to a video and optional metadata in the expected VJEPA format. | `data/surgvu/manifest_train.csv` |
| `datasets_weights` | Sampling weights for each dataset in `datasets`. Must align with `datasets`. | `[1.0]` for SurgVU-only |
| `batch_size` | Per-rank/per-GPU batch size. Global batch is `batch_size * world_size`. | `1`; try `2` only if memory allows |
| `crop_size` | Spatial crop size passed to the video transform and model. | `384` final; use `256` for smoke/debug |
| `patch_size` | ViT patch size. Must match the checkpoint family. | `16` |
| `dataset_fpcs` | Frames per clip sampled from each dataset. One value per dataset. | Stage 1 `[16]`; Stage 2 `[64]` |
| `tubelet_size` | Number of frames grouped into each temporal patch token. | `2` |
| `fps` | Temporal sampling rate used by the video loader. | `4`, matching the paper |
| `filter_long_videos` | Maximum source video file size in bytes. The upstream default is `1e9`, which can skip full SurgVU clips larger than about 1 GB. `null` is handled by the SurgVU training path as "effectively disabled". | `null` for full SurgVU |
| `filter_short_videos` | If `true`, skips videos that cannot provide enough frames for the requested clip length. | `false` |
| `preprocess_metadata` | Path, or list of paths, to `preprocess_metadata.json` files generated by `scripts/preprocess_surgvu_videos.py --metadata-only` or full preprocessing. The dataloader uses this to apply deterministic crop geometry and treat black interruptions as boundaries between separate valid video segments. | `data/surgvu_metadata_train/preprocess_metadata.json` when using metadata-only training |
| `use_preprocess_metadata` | Enables metadata-aware sampling/cropping. If omitted or `null`, the SurgVU training code enables it automatically when `preprocess_metadata` is set. | `null` for auto; `true` with metadata-only training |
| `num_workers` | DataLoader workers per process. | `8` on the SLURM node |
| `pin_mem` | Enables pinned host memory for GPU transfer. | `true` for GPU training |
| `persistent_workers` | Keeps DataLoader workers alive between epochs. Present in stock configs, but `app/vjepa_2_1/train.py` currently does not read/pass this key. | Omit unless the training code is extended |

Notes:

- For SurgVU, do not mix labels into self-supervised training. Labels can be used later for evaluation or supervised probing.
- Keep `fps: 4` first because it matches the paper. If surgical phases evolve slowly, run a later ablation with `fps: 2` to increase the real-time span per clip.

## `data_aug`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `auto_augment` | Enables stronger image-style augmentation. | `false` |
| `motion_shift` | Enables temporal motion shift augmentation. | `false` initially |
| `random_resize_aspect_ratio` | Aspect-ratio range for random resized crop. | `[0.75, 1.35]` |
| `random_resize_scale` | Area scale range for random resized crop. | `[0.3, 1.0]` |
| `reprob` | Random erasing probability. | `0.0` |

For surgical video, avoid aggressive augmentation until a stable baseline exists. Tool shape, tissue texture, and camera geometry are semantically important.

## `loss`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `loss_exp` | Exponent in the prediction loss. `1.0` gives L1-style loss, matching VJEPA-2.1. | `1.0` |
| `predict_all` | Returns all encoder tokens so the predictor can use the dense/context objective. | `true` |
| `reg_coeff` | Present in SurgVU configs for compatibility, but not currently read by `app/vjepa_2_1/train.py`. | `0.0` or omit |
| `shift_by_n` | Drops/chops the last `n` tokens in the model path when configured. | `0` |
| `weight_distance_loss` | Applies distance weighting to context loss. The paper reports this stabilizes the dense prediction objective. | `true` for Stage 1 |
| `offset_context_loss` | Computes context loss against an offset target variant. Not in the current SurgVU configs. | `false` unless deliberately experimenting |
| `computing_gram_loss` | Stock cooldown field for Gram loss. Not currently read by `app/vjepa_2_1/train.py`. | Omit |
| `gram_HighRes` | Stock cooldown Gram-loss field. Not currently read by `app/vjepa_2_1/train.py`. | Omit |
| `gram_ckpt` | Stock cooldown Gram-loss checkpoint. Not currently read by `app/vjepa_2_1/train.py`. | Omit |
| `gram_loss_weight` | Stock cooldown Gram-loss weight. Not currently read by `app/vjepa_2_1/train.py`. | Omit |

## `mask`

`mask` is a list. Each entry defines one masking pattern. The current recipe uses two masks:

- 8 small spatial blocks with `spatial_scale: [0.15, 0.15]`
- 2 large spatial blocks with `spatial_scale: [0.7, 0.7]`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `aspect_ratio` | Aspect-ratio range for sampled mask blocks. | `[0.75, 1.5]` |
| `full_complement` | If `true`, predictor masks are exactly the complement of context masks. | `false` |
| `pred_full_complement` | Optional mask-collator mode where predictor mask uses the full complement. | Omit or `false` |
| `inv_block` | Optional mode that inverts selected blocks. | Omit or `false` |
| `max_keep` | Maximum number of context patches kept. `null` lets the collator decide. | `null` |
| `max_temporal_keep` | Maximum ratio of temporal tokens kept in context. | `1.0` |
| `num_blocks` | Number of mask blocks sampled for this mask entry. | `8` for small-block entry, `2` for large-block entry |
| `spatial_scale` | Fractional spatial area range for mask blocks. | `[0.15, 0.15]` and `[0.7, 0.7]` |
| `temporal_scale` | Fractional temporal span range for mask blocks. VJEPA-2.1 masks full temporal tubes. | `[1.0, 1.0]` |

Recommended final mask:

```yaml
mask:
- aspect_ratio: [0.75, 1.5]
  full_complement: false
  max_keep: null
  max_temporal_keep: 1.0
  num_blocks: 8
  spatial_scale: [0.15, 0.15]
  temporal_scale: [1.0, 1.0]
- aspect_ratio: [0.75, 1.5]
  full_complement: false
  max_keep: null
  max_temporal_keep: 1.0
  num_blocks: 2
  spatial_scale: [0.7, 0.7]
  temporal_scale: [1.0, 1.0]
```

## `img_data` and `img_mask`

These sections appear in stock VJEPA-2.1 configs for joint image/video pretraining.

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `img_data` | Optional image dataset branch. Used by the train loop if present. | Omit for SurgVU-only domain adaptation |
| `img_data.dataset_type` | Dataset type for image branch. Stock configs use `VideoDataset` with one-frame clips. | Omit |
| `img_data.datasets` | Image manifest list. | Omit |
| `img_data.datasets_weights` | Image dataset sampling weights. | Omit |
| `img_data.dataset_fpcs` | Usually `[1]` for images. | Omit |
| `img_data.batch_size` | Image branch batch size before rank redistribution. | Omit |
| `img_data.crop_size` | Image branch crop size. | Omit |
| `img_data.rank_ratio` | Fraction of distributed ranks assigned to image data. | Omit |
| `img_data.num_workers` | Image branch workers. Falls back to video `num_workers` if omitted. | Omit |
| `img_mask` | Mask list for image branch. Same fields as `mask`. | Omit |

For the first meaningful SurgVU representation, train only on SurgVU videos. Add image data later only if you intentionally build a mixed surgical-image/video corpus.

## `meta`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `dtype` | Mixed precision dtype. Supported values depend on GPU and PyTorch. | `float16` for broad cluster compatibility; use `bfloat16` if hardware supports it reliably |
| `eval_freq` | Stock metadata field. The current `app/vjepa_2_1/train.py` does not run an online eval loop from this key. | Keep `100` or omit |
| `load_checkpoint` | Enables checkpoint loading. | `true` |
| `read_checkpoint` | Checkpoint path to load. Public pretrained checkpoint for Stage 1; Stage 1 output for Stage 2. | Stage 1 `checkpoints/vjepa2_1_vitb_dist_vitG_384.pt` |
| `save_every_freq` | Save checkpoint every N epochs. | `5` |
| `seed` | Random seed for reproducibility. | `239` |
| `use_sdpa` | Enables PyTorch scaled dot-product attention where supported. | `true` |
| `skip_batches` | Optional number of initial batches to skip after resume/debug. | Omit or `-1` |
| `sync_gc` | Optional distributed garbage collection sync path. | Omit or `false` |

## `model`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `compile_model` | Calls `torch.compile()` on encoder, target encoder, and predictor. Can add startup time and make debugging harder. | `false` |
| `has_cls_first` | Whether checkpoint/model expects a class token first. | `false` |
| `img_temporal_dim_size` | Temporal size used for image modality tokenization. | `1` |
| `interpolate_rope` | Interpolates RoPE positions when resolution/frame count differs from checkpoint. | `true` |
| `is_causal` | Makes encoder attention causal. | `false` |
| `lambda_progressive` | Progressively warms in dense/context loss weighting. | `true` for Stage 1; `false` is acceptable for short cooldown |
| `lambda_value_img` | Image context-loss weight. Only relevant with `img_data`. | `0.5` for checkpoint compatibility |
| `lambda_value_vid` | Video context-loss weight. | `0.5` |
| `local_window` | Present in SurgVU config for compatibility. Not currently passed by `app/vjepa_2_1/train.py` into model construction. | `[-1, -1, -1]` or omit |
| `modality_embedding` | Adds modality embeddings, part of VJEPA-2.1 multimodal tokenizer design. | `true` |
| `model_name` | Encoder variant. | `vit_base` for the smallest VJEPA-2.1 model used here |
| `n_registers` | Number of encoder register tokens. | `0` |
| `n_registers_predictor` | Number of predictor register tokens. | `0` |
| `normalize_predictor` | Normalizes predictor output path. | `false` |
| `pred_depth` | Number of predictor transformer blocks. | `12` for public ViT-B distilled checkpoint |
| `pred_embed_dim` | Predictor embedding dimension. | `384` |
| `pred_is_causal` | Makes predictor attention causal. | `false` |
| `pred_local_window` | Present in SurgVU config for compatibility. Not currently passed by `app/vjepa_2_1/train.py`. | `[-1, -1, -1]` or omit |
| `pred_num_heads` | Predictor attention heads. | `12` |
| `uniform_power` | Uses uniform-power ViT scaling helpers from the repo. | `true` |
| `use_activation_checkpointing` | Saves GPU memory by recomputing activations during backward. | `true` |
| `use_mask_tokens` | Uses learnable mask tokens in predictor. | `true` |
| `use_rope` | Enables rotary positional embeddings. | `true` |
| `vit_conv` | Present in SurgVU config for compatibility. Not currently read by `app/vjepa_2_1/train.py`. | `false` or omit |
| `zero_init_mask_tokens` | Initializes mask tokens to zero. | `true` |
| `use_silu` | Optional SiLU MLP activation for encoder. Not in current SurgVU configs. | Omit or `false` |
| `use_pred_silu` | Optional SiLU MLP activation for predictor. Not in current SurgVU configs. | Omit or `false` |
| `wide_silu` | Optional wider SiLU MLP mode. Only relevant when SiLU options are used. | Omit or `true` default |
| `init_type` | Optional model initialization mode passed to model construction. | Omit for checkpoint fine-tuning |
| `levels_predictor` | Number of encoder levels to expose to predictor. Read by train config but currently not passed into model construction in this path. | Omit |

Do not use the paper's full 24-block predictor for this ViT-B public checkpoint. That is for the full pretraining setup, not this distilled ViT-B model shape.

## `optimization`

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `ema` | Start/end EMA momentum for target encoder. | `[0.99925, 0.99925]` |
| `epochs` | Number of epochs to train. | Stage 1 `20`; Stage 2 `5` to `10` |
| `final_lr` | Final learning rate after schedule. | `1.0e-6` |
| `final_weight_decay` | Final weight decay. | `0.04` |
| `ipe` | Iterations per epoch override. `null` uses the loader length. Stock paper configs use `300`. | `null` for full SurgVU pass; small integer only for debugging |
| `ipe_scale` | Multiplies loader length when `ipe` is not explicitly set in scheduler setup. | `1.25` |
| `lr` | Peak/base learning rate after warmup. | `1.0e-5` Stage 1; `5.0e-6` to `1.0e-5` Stage 2 |
| `start_lr` | Initial warmup learning rate. | `1.0e-6` |
| `warmup` | Warmup epochs. | `2` Stage 1; `0` or `1` Stage 2 |
| `weight_decay` | Initial weight decay. | `0.04` |
| `is_anneal` | Cooldown mode using `anneal_ckpt`. Stock cooldown configs use this. | Optional for Stage 2; not needed if using `read_checkpoint` directly |
| `anneal_ckpt` | Checkpoint used by stock cooldown/anneal path. | Stage 1 `latest.pth.tar` if using `is_anneal: true` |
| `resume_anneal` | Resume cooldown state from `anneal_ckpt`. | `true` only with `is_anneal: true` |
| `use_radamw` | Uses RAdamW optimizer variant instead of default AdamW path. | `false` unless experimenting |
| `betas` | Adam/RAdam beta values. | Omit for default `(0.9, 0.999)` |
| `eps` | Adam/RAdam epsilon. | Omit for default `1.0e-8` |
| `loss_reg_std_mult` | Optional dynamic loss regularization trigger. | Omit |
| `loss_reg_num_tracking_steps` | Tracking window for optional loss regularization. | Omit |
| `loss_reg_min_epoch` | Minimum epoch before optional loss regularization. | Omit |

The paper's LR reaches around `5.25e-4` to `6e-4` during large-scale training. That is too high for SurgVU domain adaptation from a pretrained checkpoint. Use `1e-5` as the first final recipe.

## `wandb`

This section is optional. It was added for SurgVU logging.

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `enable` | Enables W&B initialization on rank 0. | `true` if credentials/cache are available |
| `project` | W&B project name. | `surgvu-vjepa2` |
| `entity` | W&B account/team. If wrong, W&B can fail with permission denied. | Omit unless you know the exact entity |
| `name` | Run name. | Include model, frames, crop, and stage |
| `mode` | W&B mode: `online`, `offline`, or `disabled`. | `offline` on compute nodes without internet; sync later |
| `log_freq` | Log every N iterations. | `10` to `50` |
| `strict` | If `true`, W&B init failures abort training. If `false`, training continues without W&B. | `true` only after offline/online permissions are verified |
| `dir` | Local W&B directory. | `${folder}` or a scratch path |
| `group` | W&B group for related runs. | `surgvu-vjepa2.1-vitb` |
| `job_type` | W&B job type. | `train` |
| `tags` | W&B tags. | `["surgvu", "vjepa2.1", "vitb"]` |
| `notes` | Free text notes. | Briefly record data split/checkpoint |
| `resume` | W&B resume behavior. | Omit unless resuming a known run |
| `id` | W&B run id. Required for exact W&B resume. | Omit for new runs |

### `wandb.pca`

This nested section logs dense encoder PCA feature maps during training. The implementation uses the current training batch after the optimizer step, runs a no-grad encoder forward on rank 0, reshapes patch/tubelet tokens to `(T, H, W)`, fits the first three PCA components, and logs a contact sheet to W&B under `pca/feature_maps`. It follows the same visualization logic as `scripts/visualize_vjepa2_1_pca.py`.

| Entry | Meaning | Recommended SurgVU value |
| --- | --- | --- |
| `enable` | Enables PCA feature-map image logging. Requires `wandb.enable: true` and successful W&B init. | `true` for smoke/debug and final monitored runs |
| `log_freq` | Logs PCA maps every N global training steps. This is independent of scalar `wandb.log_freq`. | `100` for full training; `1` for smoke tests |
| `max_samples` | Number of samples from the first clip batch to visualize at each PCA step. | `1`; increase only for short debugging runs |
| `max_pca_tokens` | Maximum tokens used to fit PCA. All tokens are projected after fitting. | `50000` |
| `encoder` | Which encoder to visualize: `target_encoder` for EMA features or `encoder` for the online encoder. | `target_encoder` |
| `overlay` | Also logs input/PCA blended overlay columns in the contact sheet. | `false`; use `true` for qualitative inspection |
| `alpha` | Overlay opacity when `overlay: true`. | `0.55` |

On SLURM compute nodes without internet, use `mode: offline`. If `strict: true` and W&B lacks permission for the selected `entity` or cannot access its local directory, training will fail by design.

## Final Stage 1 YAML Skeleton

This is the recommended first full SurgVU run on the current 1-node, 4-GPU SLURM target.

```yaml
app: vjepa_2_1
nodes: 1
tasks_per_node: 4
cpus_per_task: 8
mem_per_gpu: 120G
exclusive: true
folder: output/surgvu_slurm/vjepa2_1_vitb_384px_16f

data:
  dataset_type: VideoDataset
  datasets: [data/surgvu/manifest_train.csv]
  datasets_weights: [1.0]
  batch_size: 1
  crop_size: 384
  patch_size: 16
  dataset_fpcs: [16]
  tubelet_size: 2
  fps: 4
  num_workers: 8
  pin_mem: true

data_aug:
  auto_augment: false
  motion_shift: false
  random_resize_aspect_ratio: [0.75, 1.35]
  random_resize_scale: [0.3, 1.0]
  reprob: 0.0

loss:
  loss_exp: 1.0
  predict_all: true
  shift_by_n: 0
  weight_distance_loss: true

mask:
- aspect_ratio: [0.75, 1.5]
  full_complement: false
  max_keep: null
  max_temporal_keep: 1.0
  num_blocks: 8
  spatial_scale: [0.15, 0.15]
  temporal_scale: [1.0, 1.0]
- aspect_ratio: [0.75, 1.5]
  full_complement: false
  max_keep: null
  max_temporal_keep: 1.0
  num_blocks: 2
  spatial_scale: [0.7, 0.7]
  temporal_scale: [1.0, 1.0]

meta:
  dtype: float16
  load_checkpoint: true
  read_checkpoint: checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
  save_every_freq: 5
  seed: 239
  use_sdpa: true

model:
  compile_model: false
  has_cls_first: false
  img_temporal_dim_size: 1
  interpolate_rope: true
  is_causal: false
  lambda_progressive: true
  lambda_value_img: 0.5
  lambda_value_vid: 0.5
  modality_embedding: true
  model_name: vit_base
  n_registers: 0
  n_registers_predictor: 0
  normalize_predictor: false
  pred_depth: 12
  pred_embed_dim: 384
  pred_is_causal: false
  pred_num_heads: 12
  uniform_power: true
  use_activation_checkpointing: true
  use_mask_tokens: true
  use_rope: true
  zero_init_mask_tokens: true

optimization:
  ema: [0.99925, 0.99925]
  epochs: 20
  final_lr: 1.0e-6
  final_weight_decay: 0.04
  ipe: null
  ipe_scale: 1.25
  lr: 1.0e-5
  start_lr: 1.0e-6
  warmup: 2
  weight_decay: 0.04

wandb:
  enable: true
  project: surgvu-vjepa2
  name: surgvu-vjepa2.1-vitb-384px-16f-stage1
  mode: offline
  log_freq: 20
  strict: false
  group: surgvu-vjepa2.1-vitb
  job_type: train
  tags: [surgvu, vjepa2.1, vitb, stage1]
  pca:
    enable: true
    log_freq: 100
    max_samples: 1
    max_pca_tokens: 50000
    encoder: target_encoder
    overlay: false
    alpha: 0.55
```

## Final Stage 2 Cooldown Changes

Start from the Stage 1 output and change only these fields first:

```yaml
folder: output/surgvu_slurm/vjepa2_1_vitb_384px_64f_cooldown

data:
  dataset_fpcs: [64]
  crop_size: 384
  batch_size: 1

meta:
  load_checkpoint: true
  read_checkpoint: output/surgvu_slurm/vjepa2_1_vitb_384px_16f/latest.pth.tar

model:
  lambda_progressive: false

optimization:
  epochs: 10
  lr: 5.0e-6
  start_lr: 1.0e-6
  final_lr: 1.0e-6
  warmup: 0
```

If 64-frame cooldown does not fit in memory, use `dataset_fpcs: [32]` before reducing crop size. If it is still unstable, stay with the Stage 1 16-frame model and proceed to downstream linear probing/evaluation.
