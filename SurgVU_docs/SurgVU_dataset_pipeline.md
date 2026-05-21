# SurgVU Dataset Structure and Extraction Pipeline Guide

This guide explains the SurgVU dataset structure from `SurgVU_dataset.pdf` and how the local extraction pipeline prepares the data for VJEPA-2.1 fine-tuning.

The companion training/config guides are:

- `SurgVU_finetuning.md`
- `SurgVU_yaml_config_guide.md`

## Dataset Summary

`SurgVU_dataset.pdf` describes SurgVU as a large robotic-surgery video dataset collected from da Vinci robotic surgery training sessions. The videos show the surgeon console/endoscope view while trainees and expert surgeons perform standardized surgical steps on porcine tissues.

Key dataset facts:

- 280 video clips from 155 training sessions.
- More than 840 hours of video.
- 60 FPS endoscopic video.
- About 18 million labeled frames.
- 720p resolution, reported as 1280 x 720.
- Up to three robotic tools can appear in the surgical field.
- Up to twelve installed/visible tool classes are represented.
- Labels include tool presence, surgical step/task, detailed step descriptions, and a public visual question answering sample set.

For VJEPA-2.1 self-supervised fine-tuning, the pipeline uses the videos as the training signal. Labels are not used as supervision; they are only parsed to create deterministic numeric labels because the repo's `VideoDataset` manifest format expects a second column.

## Public Files Mentioned in the PDF

| File | URL | Purpose in this repo |
| --- | --- | --- |
| `surgvu24_videos_only.zip` | `https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip` | Main full video archive for final self-supervised training |
| `surgvu24_labels_updated_v2.zip` | `https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip` | Labels archive; parsed for metadata and stable manifest labels |
| `cat1_test_set_public.zip` | `https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip` | Public tool-detection validation set; not used by VJEPA pretraining pipeline |
| `SURGVU25_cat_2_sample_set_public.zip` | `https://storage.googleapis.com/isi-surgvu/SURGVU25_cat_2_sample_set_public.zip` | Public VQA sample set; used here for smoke tests when full videos are unavailable |

Current local files observed under `data/`:

```text
data/surgvu24_labels_updated_v2.zip
data/SURGVU25_cat_2_sample_set_public.zip
data/cat1_test_set_public.zip
data/surgvu/
data/surgvu_smoke/
```

Important: the current `data/surgvu/` directory was prepared from the public sample set unless you rerun `scripts/prepare_surgvu.py` with the full `surgvu24_videos_only.zip` or a full extracted video root.

## Dataset Label Semantics

### Tools

The PDF lists these tool categories:

- needle driver
- cadiere forceps
- prograsp forceps
- monopolar curved scissors
- bipolar forceps
- stapler
- force bipolar
- vessel sealer
- permanent cautery hook/spatula
- clip applier
- tip-up fenestrated grasper
- grasping retractor

Tool labels are presence labels derived from system data. The PDF notes that these labels can be noisy because a tool can be installed but temporarily obscured or outside the visible field. Tool frequency is also imbalanced.

In the labels archive, each case has a `tools.csv` file. Example columns observed locally:

```text
index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname
```

The VJEPA extraction pipeline does not currently use tool labels for training.

### Surgical Steps / Tasks

The PDF describes eight surgical step categories:

- Suturing
- Uterine horn
- Rectal artery/vein manipulation
- Suspensory ligaments
- General skills application
- Retraction and collision avoidance
- Range of motion
- Other

In the labels archive, each case has a `tasks.csv` file. Example columns observed locally:

```text
index,start_part,start_time,stop_part,stop_time,groundtruth_taskname
```

The PDF text refers to install/uninstall-style timing fields for task labels, but the local `surgvu24_labels_updated_v2.zip` uses `start_part`, `start_time`, `stop_part`, and `stop_time` for `tasks.csv`.

The extraction script reads `groundtruth_taskname` from each `tasks.csv`, chooses the dominant task name per case, and maps task names to integer labels. These integer labels are written into VJEPA manifests only to satisfy the dataset loader's expected manifest shape.

### Step Descriptions

The PDF says the labels were expanded with detailed descriptions of surgical steps in a `matched description` column. These descriptions are useful for vision-language modeling and retrieval/captioning work.

The current VJEPA self-supervised pipeline does not consume text descriptions.

### VQA Sample Set

The public `SURGVU25_cat_2_sample_set_public.zip` contains short sample clips and question/answer JSON files. The extracted structure looks like this:

```text
case124/
  case124.mp4
  case124_question.json
  case124.json
```

Observed sample content:

```text
case124_question.json: "What type of forceps is mentioned?"
case124.json: five acceptable ground-truth answer variants
```

This sample set is useful for smoke testing because it is much smaller than the full training archive. It is not the full SurgVU video training set.

### Category 1 Public Validation Set

The public `cat1_test_set_public.zip` contains downsampled tool-detection validation clips and JSON annotations. The archive layout observed locally:

```text
1_fps1.mp4
1_fps1_gc.json
1_fps1_coco.json
...
7_fps1.mp4
7_fps1_gc.json
7_fps1_coco.json
```

The PDF says this set was downsampled to 1 FPS and contains bounding-box annotations for tool detection. The current VJEPA self-supervised preparation script does not use this archive.

## Expected Local Layout

Recommended raw archive layout:

```text
data/
  surgvu24_videos_only.zip
  surgvu24_labels_updated_v2.zip
  SURGVU25_cat_2_sample_set_public.zip
  cat1_test_set_public.zip
```

Recommended prepared full-training layout after running `scripts/prepare_surgvu.py`:

```text
data/surgvu/
  labels/
    labels/
      case_001/
        tasks.csv
        tools.csv
      case_002/
        tasks.csv
        tools.csv
      ...
  videos_extracted/
    ...
  manifest_all.csv
  manifest_train.csv
  manifest_val.csv
  metadata.csv
  summary.json
```

Prepared smoke-test layout:

```text
data/surgvu_smoke/
  labels/
  sample_videos/
    case122/
      case122.mp4
      case122_question.json
      case122.json
    ...
  manifest_all.csv
  manifest_train.csv
  manifest_val.csv
  metadata.csv
  summary.json
```

The extraction script ignores files inside `__MACOSX` directories when discovering videos.

## Extraction Script

Main script:

```text
scripts/prepare_surgvu.py
```

The script performs these steps:

1. Creates the output directory.
2. Extracts `--labels-zip` into `<out-dir>/labels` unless `--labels-root` is provided.
3. Selects the video source:
   - `--videos-root` if already extracted videos are available.
   - `--video-zip` if a full video zip should be extracted.
   - `--sample-zip` if neither full video source is provided and the sample zip exists.
4. Recursively discovers video files with these extensions:
   - `.mp4`
   - `.mov`
   - `.avi`
   - `.mkv`
   - `.webm`
   - `.m4v`
5. Normalizes case IDs from paths using patterns like `case124`, `case_124`, or `case-124`, producing IDs like `case_124`.
6. Reads every `tasks.csv` under the labels root.
7. Extracts `groundtruth_taskname` values.
8. Chooses the dominant task per case.
9. Builds a sorted task-name-to-integer `label_map`.
10. Assigns `unknown` to videos without a matching task label.
11. Shuffles videos deterministically with `--seed`.
12. Applies `--limit` if requested.
13. Optionally copies videos into `<out-dir>/videos` if `--copy-videos` is set.
14. Splits rows into train and validation sets with `--val-fraction`.
15. Writes VJEPA manifests and metadata files.

## Script Arguments

| Argument | Meaning | Recommended use |
| --- | --- | --- |
| `--out-dir` | Prepared dataset output directory. | `data/surgvu` for full data; `data/surgvu_smoke` for smoke |
| `--videos-root` | Directory containing already extracted videos. | Use when the full zip is already extracted on shared storage |
| `--video-zip` | Full SurgVU video zip to extract and index. | `data/surgvu24_videos_only.zip` for final training |
| `--sample-zip` | Public VQA sample zip. Used when no full video source is provided. | `data/SURGVU25_cat_2_sample_set_public.zip` for smoke only |
| `--labels-zip` | Labels zip to extract and summarize. | `data/surgvu24_labels_updated_v2.zip` |
| `--labels-root` | Directory containing extracted labels and `case_*/tasks.csv` files. | Use to avoid re-extracting labels |
| `--manifest-prefix` | Prefix for generated manifest filenames. | Keep default `manifest` |
| `--limit` | Deterministic limit after shuffling. | Use only for smoke/debug |
| `--val-fraction` | Fraction of videos assigned to validation. | `0.1` full data; `0.25` smoke |
| `--seed` | Deterministic shuffle seed. | `239` |
| `--copy-videos` | Copy videos into `<out-dir>/videos` instead of indexing in place. | Usually omit to avoid duplicating large videos |
| `--force` | Recreate extracted directories. | Use when intentionally rebuilding prepared data |

## Generated Files

### `manifest_all.csv`

Space-delimited VJEPA manifest containing every indexed video:

```text
/absolute/path/to/video.mp4 5
```

Column meanings:

- Column 1: absolute video path.
- Column 2: integer label derived from dominant case-level task, or `unknown`.

For self-supervised VJEPA training, the numeric label is not the learning target.

### `manifest_train.csv`

Training split written in the same two-column format. This is the path used by the SurgVU training YAML:

```yaml
data:
  datasets:
  - data/surgvu/manifest_train.csv
```

### `manifest_val.csv`

Validation split written in the same format. The current VJEPA training loop does not run online validation from this file, but it is useful for later representation evaluation or sanity checks.

### `metadata.csv`

Human-readable metadata for each discovered video:

```text
path,case_id,label,task_name
```

Example:

```text
/home/matteo/Desktop/vjepa2/data/surgvu/sample_videos/case124/case124.mp4,case_124,5,Suturing
```

### `summary.json`

Machine-readable preparation summary:

```json
{
  "videos_root": "...",
  "labels_root": "...",
  "num_videos": 11,
  "num_train": 8,
  "num_val": 3,
  "label_map": {
    "Range of motion": 0,
    "Rectal artery/vein": 1,
    "Retraction and collision avoidance": 2,
    "Skills application": 3,
    "Suspensory ligaments": 4,
    "Suturing": 5,
    "Uterine horn": 6,
    "unknown": 7
  },
  "manifests": {
    "all": ".../manifest_all.csv",
    "train": ".../manifest_train.csv",
    "val": ".../manifest_val.csv"
  }
}
```

The exact `label_map` depends on the labels found under the selected labels root.

## Full Dataset Preparation

Use this when `data/surgvu24_videos_only.zip` is available:

```bash
conda activate vjepa2-312

python scripts/prepare_surgvu.py \
  --video-zip data/surgvu24_videos_only.zip \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu \
  --val-fraction 0.1 \
  --seed 239
```

If the full videos are already extracted:

```bash
conda activate vjepa2-312

python scripts/prepare_surgvu.py \
  --videos-root /path/to/extracted/surgvu/videos \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu \
  --val-fraction 0.1 \
  --seed 239
```

Use `--copy-videos` only if you explicitly want a self-contained prepared directory and have enough disk space.

## Smoke Dataset Preparation

Use this when testing the extraction and training loop with the public VQA sample set:

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

This produces a tiny split suitable for `configs/train_2_1/vitb16/surgvu-smoke-256px-16f.yaml`.

## Manifest Validation

Validation script:

```text
scripts/check_surgvu_manifest.py
```

It checks that:

- The manifest exists and is non-empty.
- Listed video paths exist.
- A few videos can be opened with `decord`.
- Frame count, average FPS, and duration can be read.

Run:

```bash
conda activate vjepa2-312

python scripts/check_surgvu_manifest.py data/surgvu/manifest_train.csv --limit 4
```

Expected output shape:

```json
{
  "manifest": "data/surgvu/manifest_train.csv",
  "num_rows": 8,
  "checked": [
    {
      "path": ".../case122.mp4",
      "label": "5",
      "frames": 1800,
      "fps": 60.0,
      "duration_seconds": 30.0
    }
  ]
}
```

## Current Pipeline Limitations

- It does not create time-segment-level clips from `tasks.csv`; each discovered video file is treated as one training item.
- It does not use tool labels, bounding boxes, task intervals, matched descriptions, or VQA text during VJEPA training.
- It does not balance by task/tool distribution.
- It assigns one dominant task label per case, which is only loader metadata for self-supervised training.
- It uses a simple tail split after deterministic shuffle and sort. For strict case/session-level evaluation, add a dedicated split file or split logic.
- It does not verify that every full-video case has labels; unmatched cases receive the `unknown` label.
- It does not use `cat1_test_set_public.zip`; that archive is for tool-detection evaluation, not current representation pretraining.

## Recommended Final Data Recipe

For meaningful SurgVU VJEPA-2.1 representation learning:

1. Use the full `surgvu24_videos_only.zip`, not the VQA sample set.
2. Keep labels available through `surgvu24_labels_updated_v2.zip` for metadata and downstream probes.
3. Prepare `data/surgvu/manifest_train.csv` with `--val-fraction 0.1`.
4. Validate at least 8 to 16 videos with `scripts/check_surgvu_manifest.py`.
5. Confirm `summary.json` has the expected number of full-dataset videos before submitting SLURM training.
6. Use `data/surgvu/manifest_train.csv` in the training YAML.
7. Keep `data.dataset_fpcs: [16]`, `fps: 4`, and `crop_size: 384` for the first full run.
8. Move to `dataset_fpcs: [64]` only for the optional cooldown stage after Stage 1 succeeds.

Quick final-data checklist:

```bash
conda activate vjepa2-312

python scripts/prepare_surgvu.py \
  --video-zip data/surgvu24_videos_only.zip \
  --labels-zip data/surgvu24_labels_updated_v2.zip \
  --out-dir data/surgvu \
  --val-fraction 0.1 \
  --seed 239

python scripts/check_surgvu_manifest.py data/surgvu/manifest_train.csv --limit 8

cat data/surgvu/summary.json
```

## Long-Video Skip Warning

If cluster training prints a warning like this:

```text
skipping long video of size _fsize=... (bytes)
```

it comes from `src/datasets/video_dataset.py`. The upstream loader has a default `filter_long_videos` limit of `1e9` bytes, about 1 GB. Full SurgVU clips can be larger than that, so the loader may skip valid videos.

Do not manually edit individual manifest entries unless the video path is wrong or the file is corrupt. For full SurgVU training, set the data-loader threshold in YAML:

```yaml
data:
  filter_long_videos: null
  filter_short_videos: false
```

In this SurgVU setup, `filter_long_videos: null` is interpreted as an effectively disabled size filter by `app/vjepa_2_1/train.py`.
