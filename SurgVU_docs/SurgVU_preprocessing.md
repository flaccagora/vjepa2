# SurgVU Offline Video Preprocessing

This document describes the offline preprocessing step for SurgVU videos before VJEPA-2.1 training.

The goal is to remove deterministic non-surgical image regions before the training dataloader applies random crop and augmentation:

- black margins around the endoscopic view
- top fixed UI/header regions, if present
- bottom text/UI overlays

The cleaned videos are written to a separate directory. Original videos are not modified.

## Why Offline Preprocessing

Use offline preprocessing rather than relying on training crops for this cleanup.

Advantages:

- The crop geometry is deterministic and logged.
- Training does not repeatedly spend dataloader time removing the same margins.
- The VJEPA random resized crop operates on surgical content instead of UI/margins.
- The cleaned manifest makes A/B comparison easy: raw manifest versus cleaned manifest.

Cost:

- Videos are re-encoded, so preprocessing needs disk space and time.
- You should inspect preview sheets before launching a full preprocessing job.

## Script

```text
scripts/preprocess_surgvu_videos.py
```

Inputs:

- a VJEPA-style manifest, e.g. `data/surgvu/manifest_train.csv`
- original video paths listed in that manifest

Outputs:

- cleaned videos under `<out-dir>/videos/`
- new manifest under `<out-dir>/<input-manifest-name>`
- crop metadata under `<out-dir>/preprocess_metadata.json`
- optional before/after preview sheets under `<out-dir>/previews/`

## How Cropping Works

For each video, the script:

1. Samples a configurable number of frames across the full video.
2. Removes a configured top strip from the analysis region.
3. Removes a configured bottom strip from the analysis region.
4. Detects non-black rows and columns in the remaining region.
5. Computes a single static crop box for the whole video.
6. Adds small padding around detected content.
7. Re-encodes the cropped frames into a new `.mp4`.
8. Writes a new manifest row with the cleaned video path and original label.

This is intentionally static per video. It avoids frame-by-frame geometry changes that would introduce artificial motion.

## Main Options

| Option | Meaning | Recommended starting value |
| --- | --- | --- |
| `--manifest` | Input VJEPA manifest. | `data/surgvu/manifest_train.csv` |
| `--out-dir` | Output directory for cleaned videos/manifests. | `data/surgvu_clean_train` |
| `--output-manifest` | Optional explicit output manifest path. | Omit |
| `--sample-frames` | Frames sampled per video for crop detection. | `24` |
| `--top-crop-ratio` | Fraction of original frame height removed from top before margin detection. | `0.0` initially |
| `--top-crop-pixels` | Exact top pixels to remove; overrides ratio. | Use after inspecting previews |
| `--bottom-crop-ratio` | Fraction of original frame height removed from bottom before margin detection. | `0.10` |
| `--bottom-crop-pixels` | Exact bottom pixels to remove; overrides ratio. | Use after measuring overlay height |
| `--black-threshold` | Pixel luminance threshold for non-black content. | `12` |
| `--min-content-fraction` | Minimum non-black fraction for a row/column to count as content. | `0.01` |
| `--padding` | Pixels added around the detected content crop. | `4` |
| `--preview-count` | Number of before/after preview sheets to save. | `8` for first pass |
| `--preview-frames` | Frames shown per preview sheet. | `8` |
| `--limit` | Process only first N manifest rows. | Use for debugging |
| `--max-output-frames` | Write only first N output frames. | Use for debugging only |
| `--overwrite` | Replace existing cleaned videos. | Use when retuning crop settings |
| `--backend` | Video writing backend. `auto` uses `ffmpeg` if available, otherwise OpenCV. | `auto` |
| `--workers` | Number of videos processed in parallel. | `4` on a 32 CPU node |
| `--ffmpeg-bin` | ffmpeg executable path/name. | `ffmpeg` |
| `--ffprobe-bin` | ffprobe executable path/name, used by `--quality-mode source`. | `ffprobe` |
| `--ffmpeg-encoder` | ffmpeg encoder. `libx264` is the quality-stable CPU default; `h264_nvenc` uses NVIDIA GPU encoding when available. | `libx264` |
| `--ffmpeg-preset` | libx264 speed preset. Faster presets reduce preprocessing time; target bitrate still controls output size in source mode. | `veryfast` |
| `--nvenc-preset` | NVENC speed/quality preset when using an `*_nvenc` encoder. Lower preset numbers are faster, higher numbers are slower/higher quality. | `p4` |
| `--nvenc-gpus` | Comma-separated GPU indices for NVENC, assigned by manifest index modulo the list. | `0,1,2,3` on a 4-GPU node |
| `--ffmpeg-hwaccel` | Optional ffmpeg input hardware acceleration. Use only for benchmarking because crop/blackdetect still use CPU filters in the current ffmpeg build. | `none` |
| `--quality-mode` | ffmpeg quality mode. `source` matches the original video's bits-per-pixel-frame after crop/FPS changes; `crf` uses `--crf`; `lossless` uses `-qp 0` and can be much larger. | `source` |
| `--bitrate-scale` | Multiplier applied to source-matched bitrate. | `1.0` |
| `--remove-black-sections` | Detect and remove full-screen black video intervals with ffmpeg `blackdetect`. | Enable if SurgVU clips contain black gaps |
| `--save-black-sections` | Save detected full-screen black intervals as separate clips. Can be used with or without removal. | Enable when auditing black gaps |
| `--black-sections-dir` | Directory for saved black-section clips. | `<out-dir>/black_sections` |
| `--black-section-save-mode` | Save mode for black-section clips. `copy` is fastest and preserves original stream packets; `encode` gives exact trim boundaries. | `copy` |
| `--black-min-duration` | Minimum black interval duration to remove, in seconds. | `1.0` |
| `--black-pix-th` | ffmpeg black pixel threshold. Lower is stricter black. | `0.10` |
| `--black-pic-th` | Fraction of pixels that must be black for a frame to count as black. | `0.98` |
| `--black-section-padding` | Seconds added before/after each detected black interval. | `0.0` |
| `--output-fps` | Optional output FPS. Set this to the training FPS only if you intentionally want preprocessing to remove frames the loader would otherwise skip. | `4` for the current VJEPA config; omit to keep source FPS |
| `--crf` | libx264 quality for `--quality-mode crf`. Lower means higher quality and larger files. | `16` |
| `--ffmpeg-threads` | Threads per ffmpeg process. `0` lets ffmpeg choose. | `2` to `4` when using multiple workers |
| `--faststart` | Move MP4 metadata to the start of the file. This adds an extra file rewrite and is unnecessary for training. | Omit |
| `--codec` | OpenCV fourcc codec, used only with `--backend opencv` or debug frame limiting. | `mp4v` |

## Speed Settings

The fast path is:

```bash
--backend ffmpeg
```

or simply:

```bash
--backend auto
```

when `ffmpeg` is installed. `ffmpeg` is much faster than the Python/OpenCV frame-by-frame writer because cropping and encoding stay inside optimized native code.

For a 1-node preprocessing job with 32 CPU cores, start with:

```bash
--workers 4
--ffmpeg-threads 4
--ffmpeg-preset veryfast
--quality-mode source
--bitrate-scale 1.0
--output-fps 4
```

`--output-fps 4` is the largest speedup for SurgVU because many raw videos are 60 FPS while the current training recipe uses `data.fps: 4`. This writes about 15x fewer frames and stores only the temporal stream used by the current model recipe. If your requirement is "same video except black-section removal and cropping", omit `--output-fps` and keep the source frame rate.

When preprocessing at 4 FPS, keep the training YAML at:

```yaml
data:
  fps: 4
```

If the filesystem becomes the bottleneck, reduce `--workers`. If CPU is underused, increase `--workers` or `--ffmpeg-threads`, but avoid oversubscribing too far beyond the allocated CPU count.

Quality notes:

- Cropping, FPS conversion, and black-section removal require re-encoding; exact stream copy is not possible.
- `--quality-mode source` is the recommended default. It probes the source video bitrate, resolution, and FPS, then scales target bitrate by retained crop area and retained frame rate. This keeps compression density roughly constant while letting cropped/downsampled outputs become smaller.
- `--bitrate-scale 1.0` means "match source compression density". Use values above `1.0` only if previews show unacceptable compression artifacts; use values below `1.0` only if you intentionally accept more compression.
- `--quality-mode lossless` with `libx264 -qp 0` avoids additional compression loss on decoded/cropped frames, but it is not "same as original file quality" and can easily create outputs many times larger than the source.
- `--faststart` is intentionally off by default because it rewrites the MP4 after encoding and does not help local dataloader training.
- `--crf 16` is a high-quality near-visually-lossless setting if you intentionally choose `--quality-mode crf` to save disk.
- `--crf 18` is smaller and still usually visually strong.
- In CRF mode, `--ffmpeg-preset` changes compression efficiency/speed, not the target CRF quality. Slower presets may produce smaller files at the same CRF, but they take longer.
- Audio is dropped with the ffmpeg backend because VJEPA training does not use audio.

## NVIDIA GPU Acceleration

Your cluster ffmpeg build exposes CUDA hardware acceleration and NVENC encoders:

```bash
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -encoders | grep nvenc
```

That is useful, but it is not a complete GPU preprocessing pipeline. The listed CUDA filters do not include `crop_cuda`, and `blackdetect` is also a CPU filter. In practice:

- Decoding, black-section detection, FPS selection, and crop filtering remain CPU-side or require CPU/GPU frame transfers.
- Encoding can be accelerated with NVENC.
- NVENC is much faster than `libx264`, but at the same bitrate it can be less compression-efficient. Keep `--quality-mode source` and inspect previews before using it for the final dataset.

Recommended benchmark command:

```bash
python scripts/preprocess_surgvu_videos.py \
  --manifest data/surgvu/manifest_train.csv \
  --out-dir output/surgvu_preprocess_nvenc_benchmark \
  --limit 3 \
  --preview-count 3 \
  --sample-frames 24 \
  --top-crop-pixels 24 \
  --bottom-crop-pixels 72 \
  --backend ffmpeg \
  --ffmpeg-encoder h264_nvenc \
  --nvenc-preset p4 \
  --nvenc-gpus 0,1,2,3 \
  --quality-mode source \
  --bitrate-scale 1.0 \
  --remove-black-sections \
  --black-min-duration 1.0 \
  --output-fps 4 \
  --workers 4 \
  --ffmpeg-threads 1 \
  --overwrite
```

Start with `--workers 1` for a single-video benchmark, then try `--workers 4 --nvenc-gpus 0,1,2,3` on a 4-GPU node. NVENC sessions can saturate quickly, and too many parallel encoders can reduce throughput or fail depending on GPU/driver limits.

`--ffmpeg-hwaccel cuda` is available as an experimental option, but benchmark it rather than assuming it is faster. Because the active crop/blackdetect filters are CPU filters, hardware decode can add frame-transfer overhead and may not improve end-to-end time.

If ffmpeg fails with `CUDA_ERROR_NO_DEVICE`, the encoder is compiled in but the current process cannot see a CUDA-capable GPU. On Slurm, verify the job requested GPUs and that `CUDA_VISIBLE_DEVICES` is populated inside the allocation.

## Removing Black Screen Sections

Some long SurgVU videos can contain full-screen black intervals. Use:

```bash
--remove-black-sections
```

This runs ffmpeg `blackdetect` before encoding each video, then removes the detected intervals in the crop/re-encode pass. To also archive those intervals separately, add:

```bash
--save-black-sections
```

By default, saved black clips are written under:

```text
<out-dir>/black_sections/
```

The output metadata records what was detected, removed, and saved:

```json
{
  "black_sections_detected": [
    {"start": 120.4, "end": 134.8}
  ],
  "black_sections_removed": [
    {"start": 120.4, "end": 134.8}
  ],
  "black_sections_saved": [
    {
      "index": 0,
      "start": 120.4,
      "end": 134.8,
      "duration": 14.4,
      "path": "data/surgvu_clean_train/black_sections/case001_black_000_000120400ms_000134800ms.mp4",
      "mode": "copy"
    }
  ],
  "black_duration_removed": 14.4
}
```

Recommended starting settings:

```bash
--remove-black-sections
--save-black-sections
--black-min-duration 1.0
--black-pix-th 0.10
--black-pic-th 0.98
--black-section-padding 0.0
```

Raise `--black-min-duration` if short dark transitions are being removed. Lower `--black-pic-th` only if black screens contain logos or overlays and are not detected. Keep it high for safety because surgical scenes can naturally contain dark regions.

`--black-section-save-mode copy` is fastest and keeps the original encoded stream for the black interval, but because video packet copying is keyframe-bound, clip boundaries can be slightly approximate. Use `--black-section-save-mode encode` if you need exact interval boundaries.

## Debug One Video

Always start with a short debug run:

```bash
conda activate vjepa2-312

python scripts/preprocess_surgvu_videos.py \
  --manifest data/surgvu/manifest_train.csv \
  --out-dir output/surgvu_preprocess_debug \
  --limit 1 \
  --max-output-frames 16 \
  --preview-count 1 \
  --sample-frames 4 \
  --bottom-crop-ratio 0.10 \
  --top-crop-ratio 0.0 \
  --overwrite
```

Inspect:

```text
output/surgvu_preprocess_debug/previews/
output/surgvu_preprocess_debug/preprocess_metadata.json
```

The metadata records per-video crop geometry:

```json
{
  "source_width": 1280,
  "source_height": 720,
  "top_crop_pixels": 0,
  "bottom_crop_pixels": 72,
  "crop_x": 188,
  "crop_y": 0,
  "crop_width": 904,
  "crop_height": 648
}
```

## Tuning Top and Bottom Crops

Use ratio controls for a first pass:

```bash
--top-crop-ratio 0.03
--bottom-crop-ratio 0.10
```

For 720p videos:

- `--top-crop-ratio 0.03` removes about 22 pixels.
- `--bottom-crop-ratio 0.10` removes about 72 pixels.

Once you know the exact overlay height, prefer pixel controls:

```bash
--top-crop-pixels 24
--bottom-crop-pixels 72
```

Pixel controls are easier to reproduce across runs. Ratio controls are more robust if you mix resolutions.

## Full Training Preprocessing

After previews look correct:

```bash
conda activate vjepa2-312

python scripts/preprocess_surgvu_videos.py \
  --manifest data/surgvu/manifest_train.csv \
  --out-dir data/surgvu_clean_train \
  --preview-count 8 \
  --sample-frames 24 \
  --top-crop-ratio 0.0 \
  --bottom-crop-ratio 0.10 \
  --backend auto \
  --workers 4 \
  --ffmpeg-threads 4 \
  --quality-mode source \
  --bitrate-scale 1.0 \
  --ffmpeg-preset veryfast \
  --remove-black-sections \
  --save-black-sections \
  --black-min-duration 1.0 \
  --output-fps 4
```

If the top also contains fixed non-surgical content:

```bash
python scripts/preprocess_surgvu_videos.py \
  --manifest data/surgvu/manifest_train.csv \
  --out-dir data/surgvu_clean_train \
  --preview-count 8 \
  --sample-frames 24 \
  --top-crop-pixels 24 \
  --bottom-crop-pixels 72 \
  --backend auto \
  --workers 4 \
  --ffmpeg-threads 4 \
  --quality-mode source \
  --bitrate-scale 1.0 \
  --ffmpeg-preset veryfast \
  --remove-black-sections \
  --save-black-sections \
  --black-min-duration 1.0 \
  --output-fps 4 \
  --overwrite
```

## Use Cleaned Videos For Training

Point the training config at the cleaned manifest:

```yaml
data:
  datasets:
  - data/surgvu_clean_train/manifest_train.csv
```

Then run the dataloader visualizer on the cleaned manifest/config:

```bash
python scripts/visualize_surgvu_dataloader.py \
  --config configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm.yaml \
  --out-dir output/surgvu_dataloader_preview_clean \
  --num-batches 2 \
  --max-samples 4 \
  --num-workers 0
```

If the config still points to the raw manifest, either edit the config temporarily or copy the cleaned manifest path into a dedicated clean-data YAML.

## Recommended Workflow

1. Run `preprocess_surgvu_videos.py` with `--limit 1 --max-output-frames 16 --preview-count 1`.
2. Inspect the preview image and `preprocess_metadata.json`.
3. Tune `--top-crop-*`, `--bottom-crop-*`, and `--padding`.
4. Run with `--limit 8 --max-output-frames 64 --preview-count 8`.
5. Inspect previews across several cases.
6. Run full preprocessing without `--limit` or `--max-output-frames`.
7. Point training YAML to the cleaned manifest.
8. Run `visualize_surgvu_dataloader.py` against the cleaned manifest.
9. Submit VJEPA training.

## Notes

- The crop is static per video, not dynamic per frame.
- Audio is not preserved. VJEPA training does not use audio.
- Original videos are left untouched.
- The second manifest column is preserved exactly.
- If a video already has no detectable black margin, only the configured top/bottom strip and padding-adjusted content crop will apply.
- Avoid aggressive top/bottom crops until preview sheets confirm the UI/text region is actually fixed and non-surgical.
