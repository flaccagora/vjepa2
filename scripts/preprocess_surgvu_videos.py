#!/usr/bin/env python3
"""Preprocess SurgVU videos before VJEPA training.

The script reads a VJEPA-style manifest, removes static black margins and a
bottom overlay region from each video, writes cleaned videos, and emits a new
manifest with the same labels.

Metadata only preprocessing:
    python scripts/preprocess_surgvu_videos.py \
        --manifest data/surgvu_metadata/manifest_all.csv \
        --out-dir surgvu_metadata_train \
        --metadata-only \
        --sample-frames 24 \
        --remove-black-sections \
        --black-detection-mode sampled \
        --black-sample-fps 1 \
        --black-sample-width 64 \
        --black-sample-height 36 \
        --black-sample-threshold 12 \
        --black-pic-th 0.95 \
        --black-section-padding 0.5 \
        --workers 1 \
        --overwrite

"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("surgvu_preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input VJEPA manifest.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for cleaned videos and metadata.")
    parser.add_argument("--output-manifest", type=Path, help="Output manifest path. Defaults to <out-dir>/<manifest-name>.")
    parser.add_argument("--sample-frames", type=int, default=24, help="Frames sampled per video for black-margin detection.")
    parser.add_argument("--black-threshold", type=int, default=12, help="Pixel luminance threshold for non-black content.")
    parser.add_argument(
        "--min-content-fraction",
        type=float,
        default=0.01,
        help="Minimum non-black fraction for a row/column to be considered content.",
    )
    parser.add_argument(
        "--bottom-crop-ratio",
        type=float,
        default=0.06,
        help="Fraction of original frame height removed from the bottom before margin detection.",
    )
    parser.add_argument("--bottom-crop-pixels", type=int, help="Overrides --bottom-crop-ratio when set.")
    parser.add_argument(
        "--top-crop-ratio",
        type=float,
        default=0.06,
        help="Fraction of original frame height removed from the top before margin detection.",
    )
    parser.add_argument("--top-crop-pixels", type=int, help="Overrides --top-crop-ratio when set.")
    parser.add_argument("--padding", type=int, default=4, help="Pixels added around detected content crop.")
    parser.add_argument("--limit", type=int, help="Process only the first N manifest rows.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite cleaned videos if they already exist.")
    parser.add_argument(
        "--max-output-frames",
        type=int,
        help="Debug option: write only the first N frames of each output video.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only detect crop/black intervals and write metadata; do not encode cleaned videos.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing preprocess_metadata.json and process all selected manifest rows.",
    )
    parser.add_argument("--preview-count", type=int, default=0, help="Write before/after contact sheets for first N videos.")
    parser.add_argument("--preview-frames", type=int, default=8, help="Frames per preview sheet.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec for output mp4 files.")
    parser.add_argument(
        "--backend",
        choices=("auto", "ffmpeg", "opencv"),
        default="ffmpeg",
        help="Video writing backend. 'auto' uses ffmpeg when available, otherwise OpenCV.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Videos to preprocess in parallel.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable.")
    parser.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe executable.")
    parser.add_argument(
        "--ffmpeg-encoder",
        choices=("libx264", "h264_nvenc", "hevc_nvenc", "av1_nvenc"),
        default="libx264",
        help="ffmpeg video encoder. Use h264_nvenc for faster GPU encoding when available.",
    )
    parser.add_argument("--ffmpeg-preset", default="veryfast", help="libx264 preset used by ffmpeg backend.")
    parser.add_argument("--nvenc-preset", default="p4", help="NVENC preset used when --ffmpeg-encoder is *_nvenc.")
    parser.add_argument(
        "--nvenc-gpus",
        default="",
        help="Comma-separated NVENC GPU indices, e.g. 0,1,2,3. Videos are assigned by manifest index modulo this list.",
    )
    parser.add_argument(
        "--ffmpeg-hwaccel",
        choices=("none", "auto", "cuda"),
        default="none",
        help="Optional ffmpeg input hardware acceleration. Experimental with CPU crop/blackdetect filters.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=("source", "crf", "lossless"),
        default="source",
        help="source matches original bits-per-pixel-frame; crf uses --crf; lossless uses qp=0.",
    )
    parser.add_argument(
        "--bitrate-scale",
        type=float,
        default=1.0,
        help="Multiplier for source-matched bitrate. Keep 1.0 to preserve original compression density.",
    )
    parser.add_argument(
        "--remove-black-sections",
        action="store_true",
        help="Use ffmpeg blackdetect to remove full-screen black intervals before writing cleaned videos.",
    )
    parser.add_argument(
        "--save-black-sections",
        action="store_true",
        help="Save detected full-screen black intervals as separate video clips.",
    )
    parser.add_argument(
        "--black-sections-dir",
        type=Path,
        help="Directory for saved black-section clips. Defaults to <out-dir>/black_sections.",
    )
    parser.add_argument(
        "--black-section-save-mode",
        choices=("copy", "encode"),
        default="copy",
        help="How to save black-section clips. copy is fastest/original stream; encode gives exact trim boundaries.",
    )
    parser.add_argument(
        "--black-detection-mode",
        choices=("sampled", "full"),
        default="sampled",
        help="Black-section detector. sampled is much faster; full uses ffmpeg blackdetect over the full stream.",
    )
    parser.add_argument(
        "--black-sample-fps",
        type=float,
        default=1.0,
        help="FPS for sampled black-section detection.",
    )
    parser.add_argument(
        "--black-sample-width",
        type=int,
        default=64,
        help="Width for downscaled sampled black-frame detection.",
    )
    parser.add_argument(
        "--black-sample-height",
        type=int,
        default=36,
        help="Height for downscaled sampled black-frame detection.",
    )
    parser.add_argument(
        "--black-sample-filter-backend",
        choices=("cpu", "cuda", "cvcuda"),
        default="cpu",
        help="Filter backend for sampled black detection. cuda uses ffmpeg scale_cuda; cvcuda uses CV-CUDA resize.",
    )
    parser.add_argument(
        "--black-cvcuda-batch-size",
        type=int,
        default=64,
        help="Sampled RGB frames per CVCUDA batch when --black-sample-filter-backend cvcuda.",
    )
    parser.add_argument(
        "--black-sample-threshold",
        type=int,
        default=12,
        help="Pixel threshold [0,255] for sampled black-frame detection.",
    )
    parser.add_argument(
        "--black-refine-boundaries",
        action="store_true",
        help="Refine sampled black-section boundaries with a higher-FPS scan around candidate boundaries.",
    )
    parser.add_argument(
        "--black-refine-fps",
        type=float,
        default=8.0,
        help="FPS for boundary refinement when --black-refine-boundaries is set.",
    )
    parser.add_argument(
        "--black-refine-window",
        type=float,
        default=2.0,
        help="Seconds around each sampled candidate interval to rescan during boundary refinement.",
    )
    parser.add_argument(
        "--black-min-duration",
        type=float,
        default=0.5,
        help="Minimum black interval duration in seconds for ffmpeg blackdetect.",
    )
    parser.add_argument(
        "--black-pix-th",
        type=float,
        default=0.10,
        help="ffmpeg blackdetect pixel threshold. Lower is stricter black.",
    )
    parser.add_argument(
        "--black-pic-th",
        type=float,
        default=0.98,
        help="ffmpeg blackdetect picture threshold: fraction of pixels that must be black.",
    )
    parser.add_argument(
        "--black-section-padding",
        type=float,
        default=0.0,
        help="Seconds added before/after detected black intervals when cutting them out.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Optional output FPS. Set to the training FPS, e.g. 4, to avoid encoding unused 60 FPS frames.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=16,
        help="ffmpeg libx264 CRF. Lower is higher quality/larger files. 16 is near-visually-lossless.",
    )
    parser.add_argument(
        "--faststart",
        action="store_true",
        help="Move MP4 metadata to the start of the file. Useful for streaming, slower for preprocessing.",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Threads per ffmpeg process. 0 lets ffmpeg decide; set 1-4 when using many workers.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s][%(asctime)s][%(processName)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def timed_step(timings: dict[str, float], name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)


def round_timings(timings: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in sorted(timings.items())}


def read_manifest(path: Path) -> list[tuple[Path, str]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 1:
                rows.append((Path(parts[0]), "0"))
            else:
                rows.append((Path(parts[0]), parts[1]))
    return rows


def write_manifest(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        for video_path, label in rows:
            writer.writerow([str(video_path), label])
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp_path, path)


def sample_indices(total_frames: int, count: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    count = max(1, min(count, total_frames))
    return np.unique(np.linspace(0, total_frames - 1, count, dtype=np.int64))


def read_frame(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    return frame if ok else None


def even_crop(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 2, min(x2, width))
    y2 = max(y1 + 2, min(y2, height))
    if (x2 - x1) % 2:
        x2 -= 1
    if (y2 - y1) % 2:
        y2 -= 1
    return x1, y1, x2, y2


def detect_crop(path: Path, args: argparse.Namespace) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bottom_crop = args.bottom_crop_pixels
    if bottom_crop is None:
        bottom_crop = int(round(height * args.bottom_crop_ratio))
    top_crop = args.top_crop_pixels
    if top_crop is None:
        top_crop = int(round(height * args.top_crop_ratio))
    top_crop = max(0, min(top_crop, height - 2))
    bottom_crop = max(0, min(bottom_crop, height - top_crop - 2))
    analysis_top = top_crop
    analysis_bottom = height - bottom_crop
    analysis_height = max(2, analysis_bottom - analysis_top)

    boxes = []
    for idx in sample_indices(frames, args.sample_frames):
        frame = read_frame(cap, int(idx))
        if frame is None:
            continue
        roi = frame[analysis_top:analysis_bottom, :, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = gray > args.black_threshold
        if mask.mean() < 0.001:
            continue

        rows = np.where(mask.mean(axis=1) >= args.min_content_fraction)[0]
        cols = np.where(mask.mean(axis=0) >= args.min_content_fraction)[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        boxes.append((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))

    cap.release()

    if boxes:
        x1 = min(box[0] for box in boxes) - args.padding
        y1 = analysis_top + min(box[1] for box in boxes) - args.padding
        x2 = max(box[2] for box in boxes) + args.padding
        y2 = analysis_top + max(box[3] for box in boxes) + args.padding
    else:
        x1, y1, x2, y2 = 0, analysis_top, width, analysis_bottom

    x1, y1, x2, y2 = even_crop(x1, y1, x2, y2, width, height)
    y1 = max(y1, analysis_top)
    y2 = min(y2, analysis_bottom)
    x1, y1, x2, y2 = even_crop(x1, y1, x2, y2, width, height)
    return {
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_frames": frames,
        "top_crop_pixels": top_crop,
        "bottom_crop_pixels": bottom_crop,
        "crop_x": x1,
        "crop_y": y1,
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
    }


def common_parent(paths: list[Path]) -> Path:
    parents = [str(p.resolve().parent) for p in paths]
    return Path(os.path.commonpath(parents))


def output_path_for(source: Path, common_root: Path, videos_dir: Path) -> Path:
    resolved = source.resolve()
    try:
        rel = resolved.relative_to(common_root)
    except ValueError:
        rel = Path(resolved.name)
    return (videos_dir / rel).with_suffix(".mp4")


def in_intervals(timestamp: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def process_video(
    source: Path,
    output: Path,
    crop: dict,
    args: argparse.Namespace,
    black_sections: list[tuple[float, float]] | None = None,
) -> int:
    if output.exists() and not args.overwrite:
        return -1

    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = args.output_fps or crop["source_fps"] or 60.0
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, (crop["crop_width"], crop["crop_height"]))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output}")

    frame_stride = 1
    if args.output_fps is not None and crop["source_fps"]:
        frame_stride = max(1, int(round(crop["source_fps"] / args.output_fps)))

    x1 = crop["crop_x"]
    y1 = crop["crop_y"]
    x2 = x1 + crop["crop_width"]
    y2 = y1 + crop["crop_height"]

    black_sections = black_sections or []
    read_frames = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = read_frames / crop["source_fps"] if crop["source_fps"] else 0.0
        if read_frames % frame_stride == 0 and not in_intervals(timestamp, black_sections):
            writer.write(frame[y1:y2, x1:x2, :])
            written += 1
        read_frames += 1
        if args.max_output_frames is not None and written >= args.max_output_frames:
            break

    writer.release()
    cap.release()
    return written


def merge_intervals(intervals: list[tuple[float, float]], duration: float | None = None) -> list[tuple[float, float]]:
    cleaned = []
    for start, end in intervals:
        start = max(0.0, float(start))
        end = max(start, float(end))
        if duration is not None:
            start = min(start, duration)
            end = min(end, duration)
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()

    merged = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else None
    return float(value)


def ffprobe_video(source: Path, args: argparse.Namespace) -> dict:
    if not shutil.which(args.ffprobe_bin):
        raise RuntimeError(f"ffprobe executable not found: {args.ffprobe_bin}")
    cmd = [
        args.ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,bit_rate,avg_frame_rate,r_frame_rate,width,height,nb_frames,duration:format=bit_rate,duration,size",
        "-of",
        "json",
        str(source),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def source_matched_encoding(source: Path, crop: dict, args: argparse.Namespace) -> dict:
    probe = ffprobe_video(source, args)
    stream = (probe.get("streams") or [{}])[0]
    fmt = probe.get("format") or {}

    duration = None
    for value in (stream.get("duration"), fmt.get("duration")):
        if value not in (None, "N/A"):
            duration = float(value)
            break
    if duration is None and crop["source_fps"]:
        duration = crop["source_frames"] / crop["source_fps"]

    source_bitrate = None
    for value in (stream.get("bit_rate"), fmt.get("bit_rate")):
        if value not in (None, "N/A"):
            source_bitrate = float(value)
            break
    if source_bitrate is None:
        size = fmt.get("size")
        if size not in (None, "N/A") and duration:
            source_bitrate = float(size) * 8.0 / duration
    if source_bitrate is None:
        raise RuntimeError(f"Could not determine source bitrate for source-matched encoding: {source}")

    source_width = int(stream.get("width") or crop["source_width"])
    source_height = int(stream.get("height") or crop["source_height"])
    source_fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")) or crop["source_fps"]
    output_fps = args.output_fps or source_fps

    source_area = max(1, source_width * source_height)
    output_area = max(1, crop["crop_width"] * crop["crop_height"])
    area_ratio = output_area / source_area
    fps_ratio = output_fps / source_fps if source_fps else 1.0
    target_bitrate = max(1, int(source_bitrate * area_ratio * fps_ratio * args.bitrate_scale))

    source_bpppf = source_bitrate / (source_area * source_fps) if source_fps else None
    target_bpppf = target_bitrate / (output_area * output_fps) if output_fps else None
    return {
        "source_codec": stream.get("codec_name"),
        "encoder": args.ffmpeg_encoder,
        "source_bitrate": int(source_bitrate),
        "target_bitrate": target_bitrate,
        "source_bits_per_pixel_frame": source_bpppf,
        "target_bits_per_pixel_frame": target_bpppf,
        "source_encoding_width": source_width,
        "source_encoding_height": source_height,
        "source_encoding_fps": source_fps,
        "target_encoding_fps": output_fps,
        "area_ratio": area_ratio,
        "fps_ratio": fps_ratio,
        "bitrate_scale": args.bitrate_scale,
    }


def detect_black_sections_ffmpeg(source: Path, crop: dict, args: argparse.Namespace) -> list[tuple[float, float]]:
    if not (args.remove_black_sections or args.save_black_sections):
        return []
    if args.black_detection_mode == "sampled":
        return detect_black_sections_sampled(source, crop, args)
    return detect_black_sections_full(source, crop, args)


def detect_black_sections_full(source: Path, crop: dict, args: argparse.Namespace) -> list[tuple[float, float]]:
    """Detect full-screen black intervals with ffmpeg blackdetect."""
    if not shutil.which(args.ffmpeg_bin):
        raise RuntimeError(f"ffmpeg executable not found: {args.ffmpeg_bin}")

    duration = None
    if crop["source_fps"]:
        duration = crop["source_frames"] / crop["source_fps"]

    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-hwaccel", "auto",             # Automatically select the best HW decoder
        "-i",
        str(source),
        "-vf",
        f"blackdetect=d={args.black_min_duration}:pix_th={args.black_pix_th}:pic_th={args.black_pic_th}",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg blackdetect failed for {source}:\n{proc.stderr}")

    intervals = []
    pattern = re.compile(
        r"black_start:(?P<start>[0-9.]+)\s+black_end:(?P<end>[0-9.]+)\s+black_duration:(?P<duration>[0-9.]+)"
    )
    for match in pattern.finditer(proc.stderr):
        start = float(match.group("start")) - args.black_section_padding
        end = float(match.group("end")) + args.black_section_padding
        intervals.append((start, end))
    return merge_intervals(intervals, duration=duration)


def detect_black_sections_sampled(source: Path, crop: dict, args: argparse.Namespace) -> list[tuple[float, float]]:
    """Detect black intervals by sampling tiny grayscale frames instead of scanning every frame."""
    if not shutil.which(args.ffmpeg_bin):
        raise RuntimeError(f"ffmpeg executable not found: {args.ffmpeg_bin}")

    duration = None
    if crop["source_fps"]:
        duration = crop["source_frames"] / crop["source_fps"]

    intervals = sampled_black_pass(
        source=source,
        args=args,
        fps=args.black_sample_fps,
        start_time=0.0,
        duration=None,
        min_duration=args.black_min_duration,
        source_width=crop["source_width"],
        source_height=crop["source_height"],
    )

    if args.black_refine_boundaries and intervals:
        refined = []
        for start, end in intervals:
            scan_start = max(0.0, start - args.black_refine_window)
            scan_end = end + args.black_refine_window
            if duration is not None:
                scan_end = min(duration, scan_end)
            candidates = sampled_black_pass(
                source=source,
                args=args,
                fps=args.black_refine_fps,
                start_time=scan_start,
                duration=max(0.0, scan_end - scan_start),
                min_duration=0.0,
                source_width=crop["source_width"],
                source_height=crop["source_height"],
            )
            overlapping = [
                (cand_start, cand_end)
                for cand_start, cand_end in candidates
                if cand_end > start and cand_start < end
            ]
            refined.extend(overlapping or [(start, end)])
        intervals = merge_intervals(refined, duration=duration)

    padded = [
        (start - args.black_section_padding, end + args.black_section_padding)
        for start, end in intervals
        if end - start >= args.black_min_duration
    ]
    return merge_intervals(padded, duration=duration)


def sampled_black_pass(
    source: Path,
    args: argparse.Namespace,
    fps: float,
    start_time: float = 0.0,
    duration: float | None = None,
    min_duration: float = 0.0,
    source_width: int | None = None,
    source_height: int | None = None,
) -> list[tuple[float, float]]:
    if args.black_sample_filter_backend == "cvcuda":
        return sampled_black_pass_cvcuda(
            source=source,
            args=args,
            fps=fps,
            start_time=start_time,
            duration=duration,
            min_duration=min_duration,
            source_width=source_width,
            source_height=source_height,
        )

    width = int(args.black_sample_width)
    height = int(args.black_sample_height)
    frame_size = width * height
    if fps <= 0:
        raise ValueError("--black-sample-fps and --black-refine-fps must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("--black-sample-width and --black-sample-height must be positive")

    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if args.black_sample_filter_backend == "cuda":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif args.ffmpeg_hwaccel != "none":
        cmd += ["-hwaccel", args.ffmpeg_hwaccel]
    if start_time > 0:
        cmd += ["-ss", f"{start_time:.6f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    if args.black_sample_filter_backend == "cuda":
        video_filter = f"scale_cuda={width}:{height},hwdownload,format=gray,fps={fps:g}"
    else:
        video_filter = f"fps={fps:g},scale={width}:{height}:flags=fast_bilinear,format=gray"

    cmd += [
        "-i",
        str(source),
        "-vf",
        video_filter,
        "-an",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"sampled black detection failed for {source}:\n{proc.stderr.decode(errors='replace')}")

    if not proc.stdout:
        return []
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    num_frames = raw.size // frame_size
    if num_frames == 0:
        return []
    raw = raw[: num_frames * frame_size]
    frames = raw.reshape(num_frames, frame_size)
    black_fraction = (frames <= int(args.black_sample_threshold)).mean(axis=1)
    is_black = black_fraction >= float(args.black_pic_th)

    intervals = []
    current_start = None
    sample_period = 1.0 / fps
    for i, black in enumerate(is_black):
        t = start_time + i * sample_period
        if black and current_start is None:
            current_start = t
        elif not black and current_start is not None:
            end = t
            if end - current_start >= min_duration:
                intervals.append((current_start, end))
            current_start = None
    if current_start is not None:
        end = start_time + num_frames * sample_period
        if end - current_start >= min_duration:
            intervals.append((current_start, end))
    return intervals


def _read_exact(pipe, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def sampled_black_pass_cvcuda(
    source: Path,
    args: argparse.Namespace,
    fps: float,
    start_time: float = 0.0,
    duration: float | None = None,
    min_duration: float = 0.0,
    source_width: int | None = None,
    source_height: int | None = None,
) -> list[tuple[float, float]]:
    if source_width is None or source_height is None:
        raise ValueError("CVCUDA sampled detection requires source_width and source_height")
    if fps <= 0:
        raise ValueError("--black-sample-fps and --black-refine-fps must be positive")

    try:
        import torch
        import cvcuda
    except Exception as exc:
        raise RuntimeError("CVCUDA backend requested but cvcuda/torch could not be imported") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CVCUDA backend requested but torch.cuda.is_available() is false")

    out_width = int(args.black_sample_width)
    out_height = int(args.black_sample_height)
    if out_width <= 0 or out_height <= 0:
        raise ValueError("--black-sample-width and --black-sample-height must be positive")

    frame_size = int(source_width) * int(source_height) * 3
    batch_size = max(1, int(args.black_cvcuda_batch_size))
    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if args.ffmpeg_hwaccel != "none":
        cmd += ["-hwaccel", args.ffmpeg_hwaccel]
    if start_time > 0:
        cmd += ["-ss", f"{start_time:.6f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-i",
        str(source),
        "-vf",
        f"fps={fps:g},format=rgb24",
        "-an",
        "-f",
        "rawvideo",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    intervals = []
    current_start = None
    sample_period = 1.0 / fps
    frame_offset = 0

    try:
        while True:
            chunk = _read_exact(proc.stdout, frame_size * batch_size)
            if not chunk:
                break
            if len(chunk) % frame_size != 0:
                raise RuntimeError(f"partial raw frame from ffmpeg while reading {source}")
            num_frames = len(chunk) // frame_size
            frames_np = np.frombuffer(chunk, dtype=np.uint8).reshape(
                num_frames, int(source_height), int(source_width), 3
            )
            frames_torch = torch.as_tensor(frames_np, device="cuda")
            cvcuda_in = cvcuda.as_tensor(frames_torch, "NHWC")
            resized = cvcuda.resize(cvcuda_in, (num_frames, out_height, out_width, 3), cvcuda.Interp.LINEAR)
            resized_torch = torch.utils.dlpack.from_dlpack(resized)
            # Integer luma approximation: Y ~= 0.299R + 0.587G + 0.114B.
            rgb = resized_torch.to(torch.int16)
            gray = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8
            black_fraction = (gray <= int(args.black_sample_threshold)).float().mean(dim=(1, 2))
            is_black = (black_fraction >= float(args.black_pic_th)).detach().cpu().numpy().astype(bool)

            for i, black in enumerate(is_black):
                t = start_time + (frame_offset + i) * sample_period
                if black and current_start is None:
                    current_start = t
                elif not black and current_start is not None:
                    end = t
                    if end - current_start >= min_duration:
                        intervals.append((current_start, end))
                    current_start = None
            frame_offset += num_frames
    finally:
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(f"CVCUDA sampled black detection ffmpeg failed for {source}:\n{stderr.decode(errors='replace')}")
    if current_start is not None:
        end = start_time + frame_offset * sample_period
        if end - current_start >= min_duration:
            intervals.append((current_start, end))
    return intervals


def ffmpeg_select_non_black_filter(black_sections: list[tuple[float, float]]) -> str | None:
    if not black_sections:
        return None
    terms = [f"between(t\\,{start:.6f}\\,{end:.6f})" for start, end in black_sections]
    return f"select='not({'+'.join(terms)})'"


def nvenc_gpu_for_index(args: argparse.Namespace, idx: int) -> str | None:
    if not args.nvenc_gpus:
        return None
    gpus = [gpu.strip() for gpu in args.nvenc_gpus.split(",") if gpu.strip()]
    if not gpus:
        return None
    return gpus[idx % len(gpus)]


def black_section_output_path(output: Path, args: argparse.Namespace, section_idx: int, start: float, end: float) -> Path:
    videos_root = (args.out_dir / "videos").resolve()
    try:
        rel_output = output.resolve().relative_to(videos_root)
    except ValueError:
        rel_output = Path(output.name)

    start_ms = int(round(start * 1000))
    end_ms = int(round(end * 1000))
    filename = f"{rel_output.stem}_black_{section_idx:03d}_{start_ms:010d}ms_{end_ms:010d}ms.mp4"
    return args.black_sections_dir / rel_output.parent / filename


def save_black_sections_ffmpeg(
    idx: int,
    source: Path,
    output: Path,
    args: argparse.Namespace,
    black_sections: list[tuple[float, float]],
) -> list[dict]:
    if not args.save_black_sections or not black_sections:
        return []
    if not shutil.which(args.ffmpeg_bin):
        raise RuntimeError(f"ffmpeg executable not found: {args.ffmpeg_bin}")

    saved = []
    for section_idx, (start, end) in enumerate(black_sections):
        duration = max(0.0, end - start)
        if duration <= 0:
            continue

        section_output = black_section_output_path(output, args, section_idx, start, end)
        section_output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if args.overwrite else "-n",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
        ]
        if args.black_section_save_mode == "copy":
            cmd += ["-c:v", "copy"]
        else:
            cmd += [
                "-c:v",
                args.ffmpeg_encoder,
            ]
            if args.ffmpeg_encoder == "libx264":
                cmd += ["-preset", args.ffmpeg_preset, "-crf", str(args.crf)]
            else:
                cmd += ["-preset", args.nvenc_preset]
                nvenc_gpu = nvenc_gpu_for_index(args, idx)
                if nvenc_gpu is not None:
                    cmd += ["-gpu", nvenc_gpu]
                cmd += ["-rc", "vbr", "-cq:v", str(args.crf)]
            cmd += ["-pix_fmt", "yuv420p"]

        cmd += [str(section_output)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed while saving black section {section_idx} from {source}") from exc

        saved.append(
            {
                "index": section_idx,
                "start": start,
                "end": end,
                "duration": duration,
                "path": str(section_output),
                "mode": args.black_section_save_mode,
            }
        )
    return saved


def process_video_ffmpeg(
    idx: int,
    source: Path,
    output: Path,
    crop: dict,
    args: argparse.Namespace,
    black_sections: list[tuple[float, float]] | None = None,
) -> tuple[int, dict]:
    if output.exists() and not args.overwrite:
        return -1, {}

    output.parent.mkdir(parents=True, exist_ok=True)
    crop_filter = (
        f"crop={crop['crop_width']}:{crop['crop_height']}:"
        f"{crop['crop_x']}:{crop['crop_y']}"
    )
    black_sections = black_sections or []
    video_filters = []
    select_filter = ffmpeg_select_non_black_filter(black_sections)
    if select_filter is not None:
        video_filters += [select_filter, "setpts=N/FRAME_RATE/TB"]
    if args.output_fps is not None:
        video_filters += [f"fps={args.output_fps:g}"]
    video_filters += [crop_filter]
    encoding_info = {}
    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
    ]
    if args.ffmpeg_hwaccel != "none":
        cmd += ["-hwaccel", args.ffmpeg_hwaccel]
    cmd += [
        "-i",
        str(source),
        "-vf",
        ",".join(video_filters),
        "-an",
        "-c:v",
        args.ffmpeg_encoder,
    ]
    if args.ffmpeg_encoder == "libx264":
        cmd += ["-preset", args.ffmpeg_preset]
    else:
        cmd += ["-preset", args.nvenc_preset]
        nvenc_gpu = nvenc_gpu_for_index(args, idx)
        if nvenc_gpu is not None:
            cmd += ["-gpu", nvenc_gpu]

    if args.quality_mode == "source":
        encoding_info = source_matched_encoding(source, crop, args)
        if args.ffmpeg_encoder != "libx264":
            encoding_info["encoder_gpu"] = nvenc_gpu_for_index(args, idx)
        target_bitrate = encoding_info["target_bitrate"]
        if args.ffmpeg_encoder != "libx264":
            cmd += ["-rc", "vbr"]
        cmd += [
            "-b:v",
            str(target_bitrate),
            "-maxrate",
            str(target_bitrate),
            "-bufsize",
            str(max(target_bitrate * 2, 1)),
        ]
    elif args.quality_mode == "lossless":
        if args.ffmpeg_encoder != "libx264":
            raise RuntimeError("--quality-mode lossless is only supported with --ffmpeg-encoder libx264")
        cmd += ["-qp", "0"]
    else:
        if args.ffmpeg_encoder == "libx264":
            cmd += ["-crf", str(args.crf)]
        else:
            cmd += ["-rc", "vbr", "-cq:v", str(args.crf)]
    cmd += [
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
    ]
    if args.faststart:
        cmd += ["-movflags", "+faststart"]
    if args.ffmpeg_threads > 0:
        cmd += ["-threads", str(args.ffmpeg_threads)]
    cmd.append(str(output))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {source}") from exc

    cap = cv2.VideoCapture(str(output))
    written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return written, encoding_info


def make_preview(source: Path, output: Path, crop: dict, preview_path: Path, frame_count: int) -> None:
    cap_src = cv2.VideoCapture(str(source))
    cap_out = cv2.VideoCapture(str(output))
    total = int(min(cap_src.get(cv2.CAP_PROP_FRAME_COUNT), cap_out.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = sample_indices(total, frame_count)
    src_frames, out_frames = [], []

    for idx in indices:
        src = read_frame(cap_src, int(idx))
        out = read_frame(cap_out, int(idx))
        if src is None or out is None:
            continue
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (320, 180), interpolation=cv2.INTER_AREA)
        out_h = max(1, int(round(320 * out.shape[0] / out.shape[1])))
        out = cv2.resize(out, (320, out_h), interpolation=cv2.INTER_AREA)
        src_frames.append(src)
        out_frames.append(out)

    cap_src.release()
    cap_out.release()
    if not src_frames:
        return

    row_h = max(180, max(frame.shape[0] for frame in out_frames)) + 22
    canvas = Image.new("RGB", (320 * len(src_frames), row_h * 2), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, (src, out) in enumerate(zip(src_frames, out_frames)):
        x = i * 320
        canvas.paste(Image.fromarray(src), (x, 0))
        canvas.paste(Image.fromarray(out), (x, row_h))
        draw.text((x + 4, 183), f"before frame {int(indices[i])}", fill=(235, 235, 235))
        draw.text((x + 4, row_h + out.shape[0] + 3), "after", fill=(235, 235, 235))

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path, quality=95)


def args_for_json(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def normalize_source_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def metadata_key(item: dict) -> tuple[int, str]:
    return int(item["index"]), normalize_source_path(item.get("source_path", ""))


def load_existing_metadata(metadata_path: Path, selected_sources: set[str]) -> dict[tuple[int, str], dict]:
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open() as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not resume because metadata is not valid JSON: {metadata_path}") from exc

    existing = {}
    for item in doc.get("videos", []):
        try:
            key = metadata_key(item)
        except (KeyError, TypeError, ValueError):
            continue
        if key[1] in selected_sources:
            existing[key] = item
    return existing


def build_preprocess_metadata_doc(
    args: argparse.Namespace,
    output_manifest: Path,
    videos_dir: Path,
    black_sections_dir: Path,
    root: Path,
    use_ffmpeg: bool,
    metadata: list[dict],
    main_timings: dict[str, float],
) -> dict:
    aggregate_timings: dict[str, float] = {}
    for item in metadata:
        for key, value in item.get("timings_seconds", {}).items():
            aggregate_timings[key] = aggregate_timings.get(key, 0.0) + float(value)

    return {
        "input_manifest": str(args.manifest),
        "output_manifest": str(output_manifest.resolve()),
        "videos_dir": str(videos_dir.resolve()),
        "black_sections_dir": str(black_sections_dir),
        "common_source_root": str(root),
        "num_videos": len(metadata),
        "backend": "ffmpeg" if use_ffmpeg else "opencv",
        "settings": args_for_json(args),
        "timings_seconds": {
            "main": round_timings(main_timings),
            "aggregate_video_steps": round_timings(aggregate_timings),
        },
        "videos": metadata,
    }


def checkpoint_outputs(
    args: argparse.Namespace,
    metadata_path: Path,
    output_manifest: Path,
    videos_dir: Path,
    black_sections_dir: Path,
    root: Path,
    use_ffmpeg: bool,
    metadata_by_key: dict[tuple[int, str], dict],
    main_timings: dict[str, float],
) -> list[dict]:
    metadata = sorted(metadata_by_key.values(), key=lambda item: (int(item["index"]), str(item.get("source_path", ""))))
    processed_rows = [(Path(item["output_path"]), item["label"]) for item in metadata]
    write_manifest(output_manifest, processed_rows)
    atomic_write_json(
        metadata_path,
        build_preprocess_metadata_doc(
            args=args,
            output_manifest=output_manifest,
            videos_dir=videos_dir,
            black_sections_dir=black_sections_dir,
            root=root,
            use_ffmpeg=use_ffmpeg,
            metadata=metadata,
            main_timings=main_timings,
        ),
    )
    return metadata


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend == "auto":
        return "ffmpeg" if shutil.which(args.ffmpeg_bin) else "opencv"
    if args.backend == "ffmpeg" and not shutil.which(args.ffmpeg_bin):
        raise SystemExit(f"ffmpeg backend requested but executable not found: {args.ffmpeg_bin}")
    return args.backend


def process_one(task: tuple[int, str, str, str, str, argparse.Namespace, bool]) -> dict:
    idx, source_str, label, output_str, preview_str, args, use_ffmpeg = task
    source = Path(source_str)
    output = Path(output_str)
    timings: dict[str, float] = {}
    logger.info("video %d start: %s", idx, source)
    with timed_step(timings, "total"):
        with timed_step(timings, "detect_crop"):
            crop = detect_crop(source, args)
        logger.info(
            "video %d crop: %dx%d -> %dx%d at x=%d y=%d top=%d bottom=%d in %.2fs",
            idx,
            crop["source_width"],
            crop["source_height"],
            crop["crop_width"],
            crop["crop_height"],
            crop["crop_x"],
            crop["crop_y"],
            crop["top_crop_pixels"],
            crop["bottom_crop_pixels"],
            timings["detect_crop"],
        )

        with timed_step(timings, "detect_black_sections"):
            black_sections = detect_black_sections_ffmpeg(source, crop, args)
        if black_sections:
            logger.info(
                "video %d black sections: %d intervals detected, %.2fs total, detection %.2fs",
                idx,
                len(black_sections),
                sum(end - start for start, end in black_sections),
                timings["detect_black_sections"],
            )
        else:
            logger.info("video %d black sections: none, detection %.2fs", idx, timings["detect_black_sections"])

        with timed_step(timings, "save_black_sections"):
            saved_black_sections = save_black_sections_ffmpeg(idx, source, output, args, black_sections)
        if saved_black_sections:
            logger.info(
                "video %d saved black sections: %d clips in %.2fs",
                idx,
                len(saved_black_sections),
                timings["save_black_sections"],
            )

        cut_black_sections = black_sections if (args.remove_black_sections and not args.metadata_only) else []
        with timed_step(timings, "encode"):
            if args.metadata_only:
                written = crop["source_frames"]
                encoding_info = {}
                backend_used = "metadata_only"
            elif use_ffmpeg and args.max_output_frames is None:
                written, encoding_info = process_video_ffmpeg(
                    idx,
                    source,
                    output,
                    crop,
                    args,
                    black_sections=cut_black_sections,
                )
                backend_used = "ffmpeg"
            else:
                written = process_video(source, output, crop, args, black_sections=cut_black_sections)
                encoding_info = {}
                backend_used = "opencv"
        logger.info("video %d encode: backend=%s frames=%s in %.2fs", idx, backend_used, written, timings["encode"])

        if written < 0:
            with timed_step(timings, "probe_existing_output"):
                cap = cv2.VideoCapture(str(output))
                written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

        item = {
            "index": idx,
            "source_path": str(source),
            "output_path": str(source if args.metadata_only else output),
            "label": label,
            "written_frames": written,
            "backend": backend_used,
            "output_fps": args.output_fps or crop["source_fps"],
            "quality_mode": args.quality_mode if backend_used == "ffmpeg" else backend_used,
            "encoding": encoding_info,
            "black_sections_detected": [{"start": start, "end": end} for start, end in black_sections],
            "black_sections_removed": [{"start": start, "end": end} for start, end in cut_black_sections],
            "black_sections_saved": saved_black_sections,
            "black_duration_removed": sum(end - start for start, end in cut_black_sections),
            **crop,
        }
        if preview_str and not args.metadata_only:
            with timed_step(timings, "preview"):
                preview = Path(preview_str)
                make_preview(source, output, crop, preview, args.preview_frames)
                item["preview_path"] = str(preview)
            logger.info("video %d preview: %s in %.2fs", idx, preview, timings["preview"])

    item["timings_seconds"] = round_timings(timings)
    logger.info(
        "video %d done: total=%.2fs detect_crop=%.2fs blackdetect=%.2fs encode=%.2fs",
        idx,
        timings.get("total", 0.0),
        timings.get("detect_crop", 0.0),
        timings.get("detect_black_sections", 0.0),
        timings.get("encode", 0.0),
    )
    return item


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    main_timings: dict[str, float] = {}
    with timed_step(main_timings, "total"):
        with timed_step(main_timings, "setup"):
            args.out_dir.mkdir(parents=True, exist_ok=True)
            videos_dir = args.out_dir / "videos"
            if args.black_sections_dir is None:
                args.black_sections_dir = args.out_dir / "black_sections"
            args.black_sections_dir = args.black_sections_dir.resolve()
            output_manifest = args.output_manifest or (args.out_dir / args.manifest.name)
            metadata_path = args.out_dir / "preprocess_metadata.json"

            rows = read_manifest(args.manifest)
            if args.limit is not None:
                rows = rows[: args.limit]
            if not rows:
                raise SystemExit(f"No rows found in manifest: {args.manifest}")

            root = common_parent([path for path, _ in rows])
            backend = resolve_backend(args)
            use_ffmpeg = backend == "ffmpeg"
            if args.max_output_frames is not None and use_ffmpeg:
                logger.info("max-output-frames is set; using OpenCV backend for debug frame limiting.")
                use_ffmpeg = False

            selected_sources = {normalize_source_path(source) for source, _ in rows}
            existing_metadata = {}
            if not args.no_resume:
                existing_metadata = load_existing_metadata(metadata_path, selected_sources)
                if existing_metadata:
                    logger.info("resume enabled: loaded %d completed metadata entries", len(existing_metadata))

            tasks = []
            for idx, (source, label) in enumerate(rows):
                resume_key = (idx, normalize_source_path(source))
                if resume_key in existing_metadata:
                    continue
                output = output_path_for(source, root, videos_dir).resolve()
                preview = args.out_dir / "previews" / f"{output.stem}_preview.jpg" if idx < args.preview_count else None
                tasks.append((idx, str(source), label, str(output), str(preview) if preview else "", args, use_ffmpeg))

            metadata_by_key = dict(existing_metadata)
            if metadata_by_key:
                checkpoint_outputs(
                    args=args,
                    metadata_path=metadata_path,
                    output_manifest=output_manifest,
                    videos_dir=videos_dir,
                    black_sections_dir=args.black_sections_dir,
                    root=root,
                    use_ffmpeg=use_ffmpeg,
                    metadata_by_key=metadata_by_key,
                    main_timings=main_timings,
                )

        logger.info(
            "preprocessing start: pending=%d completed=%d backend=%s workers=%d output_fps=%s remove_black=%s save_black=%s",
            len(tasks),
            len(metadata_by_key),
            "ffmpeg" if use_ffmpeg else "opencv",
            max(1, args.workers),
            args.output_fps,
            args.remove_black_sections,
            args.save_black_sections,
        )

        max_workers = max(1, args.workers)
        with timed_step(main_timings, "process_videos"):
            if max_workers == 1:
                for task in tqdm(tasks, total=len(tasks)):
                    item = process_one(task)
                    metadata_by_key[metadata_key(item)] = item
                    checkpoint_outputs(
                        args=args,
                        metadata_path=metadata_path,
                        output_manifest=output_manifest,
                        videos_dir=videos_dir,
                        black_sections_dir=args.black_sections_dir,
                        root=root,
                        use_ffmpeg=use_ffmpeg,
                        metadata_by_key=metadata_by_key,
                        main_timings=main_timings,
                    )
                    print(json.dumps(item), flush=True)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_one, task) for task in tasks]
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        item = future.result()
                        metadata_by_key[metadata_key(item)] = item
                        checkpoint_outputs(
                            args=args,
                            metadata_path=metadata_path,
                            output_manifest=output_manifest,
                            videos_dir=videos_dir,
                            black_sections_dir=args.black_sections_dir,
                            root=root,
                            use_ffmpeg=use_ffmpeg,
                            metadata_by_key=metadata_by_key,
                            main_timings=main_timings,
                        )
                        print(json.dumps(item), flush=True)

        with timed_step(main_timings, "write_outputs"):
            metadata = checkpoint_outputs(
                args=args,
                metadata_path=metadata_path,
                output_manifest=output_manifest,
                videos_dir=videos_dir,
                black_sections_dir=args.black_sections_dir,
                root=root,
                use_ffmpeg=use_ffmpeg,
                metadata_by_key=metadata_by_key,
                main_timings=main_timings,
            )

    logger.info(
        "preprocessing done: videos=%d total=%.2fs process_videos=%.2fs write_outputs=%.2fs",
        len(metadata),
        main_timings.get("total", 0.0),
        main_timings.get("process_videos", 0.0),
        main_timings.get("write_outputs", 0.0),
    )
    print(
        json.dumps(
            {
                "output_manifest": str(output_manifest.resolve()),
                "num_videos": len(metadata),
                "timings_seconds": round_timings(main_timings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
