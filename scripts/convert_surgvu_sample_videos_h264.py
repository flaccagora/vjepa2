#!/usr/bin/env python3
"""Convert SurgVU sample videos to H.264 while preserving directory layout.

Default usage:

    python3 scripts/convert_surgvu_sample_videos_h264.py

This reads videos from data/surgvu/sample_videos and writes converted videos to
data/surgvu/sample_video_h264 with the same relative paths.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEFAULT_INPUT_ROOT = Path("data/surgvu/sample_videos")
DEFAULT_OUTPUT_ROOT = Path("data/surgvu/sample_video_h264")


@dataclass(frozen=True)
class ConversionTask:
    source: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Directory containing SurgVU sample videos. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for converted videos. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable.")
    parser.add_argument("--encoder", default="libx264", help="H.264 encoder, e.g. libx264 or h264_nvenc.")
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="Quality target. Lower is higher quality/larger files; 28 is streaming-friendly.",
    )
    parser.add_argument("--preset", default="veryfast", help="libx264 preset.")
    parser.add_argument(
        "--nvenc-preset",
        default="p4",
        help="NVENC preset when --encoder is *_nvenc (p1 fastest, p7 highest quality).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Limit output width for streaming (0 disables the limit).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Limit output height for streaming (0 disables the limit).",
    )
    parser.add_argument(
        "--half-res",
        action="store_true",
        help="Scale videos to half resolution before applying max size limits.",
    )
    parser.add_argument(
        "--quarter-res",
        action="store_true",
        help="Scale videos to quarter resolution before applying max size limits.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output frames per second (0 keeps source FPS).",
    )
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        help="Output pixel format. yuv420p is broadly compatible for H.264 MP4.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of videos to convert in parallel.")
    parser.add_argument("--limit", type=int, help="Convert only the first N videos after sorting.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite converted videos if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without creating output files.")
    return parser.parse_args()


def require_ffmpeg(ffmpeg_bin: str) -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise SystemExit(f"ffmpeg executable not found: {ffmpeg_bin}")


def resolve_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise SystemExit(f"Input root is not a directory: {input_root}")
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise SystemExit("Output root must not be the input root or a child of it.")

    return input_root, output_root


def mirror_directories(input_root: Path, output_root: Path) -> None:
    for directory in sorted(p for p in input_root.rglob("*") if p.is_dir()):
        (output_root / directory.relative_to(input_root)).mkdir(parents=True, exist_ok=True)


def discover_videos(input_root: Path, output_root: Path, limit: int | None) -> list[ConversionTask]:
    videos = sorted(
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and "__MACOSX" not in p.parts
    )
    if limit is not None:
        videos = videos[:limit]

    return [ConversionTask(source=source, output=output_root / source.relative_to(input_root)) for source in videos]


def normalize_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def normalize_fps(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return value


def resolve_scale_factor(half_res: bool, quarter_res: bool) -> float | None:
    if half_res and quarter_res:
        raise SystemExit("--half-res and --quarter-res are mutually exclusive")
    if quarter_res:
        return 0.25
    if half_res:
        return 0.5
    return None


def build_scale_filter(
    max_width: int | None,
    max_height: int | None,
    scale_factor: float | None,
) -> str | None:
    max_width = normalize_limit(max_width)
    max_height = normalize_limit(max_height)
    if max_width is None and max_height is None and scale_factor is None:
        return None

    if scale_factor is None:
        base_width_expr = "iw"
        base_height_expr = "ih"
    else:
        scale_expr = f"{scale_factor:g}"
        base_width_expr = f"iw*{scale_expr}"
        base_height_expr = f"ih*{scale_expr}"
    max_width_expr = f"min({base_width_expr},{max_width})" if max_width is not None else base_width_expr
    max_height_expr = f"min({base_height_expr},{max_height})" if max_height is not None else base_height_expr
    return (
        "scale="
        f"'{max_width_expr}':'{max_height_expr}'"
        ":force_original_aspect_ratio=decrease"
    )


def build_filter_chain(args: argparse.Namespace) -> str | None:
    filters: list[str] = []

    fps_value = normalize_fps(args.fps)
    if fps_value is not None:
        filters.append(f"fps={fps_value:g}")

    scale_factor = resolve_scale_factor(args.half_res, args.quarter_res)
    scale_filter = build_scale_filter(args.max_width, args.max_height, scale_factor)
    if scale_filter:
        filters.append(scale_filter)

    if args.pix_fmt:
        filters.append(f"format={args.pix_fmt}")

    if not filters:
        return None
    return ",".join(filters)


def ffmpeg_command(task: ConversionTask, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y" if args.overwrite else "-n",
        "-i",
        str(task.source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        args.encoder,
    ]

    if args.encoder == "libx264":
        cmd += ["-preset", args.preset, "-crf", str(args.crf)]
    elif args.encoder.endswith("_nvenc"):
        cmd += ["-preset", args.nvenc_preset, "-rc", "vbr", "-cq", str(args.crf)]

    filter_chain = build_filter_chain(args)
    if filter_chain:
        cmd += ["-vf", filter_chain]

    cmd += [
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(task.output),
    ]
    return cmd


def convert_one(task: ConversionTask, args: argparse.Namespace) -> tuple[ConversionTask, bool, str]:
    if task.output.exists() and not args.overwrite:
        return task, False, "exists"

    task.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ffmpeg_command(task, args)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return task, False, proc.stderr.strip() or f"ffmpeg exited with code {proc.returncode}"

    return task, True, "converted"


def print_plan(tasks: list[ConversionTask], input_root: Path, output_root: Path) -> None:
    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Videos:      {len(tasks)}")
    for task in tasks[:10]:
        print(f"  {task.source.relative_to(input_root)} -> {task.output.relative_to(output_root)}")
    if len(tasks) > 10:
        print(f"  ... {len(tasks) - 10} more")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    input_root, output_root = resolve_roots(args.input_root, args.output_root)
    tasks = discover_videos(input_root, output_root, args.limit)
    if not tasks:
        raise SystemExit(f"No videos with extensions {sorted(VIDEO_EXTENSIONS)} found under {input_root}")

    print_plan(tasks, input_root, output_root)
    if args.dry_run:
        return

    require_ffmpeg(args.ffmpeg_bin)
    output_root.mkdir(parents=True, exist_ok=True)
    mirror_directories(input_root, output_root)

    converted = 0
    skipped = 0
    failures: list[tuple[ConversionTask, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(convert_one, task, args): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task, did_convert, status = future.result()
            relpath = task.source.relative_to(input_root)
            if did_convert:
                converted += 1
                print(f"[converted] {relpath}")
            elif status == "exists":
                skipped += 1
                print(f"[skipped]   {relpath}")
            else:
                failures.append((task, status))
                print(f"[failed]    {relpath}")

    print(f"Done. converted={converted} skipped={skipped} failed={len(failures)}")
    if failures:
        print("\nFailures:")
        for task, message in failures:
            print(f"\n{task.source}")
            print(message)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
