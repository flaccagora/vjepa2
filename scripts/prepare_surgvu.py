#!/usr/bin/env python3
"""Prepare SurgVU videos for V-JEPA2.1 self-supervised fine-tuning.

The V-JEPA VideoDataset expects a space-delimited manifest with two columns:

    /absolute/path/to/video.mp4 0

The numeric label is not used by the self-supervised training loop, but keeping
it deterministic makes the manifest compatible with the existing dataset class.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/surgvu"), help="Output directory.")
    parser.add_argument("--videos-root", type=Path, help="Directory containing extracted SurgVU videos.")
    parser.add_argument("--video-zip", type=Path, help="Optional SurgVU video zip to extract and index.")
    parser.add_argument(
        "--sample-zip",
        type=Path,
        default=Path("data/SURGVU25_cat_2_sample_set_public.zip"),
        help="Optional public sample zip used when --videos-root/--video-zip are not supplied.",
    )
    parser.add_argument(
        "--labels-zip",
        type=Path,
        default=Path("data/surgvu24_labels_updated_v2.zip"),
        help="Optional SurgVU labels zip to extract and summarize.",
    )
    parser.add_argument("--labels-root", type=Path, help="Directory containing extracted labels/case_*/tasks.csv files.")
    parser.add_argument("--manifest-prefix", default="manifest", help="Manifest filename prefix.")
    parser.add_argument("--limit", type=int, default=None, help="Limit indexed videos after deterministic shuffling.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of videos assigned to validation.")
    parser.add_argument("--seed", type=int, default=239, help="Shuffle seed.")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos into out-dir/videos instead of indexing in place.")
    parser.add_argument("--force", action="store_true", help="Overwrite extracted directories if needed.")
    return parser.parse_args()


def normalize_case_id(value: str) -> str | None:
    match = re.search(r"case[_-]?(\d+)", value.lower())
    if match:
        return f"case_{int(match.group(1)):03d}"
    return None


def extract_zip(zip_path: Path, destination: Path, force: bool = False) -> Path:
    if destination.exists() and force:
        shutil.rmtree(destination)
    if destination.exists() and any(destination.iterdir()):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destination)
    return destination


def discover_videos(root: Path) -> list[Path]:
    return sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and "__MACOSX" not in p.parts
    )


def read_task_labels(labels_root: Path | None) -> tuple[dict[str, str], dict[str, int]]:
    if labels_root is None or not labels_root.exists():
        return {}, {}

    task_by_case: dict[str, str] = {}
    task_names: set[str] = set()
    for task_file in sorted(labels_root.rglob("tasks.csv")):
        case_id = normalize_case_id(str(task_file))
        if not case_id:
            continue
        with task_file.open(newline="") as f:
            reader = csv.DictReader(f)
            tasks = [row.get("groundtruth_taskname", "").strip() for row in reader]
        tasks = [task for task in tasks if task]
        if tasks:
            dominant_task = Counter(tasks).most_common(1)[0][0]
            task_by_case[case_id] = dominant_task
            task_names.update(tasks)

    label_map = {name: idx for idx, name in enumerate(sorted(task_names))}
    return task_by_case, label_map


def maybe_copy_videos(videos: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in videos:
        case_id = normalize_case_id(str(src)) or src.stem
        dst = destination / f"{case_id}{src.suffix.lower()}"
        if dst.exists():
            stem = dst.stem
            counter = 1
            while dst.exists():
                dst = destination / f"{stem}_{counter}{src.suffix.lower()}"
                counter += 1
        shutil.copy2(src, dst)
        copied.append(dst.resolve())
    return copied


def write_manifest(path: Path, rows: list[tuple[Path, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        for video_path, label in rows:
            writer.writerow([str(video_path), label])


def write_metadata(path: Path, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_root = args.labels_root
    if labels_root is None and args.labels_zip and args.labels_zip.exists():
        labels_root = extract_zip(args.labels_zip, args.out_dir / "labels", force=args.force)

    videos_root = args.videos_root
    if videos_root is None and args.video_zip is not None:
        videos_root = extract_zip(args.video_zip, args.out_dir / "videos_extracted", force=args.force)
    elif videos_root is None and args.sample_zip is not None and args.sample_zip.exists():
        videos_root = extract_zip(args.sample_zip, args.out_dir / "sample_videos", force=args.force)

    if videos_root is None:
        raise SystemExit("No videos found. Provide --videos-root, --video-zip, or --sample-zip.")
    if not videos_root.exists():
        raise SystemExit(f"Video root does not exist: {videos_root}")

    videos = discover_videos(videos_root)
    if not videos:
        raise SystemExit(f"No videos with extensions {sorted(VIDEO_EXTENSIONS)} found under {videos_root}")

    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.limit is not None:
        videos = videos[: args.limit]
    videos = sorted(videos)
    if args.copy_videos:
        videos = maybe_copy_videos(videos, args.out_dir / "videos")

    task_by_case, label_map = read_task_labels(labels_root)
    unknown_label = len(label_map)

    rows: list[tuple[Path, int]] = []
    metadata: list[dict[str, str | int]] = []
    for video in videos:
        case_id = normalize_case_id(str(video)) or video.stem
        task_name = task_by_case.get(case_id, "unknown")
        label = label_map.get(task_name, unknown_label)
        rows.append((video, label))
        metadata.append(
            {
                "path": str(video),
                "case_id": case_id,
                "label": label,
                "task_name": task_name,
            }
        )

    val_count = int(round(len(rows) * args.val_fraction))
    if len(rows) > 1 and val_count == 0 and args.val_fraction > 0:
        val_count = 1
    val_count = min(val_count, max(0, len(rows) - 1))
    train_rows = rows[:-val_count] if val_count else rows
    val_rows = rows[-val_count:] if val_count else []

    write_manifest(args.out_dir / f"{args.manifest_prefix}_all.csv", rows)
    write_manifest(args.out_dir / f"{args.manifest_prefix}_train.csv", train_rows)
    write_manifest(args.out_dir / f"{args.manifest_prefix}_val.csv", val_rows)
    write_metadata(args.out_dir / "metadata.csv", metadata)

    summary = {
        "videos_root": str(videos_root.resolve()),
        "labels_root": str(labels_root.resolve()) if labels_root else None,
        "num_videos": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "label_map": label_map | {"unknown": unknown_label},
        "manifests": {
            "all": str((args.out_dir / f"{args.manifest_prefix}_all.csv").resolve()),
            "train": str((args.out_dir / f"{args.manifest_prefix}_train.csv").resolve()),
            "val": str((args.out_dir / f"{args.manifest_prefix}_val.csv").resolve()),
        },
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
