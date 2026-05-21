#!/usr/bin/env python3
"""Validate a SurgVU/V-JEPA manifest by opening a few videos with decord."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from decord import VideoReader, cpu
    except Exception as exc:
        raise SystemExit("decord is required for manifest validation. Install the repo environment first.") from exc

    rows: list[tuple[Path, str]] = []
    with args.manifest.open(newline="") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if not row:
                continue
            rows.append((Path(row[0]), row[1] if len(row) > 1 else "0"))

    if not rows:
        raise SystemExit(f"Manifest is empty: {args.manifest}")

    checks = []
    for path, label in rows[: args.limit]:
        if not path.exists():
            raise SystemExit(f"Missing video listed in manifest: {path}")
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        frames = len(vr)
        checks.append(
            {
                "path": str(path),
                "label": label,
                "frames": frames,
                "fps": fps,
                "duration_seconds": frames / fps if fps else None,
            }
        )

    print(json.dumps({"manifest": str(args.manifest), "num_rows": len(rows), "checked": checks}, indent=2))


if __name__ == "__main__":
    main()
