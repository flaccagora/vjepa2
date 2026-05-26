#!/usr/bin/env python3
"""Fast SurgVU video preprocessor — dual CPU/GPU pipeline, no ffmpeg required.

Architecture
------------
* One OS process per video (ProcessPoolExecutor) — escapes the GIL and lets
  the OS scheduler spread work across all physical cores.
* Inside each worker a two-thread pipeline:
    decode-thread  →  bounded Queue  →  encode-thread
  so decode and encode overlap completely.
* GPU workers use cv2.cuda for upload/crop/download; CPU workers use a plain
  NumPy slice — both share the same queue/writer plumbing.
* Crop detection samples frames *sequentially* (no random seeks) using a
  dedicated forward pass so the OS read-ahead cache is hot for the encode pass.
* The crop box is the union across all sample frames, guaranteeing no content
  is ever clipped.
* Progress is reported via a multiprocessing.Queue back to the main process,
  which owns all tqdm bars (one batch bar + one bar per worker slot).

Usage (quick start)
-------------------
  # all CPU, 8 parallel videos
  python surgvu_preprocess_fast.py \
      --manifest data/train.txt \
      --out-dir  data/clean \
      --workers  8

  # 2 CUDA GPUs + CPU overflow
  python surgvu_preprocess_fast.py \
      --manifest data/train.txt \
      --out-dir  data/clean \
      --gpu-ids  0,1 \
      --workers  16

python scripts/fast_preprocessing.py   --manifest data/surgvu_full/manifest_train.csv   --out-dir data/surgvu_clean_train   --preview-count 8   --sample-frames 24  --limit 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import multiprocessing
import os
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger("surgvu_preprocess")


# ---------------------------------------------------------------------------
# Logging handler that routes through tqdm.write (avoids bar corruption)
# ---------------------------------------------------------------------------

class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    if not root.handlers:
        h = _TqdmHandler()
        h.setFormatter(logging.Formatter(
            "[%(levelname)s][%(asctime)s][%(processName)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(h)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    p.add_argument("--manifest",        type=Path, required=True)
    p.add_argument("--out-dir",         type=Path, required=True)
    p.add_argument("--output-manifest", type=Path)
    p.add_argument("--limit",           type=int,
                   help="Process only the first N manifest rows.")
    p.add_argument("--overwrite",       action="store_true", default=True)

    # Crop detection
    p.add_argument("--sample-frames",        type=int,   default=32,
                   help="Frames sampled per video for black-margin detection.")
    p.add_argument("--black-threshold",      type=int,   default=12,
                   help="Pixel luminance threshold for non-black content.")
    p.add_argument("--min-content-fraction", type=float, default=0.01,
                   help="Min non-black fraction for a row/col to count as content.")
    p.add_argument("--bottom-crop-ratio",    type=float, default=0.06,
                   help="Fraction of frame height stripped from the bottom before detection.")
    p.add_argument("--bottom-crop-pixels",   type=int,
                   help="Overrides --bottom-crop-ratio.")
    p.add_argument("--top-crop-ratio",       type=float, default=0.06,
                   help="Fraction of frame height stripped from the top before detection.")
    p.add_argument("--top-crop-pixels",      type=int,
                   help="Overrides --top-crop-ratio.")
    p.add_argument("--padding",              type=int,   default=4,
                   help="Extra pixels kept around the detected content box.")

    # Encode
    p.add_argument("--codec", default="auto",
                   help="OpenCV fourcc codec string. 'auto' tries avc1→mp4v. "
                        "Use 'avc1'/'H264' for hardware H.264, 'mp4v' as a safe fallback.")
    p.add_argument("--queue-depth", type=int, default=64,
                   help="Frames buffered between decode and encode threads.")

    # Progress
    p.add_argument("--progress-interval", type=int, default=30,
                   help="Frames between progress queue updates. Lower = smoother bars, "
                        "slightly higher IPC overhead.")

    # GPU
    p.add_argument("--gpu-ids", default="",
                   help="Comma-separated CUDA device indices, e.g. '0,1'. "
                        "Empty string → CPU-only mode.")

    # Parallelism
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                   help="Videos processed in parallel.")

    # Debug / preview
    p.add_argument("--max-output-frames", type=int,
                   help="Write only the first N frames per video (debug).")
    p.add_argument("--preview-count",  type=int, default=0)
    p.add_argument("--preview-frames", type=int, default=8)

    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def timed(timings: dict, name: str) -> Generator:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - t0, 4)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            rows.append((Path(parts[0]), parts[1] if len(parts) == 2 else "0"))
    return rows


def write_manifest(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=" ")
        for video_path, label in rows:
            w.writerow([str(video_path), label])


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _even(v: int) -> int:
    """Round down to nearest even number (required by YUV encoders)."""
    return v & ~1


def even_crop(
    x1: int, y1: int, x2: int, y2: int, W: int, H: int
) -> tuple[int, int, int, int]:
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(W, x2);  y2 = min(H, y2)
    x2 = max(x1 + 2, x2);  y2 = max(y1 + 2, y2)
    x1 = _even(x1);   y1 = _even(y1)
    x2 = _even(x2);   y2 = _even(y2)
    x2 = min(W, x2);  y2 = min(H, y2)
    return x1, y1, x2, y2


def sample_indices(total: int, count: int) -> np.ndarray:
    count = max(1, min(count, total))
    return np.unique(np.linspace(0, total - 1, count, dtype=np.int64))


# ---------------------------------------------------------------------------
# Crop detection  (sequential read — maximises OS read-ahead)
# ---------------------------------------------------------------------------

def detect_crop(path: Path, args: argparse.Namespace) -> dict:
    """Sample `args.sample_frames` frames sequentially, return union crop box."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")

    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bot_px = (args.bottom_crop_pixels if args.bottom_crop_pixels is not None
              else int(round(H * args.bottom_crop_ratio)))
    top_px = (args.top_crop_pixels if args.top_crop_pixels is not None
              else int(round(H * args.top_crop_ratio)))
    top_px = max(0, min(top_px, H - 2))
    bot_px = max(0, min(bot_px, H - top_px - 2))

    analysis_top    = top_px
    analysis_bottom = H - bot_px

    targets = set(sample_indices(n_frames, args.sample_frames).tolist())
    boxes: list[tuple[int, int, int, int]] = []

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in targets:
            roi  = frame[analysis_top:analysis_bottom, :, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mask = gray > args.black_threshold

            rows_idx = np.where(mask.mean(axis=1) >= args.min_content_fraction)[0]
            cols_idx = np.where(mask.mean(axis=0) >= args.min_content_fraction)[0]

            if len(rows_idx) and len(cols_idx):
                boxes.append((
                    int(cols_idx[0]),
                    int(rows_idx[0]),
                    int(cols_idx[-1]) + 1,
                    int(rows_idx[-1]) + 1,
                ))
        fi += 1

    cap.release()

    if boxes:
        pad = args.padding
        rx1 = min(b[0] for b in boxes) - pad
        ry1 = min(b[1] for b in boxes) - pad
        rx2 = max(b[2] for b in boxes) + pad
        ry2 = max(b[3] for b in boxes) + pad
        x1, y1, x2, y2 = rx1, analysis_top + ry1, rx2, analysis_top + ry2
    else:
        x1, y1, x2, y2 = 0, analysis_top, W, analysis_bottom

    y1 = max(y1, analysis_top)
    y2 = min(y2, analysis_bottom)
    x1, y1, x2, y2 = even_crop(x1, y1, x2, y2, W, H)

    return {
        "source_width":    W,
        "source_height":   H,
        "source_fps":      fps,
        "source_frames":   n_frames,
        "top_crop_pixels": top_px,
        "bot_crop_pixels": bot_px,
        "crop_x":  x1,
        "crop_y":  y1,
        "crop_w":  x2 - x1,
        "crop_h":  y2 - y1,
    }


# ---------------------------------------------------------------------------
# GPU availability probe (called once per worker process)
# ---------------------------------------------------------------------------

def _probe_gpu(device_id: int) -> bool:
    """Return True if cv2.cuda is functional on this device."""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return False
        cv2.cuda.setDevice(device_id)
        dummy   = np.zeros((8, 8, 3), dtype=np.uint8)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(dummy)
        _ = gpu_mat.download()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Codec probe
# ---------------------------------------------------------------------------

_CODEC_PREFERENCE = ("avc1", "H264", "X264", "mp4v")
_RESOLVED_CODEC: str | None = None


def probe_best_codec(w: int = 64, h: int = 64, fps: float = 25.0) -> str:
    """Return the first fourcc that OpenCV can actually open.
    Cached per-process so the probe only runs once per worker."""
    global _RESOLVED_CODEC
    if _RESOLVED_CODEC is not None:
        return _RESOLVED_CODEC
    import tempfile
    for codec in _CODEC_PREFERENCE:
        tmp = tempfile.mktemp(suffix=".mp4")
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
            if writer.isOpened():
                writer.release()
                Path(tmp).unlink(missing_ok=True)
                logger.debug("codec probe: selected %s", codec)
                _RESOLVED_CODEC = codec
                return codec
            writer.release()
        except Exception:
            pass
        finally:
            Path(tmp).unlink(missing_ok=True)
    raise RuntimeError(
        "No working OpenCV VideoWriter codec found. "
        "Try installing opencv-python with ffmpeg support."
    )


# ---------------------------------------------------------------------------
# Progress event protocol
#
# All events are plain tuples sent over a multiprocessing.Queue.
# The main-process progress thread is the only consumer.
#
#   ("phase",    idx, phase: str)              – "detecting" | "encoding"
#   ("start",    idx, name: str, total: int,
#                device_label: str)            – begin encode bar with known total
#   ("progress", idx, delta: int)              – N more frames written
#   ("done",     idx, written: int,
#                timings: dict)                – video finished OK
#   ("error",    idx, msg: str)                – video failed
#   None                                       – shutdown signal
# ---------------------------------------------------------------------------

# Type alias for clarity
_PQ = "multiprocessing.Queue[object]"


# ---------------------------------------------------------------------------
# Encode pipeline  (decode-thread → frame-queue → encode-thread)
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _decode_cpu(
    path:       Path,
    crop:       dict,
    out_q:      "queue.Queue",
    max_frames: int | None,
) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        out_q.put(RuntimeError(f"Cannot open: {path}"))
        out_q.put(_SENTINEL)
        return

    x1, y1 = crop["crop_x"], crop["crop_y"]
    x2, y2 = x1 + crop["crop_w"], y1 + crop["crop_h"]
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_q.put(frame[y1:y2, x1:x2])
        written += 1
        if max_frames is not None and written >= max_frames:
            break
    cap.release()
    out_q.put(_SENTINEL)


def _decode_gpu(
    path:       Path,
    crop:       dict,
    device_id:  int,
    out_q:      "queue.Queue",
    max_frames: int | None,
) -> None:
    try:
        cv2.cuda.setDevice(device_id)
        cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")

        x1, y1   = crop["crop_x"], crop["crop_y"]
        w,  h    = crop["crop_w"], crop["crop_h"]
        roi_rect = (x1, y1, w, h)
        gpu_mat  = cv2.cuda_GpuMat()
        written  = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gpu_mat.upload(frame)
            out_q.put(gpu_mat(roi_rect).download())
            written += 1
            if max_frames is not None and written >= max_frames:
                break
        cap.release()
    except Exception as exc:
        out_q.put(exc)
    finally:
        out_q.put(_SENTINEL)


def _encode(
    out_path:  Path,
    crop:      dict,
    codec:     str,
    in_q:      "queue.Queue",
    prog_q:    _PQ | None,
    idx:       int,
    interval:  int,
) -> int:
    """Drain frame queue, write video, send progress events."""
    if codec == "auto":
        codec = probe_best_codec()

    fourcc = cv2.VideoWriter_fourcc(*codec)
    fps    = crop["source_fps"] or 25.0
    size   = (crop["crop_w"], crop["crop_h"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        while in_q.get() is not _SENTINEL:   # drain so decoder unblocks
            pass
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")

    written   = 0
    since_upd = 0
    while True:
        item = in_q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            writer.release()
            raise item
        writer.write(item)
        written   += 1
        since_upd += 1
        if prog_q is not None and since_upd >= interval:
            prog_q.put(("progress", idx, since_upd))
            since_upd = 0

    # flush remainder
    if prog_q is not None and since_upd > 0:
        prog_q.put(("progress", idx, since_upd))

    writer.release()
    return written


def encode_video(
    source:    Path,
    output:    Path,
    crop:      dict,
    args:      argparse.Namespace,
    device_id: int | None,
    prog_q:    _PQ | None,
    idx:       int,
) -> int:
    """Full decode→encode pipeline; returns frames written."""
    fq: "queue.Queue" = queue.Queue(maxsize=args.queue_depth)

    dec_fn = (
        (lambda: _decode_gpu(source, crop, device_id, fq, args.max_output_frames))
        if device_id is not None
        else (lambda: _decode_cpu(source, crop, fq, args.max_output_frames))
    )
    dec_thread = threading.Thread(target=dec_fn, daemon=True)
    dec_thread.start()
    try:
        written = _encode(
            output, crop, args.codec, fq,
            prog_q, idx, args.progress_interval,
        )
    finally:
        dec_thread.join()
    return written


# ---------------------------------------------------------------------------
# Preview sheet
# ---------------------------------------------------------------------------

def make_preview(
    source: Path,
    output: Path,
    crop:   dict,
    dest:   Path,
    n:      int,
) -> None:
    cap_s = cv2.VideoCapture(str(source))
    cap_o = cv2.VideoCapture(str(output))
    total = int(min(cap_s.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap_o.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = sample_indices(total, n)
    srcs, outs = [], []

    for i in idxs:
        for cap, lst in ((cap_s, srcs), (cap_o, outs)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                lst.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    cap_s.release();  cap_o.release()
    if not srcs or not outs:
        return

    th    = 180
    row_h = th + 22
    canvas = Image.new("RGB", (320 * len(srcs), row_h * 2), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    for i, (s, o) in enumerate(zip(srcs, outs)):
        x   = i * 320
        s_r = cv2.resize(s, (320, th), interpolation=cv2.INTER_AREA)
        ow  = max(1, int(round(320 * o.shape[0] / o.shape[1])))
        o_r = cv2.resize(o, (320, ow), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(s_r), (x, 0))
        canvas.paste(Image.fromarray(o_r), (x, row_h))
        draw.text((x + 4, th + 3),         f"before {int(idxs[i])}", fill=(220, 220, 220))
        draw.text((x + 4, row_h + ow + 3),  "after",                 fill=(220, 220, 220))

    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=92)


# ---------------------------------------------------------------------------
# Per-video worker  (runs inside a worker process)
# ---------------------------------------------------------------------------

def process_one(task: dict) -> dict:
    source     = Path(task["source"])
    output     = Path(task["output"])
    label      = task["label"]
    device_id  = task["device_id"]
    idx        = task["idx"]
    do_preview = task["do_preview"]
    args_dict  = task["args_dict"]
    prog_q: _PQ | None = task.get("prog_q")

    args = argparse.Namespace(**args_dict)
    timings: dict[str, float] = {}
    device_label = f"GPU{device_id}" if device_id is not None else "CPU"

    def _send(*event):
        if prog_q is not None:
            try:
                prog_q.put(event)
            except Exception:
                pass

    _send("phase", idx, source.name, "detecting", device_label)
    logger.debug("[%d] detect start  %s", idx, source.name)

    if output.exists() and not args.overwrite:
        cap     = cv2.VideoCapture(str(output))
        written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        _send("done", idx, written, {})
        return {
            "idx": idx, "source": str(source), "output": str(output),
            "label": label, "written": written, "skipped": True,
        }

    with timed(timings, "detect"):
        crop = detect_crop(source, args)

    logger.debug(
        "[%d] crop %dx%d→%dx%d x=%d y=%d  (%.2fs)",
        idx, crop["source_width"], crop["source_height"],
        crop["crop_w"], crop["crop_h"],
        crop["crop_x"], crop["crop_y"],
        timings["detect"],
    )

    # Signal that we now know the total frame count → encode bar can be created
    _send("start", idx, source.name, crop["source_frames"], device_label)

    with timed(timings, "encode"):
        written = encode_video(source, output, crop, args, device_id, prog_q, idx)

    logger.debug("[%d] done frames=%d encode=%.1fs", idx, written, timings["encode"])
    _send("done", idx, written, timings)

    item: dict = {
        "idx":     idx,
        "source":  str(source),
        "output":  str(output),
        "label":   label,
        "written": written,
        "device":  device_id,
        "timings": timings,
        **crop,
    }

    if do_preview:
        preview_path = (
            Path(args_dict["out_dir"]) / "previews" / f"{output.stem}_preview.jpg"
        )
        with timed(timings, "preview"):
            make_preview(source, output, crop, preview_path, args.preview_frames)
        item["preview"] = str(preview_path)

    return item


# ---------------------------------------------------------------------------
# Progress display — runs in a dedicated thread in the main process
# ---------------------------------------------------------------------------

# Bar colours via ANSI (graceful no-op on terminals without colour support)
_PHASE_COLOUR = {
    "detecting": "\033[33m",   # yellow
    "encoding":  "\033[36m",   # cyan
    "done":      "\033[32m",   # green
    "error":     "\033[31m",   # red
}
_RESET = "\033[0m"


def _fmt_desc(name: str, phase: str, device: str, width: int = 28) -> str:
    colour = _PHASE_COLOUR.get(phase, "")
    tag    = f"{colour}{phase:<9}{_RESET}"
    short  = name if len(name) <= width else "…" + name[-(width - 1):]
    return f"[{device:>5}] {short:<{width}} {tag}"


def _progress_thread(
    prog_q:    _PQ,
    n_total:   int,
    n_workers: int,
    batch_bar: tqdm,
) -> None:
    """
    Consume progress events and keep all tqdm bars up to date.

    Slot layout (tqdm positions):
      0            → batch_bar (owned by caller, passed in)
      1 … n_workers → per-video bars (one per concurrent slot)
    """
    # Slot pool: indices 1..n_workers available for allocation
    free_slots: list[int] = list(range(1, n_workers + 1))
    # idx → (tqdm bar, slot index)
    active: dict[int, tuple[tqdm, int]] = {}
    # idx → device label (stored during "phase" before "start")
    pending_device: dict[int, str] = {}
    pending_name:   dict[int, str] = {}

    def _acquire_slot(idx: int, name: str, total: int, device: str) -> tqdm:
        slot = free_slots.pop(0) if free_slots else n_workers  # overflow → last row
        bar  = tqdm(
            total=total,
            desc=_fmt_desc(name, "encoding", device),
            position=slot,
            leave=False,
            unit="fr",
            unit_scale=True,
            dynamic_ncols=True,
            bar_format=(
                "{desc} {bar} {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
        active[idx] = (bar, slot)
        return bar

    def _release_slot(idx: int) -> None:
        if idx in active:
            bar, slot = active.pop(idx)
            bar.close()
            if slot not in free_slots:
                free_slots.append(slot)
                free_slots.sort()

    while True:
        try:
            event = prog_q.get(timeout=0.5)
        except Exception:
            continue

        if event is None:
            # shutdown: close any lingering bars
            for bar, _ in list(active.values()):
                bar.close()
            break

        kind = event[0]

        # ── "phase"  (idx, name, phase, device) ──────────────────────────
        # Emitted immediately when the worker starts (before detect finishes).
        # We don't have a total yet, so we show an indeterminate bar.
        if kind == "phase":
            _, idx, name, phase, device = event
            pending_device[idx] = device
            pending_name[idx]   = name
            if idx not in active:
                # Create an indeterminate bar so the slot appears immediately
                slot = free_slots.pop(0) if free_slots else n_workers
                bar  = tqdm(
                    total=None,
                    desc=_fmt_desc(name, phase, device),
                    position=slot,
                    leave=False,
                    unit="fr",
                    dynamic_ncols=True,
                    bar_format="{desc} … {elapsed}",
                )
                active[idx] = (bar, slot)
            else:
                bar, _ = active[idx]
                bar.set_description(_fmt_desc(name, phase, device))

        # ── "start"  (idx, name, total, device) ──────────────────────────
        # Emitted after detect completes; we now know total frames.
        elif kind == "start":
            _, idx, name, total, device = event
            # Replace the indeterminate bar with a proper one
            _release_slot(idx)
            _acquire_slot(idx, name, total, device)

        # ── "progress"  (idx, delta) ──────────────────────────────────────
        elif kind == "progress":
            _, idx, delta = event
            if idx in active:
                active[idx][0].update(delta)

        # ── "done"  (idx, written, timings) ──────────────────────────────
        elif kind == "done":
            _, idx, written, timings = event
            if idx in active:
                bar, _ = active[idx]
                name   = pending_name.get(idx, "")
                device = pending_device.get(idx, "")
                enc_s  = timings.get("encode", 0.0)
                fps    = written / enc_s if enc_s > 0 else 0.0
                bar.set_description(_fmt_desc(name, "done", device))
                bar.set_postfix_str(f"{written:,} fr  {fps:,.0f} fr/s", refresh=True)
            _release_slot(idx)
            pending_device.pop(idx, None)
            pending_name.pop(idx, None)
            batch_bar.update(1)

        # ── "error"  (idx, msg) ──────────────────────────────────────────
        elif kind == "error":
            _, idx, msg = event
            if idx in active:
                bar, _ = active[idx]
                name   = pending_name.get(idx, "")
                device = pending_device.get(idx, "")
                bar.set_description(_fmt_desc(name, "error", device))
                bar.set_postfix_str(msg[:60], refresh=True)
            _release_slot(idx)
            pending_device.pop(idx, None)
            pending_name.pop(idx, None)
            batch_bar.update(1)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def common_parent(paths: list[Path]) -> Path:
    return Path(os.path.commonpath([str(p.resolve().parent) for p in paths]))


def output_path(source: Path, root: Path, videos_dir: Path) -> Path:
    resolved = source.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        rel = Path(resolved.name)
    return (videos_dir / rel).with_suffix(".mp4")


# ---------------------------------------------------------------------------
# GPU scheduling
# ---------------------------------------------------------------------------

def build_device_schedule(gpu_ids: list[int], n: int) -> list[int | None]:
    """Round-robin assign GPU ids to task indices; None → CPU."""
    if not gpu_ids:
        return [None] * n
    return [gpu_ids[i % len(gpu_ids)] for i in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir   = args.out_dir / "videos"
    out_manifest = args.output_manifest or (args.out_dir / args.manifest.name)

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows in manifest: {args.manifest}")

    root     = common_parent([p for p, _ in rows])
    gpu_ids  = (
        [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
        if args.gpu_ids else []
    )
    schedule = build_device_schedule(gpu_ids, len(rows))

    # Probe GPUs in main process once and warn early
    available_gpus: set[int] = set()
    for gid in gpu_ids:
        if _probe_gpu(gid):
            available_gpus.add(gid)
            logger.info("GPU %d: available ✓", gid)
        else:
            logger.warning("GPU %d: NOT available — tasks fall back to CPU", gid)

    args_dict = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    # Manager().Queue() returns a proxy object that IS picklable, so it can
    # safely be passed as a submit() argument across ProcessPoolExecutor workers.
    # A bare multiprocessing.Queue cannot be pickled this way — it can only be
    # shared via fork-inheritance or an initializer, neither of which is
    # available to ProcessPoolExecutor on all platforms.
    _manager = multiprocessing.Manager()
    prog_q   = _manager.Queue()

    n_workers = max(1, args.workers)

    tasks = []
    for i, ((source, label), device_id) in enumerate(zip(rows, schedule)):
        out = output_path(source, root, videos_dir).resolve()
        effective_device = device_id if (device_id in available_gpus) else None
        tasks.append({
            "idx":        i,
            "source":     str(source),
            "output":     str(out),
            "label":      label,
            "device_id":  effective_device,
            "do_preview": i < args.preview_count,
            "args_dict":  args_dict,
            "prog_q":     prog_q,
        })

    tqdm.write(
        f"Starting: {len(tasks)} videos  "
        f"workers={n_workers}  "
        f"gpu_ids={gpu_ids or 'none (CPU-only)'}"
    )

    t0 = time.perf_counter()
    metadata: list[dict] = []

    # Batch bar at position 0 — always visible at the top
    batch_bar = tqdm(
        total=len(tasks),
        desc="Batch",
        position=0,
        leave=True,
        unit="video",
        dynamic_ncols=True,
        bar_format=(
            "Batch  {bar} {n_fmt}/{total_fmt} videos "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
        colour="green",
    )

    # Start the progress display thread (reads from prog_q, updates tqdm)
    prog_thread = threading.Thread(
        target=_progress_thread,
        args=(prog_q, len(tasks), n_workers, batch_bar),
        daemon=True,
    )
    prog_thread.start()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                logger.error("[%d] FAILED %s: %s", task["idx"], task["source"], exc)
                item = {
                    "idx":    task["idx"],
                    "source": task["source"],
                    "output": task["output"],
                    "label":  task["label"],
                    "error":  str(exc),
                }
                # Ensure the progress thread advances the batch bar on errors
                # that weren't caught inside process_one (e.g. pickling crash)
                prog_q.put(("error", task["idx"], str(exc)))
            metadata.append(item)
            print(json.dumps(item, default=str), flush=True)

    # Shut down progress thread cleanly, then release the Manager server
    prog_q.put(None)
    prog_thread.join(timeout=5)
    batch_bar.close()
    _manager.shutdown()

    total_s = time.perf_counter() - t0
    metadata.sort(key=lambda x: x["idx"])
    processed_rows = [
        (Path(m["output"]), m["label"]) for m in metadata if "error" not in m
    ]
    write_manifest(out_manifest, processed_rows)

    n_ok  = len(processed_rows)
    n_err = len(metadata) - n_ok
    summary = {
        "input_manifest":  str(args.manifest),
        "output_manifest": str(out_manifest.resolve()),
        "videos_dir":      str(videos_dir.resolve()),
        "num_videos":      n_ok,
        "num_errors":      n_err,
        "total_seconds":   round(total_s, 2),
        "workers":         n_workers,
        "gpu_ids":         gpu_ids,
        "videos":          metadata,
    }
    (args.out_dir / "preprocess_metadata.json").write_text(
        json.dumps(summary, indent=2)
    )

    tqdm.write(
        f"Done: {n_ok}/{len(metadata)} videos  "
        f"errors={n_err}  "
        f"total={total_s:.1f}s  "
        f"({n_ok / (total_s / 60):.1f} vid/min)"
    )
    print(json.dumps({
        "output_manifest": str(out_manifest.resolve()),
        "num_videos":      n_ok,
        "num_errors":      n_err,
        "total_seconds":   round(total_s, 2),
    }, indent=2))


if __name__ == "__main__":
    main()