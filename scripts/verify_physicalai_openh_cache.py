#!/usr/bin/env python3
"""Verify a PhysicalAI Open-H cache before offline compute-node training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.physicalai_openh import DEFAULT_REPO_ID, PhysicalAIOpenHDataset


DATASET_KWARG_KEYS = (
    "repo_id",
    "revision",
    "cache_dir",
    "local_repo_root",
    "local_files_only",
    "max_episodes",
    "task_filter",
    "embodiment_filter",
    "split_strategy",
    "split_embodiments",
    "video_key",
    "random_video_key",
    "require_action",
    "require_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Training config to verify.")
    parser.add_argument("--repo-id", default=None, help=f"Hugging Face dataset repo, default {DEFAULT_REPO_ID}.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", required=True, help="Local mirror directory visible to compute nodes.")
    parser.add_argument("--roots", nargs="+", help="Dataset roots/leaves to verify. Overrides config data.datasets.")
    parser.add_argument("--split", default=None, help="Split to verify, e.g. train/val/test.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap selected episodes for pilot verification.")
    parser.add_argument("--task-filter", nargs="+", default=None)
    parser.add_argument("--embodiment-filter", nargs="+", default=None)
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--all-video-keys", action="store_true", help="Require every declared video view per episode.")
    parser.add_argument("--metadata-only", action="store_true", help="Only verify meta/*.json[l], not episode files.")
    parser.add_argument("--check-samples", type=int, default=0, help="Decode/read the first N selected samples.")
    return parser.parse_args()


def config_plan(config_path: Path | None) -> tuple[dict, list[str]]:
    if config_path is None:
        return {}, []
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = cfg.get("data", {})
    kwargs = dict(data_cfg.get("dataset_kwargs", {}) or {})
    for key in DATASET_KWARG_KEYS:
        if key in data_cfg:
            kwargs[key] = data_cfg[key]
    return kwargs, list(data_cfg.get("datasets", []) or [])


def main() -> None:
    args = parse_args()
    kwargs, roots = config_plan(args.config)
    roots = args.roots or roots
    if not roots:
        raise SystemExit("No roots were provided. Use --roots or a config with data.datasets.")

    kwargs["cache_dir"] = args.cache_dir
    kwargs["local_files_only"] = True
    kwargs["repo_id"] = args.repo_id or kwargs.get("repo_id") or DEFAULT_REPO_ID
    if args.revision is not None:
        kwargs["revision"] = args.revision
    kwargs.setdefault("revision", "main")
    if args.split is not None:
        kwargs["split"] = args.split
    kwargs.setdefault("split", "train")
    if args.max_episodes is not None:
        kwargs["max_episodes"] = args.max_episodes
    if args.task_filter is not None:
        kwargs["task_filter"] = args.task_filter
    if args.embodiment_filter is not None:
        kwargs["embodiment_filter"] = args.embodiment_filter
    if args.video_key is not None:
        kwargs["video_key"] = args.video_key

    dataset = PhysicalAIOpenHDataset(
        dataset_roots=roots,
        frames_per_clip=1,
        frame_step=1,
        transform=None,
        **kwargs,
    )
    required = (
        dataset.required_metadata_files()
        if args.metadata_only
        else dataset.required_files(True if args.all_video_keys else None)
    )
    cache_dir = Path(args.cache_dir).expanduser()
    missing = [path for path in required if not (cache_dir / path).exists()]

    print(f"cache_dir: {cache_dir}")
    print(f"split: {dataset.split}")
    print(f"roots: {len(dataset.datasets)}")
    print(f"episodes: {len(dataset.episodes)}")
    print(f"required_files: {len(required)}")
    if missing:
        print("MISSING FILES:")
        for path in missing[:100]:
            print(path)
        if len(missing) > 100:
            print(f"... {len(missing) - 100} more")
        raise SystemExit(1)

    for idx in range(min(args.check_samples, len(dataset))):
        sample = dataset[idx]
        clip = sample[0][0] if isinstance(sample[0], (list, tuple)) else sample[0]
        aux = sample[2]
        print(
            f"sample {idx}: clip_shape={tuple(clip.shape)} "
            f"episode={aux.get('episode_index')} embodiment={aux.get('embodiment')} video_key={aux.get('video_key')}"
        )

    print("CACHE OK")


if __name__ == "__main__":
    main()
