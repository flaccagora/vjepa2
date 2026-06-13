#!/usr/bin/env python3
"""Prepare a local PhysicalAI Open-H cache for offline cluster training."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--config", type=Path, help="Training config to mirror.")
    parser.add_argument("--repo-id", default=None, help=f"Hugging Face dataset repo, default {DEFAULT_REPO_ID}.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", required=True, help="Local mirror directory visible to compute nodes.")
    parser.add_argument("--roots", nargs="+", help="Dataset roots/leaves to prepare. Overrides config data.datasets.")
    parser.add_argument("--split", default=None, help="Split to prepare, e.g. train/val/test.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap selected episodes for pilot caches.")
    parser.add_argument("--task-filter", nargs="+", default=None)
    parser.add_argument("--embodiment-filter", nargs="+", default=None)
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--all-video-keys", action="store_true", help="Download every declared video view per episode.")
    parser.add_argument("--metadata-only", action="store_true", help="Only download meta/*.json[l], not episodes.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
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
    kwargs["local_files_only"] = False
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
    if args.token is not None:
        kwargs["token"] = args.token

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = PhysicalAIOpenHDataset(
        dataset_roots=roots,
        frames_per_clip=1,
        frame_step=1,
        transform=None,
        **kwargs,
    )

    files = dataset.required_metadata_files()
    if not args.metadata_only:
        files = dataset.required_files(include_all_video_keys=True if args.all_video_keys else None)

    for idx, path in enumerate(files, 1):
        resolved = dataset._repo_file_path(path)
        print(f"[{idx}/{len(files)}] {path} -> {resolved}", flush=True)

    manifest = {
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "split": dataset.split,
        "cache_dir": str(cache_dir),
        "roots": [sub.root for sub in dataset.datasets],
        "episodes": [
            {
                "root": episode.dataset.root,
                "episode_index": episode.episode_index,
                "embodiment": episode.embodiment,
                "tasks": list(episode.tasks),
            }
            for episode in dataset.episodes
        ],
        "files": files,
    }
    manifest_path = cache_dir / "physicalai_openh_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(dataset.episodes)} episodes and {len(files)} files.")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
