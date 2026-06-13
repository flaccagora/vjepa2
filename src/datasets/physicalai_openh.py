# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face Hub backed loader for NVIDIA PhysicalAI Open-H Embodiment.

The dataset is published as many LeRobot v2.1 sub-datasets. Each supported
leaf has a small metadata directory and lazy per-episode media files:

    meta/info.json
    meta/modality.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4

Only metadata is read during initialization. Episode Parquet/video files are
downloaded or opened only when a sample is requested.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import random
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from huggingface_hub import HfApi, hf_hub_download

from src.datasets.utils.dataloader import NondeterministicDataLoader

logger = getLogger()


DEFAULT_REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
SUPPORTED_CODEBASE_VERSION = "v2.1"
REQUIRED_META_FILES = ("info.json", "modality.json", "tasks.jsonl", "episodes.jsonl")
UNRESOLVED_ENV_VAR_RE = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


class PhysicalAIOpenHError(RuntimeError):
    """Raised when a PhysicalAI Open-H sub-dataset cannot be used safely."""


@dataclass(frozen=True)
class OpenHSubDataset:
    root: str
    info: dict[str, Any]
    modality: dict[str, Any]
    tasks: dict[int, str]
    episodes: list[dict[str, Any]]
    video_keys: list[str]
    action_keys: list[str]
    state_keys: list[str]
    proprio_keys: list[str]
    robot_type: str


@dataclass(frozen=True)
class OpenHEpisode:
    dataset: OpenHSubDataset
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    label: int

    @property
    def embodiment(self) -> str:
        return self.dataset.robot_type


def _normalize_repo_path(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return ""
    return str(path).strip("/")


def _expand_local_path(path: str | os.PathLike[str] | None, field_name: str) -> pathlib.Path | None:
    if path in (None, ""):
        return None
    value = os.path.expanduser(os.path.expandvars(str(path)))
    if UNRESOLVED_ENV_VAR_RE.search(value):
        raise PhysicalAIOpenHError(
            f"{field_name}={path!r} contains an unresolved environment variable. "
            "Set it before launching training or pass an explicit path."
        )
    return pathlib.Path(value)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PhysicalAIOpenHError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _parse_split_range(spec: str) -> tuple[int, int]:
    try:
        start, end = spec.split(":")
        return int(start), int(end)
    except Exception as exc:
        raise PhysicalAIOpenHError(f"Unsupported LeRobot split spec {spec!r}; expected 'start:end'.") from exc


def _fallback_episode_indices(
    total_episodes: int,
    split: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> set[int]:
    if total_episodes <= 0:
        return set()
    if split not in {"train", "val", "validation", "test"}:
        raise PhysicalAIOpenHError(
            f"Unknown split {split!r}; expected train, val/validation, or test when no split metadata exists."
        )

    rng = random.Random(seed)
    indices = list(range(total_episodes))
    rng.shuffle(indices)

    n_test = int(round(total_episodes * test_fraction))
    n_val = int(round(total_episodes * val_fraction))
    n_test = min(max(n_test, 0), total_episodes)
    n_val = min(max(n_val, 0), total_episodes - n_test)

    test_indices = set(indices[:n_test])
    val_indices = set(indices[n_test : n_test + n_val])
    train_indices = set(indices[n_test + n_val :])
    if split == "test":
        return test_indices
    if split in {"val", "validation"}:
        return val_indices
    return train_indices


def _split_episode_indices(
    info: dict[str, Any],
    split: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> set[int]:
    total_episodes = int(info.get("total_episodes", 0))
    split_key = "val" if split == "validation" else split
    splits = info.get("splits") or {}
    if split_key in splits:
        start, end = _parse_split_range(str(splits[split_key]))
        return set(range(max(0, start), min(total_episodes, end)))
    if splits:
        logger.warning(
            "Split %s is not declared in metadata; skipping this sub-dataset to avoid episode leakage.",
            split_key,
        )
        return set()
    return _fallback_episode_indices(total_episodes, split_key, val_fraction, test_fraction, seed)


def _feature_shape_size(feature: dict[str, Any]) -> int:
    shape = feature.get("shape") or []
    size = 1
    for dim in shape:
        try:
            size *= int(dim)
        except (TypeError, ValueError):
            return 0
    return size


def _is_numeric_feature(feature: dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", "")).lower()
    return any(token in dtype for token in ("float", "int", "bool"))


def _original_key(spec: dict[str, Any], default_key: str | None = None) -> str | None:
    key = spec.get("original_key", default_key)
    return str(key) if key is not None else None


def _derive_video_keys(
    features: dict[str, Any],
    modality: dict[str, Any],
    preferred_video_key: str | None,
) -> list[str]:
    by_feature = [k for k, v in features.items() if str(v.get("dtype", "")).lower() == "video"]
    by_modality = []
    for logical_name, spec in (modality.get("video") or {}).items():
        key = _original_key(spec, logical_name)
        if key in features:
            by_modality.append(key)

    keys = []
    for key in by_modality + by_feature:
        if key not in keys:
            keys.append(key)

    if not keys:
        raise PhysicalAIOpenHError("No video feature declared in meta/info.json or meta/modality.json.")
    if preferred_video_key:
        if preferred_video_key not in keys:
            raise PhysicalAIOpenHError(
                f"Requested video_key={preferred_video_key!r}, but available video keys are {keys}."
            )
        return [preferred_video_key]

    def score(key: str) -> tuple[int, str]:
        low = key.lower()
        if "depth" in low:
            return (10, key)
        for rank, token in enumerate(("endoscope", "color", "tpv", "wrist", "ultrasound")):
            if token in low:
                return (rank, key)
        return (5, key)

    return sorted(keys, key=score)


def _derive_numeric_keys(
    features: dict[str, Any],
    modality: dict[str, Any],
    modality_name: str,
    prefixes: tuple[str, ...],
    explicit_keys: list[str] | None,
    require: bool,
) -> list[str]:
    if explicit_keys:
        missing = [key for key in explicit_keys if key not in features]
        if missing:
            raise PhysicalAIOpenHError(f"Requested {modality_name} keys are missing from features: {missing}.")
        return list(explicit_keys)

    keys = []
    for logical_name, spec in (modality.get(modality_name) or {}).items():
        key = _original_key(spec, logical_name)
        if key in features and _is_numeric_feature(features[key]):
            keys.append(key)
    for key, feature in features.items():
        if key.startswith(prefixes) and _is_numeric_feature(feature):
            keys.append(key)
    deduped = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    if require and not deduped:
        raise PhysicalAIOpenHError(
            f"No {modality_name} feature found. Available numeric features are "
            f"{[k for k, v in features.items() if _is_numeric_feature(v)]}."
        )
    return deduped


def _episode_label(episode: dict[str, Any], task_to_index: dict[str, int]) -> int:
    tasks = episode.get("tasks") or []
    if tasks:
        return task_to_index.get(str(tasks[0]), 0)
    task_index = episode.get("task_index")
    if task_index is not None:
        try:
            return int(task_index)
        except (TypeError, ValueError):
            pass
    return 0


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise PhysicalAIOpenHError(f"Invalid chunks_size={chunks_size} in meta/info.json.")
    return episode_index // chunks_size


def _format_template(template: str, info: dict[str, Any], episode_index: int, video_key: str | None = None) -> str:
    values = {
        "episode_chunk": _episode_chunk(info, episode_index),
        "episode_index": episode_index,
        "video_key": video_key or "",
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise PhysicalAIOpenHError(f"Unsupported path template variable in {template!r}: {exc}") from exc


class PhysicalAIOpenHDataset(torch.utils.data.Dataset):
    """LeRobot v2.1 episode dataset for V-JEPA style video fine-tuning."""

    def __init__(
        self,
        dataset_roots: list[str] | str,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = "main",
        split: str = "train",
        local_repo_root: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = True,
        discover_roots: bool = False,
        max_discovery_depth: int = 6,
        max_episodes: int | None = None,
        task_filter: list[str] | None = None,
        embodiment_filter: list[str] | None = None,
        split_strategy: str = "metadata",
        split_embodiments: dict[str, list[str]] | None = None,
        val_embodiment_fraction: float = 0.0,
        test_embodiment_fraction: float = 0.0,
        video_key: str | None = None,
        random_video_key: bool = False,
        action_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        proprio_keys: list[str] | None = None,
        require_action: bool = False,
        require_state: bool = False,
        frames_per_clip: int = 16,
        fps: int | None = None,
        frame_step: int | None = None,
        duration: float | None = None,
        num_clips: int = 1,
        random_clip_sampling: bool = True,
        filter_short_videos: bool = False,
        transform=None,
        val_fraction: float = 0.05,
        test_fraction: float = 0.0,
        seed: int = 239,
        token: str | bool | None = None,
        max_load_retries: int = 10,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.split = split
        self.cache_dir = _expand_local_path(cache_dir, "cache_dir")
        self.local_repo_root = _expand_local_path(local_repo_root, "local_repo_root") or self.cache_dir
        self.local_files_only = bool(local_files_only)
        self.discover_roots = discover_roots
        self.max_discovery_depth = max_discovery_depth
        self.max_episodes = max_episodes
        self.video_key = video_key
        self.random_video_key = random_video_key
        self.frames_per_clip = int(frames_per_clip)
        self.fps = fps
        self.frame_step = frame_step
        self.duration = duration
        self.num_clips = int(num_clips)
        self.random_clip_sampling = random_clip_sampling
        self.filter_short_videos = filter_short_videos
        self.transform = transform
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)
        self.token = token
        self.max_load_retries = int(max_load_retries)

        if sum(v is not None for v in (self.fps, self.duration, self.frame_step)) != 1:
            raise ValueError("Specify exactly one of fps, duration, or frame_step for PhysicalAIOpenHDataset.")
        if self.frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be positive.")
        if self.num_clips != 1:
            raise NotImplementedError("PhysicalAIOpenHDataset currently supports num_clips=1 for V-JEPA training.")
        if self.local_files_only and self.local_repo_root is None:
            raise PhysicalAIOpenHError(
                "PhysicalAIOpenHDataset is running with local_files_only=True, but no cache_dir "
                "or local_repo_root was provided. Prepare the cache on an internet-connected login "
                "node, then set data.cache_dir or dataset_kwargs.local_repo_root for compute-node training."
            )

        roots = dataset_roots if isinstance(dataset_roots, (list, tuple)) else [dataset_roots]
        roots = [_normalize_repo_path(root) for root in roots]
        roots = self._discover_dataset_roots(roots) if discover_roots else roots
        if not roots:
            raise PhysicalAIOpenHError("No PhysicalAI Open-H dataset roots were provided or discovered.")

        self.task_filter = {str(t) for t in task_filter} if task_filter else None
        self.embodiment_filter = {str(e) for e in embodiment_filter} if embodiment_filter else None
        self.split_strategy = str(split_strategy)
        self.split_embodiments = split_embodiments or {}
        self.val_embodiment_fraction = float(val_embodiment_fraction)
        self.test_embodiment_fraction = float(test_embodiment_fraction)
        if self.split_strategy not in {"metadata", "episode", "embodiment"}:
            raise PhysicalAIOpenHError(
                "split_strategy must be one of 'metadata', 'episode', or 'embodiment'."
            )
        self.datasets = [
            self._load_subdataset(
                root=root,
                action_keys=action_keys,
                state_keys=state_keys,
                proprio_keys=proprio_keys,
                require_action=require_action,
                require_state=require_state,
            )
            for root in roots
        ]
        self.episodes = self._build_episode_index()
        if not self.episodes:
            raise PhysicalAIOpenHError(
                "PhysicalAI Open-H dataset produced zero episodes after split/task/embodiment filters."
            )
        if self.max_episodes is not None:
            self.episodes = self.episodes[: int(self.max_episodes)]

        logger.info(
            "PhysicalAI Open-H dataset created: split=%s roots=%d episodes=%d repo=%s",
            self.split,
            len(self.datasets),
            len(self.episodes),
            self.repo_id,
        )

    def _repo_file_path(self, path: str) -> pathlib.Path:
        local_path = self.local_repo_root / path if self.local_repo_root is not None else None
        if local_path is not None:
            if local_path.exists():
                return local_path
            if self.local_files_only:
                raise PhysicalAIOpenHError(
                    f"Required cached dataset file is missing: {local_path}. "
                    "Run scripts/prepare_physicalai_openh_cache.py on a login node and then "
                    "scripts/verify_physicalai_openh_cache.py before compute-node training."
                )
        if self.local_files_only:
            raise PhysicalAIOpenHError(
                f"Required dataset file {path!r} is not available locally and local_files_only=True."
            )

        download_kwargs = {
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "revision": self.revision,
            "filename": path,
            "token": self.token,
            "local_files_only": False,
        }
        if self.local_repo_root is not None:
            download_kwargs["local_dir"] = self.local_repo_root
        downloaded = hf_hub_download(**download_kwargs)
        downloaded_path = pathlib.Path(downloaded)
        if local_path is not None and local_path.exists():
            return local_path
        return downloaded_path

    def _repo_file_exists(self, path: str) -> bool:
        if self.local_repo_root is not None:
            local_path = self.local_repo_root / path
            if local_path.exists():
                return True
            if self.local_files_only:
                return False
        api = HfApi()
        try:
            api.hf_hub_download  # type: ignore[attr-defined]
        except AttributeError:
            pass
        try:
            self._repo_file_path(path)
            return True
        except Exception:
            return False

    def _discover_dataset_roots(self, roots: list[str]) -> list[str]:
        if self.local_files_only:
            if self.local_repo_root is None:
                raise PhysicalAIOpenHError("Cannot discover PhysicalAI Open-H roots without a local cache_dir.")
            discovered = []
            for root in roots:
                base = self.local_repo_root / root
                if not base.exists():
                    raise PhysicalAIOpenHError(
                        f"Cached dataset root is missing during discovery: {base}. "
                        "Use a narrower data.datasets list or prepare that root first."
                    )
                for info_path in base.rglob("meta/info.json"):
                    discovered.append(str(info_path.parent.parent.relative_to(self.local_repo_root)))
            return sorted(set(discovered))

        api = HfApi()
        discovered = []
        queue = [(root, 0) for root in roots]
        seen = set()
        while queue:
            root, depth = queue.pop(0)
            if root in seen or depth > self.max_discovery_depth:
                continue
            seen.add(root)
            child_paths = []
            for item in api.list_repo_tree(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                path_in_repo=root or None,
                recursive=False,
                token=self.token,
            ):
                child_paths.append(item.path)
            names = {pathlib.PurePosixPath(path).name for path in child_paths}
            if {"data", "meta", "videos"}.issubset(names) and self._repo_file_exists(f"{root}/meta/info.json"):
                discovered.append(root)
                continue
            for path in child_paths:
                if not pathlib.PurePosixPath(path).suffix:
                    queue.append((path, depth + 1))
        return sorted(set(discovered))

    def _load_subdataset(
        self,
        root: str,
        action_keys: list[str] | None,
        state_keys: list[str] | None,
        proprio_keys: list[str] | None,
        require_action: bool,
        require_state: bool,
    ) -> OpenHSubDataset:
        meta_dir = f"{root}/meta"
        for name in REQUIRED_META_FILES:
            if not self._repo_file_exists(f"{meta_dir}/{name}"):
                raise PhysicalAIOpenHError(
                    f"{root!r} is not a supported LeRobot v2.1 leaf: missing meta/{name}. "
                    "Legacy per-episode JSON layouts are not supported by this loader."
                )

        info = json.loads(self._repo_file_path(f"{meta_dir}/info.json").read_text())
        if str(info.get("codebase_version")) != SUPPORTED_CODEBASE_VERSION:
            raise PhysicalAIOpenHError(
                f"{root!r} has codebase_version={info.get('codebase_version')!r}; "
                f"only {SUPPORTED_CODEBASE_VERSION} is supported."
            )
        for template_key in ("data_path", "video_path"):
            if template_key not in info:
                raise PhysicalAIOpenHError(f"{root!r} meta/info.json is missing {template_key!r}.")

        modality = json.loads(self._repo_file_path(f"{meta_dir}/modality.json").read_text())
        task_rows = _read_jsonl(self._repo_file_path(f"{meta_dir}/tasks.jsonl"))
        episode_rows = _read_jsonl(self._repo_file_path(f"{meta_dir}/episodes.jsonl"))
        tasks = {
            int(row["task_index"]): str(row["task"])
            for row in task_rows
            if "task_index" in row and "task" in row
        }
        features = info.get("features") or {}

        video_keys = _derive_video_keys(features, modality, self.video_key)
        action_keys = _derive_numeric_keys(features, modality, "action", ("action",), action_keys, require_action)
        state_keys = _derive_numeric_keys(
            features,
            modality,
            "state",
            ("observation.state",),
            state_keys,
            require_state,
        )
        proprio_keys = _derive_numeric_keys(
            features,
            modality,
            "state",
            ("observation.state", "observation.meta.force_torque"),
            proprio_keys,
            require=False,
        )
        robot_type = str(info.get("robot_type") or root.split("/")[0])

        return OpenHSubDataset(
            root=root,
            info=info,
            modality=modality,
            tasks=tasks,
            episodes=episode_rows,
            video_keys=video_keys,
            action_keys=action_keys,
            state_keys=state_keys,
            proprio_keys=proprio_keys,
            robot_type=robot_type,
        )

    def _build_episode_index(self) -> list[OpenHEpisode]:
        all_episodes = []
        selected_embodiments = self._selected_embodiments()
        for dataset in self.datasets:
            if self.embodiment_filter and dataset.robot_type not in self.embodiment_filter:
                continue
            if selected_embodiments is not None:
                if dataset.robot_type not in selected_embodiments:
                    continue
                selected_indices = {int(row.get("episode_index", -1)) for row in dataset.episodes}
            elif self.split_strategy == "episode":
                selected_indices = _fallback_episode_indices(
                    int(dataset.info.get("total_episodes", 0)),
                    self.split,
                    val_fraction=self.val_fraction,
                    test_fraction=self.test_fraction,
                    seed=self.seed,
                )
            else:
                selected_indices = _split_episode_indices(
                    dataset.info,
                    self.split,
                    val_fraction=self.val_fraction,
                    test_fraction=self.test_fraction,
                    seed=self.seed,
                )
            task_to_index = {task: task_index for task_index, task in dataset.tasks.items()}
            for row in dataset.episodes:
                episode_index = int(row.get("episode_index", -1))
                if episode_index not in selected_indices:
                    continue
                tasks = tuple(str(t) for t in (row.get("tasks") or []))
                if self.task_filter and not (set(tasks) & self.task_filter):
                    continue
                length = int(row.get("length", 0))
                if length <= 0:
                    continue
                all_episodes.append(
                    OpenHEpisode(
                        dataset=dataset,
                        episode_index=episode_index,
                        length=length,
                        tasks=tasks,
                        label=_episode_label(row, task_to_index),
                    )
                )
        return all_episodes

    def _selected_embodiments(self) -> set[str] | None:
        if self.split_strategy != "embodiment":
            return None

        split_key = "val" if self.split == "validation" else self.split
        explicit = self.split_embodiments.get(split_key)
        if explicit is not None:
            return {str(item) for item in explicit}

        embodiments = sorted({dataset.robot_type for dataset in self.datasets})
        if not embodiments:
            return set()
        rng = random.Random(self.seed)
        shuffled = list(embodiments)
        rng.shuffle(shuffled)

        n_test = int(round(len(shuffled) * self.test_embodiment_fraction))
        n_val = int(round(len(shuffled) * self.val_embodiment_fraction))
        n_test = min(max(n_test, 0), len(shuffled))
        n_val = min(max(n_val, 0), len(shuffled) - n_test)

        test_set = set(shuffled[:n_test])
        val_set = set(shuffled[n_test : n_test + n_val])
        train_set = set(shuffled[n_test + n_val :])
        if split_key == "test":
            return test_set
        if split_key == "val":
            return val_set
        return train_set

    def _episode_data_path(self, episode: OpenHEpisode) -> str:
        rel = _format_template(
            episode.dataset.info["data_path"],
            info=episode.dataset.info,
            episode_index=episode.episode_index,
        )
        return f"{episode.dataset.root}/{rel}"

    def _episode_video_path(self, episode: OpenHEpisode, video_key: str) -> str:
        rel = _format_template(
            episode.dataset.info["video_path"],
            info=episode.dataset.info,
            episode_index=episode.episode_index,
            video_key=video_key,
        )
        return f"{episode.dataset.root}/{rel}"

    def _select_video_key(self, episode: OpenHEpisode) -> str:
        keys = episode.dataset.video_keys
        if self.random_video_key and len(keys) > 1:
            return keys[int(torch.randint(0, len(keys), (1,)).item())]
        return keys[0]

    def _sample_indices(self, episode: OpenHEpisode, video_fps: float) -> np.ndarray:
        fpc = self.frames_per_clip
        if self.duration is not None:
            fstp = max(1, int(self.duration * video_fps / fpc))
        elif self.fps is not None:
            fstp = max(1, int(round(video_fps / float(self.fps))))
        else:
            fstp = int(self.frame_step)
        clip_len = max(1, int(fpc * fstp))

        if self.filter_short_videos and episode.length < clip_len:
            raise PhysicalAIOpenHError(
                f"Episode {episode.episode_index} is too short for clip_len={clip_len}: length={episode.length}."
            )
        if episode.length > clip_len:
            end_index = clip_len
            if self.random_clip_sampling:
                end_index = int(np.random.randint(clip_len, episode.length))
            start_index = end_index - clip_len
            indices = start_index + (np.arange(fpc, dtype=np.int64) * fstp)
            indices = np.clip(indices, start_index, episode.length - 1).astype(np.int64)
        else:
            sample_points = max(1, episode.length // fstp)
            sample_points = min(fpc, sample_points)
            indices = np.arange(sample_points, dtype=np.int64) * fstp
            if sample_points < fpc:
                indices = np.concatenate(
                    (indices, np.ones(fpc - sample_points, dtype=np.int64) * (episode.length - 1))
                )
            indices = np.clip(indices, 0, episode.length - 1).astype(np.int64)
        return indices

    @staticmethod
    def _require_pyarrow():
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "PhysicalAIOpenHDataset requires pyarrow to read LeRobot Parquet kinematics. "
                "Install repository requirements or run `pip install pyarrow`."
            ) from exc
        return pq

    @staticmethod
    def _table_values_to_numpy(values: list[Any]) -> np.ndarray:
        array = np.asarray(values)
        if array.dtype == object:
            try:
                array = np.stack([np.asarray(v) for v in values], axis=0)
            except Exception:
                return np.asarray([], dtype=np.float32)
        return array

    def _load_aux_tensors(self, episode: OpenHEpisode, indices: np.ndarray) -> dict[str, Any]:
        columns = []
        for key in episode.dataset.action_keys + episode.dataset.state_keys + episode.dataset.proprio_keys:
            if key not in columns:
                columns.append(key)
        aux: dict[str, Any] = {
            "episode_index": int(episode.episode_index),
            "embodiment": episode.embodiment,
            "tasks": ";".join(episode.tasks),
        }
        if not columns:
            return aux

        pq = self._require_pyarrow()
        parquet_path = self._repo_file_path(self._episode_data_path(episode))
        try:
            table = pq.read_table(parquet_path, columns=columns)
        except Exception as exc:
            raise PhysicalAIOpenHError(f"Failed to read Parquet kinematics from {parquet_path}: {exc}") from exc

        missing_columns = [key for key in columns if key not in table.column_names]
        if missing_columns:
            raise PhysicalAIOpenHError(f"Parquet file {parquet_path} is missing columns {missing_columns}.")

        def read_group(keys: list[str]) -> torch.Tensor | None:
            parts = []
            for key in keys:
                values = [table[key][int(i)].as_py() for i in indices]
                array = self._table_values_to_numpy(values)
                if array.size == 0 or not np.issubdtype(array.dtype, np.number):
                    continue
                if array.ndim == 1:
                    array = array[:, None]
                parts.append(array.astype(np.float32))
            if not parts:
                return None
            return torch.from_numpy(np.concatenate(parts, axis=1))

        action = read_group(episode.dataset.action_keys)
        state = read_group(episode.dataset.state_keys)
        proprio = read_group(episode.dataset.proprio_keys)
        if action is not None:
            aux["action"] = action
        if state is not None:
            aux["state"] = state
        if proprio is not None:
            aux["proprio"] = proprio
        return aux

    def _load_episode(self, episode: OpenHEpisode):
        video_key = self._select_video_key(episode)
        video_path = self._repo_file_path(self._episode_video_path(episode, video_key))
        try:
            vr = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
        except Exception as exc:
            raise PhysicalAIOpenHError(f"Failed to read video {video_path}: {exc}") from exc

        video_fps = float(vr.get_avg_fps() or episode.dataset.info.get("fps") or 1.0)
        indices = self._sample_indices(episode, video_fps=video_fps)
        indices = np.clip(indices, 0, max(0, len(vr) - 1)).astype(np.int64)
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()

        if self.transform is not None:
            buffer = [self.transform(buffer)]
        else:
            buffer = [torch.from_numpy(buffer)]

        aux = self._load_aux_tensors(episode, indices)
        aux["video_key"] = video_key
        return buffer, episode.label, aux, [indices]

    def required_metadata_files(self) -> list[str]:
        files = []
        for dataset in self.datasets:
            files.extend(f"{dataset.root}/meta/{name}" for name in REQUIRED_META_FILES)
        return sorted(set(files))

    def required_episode_files(self, include_all_video_keys: bool | None = None) -> list[str]:
        include_all = self.random_video_key if include_all_video_keys is None else include_all_video_keys
        files = []
        for episode in self.episodes:
            files.append(self._episode_data_path(episode))
            video_keys = episode.dataset.video_keys if include_all else [self._select_video_key(episode)]
            for key in video_keys:
                files.append(self._episode_video_path(episode, key))
        return sorted(set(files))

    def required_files(self, include_all_video_keys: bool | None = None) -> list[str]:
        return sorted(set(self.required_metadata_files() + self.required_episode_files(include_all_video_keys)))

    def __getitem__(self, index: int):
        tries = 0
        while tries < self.max_load_retries:
            episode = self.episodes[index]
            try:
                return self._load_episode(episode)
            except PhysicalAIOpenHError as exc:
                if self.local_files_only:
                    raise
                tries += 1
                logger.warning("Failed to load PhysicalAI Open-H episode %s: %s", episode.episode_index, exc)
                index = int(np.random.randint(len(self)))
            except Exception as exc:
                tries += 1
                logger.warning("Failed to load PhysicalAI Open-H episode %s: %s", episode.episode_index, exc)
                index = int(np.random.randint(len(self)))
        raise PhysicalAIOpenHError(f"Failed to load a PhysicalAI Open-H sample after {self.max_load_retries} retries.")

    def __len__(self) -> int:
        return len(self.episodes)


def make_physicalai_openh(
    data_paths,
    batch_size,
    frames_per_clip=16,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    filter_short_videos=False,
    transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    num_batches_per_epoch=None,
    dataset_kwargs=None,
):
    if dataset_fpcs is not None:
        if len(set(dataset_fpcs)) != 1:
            raise ValueError("PhysicalAIOpenH currently expects one frames-per-clip value across dataset roots.")
        frames_per_clip = int(dataset_fpcs[0])
    dataset_kwargs = dict(dataset_kwargs or {})
    dataset = PhysicalAIOpenHDataset(
        dataset_roots=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        duration=duration,
        fps=fps,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        filter_short_videos=filter_short_videos,
        transform=transform,
        **dataset_kwargs,
    )

    if num_batches_per_epoch is not None:
        logger.warning(
            "num_batches_per_epoch is controlled by the training loop for PhysicalAIOpenH; "
            "standard DistributedSampler does not cap samples per replica."
        )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader_cls = torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    data_loader = loader_cls(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("PhysicalAI Open-H unsupervised data loader created")
    return dataset, data_loader, dist_sampler
