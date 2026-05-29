# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import json
import shutil
import subprocess
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torchvision
from decord import cpu, VideoReader

from src.datasets.utils.dataloader import (
    ConcatIndices,
    MonitoredDataset,
    NondeterministicDataLoader,
)
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    preprocess_metadata=None,
    use_preprocess_metadata=False,
    length_weighted_sampling=False,
    video_backend="decord",
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        preprocess_metadata=preprocess_metadata,
        use_preprocess_metadata=use_preprocess_metadata,
        length_weighted_sampling=length_weighted_sampling,
        video_backend=video_backend,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")
    if dataset.sample_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        preprocess_metadata=None,
        use_preprocess_metadata=False,
        length_weighted_sampling=False,
        video_backend="decord",
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.use_preprocess_metadata = use_preprocess_metadata
        self.preprocess_metadata = self._load_preprocess_metadata(preprocess_metadata) if use_preprocess_metadata else {}
        self.length_weighted_sampling = length_weighted_sampling
        self.video_backend = str(video_backend or "decord").lower()

        if self.video_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
            logger.warning("video_backend=ffmpeg requested, but ffmpeg is not available on PATH")

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(
                f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
            )

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError(
                    "Frames per clip not properly specified for NFS data paths"
                )
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.'
            )

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == ".csv":
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    # In image captioning datasets where we have space, we use :: as delimiter.
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == ".npy":
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.samples = samples
        self.labels = labels
        self.sample_weights = self._build_sample_weights()

    @staticmethod
    def _path_keys(path):
        path = pathlib.Path(path)
        keys = {str(path)}
        try:
            keys.add(str(path.resolve()))
        except Exception:
            pass
        return keys

    def _load_preprocess_metadata(self, metadata_path):
        if metadata_path is None:
            return {}
        metadata_paths = metadata_path if isinstance(metadata_path, (list, tuple)) else [metadata_path]
        metadata = {}
        for path in metadata_paths:
            with open(path, "r") as f:
                doc = json.load(f)
            for item in doc.get("videos", []):
                value = item.get("source_path")
                if value:
                    for key in self._path_keys(value):
                        metadata[key] = item
        logger.info("Loaded preprocess metadata for %d video path keys", len(metadata))
        return metadata

    def _metadata_for_sample(self, sample):
        if not self.preprocess_metadata:
            return None
        for key in self._path_keys(sample):
            item = self.preprocess_metadata.get(key)
            if item is not None:
                return item
        return None

    @staticmethod
    def _duration_from_metadata(metadata):
        if not metadata:
            return None
        duration = metadata.get("duration")
        if duration is None:
            for frames_key, fps_key in (
                ("source_frames", "source_fps"),
                ("written_frames", "output_fps"),
            ):
                frames = metadata.get(frames_key)
                fps = metadata.get(fps_key)
                try:
                    if frames is not None and fps:
                        duration = float(frames) / float(fps)
                        break
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        if duration is None:
            return None
        try:
            return max(float(duration), 0.0)
        except (TypeError, ValueError):
            return None

    def _duration_weight_for_sample(self, sample):
        metadata = self._metadata_for_sample(sample)
        duration = self._duration_from_metadata(metadata)
        if duration is None:
            return 1.0

        black_duration = 0.0
        for start, end in self._black_intervals_from_metadata(metadata):
            black_duration += max(0.0, end - start)
        return max(duration - black_duration, 1.0)

    def _build_sample_weights(self):
        if not self.length_weighted_sampling and self.datasets_weights is None:
            return None

        sample_weights = []
        offset = 0
        for dataset_idx, ns in enumerate(self.num_samples_per_dataset):
            dataset_samples = self.samples[offset : offset + ns]
            if self.length_weighted_sampling:
                weights = [self._duration_weight_for_sample(sample) for sample in dataset_samples]
                raw_weights = list(weights)
                total_weight = sum(weights)
                if total_weight <= 0:
                    weights = [1.0 for _ in dataset_samples]
                    raw_weights = list(weights)
                    total_weight = sum(weights)
            else:
                weights = [1.0 for _ in dataset_samples]
                raw_weights = list(weights)
                total_weight = float(ns)

            dataset_weight = 1.0
            if self.datasets_weights is not None:
                dataset_weight = float(self.datasets_weights[dataset_idx])
                weights = [dataset_weight * weight / total_weight for weight in weights]

            sample_weights.extend(weights)
            if self.length_weighted_sampling:
                logger.info(
                    "Length-weighted sampling for dataset %d: samples=%d "
                    "min=%.2fs max=%.2fs mean=%.2fs dataset_weight=%.4g",
                    dataset_idx,
                    ns,
                    min(raw_weights) if raw_weights else 0.0,
                    max(raw_weights) if raw_weights else 0.0,
                    (sum(raw_weights) / len(raw_weights)) if raw_weights else 0.0,
                    dataset_weight,
                )
            offset += ns

        return sample_weights

    @staticmethod
    def _black_intervals_from_metadata(metadata):
        if not metadata:
            return []
        intervals = metadata.get("black_sections_removed") or metadata.get("black_sections_detected") or []
        result = []
        for item in intervals:
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                result.append((start, end))
        return result

    @staticmethod
    def _valid_frame_segments(num_frames, video_fps, black_intervals):
        if not black_intervals or not video_fps:
            return None
        black_frame_intervals = []
        for start, end in black_intervals:
            start_idx = max(0, int(math.floor(start * video_fps)))
            end_idx = min(num_frames, int(math.ceil(end * video_fps)))
            if end_idx > start_idx:
                black_frame_intervals.append((start_idx, end_idx))
        if not black_frame_intervals:
            return None

        black_frame_intervals.sort()
        segments = []
        cursor = 0
        for start_idx, end_idx in black_frame_intervals:
            if start_idx > cursor:
                segments.append((cursor, start_idx))
            cursor = max(cursor, end_idx)
        if cursor < num_frames:
            segments.append((cursor, num_frames))
        return [(start, end) for start, end in segments if end > start]

    def _select_valid_segment(self, valid_segments, clip_idx):
        if self.random_clip_sampling:
            lengths = np.array([end - start for start, end in valid_segments], dtype=np.float64)
            probs = lengths / lengths.sum()
            return valid_segments[int(np.random.choice(len(valid_segments), p=probs))]
        segment_idx = min(
            int(clip_idx * len(valid_segments) / max(1, self.num_clips)),
            len(valid_segments) - 1,
        )
        return valid_segments[segment_idx]

    def _sample_indices_from_segment(self, segment_start, segment_end, fpc, fstp, clip_len):
        segment_len = max(1, segment_end - segment_start)
        if segment_len > clip_len:
            end_indx = clip_len
            if self.random_clip_sampling:
                end_indx = np.random.randint(clip_len, segment_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=fpc)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
        else:
            sample_points = max(1, segment_len // fstp)
            indices = np.linspace(0, segment_len, num=sample_points)
            indices = np.concatenate(
                (
                    indices,
                    np.ones(fpc - sample_points) * segment_len,
                )
            )
            indices = np.clip(indices, 0, segment_len - 1).astype(np.int64)
        return indices + segment_start

    def _sample_video_indices(self, num_frames, video_fps, metadata, fpc, fstp, clip_len):
        valid_segments = None
        if metadata is not None:
            valid_segments = self._valid_frame_segments(
                num_frames=num_frames,
                video_fps=video_fps,
                black_intervals=self._black_intervals_from_metadata(metadata),
            )
        if valid_segments is not None and not valid_segments:
            return None, None, 0

        sample_len = num_frames
        max_segment_len = num_frames
        if valid_segments is not None:
            sample_len = sum(end - start for start, end in valid_segments)
            max_segment_len = max(end - start for start, end in valid_segments)

        partition_len = max(1, sample_len // self.num_clips)
        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if valid_segments is not None:
                segment_start, segment_end = self._select_valid_segment(valid_segments, i)
                indices = self._sample_indices_from_segment(segment_start, segment_end, fpc, fstp, clip_len)
            elif partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            elif not self.allow_clip_overlap:
                indices = np.linspace(0, partition_len, num=partition_len // fstp)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(fpc - partition_len // fstp) * partition_len,
                    )
                )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                sample_window_len = max(1, min(clip_len, sample_len)) - 1
                sample_points = max(1, sample_window_len // fstp)
                indices = np.linspace(0, sample_window_len, num=sample_points)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(fpc - sample_points) * sample_window_len,
                    )
                )
                indices = np.clip(indices, 0, sample_window_len).astype(np.int64)
                clip_step = 0
                if sample_len > clip_len and self.num_clips > 1:
                    clip_step = (sample_len - clip_len) // (self.num_clips - 1)
                indices = indices + i * clip_step

            indices = np.clip(indices, 0, num_frames - 1).astype(np.int64)
            clip_indices.append(indices)
            all_indices.extend(list(indices))

        return all_indices, clip_indices, max_segment_len

    @staticmethod
    def _apply_metadata_crop(buffer, metadata):
        if not metadata:
            return buffer
        required = ("crop_x", "crop_y", "crop_width", "crop_height")
        if not all(k in metadata for k in required):
            return buffer
        height, width = buffer.shape[1], buffer.shape[2]
        x1 = max(0, int(metadata["crop_x"]))
        y1 = max(0, int(metadata["crop_y"]))
        x2 = min(width, x1 + max(1, int(metadata["crop_width"])))
        y2 = min(height, y1 + max(1, int(metadata["crop_height"])))
        if x2 <= x1 or y2 <= y1:
            return buffer
        return buffer[:, y1:y2, x1:x2, :]

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        # Keep trying to load videos until you find a valid sample
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                if sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                    loaded_sample = self.get_item_image(index)
                else:
                    loaded_sample = self.get_item_video(index)

            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return loaded_sample

    def get_item_video(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        buffer, clip_indices = self.loadvideo_decord(
            sample, frames_per_clip
        )  # [T H W 3]
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def get_item_image(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            image_tensor = torchvision.io.read_image(
                path=sample, mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception:
            return
        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding the input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]

        if self.shared_transform is not None:
            # Technically we can have only transform, doing this just for the sake of consistency with videos.
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """Load video content using Decord"""

        fname = sample
        metadata = self._metadata_for_sample(fname)
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        if self.video_backend == "ffmpeg":
            loaded = self._loadvideo_ffmpeg_metadata(fname, metadata, fpc)
            if loaded is not None:
                return loaded

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fstp = self.frame_step
        video_fps = None
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)
        if video_fps is None:
            try:
                video_fps = float(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)
                video_fps = None

        all_indices, clip_indices, max_segment_len = self._sample_video_indices(
            num_frames=len(vr),
            video_fps=video_fps,
            metadata=metadata,
            fpc=fpc,
            fstp=fstp,
            clip_len=clip_len,
        )
        if all_indices is None:
            warnings.warn(f"skipping video with no valid non-black frames {fname=}")
            return [], None

        if self.filter_short_videos and max_segment_len < clip_len:
            warnings.warn(f"skipping video with max valid segment length {max_segment_len}")
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        buffer = vr.get_batch(all_indices).asnumpy()
        buffer = self._apply_metadata_crop(buffer, metadata)
        return buffer, clip_indices

    def _loadvideo_ffmpeg_metadata(self, fname, metadata, fpc):
        if not metadata or shutil.which("ffmpeg") is None:
            return None
        required = ("source_frames", "source_fps", "crop_x", "crop_y", "crop_width", "crop_height")
        if not all(k in metadata for k in required):
            return None
        if self.duration is not None:
            return None

        try:
            num_frames = int(metadata["source_frames"])
            video_fps = float(metadata["source_fps"])
            crop_x = max(0, int(metadata["crop_x"]))
            crop_y = max(0, int(metadata["crop_y"]))
            crop_width = max(1, int(metadata["crop_width"]))
            crop_height = max(1, int(metadata["crop_height"]))
        except (TypeError, ValueError):
            return None
        if num_frames <= 0 or video_fps <= 0:
            return None

        fstp = self.frame_step
        if self.fps is not None:
            fstp = max(1, int(video_fps) // int(self.fps))
        if fstp is None or fstp <= 0:
            return None
        clip_len = int(fpc * fstp)

        all_indices, clip_indices, max_segment_len = self._sample_video_indices(
            num_frames=num_frames,
            video_fps=video_fps,
            metadata=metadata,
            fpc=fpc,
            fstp=fstp,
            clip_len=clip_len,
        )
        if all_indices is None:
            warnings.warn(f"skipping video with no valid non-black frames {fname=}")
            return [], None
        if self.filter_short_videos and max_segment_len < clip_len:
            warnings.warn(f"skipping video with max valid segment length {max_segment_len}")
            return [], None

        requested_indices = np.asarray(all_indices, dtype=np.int64)
        decode_indices = np.unique(requested_indices)
        start_frame = int(decode_indices.min())
        rel_indices = (decode_indices - start_frame).astype(np.int64)
        duration = (int(rel_indices.max()) + 2) / video_fps
        select = "+".join(f"eq(n\\,{int(i)})" for i in rel_indices)
        vf = f"select={select},crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_frame / video_fps:.6f}",
            "-i",
            fname,
            "-t",
            f"{duration:.6f}",
            "-vf",
            vf,
            "-vsync",
            "0",
            "-frames:v",
            str(len(decode_indices)),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as exc:
            logger.warning("ffmpeg video load failed for %s; falling back to decord: %s", fname, exc)
            return None

        expected = len(decode_indices) * crop_height * crop_width * 3
        if len(proc.stdout) != expected:
            logger.warning(
                "ffmpeg video load returned %d bytes, expected %d for %s; falling back to decord",
                len(proc.stdout),
                expected,
                fname,
            )
            return None
        decoded = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
            len(decode_indices), crop_height, crop_width, 3
        )
        if np.array_equal(requested_indices, decode_indices):
            buffer = decoded
        else:
            decode_positions = {int(index): pos for pos, index in enumerate(decode_indices)}
            reorder = [decode_positions[int(index)] for index in requested_indices]
            buffer = decoded[reorder]
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
