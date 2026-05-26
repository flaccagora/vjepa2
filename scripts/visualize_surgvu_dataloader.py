#!/usr/bin/env python3
"""Save contact sheets from the VJEPA-2.1 SurgVU training dataloader.

This uses the same VideoDataset, data augmentations, temporal sampling, and mask
collator path as app/vjepa_2_1/train.py, then writes denormalized frame grids so
the training input can be inspected before launching long cluster jobs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_2_1.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training YAML config.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/surgvu_dataloader_preview"))
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=4, help="Maximum samples to save per batch.")
    parser.add_argument("--frames-per-row", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None, help="Override config data.batch_size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for easier local debugging.")
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument(
        "--disable-augment",
        action="store_true",
        help="Use deterministic resize/crop-style settings for easier inspection. By default, config augmentations are used.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def denormalize_clip(clip: torch.Tensor) -> torch.Tensor:
    """Convert [C, T, H, W] normalized training tensor to [T, H, W, C] uint8."""
    clip = clip.detach().cpu().float()
    clip = (clip * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    clip = (clip.permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8)
    return clip


def make_contact_sheet(frames: torch.Tensor, frame_indices: list[int], frames_per_row: int) -> Image.Image:
    """Create a PIL contact sheet from [T, H, W, C] uint8 frames."""
    num_frames, height, width, _ = frames.shape
    rows = int(np.ceil(num_frames / frames_per_row))
    label_h = 20
    canvas = Image.new("RGB", (frames_per_row * width, rows * (height + label_h)), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i in range(num_frames):
        row = i // frames_per_row
        col = i % frames_per_row
        x = col * width
        y = row * (height + label_h)
        frame = Image.fromarray(frames[i].numpy(), mode="RGB")
        canvas.paste(frame, (x, y))
        label = f"t={i}"
        if i < len(frame_indices):
            label += f" src={frame_indices[i]}"
        draw.text((x + 4, y + height + 3), label, fill=(235, 235, 235))

    return canvas


def summarize_masks(masks_enc: list[torch.Tensor], masks_pred: list[torch.Tensor]) -> list[dict[str, int | list[int]]]:
    summary = []
    for i, (enc, pred) in enumerate(zip(masks_enc, masks_pred)):
        summary.append(
            {
                "mask_id": i,
                "encoder_mask_shape": list(enc.shape),
                "predictor_mask_shape": list(pred.shape),
                "encoder_tokens_per_sample": int(enc.shape[-1]),
                "predictor_tokens_per_sample": int(pred.shape[-1]),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    cfgs_data = cfg["data"]
    cfgs_aug = dict(cfg.get("data_aug", {}))
    cfgs_mask = cfg["mask"]

    if args.disable_augment:
        cfgs_aug.update(
            {
                "auto_augment": False,
                "motion_shift": False,
                "random_resize_aspect_ratio": [1.0, 1.0],
                "random_resize_scale": [1.0, 1.0],
                "reprob": 0.0,
            }
        )

    batch_size = args.batch_size if args.batch_size is not None else cfgs_data.get("batch_size", 1)
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size", 16)
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    filter_long_videos = cfgs_data.get("filter_long_videos", int(1e9))
    if filter_long_videos is None:
        filter_long_videos = int(1e18)

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=cfgs_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=cfgs_aug.get("random_resize_scale", [0.3, 1.0]),
        reprob=cfgs_aug.get("reprob", 0.0),
        auto_augment=cfgs_aug.get("auto_augment", False),
        motion_shift=cfgs_aug.get("motion_shift", False),
        crop_size=crop_size,
    )

    loader, sampler = init_data(
        data=cfgs_data.get("dataset_type", "VideoDataset"),
        root_path=cfgs_data.get("datasets", []),
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=cfgs_data.get("fps"),
        transform=transform,
        rank=0,
        world_size=1,
        datasets_weights=cfgs_data.get("datasets_weights"),
        collator=mask_collator,
        num_workers=args.num_workers,
        pin_mem=False,
        filter_short_videos=cfgs_data.get("filter_short_videos", False),
        filter_long_videos=filter_long_videos,
        preprocess_metadata=cfgs_data.get("preprocess_metadata"),
        use_preprocess_metadata=(
            bool(cfgs_data.get("preprocess_metadata"))
            if cfgs_data.get("use_preprocess_metadata") is None
            else cfgs_data.get("use_preprocess_metadata")
        ),
        persistent_workers=False,
    )
    sampler.set_epoch(0)

    manifest = {
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "dataset_fpcs": dataset_fpcs,
        "crop_size": crop_size,
        "fps": cfgs_data.get("fps"),
        "disable_augment": args.disable_augment,
        "saved": [],
    }

    iterator = iter(loader)
    saved = 0
    for batch_idx in range(args.num_batches):
        batch = next(iterator)
        for fpc_group_idx, fpc_sample in enumerate(batch):
            udata, masks_enc, masks_pred = fpc_sample
            clip_list, labels, clip_indices = udata
            mask_summary = summarize_masks(masks_enc, masks_pred)

            for clip_id, clips in enumerate(clip_list):
                batch_count = min(clips.shape[0], args.max_samples)
                for sample_idx in range(batch_count):
                    clip = clips[sample_idx]
                    frames = denormalize_clip(clip)
                    src_indices = clip_indices[clip_id][sample_idx].detach().cpu().numpy().astype(int).tolist()
                    label = int(labels[sample_idx].item()) if torch.is_tensor(labels[sample_idx]) else int(labels[sample_idx])

                    sheet = make_contact_sheet(frames, src_indices, frames_per_row=args.frames_per_row)
                    filename = (
                        f"batch{batch_idx:03d}_group{fpc_group_idx}_clip{clip_id}_"
                        f"sample{sample_idx}_label{label}.jpg"
                    )
                    output_path = args.out_dir / filename
                    sheet.save(output_path, quality=95)
                    manifest["saved"].append(
                        {
                            "path": str(output_path),
                            "batch": batch_idx,
                            "fpc_group": fpc_group_idx,
                            "clip_id": clip_id,
                            "sample_idx": sample_idx,
                            "label": label,
                            "tensor_shape": list(clip.shape),
                            "source_frame_indices": src_indices,
                            "masks": mask_summary,
                        }
                    )
                    saved += 1

    with (args.out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"saved_images": saved, "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
