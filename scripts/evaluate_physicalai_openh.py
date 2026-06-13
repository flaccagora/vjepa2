#!/usr/bin/env python3
"""Evaluate V-JEPA 2.1 representations on PhysicalAI Open-H episodes.

This script intentionally reads a bounded number of episodes. It is suitable
for smoke tests and lightweight representation checks, not full benchmark runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_2_1.transforms import make_transforms
from app.vjepa_2_1.utils import init_video_model
from src.datasets.physicalai_openh import PhysicalAIOpenHDataset
from src.utils.checkpoint_loader import robust_checkpoint_loader


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="Training config used to build the encoder.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Fine-tuned or pretrained checkpoint.")
    parser.add_argument("--split", default="val", help="Episode split to evaluate.")
    parser.add_argument("--max-samples", type=int, default=32, help="Maximum episode clips to read.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz output for representations.")
    parser.add_argument(
        "--action-readout",
        action="store_true",
        help="Fit a simple linear action readout if actions exist.",
    )
    return parser.parse_args()


def strip_prefixes(key: str) -> str:
    for prefix in ("module.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    if key.startswith("module.backbone."):
        key = key[len("module.backbone.") :]
    return key


def load_encoder_state(encoder: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location=torch.device("cpu"))
    state = None
    for key in ("encoder", "target_encoder", "ema_encoder"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    if state is None:
        raise RuntimeError(f"No encoder, target_encoder, or ema_encoder state found in {checkpoint_path}.")

    normalized = {strip_prefixes(k): v for k, v in state.items()}
    compatible = {}
    for key, value in encoder.state_dict().items():
        candidate = state.get(key)
        if candidate is None:
            candidate = normalized.get(strip_prefixes(key))
        compatible[key] = value if candidate is None or candidate.shape != value.shape else candidate
    msg = encoder.load_state_dict(compatible, strict=False)
    print(f"Loaded encoder state from {checkpoint_path}: {msg}")


def build_dataset(cfg: dict, split: str, max_samples: int):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    data_aug = cfg.get("data_aug", {})
    dataset_kwargs = dict(data_cfg.get("dataset_kwargs", {}))
    for key in DATASET_KWARG_KEYS:
        if key in data_cfg:
            dataset_kwargs[key] = data_cfg[key]
    dataset_kwargs["split"] = split
    dataset_kwargs["max_episodes"] = max_samples
    dataset_kwargs.setdefault("require_action", False)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=data_aug.get("eval_resize_aspect_ratio", [1.0, 1.0]),
        random_resize_scale=data_aug.get("eval_resize_scale", [1.0, 1.0]),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg.get("crop_size", 384),
    )
    return PhysicalAIOpenHDataset(
        dataset_roots=data_cfg["datasets"],
        frames_per_clip=data_cfg["dataset_fpcs"][0],
        fps=data_cfg.get("fps"),
        frame_step=data_cfg.get("frame_step"),
        duration=data_cfg.get("duration"),
        transform=transform,
        **dataset_kwargs,
    )


def build_encoder(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    encoder, _ = init_video_model(
        uniform_power=model_cfg.get("uniform_power", False),
        use_mask_tokens=model_cfg.get("use_mask_tokens", True),
        num_mask_tokens=1,
        zero_init_mask_tokens=model_cfg.get("zero_init_mask_tokens", True),
        device=device,
        patch_size=data_cfg.get("patch_size", 16),
        max_num_frames=max(data_cfg["dataset_fpcs"]),
        tubelet_size=data_cfg.get("tubelet_size", 2),
        model_name=model_cfg["model_name"],
        crop_size=data_cfg.get("crop_size", 384),
        pred_depth=model_cfg.get("pred_depth", 12),
        pred_num_heads=model_cfg.get("pred_num_heads"),
        pred_embed_dim=model_cfg.get("pred_embed_dim", 384),
        is_causal=model_cfg.get("is_causal", False),
        pred_is_causal=model_cfg.get("pred_is_causal", False),
        use_sdpa=cfg.get("meta", {}).get("use_sdpa", False),
        use_silu=model_cfg.get("use_silu", False),
        use_pred_silu=model_cfg.get("use_pred_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        use_rope=model_cfg.get("use_rope", False),
        use_activation_checkpointing=False,
        return_all_tokens=True,
        chop_last_n_tokens=cfg.get("loss", {}).get("shift_by_n", 0),
        init_type=model_cfg.get("init_type", "default"),
        img_temporal_dim_size=model_cfg.get("img_temporal_dim_size"),
        n_registers=model_cfg.get("n_registers", 0),
        n_registers_predictor=model_cfg.get("n_registers_predictor", 0),
        has_cls_first=model_cfg.get("has_cls_first", False),
        interpolate_rope=model_cfg.get("interpolate_rope", False),
        modality_embedding=model_cfg.get("modality_embedding", False),
    )
    encoder.eval()
    return encoder


def pooled_representation(encoder, clip: torch.Tensor, device: torch.device) -> np.ndarray:
    clip = clip.unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        features = encoder([clip], gram_mode=False, training_mode=False)
        if isinstance(features, list):
            features = features[-1]
        features = F.layer_norm(features, (features.size(-1),))
        pooled = features.mean(dim=1)
    return pooled.squeeze(0).float().cpu().numpy()


def fit_action_readout(features: np.ndarray, actions: np.ndarray) -> dict[str, float]:
    if len(features) < 4:
        raise RuntimeError("Need at least 4 samples for a linear action readout smoke evaluation.")
    split = max(1, int(round(0.8 * len(features))))
    split = min(split, len(features) - 1)
    x_train, x_val = features[:split], features[split:]
    y_train, y_val = actions[:split], actions[split:]
    x_train_aug = np.concatenate([x_train, np.ones((len(x_train), 1), dtype=x_train.dtype)], axis=1)
    x_val_aug = np.concatenate([x_val, np.ones((len(x_val), 1), dtype=x_val.dtype)], axis=1)
    weights = np.linalg.pinv(x_train_aug) @ y_train
    pred = x_val_aug @ weights
    mse = float(np.mean((pred - y_val) ** 2))
    return {"action_readout_mse": mse, "train_samples": float(len(x_train)), "val_samples": float(len(x_val))}


def main() -> None:
    args = parse_args()
    with args.config.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(args.device)
    dataset = build_dataset(cfg, split=args.split, max_samples=args.max_samples)
    encoder = build_encoder(cfg, device=device)
    load_encoder_state(encoder, args.checkpoint)

    reps, labels, action_targets = [], [], []
    missing_action = False
    for idx in range(min(args.max_samples, len(dataset))):
        clips, label, aux, _ = dataset[idx]
        reps.append(pooled_representation(encoder, clips[0], device=device))
        labels.append(int(label))
        if "action" in aux:
            action_targets.append(aux["action"].float().mean(dim=0).numpy())
        else:
            missing_action = True

    reps_np = np.stack(reps, axis=0)
    labels_np = np.asarray(labels, dtype=np.int64)
    print(f"representations: shape={reps_np.shape} labels={labels_np.shape} split={args.split}")

    metrics = {}
    if args.action_readout:
        if missing_action or not action_targets:
            raise RuntimeError(
                "Action readout requested, but at least one sampled episode has no action tensor. "
                "Set dataset_kwargs.require_action: true to fail earlier during dataset construction."
            )
        metrics.update(fit_action_readout(reps_np, np.stack(action_targets, axis=0)))
        print(metrics)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"features": reps_np, "labels": labels_np}
        if action_targets:
            payload["actions"] = np.stack(action_targets, axis=0)
        np.savez(args.output, **payload)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
