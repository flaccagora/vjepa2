#!/usr/bin/env python3
"""Save PCA visualizations of V-JEPA 2.1 dense feature maps.

Examples:

  # First cache the pretrained checkpoint you want to visualize.
  python scripts/cache_vjepa_models.py --model vjepa2_1_vitb
  python scripts/cache_vjepa_models.py --model vjepa2_1_vitl
  python scripts/cache_vjepa_models.py --model vjepa2_1_vitg
  python scripts/cache_vjepa_models.py --model vjepa2_1_vitG

  # Pretrained V-JEPA 2.1 ViT-B on one PNG/JPEG image.
  python scripts/visualize_vjepa2_1_pca.py \
    --config configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml \
    --model-name vit_base \
    --checkpoint checkpoints/vjepa2_1_vitb_dist_vitG_384.pt \
    --images data/cat.jpg \
    --out-dir output/vjepa2_1_pca_vitb

  # Pretrained V-JEPA 2.1 ViT-L on one PNG/JPEG image.
  python scripts/visualize_vjepa2_1_pca.py \
    --config configs/train_2_1/vitl16/pretrain-256px-16f.yaml \
    --model-name vit_large \
    --crop-size 384 \
    --checkpoint checkpoints/vjepa2_1_vitl_dist_vitG_384.pt \
    --images data/cat.jpg \
    --out-dir output/vjepa2_1_pca_vitl

  # Pretrained V-JEPA 2.1 ViT-g on one PNG/JPEG image.
  python scripts/visualize_vjepa2_1_pca.py \
    --config configs/train_2_1/vitg16/pretrain-256px-16f.yaml \
    --model-name vit_giant_xformers \
    --crop-size 384 \
    --checkpoint checkpoints/vjepa2_1_vitg_384.pt \
    --images data/cat.jpg \
    --out-dir output/vjepa2_1_pca_vitg

  # Pretrained V-JEPA 2.1 ViT-G on one PNG/JPEG image.
  python scripts/visualize_vjepa2_1_pca.py \
    --config configs/train_2_1/vitG16/pretrain-256px-16f.yaml \
    --model-name vit_gigantic_xformers \
    --crop-size 384 \
    --checkpoint checkpoints/vjepa2_1_vitG_384.pt \
    --images data/cat.jpg \
    --out-dir output/vjepa2_1_pca_vitG

  # Fine-tuned SurgVU checkpoint on samples from the SurgVU manifest.
  python scripts/visualize_vjepa2_1_pca.py \
    --config configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml \
    --checkpoint output/surgvu_slurm/vjepa2_1_vitb_384px_16f/latest.pth.tar \
    --manifest data/surgvu/manifest_train.csv \
    --num-manifest-samples 4 \
    --out-dir output/vjepa2_1_pca_surgvu_finetuned

  # Side-by-side PCA maps for the same image/video frames across models.
  python scripts/visualize_vjepa2_1_pca.py \
    --compare \
    --compare-pretrained vjepa2_1_vitb vjepa2_1_vitg vjepa2_1_vitG \
    --checkpoint output/surgvu_slurm/vjepa2_1_vitb_384px_16f/latest.pth.tar \
    --images data/cat.jpg \
    --out-dir output/vjepa2_1_pca_compare
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_2_1.models import vision_transformer as video_vit


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


PRETRAINED_MODELS = {
    "vjepa2_1_vitb": {
        "label": "pretrained_vitb",
        "model_name": "vit_base",
        "checkpoint": Path("checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"),
    },
    "vjepa2_1_vitl": {
        "label": "pretrained_vitl",
        "model_name": "vit_large",
        "checkpoint": Path("checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"),
    },
    "vjepa2_1_vitg": {
        "label": "pretrained_vitg",
        "model_name": "vit_giant_xformers",
        "checkpoint": Path("checkpoints/vjepa2_1_vitg_384.pt"),
    },
    "vjepa2_1_vitG": {
        "label": "pretrained_vitG",
        "model_name": "vit_gigantic_xformers",
        "checkpoint": Path("checkpoints/vjepa2_1_vitG_384.pt"),
    },
}


@dataclass
class Sample:
    name: str
    frames: list[Image.Image]
    source: str
    frame_indices: list[int] | None = None


@dataclass
class ModelSpec:
    label: str
    model_name: str | None
    checkpoint: Path
    checkpoint_key: str = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/train_2_1/vitb16/surgvu-finetune-384px-16f.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Pretrained or fine-tuned checkpoint. Defaults to meta.read_checkpoint from --config.",
    )
    parser.add_argument(
        "--checkpoint-key",
        default="auto",
        help="Encoder key to load from checkpoint. Use auto for target_encoder/ema_encoder/encoder fallback.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List named pretrained models and whether their checkpoint files are cached.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Write side-by-side PCA sheets for multiple models on the same samples.",
    )
    parser.add_argument(
        "--compare-pretrained",
        nargs="*",
        choices=sorted(PRETRAINED_MODELS),
        default=None,
        help="Named pretrained checkpoints to include. With --compare, defaults to cached named checkpoints.",
    )
    parser.add_argument(
        "--compare-finetuned-checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint to include in --compare. Defaults to --checkpoint if provided.",
    )
    parser.add_argument(
        "--compare-finetuned-label",
        default="finetuned",
        help="Label for the fine-tuned checkpoint column.",
    )
    parser.add_argument(
        "--compare-checkpoint",
        action="append",
        default=[],
        metavar="LABEL:MODEL_NAME:PATH",
        help="Additional model to compare, e.g. surgvu_vitb:vit_base:output/run/latest.pth.tar.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output/vjepa2_1_pca"))
    parser.add_argument("--images", type=Path, nargs="*", default=[], help="PNG/JPEG images to visualize.")
    parser.add_argument("--video", type=Path, default=None, help="Video file to visualize.")
    parser.add_argument("--manifest", type=Path, default=None, help="SurgVU/V-JEPA manifest containing video paths.")
    parser.add_argument("--manifest-index", type=int, default=0, help="First manifest row to visualize.")
    parser.add_argument("--num-manifest-samples", type=int, default=1)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames sampled from each video. Defaults to config data.dataset_fpcs[0].",
    )
    parser.add_argument(
        "--frame-indices",
        default=None,
        help="Comma-separated video frame indices. Overrides uniform sampling.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default=None, help="Override config model.model_name.")
    parser.add_argument("--crop-size", type=int, default=None, help="Override config data.crop_size.")
    parser.add_argument("--patch-size", type=int, default=None, help="Override config data.patch_size.")
    parser.add_argument("--tubelet-size", type=int, default=None, help="Override config data.tubelet_size.")
    parser.add_argument("--overlay", action="store_true", help="Include input/PCA overlay columns in sheets.")
    parser.add_argument("--alpha", type=float, default=0.55, help="PCA overlay opacity (used with --overlay).")
    parser.add_argument(
        "--pca-scope",
        choices=["batch", "sample"],
        default="batch",
        help="Fit PCA once for all samples or separately per sample.",
    )
    parser.add_argument(
        "--max-pca-tokens",
        type=int,
        default=50000,
        help="Subsample tokens used to fit PCA; all tokens are projected.",
    )
    parser.add_argument("--save-npy", action="store_true", help="Also save token feature maps as .npy files.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def safe_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"


def list_named_models() -> None:
    rows = []
    for key, spec in PRETRAINED_MODELS.items():
        checkpoint = spec["checkpoint"]
        rows.append(
            {
                "key": key,
                "label": spec["label"],
                "model_name": spec["model_name"],
                "checkpoint": str(checkpoint),
                "cached": checkpoint.exists(),
            }
        )
    print(json.dumps(rows, indent=2))


def resolve_checkpoint(args: argparse.Namespace, cfg: dict) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    checkpoint = cfg.get("meta", {}).get("read_checkpoint")
    if not checkpoint:
        raise SystemExit("No --checkpoint provided and config meta.read_checkpoint is empty.")
    return Path(checkpoint)


def parse_compare_checkpoint(value: str) -> ModelSpec:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise SystemExit(
            "--compare-checkpoint must use LABEL:MODEL_NAME:PATH, "
            f"got {value!r}"
        )
    label, model_name, checkpoint = parts
    return ModelSpec(label=safe_stem(label), model_name=model_name, checkpoint=Path(checkpoint))


def compare_model_specs(args: argparse.Namespace, cfg: dict) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    pretrained_keys = args.compare_pretrained
    if pretrained_keys is None:
        pretrained_keys = [
            key for key, spec in PRETRAINED_MODELS.items() if spec["checkpoint"].exists()
        ]
    for key in pretrained_keys:
        spec = PRETRAINED_MODELS[key]
        specs.append(
            ModelSpec(
                label=spec["label"],
                model_name=spec["model_name"],
                checkpoint=spec["checkpoint"],
                checkpoint_key=args.checkpoint_key,
            )
        )

    finetuned_checkpoint = args.compare_finetuned_checkpoint or args.checkpoint
    if finetuned_checkpoint is not None:
        specs.append(
            ModelSpec(
                label=safe_stem(args.compare_finetuned_label),
                model_name=args.model_name,
                checkpoint=finetuned_checkpoint,
                checkpoint_key=args.checkpoint_key,
            )
        )

    specs.extend(parse_compare_checkpoint(value) for value in args.compare_checkpoint)

    if len(specs) < 2:
        raise SystemExit(
            "--compare needs at least two models. Cache named pretrained checkpoints "
            "with scripts/cache_vjepa_models.py, pass --checkpoint for a fine-tuned "
            "model, or add --compare-checkpoint LABEL:MODEL_NAME:PATH."
        )
    for spec in specs:
        if not spec.checkpoint.exists():
            raise SystemExit(
                f"Checkpoint for {spec.label!r} not found: {spec.checkpoint}. "
                "Use scripts/cache_vjepa_models.py to cache pretrained models."
            )
    return specs


def config_frame_counts(cfg: dict) -> list[int]:
    frame_counts = cfg.get("data", {}).get("dataset_fpcs") or []
    return [int(value) for value in frame_counts]


def model_num_frames(cfg: dict, args: argparse.Namespace) -> int:
    frame_counts = config_frame_counts(cfg)
    if args.num_frames is not None:
        frame_counts.append(int(args.num_frames))
    return max(frame_counts or [16])


def sample_num_frames(cfg: dict, args: argparse.Namespace) -> int:
    if args.num_frames is not None:
        return int(args.num_frames)
    frame_counts = config_frame_counts(cfg)
    return frame_counts[0] if frame_counts else 16


def build_encoder(
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    model_name_override: str | None = None,
) -> torch.nn.Module:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    model_name = model_name_override or args.model_name or model_cfg.get("model_name", "vit_base")
    crop_size = args.crop_size or data_cfg.get("crop_size", 384)
    patch_size = args.patch_size or data_cfg.get("patch_size", 16)
    tubelet_size = args.tubelet_size or data_cfg.get("tubelet_size", 2)
    num_frames = model_num_frames(cfg, args)

    if model_name not in video_vit.__dict__:
        raise SystemExit(f"Unknown V-JEPA 2.1 model_name: {model_name}")

    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=model_cfg.get("uniform_power", False),
        use_sdpa=model_cfg.get("use_sdpa", cfg.get("meta", {}).get("use_sdpa", False)),
        use_silu=model_cfg.get("use_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        use_activation_checkpointing=False,
        is_causal=model_cfg.get("is_causal", False),
        use_rope=model_cfg.get("use_rope", False),
        init_type=model_cfg.get("init_type", "default"),
        img_temporal_dim_size=model_cfg.get("img_temporal_dim_size"),
        n_registers=model_cfg.get("n_registers", 0),
        has_cls_first=model_cfg.get("has_cls_first", False),
        interpolate_rope=model_cfg.get("interpolate_rope", False),
        modality_embedding=model_cfg.get("modality_embedding", False),
    )
    encoder.to(device)
    encoder.eval()
    return encoder


def strip_prefixes(key: str) -> str:
    for prefix in ("module.backbone.", "module.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def select_checkpoint_state(checkpoint: dict, checkpoint_key: str) -> tuple[str, dict]:
    if checkpoint_key != "auto":
        if checkpoint_key not in checkpoint:
            raise SystemExit(f'Checkpoint key "{checkpoint_key}" not found. Available keys: {sorted(checkpoint.keys())[:30]}')
        return checkpoint_key, checkpoint[checkpoint_key]
    for key in ("target_encoder", "ema_encoder", "encoder"):
        if key in checkpoint:
            return key, checkpoint[key]
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return "state_dict", checkpoint
    raise SystemExit(f"Could not find an encoder state in checkpoint keys: {sorted(checkpoint.keys())[:30]}")


def load_encoder_checkpoint(encoder: torch.nn.Module, checkpoint_path: Path, checkpoint_key: str) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Unsupported checkpoint type in {checkpoint_path}: {type(checkpoint)}")
    selected_key, state = select_checkpoint_state(checkpoint, checkpoint_key)
    normalized = {strip_prefixes(k): v for k, v in state.items()}
    model_state = encoder.state_dict()
    compatible = {}
    loaded = 0
    skipped = 0
    for key, value in model_state.items():
        candidate = normalized.get(strip_prefixes(key))
        if candidate is None or tuple(candidate.shape) != tuple(value.shape):
            compatible[key] = value
            skipped += 1
        else:
            compatible[key] = candidate
            loaded += 1
    encoder.load_state_dict(compatible, strict=False)
    return f"{selected_key}: loaded {loaded} tensors, kept initialized {skipped} tensors"


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    image = image.resize(new_size, Image.Resampling.BICUBIC)
    left = max(0, (image.width - size) // 2)
    top = max(0, (image.height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def frames_to_tensor(frames: list[Image.Image], crop_size: int) -> tuple[torch.Tensor, list[Image.Image]]:
    processed = [center_crop_resize(frame, crop_size) for frame in frames]
    arrays = [np.asarray(frame, dtype=np.float32) / 255.0 for frame in processed]
    tensor = torch.from_numpy(np.stack(arrays, axis=0)).permute(3, 0, 1, 2)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, processed


def parse_frame_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise SystemExit("--frame-indices was provided but no valid indices were parsed.")
    return indices


def read_video(path: Path, num_frames: int, frame_indices: list[int] | None) -> tuple[list[Image.Image], list[int]]:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise SystemExit("Reading videos requires decord. Install repo requirements or pass --images instead.") from exc

    vr = VideoReader(str(path), num_threads=1, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise SystemExit(f"Video has no readable frames: {path}")
    if frame_indices is None:
        frame_indices = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
    frame_indices = [min(max(0, idx), total - 1) for idx in frame_indices]
    batch = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frame).convert("RGB") for frame in batch]
    return frames, frame_indices


def manifest_video_paths(path: Path, start: int, count: int) -> list[Path]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",")[0].split()[0]
            if first.lower() in {"path", "video", "video_path"}:
                continue
            rows.append(Path(first))
    return rows[start : start + count]


def collect_samples(args: argparse.Namespace, cfg: dict) -> list[Sample]:
    num_frames = sample_num_frames(cfg, args)
    explicit_indices = parse_frame_indices(args.frame_indices)
    samples: list[Sample] = []

    for image_path in args.images:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported image suffix: {image_path}")
        samples.append(Sample(name=safe_stem(str(image_path)), frames=[Image.open(image_path).convert("RGB")], source=str(image_path)))

    if args.video is not None:
        frames, indices = read_video(args.video, num_frames, explicit_indices)
        samples.append(Sample(name=safe_stem(str(args.video)), frames=frames, source=str(args.video), frame_indices=indices))

    if args.manifest is not None:
        for video_path in manifest_video_paths(args.manifest, args.manifest_index, args.num_manifest_samples):
            frames, indices = read_video(video_path, num_frames, explicit_indices)
            samples.append(Sample(name=safe_stem(str(video_path)), frames=frames, source=str(video_path), frame_indices=indices))

    if not samples:
        raise SystemExit("Provide at least one input via --images, --video, or --manifest.")
    return samples


@torch.inference_mode()
def extract_features(encoder: torch.nn.Module, clip: torch.Tensor, device: torch.device) -> torch.Tensor:
    output = encoder(clip.unsqueeze(0).to(device))
    if isinstance(output, list):
        output = output[-1]
    return output.squeeze(0).detach().float().cpu()


def fit_pca(tokens: torch.Tensor, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2:
        tokens = tokens.reshape(-1, tokens.shape[-1])
    if tokens.shape[0] > max_tokens:
        step = math.ceil(tokens.shape[0] / max_tokens)
        fit_tokens = tokens[::step][:max_tokens]
    else:
        fit_tokens = tokens
    mean = fit_tokens.mean(dim=0, keepdim=True)
    centered = fit_tokens - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:3].T.contiguous()
    return mean.squeeze(0), components


def pca_limits(projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projected = projected.reshape(-1, 3)
    return torch.quantile(projected, 0.01, dim=0), torch.quantile(projected, 0.99, dim=0)


def project_pca(
    tokens: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    lo: torch.Tensor | None = None,
    hi: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = (tokens - mean) @ components
    if lo is None or hi is None:
        lo, hi = pca_limits(projected)
    projected = (projected - lo) / (hi - lo).clamp_min(1e-6)
    return projected.clamp(0.0, 1.0)


def reshape_feature_map(tokens: torch.Tensor, clip_shape: tuple[int, int, int, int], patch_size: int, tubelet_size: int) -> torch.Tensor:
    _, num_frames, height, width = clip_shape
    h_tokens = height // patch_size
    w_tokens = width // patch_size
    if num_frames == 1:
        t_tokens = 1
    else:
        t_tokens = max(1, num_frames // tubelet_size)
    expected = t_tokens * h_tokens * w_tokens
    if tokens.shape[0] != expected:
        raise RuntimeError(f"Cannot reshape {tokens.shape[0]} tokens to T,H,W={t_tokens},{h_tokens},{w_tokens} ({expected}).")
    return tokens.view(t_tokens, h_tokens, w_tokens, -1)


def to_uint8_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")


def overlay(raw: Image.Image, pca: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(raw.convert("RGB"), pca.resize(raw.size, Image.Resampling.BILINEAR).convert("RGB"), alpha)


def make_contact_sheet(
    raw_frames: list[Image.Image],
    pca_maps: torch.Tensor,
    tubelet_size: int,
    alpha: float,
    title: str,
    include_overlay: bool,
) -> Image.Image:
    tile_w, tile_h = raw_frames[0].size
    label_h = 22
    rows = pca_maps.shape[0]
    headers = ["input", "PCA RGB"]
    if include_overlay:
        headers.append("overlay")
    cols = len(headers)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h) + label_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(240, 240, 240))
    for col, header in enumerate(headers):
        draw.text((col * tile_w + 6, label_h + 3), header, fill=(240, 240, 240))

    for t in range(rows):
        raw_idx = 0 if len(raw_frames) == 1 else min(len(raw_frames) - 1, int(round((t + 0.5) * tubelet_size - 0.5)))
        raw = raw_frames[raw_idx]
        pca = to_uint8_image(pca_maps[t].numpy()).resize(raw.size, Image.Resampling.BILINEAR)
        images = [raw, pca]
        if include_overlay:
            images.append(overlay(raw, pca, alpha))
        y = label_h + t * (tile_h + label_h) + label_h
        for col, image in enumerate(images):
            canvas.paste(image, (col * tile_w, y))
        draw.text((6, y + tile_h + 3), f"feature_t={t} input_frame={raw_idx}", fill=(230, 230, 230))
    return canvas


def make_compare_sheet(
    raw_frames: list[Image.Image],
    model_maps: list[tuple[str, torch.Tensor]],
    tubelet_size: int,
    alpha: float,
    title: str,
    include_overlay: bool,
) -> Image.Image:
    tile_w, tile_h = raw_frames[0].size
    label_h = 24
    rows = max(pca_maps.shape[0] for _, pca_maps in model_maps)
    if include_overlay:
        cols = 1 + len(model_maps) * 2
    else:
        cols = 1 + len(model_maps)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h) + label_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(240, 240, 240))
    draw.text((6, label_h + 3), "input", fill=(240, 240, 240))
    for model_idx, (label, _) in enumerate(model_maps):
        if include_overlay:
            pca_col = 1 + model_idx * 2
            draw.text((pca_col * tile_w + 6, label_h + 3), f"{label} PCA", fill=(240, 240, 240))
            draw.text(((pca_col + 1) * tile_w + 6, label_h + 3), f"{label} overlay", fill=(240, 240, 240))
        else:
            pca_col = 1 + model_idx
            draw.text((pca_col * tile_w + 6, label_h + 3), f"{label} PCA", fill=(240, 240, 240))

    for t in range(rows):
        raw_idx = 0 if len(raw_frames) == 1 else min(len(raw_frames) - 1, int(round((t + 0.5) * tubelet_size - 0.5)))
        raw = raw_frames[raw_idx]
        y = label_h + t * (tile_h + label_h) + label_h
        canvas.paste(raw, (0, y))
        for model_idx, (_, pca_maps) in enumerate(model_maps):
            map_idx = min(t, pca_maps.shape[0] - 1)
            pca = to_uint8_image(pca_maps[map_idx].numpy()).resize(raw.size, Image.Resampling.BILINEAR)
            if include_overlay:
                over = overlay(raw, pca, alpha)
                pca_col = 1 + model_idx * 2
                canvas.paste(pca, (pca_col * tile_w, y))
                canvas.paste(over, ((pca_col + 1) * tile_w, y))
            else:
                pca_col = 1 + model_idx
                canvas.paste(pca, (pca_col * tile_w, y))
        draw.text((6, y + tile_h + 3), f"feature_t={t} input_frame={raw_idx}", fill=(230, 230, 230))
    return canvas


def pca_maps_for_features(
    feature_maps: list[torch.Tensor],
    pca_scope: str,
    max_pca_tokens: int,
) -> list[torch.Tensor]:
    all_tokens = [feature_map.reshape(-1, feature_map.shape[-1]) for feature_map in feature_maps]
    batch_mean = batch_components = batch_lo = batch_hi = None
    if pca_scope == "batch":
        batch_mean, batch_components = fit_pca(torch.cat(all_tokens, dim=0), max_pca_tokens)
        batch_projected = (torch.cat(all_tokens, dim=0) - batch_mean) @ batch_components
        batch_lo, batch_hi = pca_limits(batch_projected)

    pca_maps = []
    for feature_map, flat in zip(feature_maps, all_tokens):
        if pca_scope == "sample":
            mean, components = fit_pca(flat, max_pca_tokens)
            lo = hi = None
        else:
            mean, components = batch_mean, batch_components
            lo, hi = batch_lo, batch_hi
        pca_maps.append(project_pca(flat, mean, components, lo, hi).view(*feature_map.shape[:3], 3))
    return pca_maps


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_named_models()
        return

    cfg = load_yaml(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    data_cfg = cfg.get("data", {})
    crop_size = args.crop_size or data_cfg.get("crop_size", 384)
    patch_size = args.patch_size or data_cfg.get("patch_size", 16)
    tubelet_size = args.tubelet_size or data_cfg.get("tubelet_size", 2)

    samples = collect_samples(args, cfg)

    if args.compare:
        model_specs = compare_model_specs(args, cfg)
        sample_frames: dict[str, tuple[Sample, list[Image.Image], torch.Tensor]] = {}
        for sample in samples:
            clip, raw_frames = frames_to_tensor(sample.frames, crop_size)
            sample_frames[sample.name] = (sample, raw_frames, clip)

        model_records = []
        per_model_feature_maps: dict[str, list[torch.Tensor]] = {}
        for spec in model_specs:
            encoder = build_encoder(cfg, args, device, model_name_override=spec.model_name)
            load_summary = load_encoder_checkpoint(encoder, spec.checkpoint, spec.checkpoint_key)
            feature_maps = []
            for _, _, clip in sample_frames.values():
                tokens = extract_features(encoder, clip, device)
                feature_maps.append(reshape_feature_map(tokens, tuple(clip.shape), patch_size, tubelet_size))
            per_model_feature_maps[spec.label] = feature_maps
            model_records.append(
                {
                    "label": spec.label,
                    "model_name": spec.model_name or args.model_name or cfg.get("model", {}).get("model_name", "vit_base"),
                    "checkpoint": str(spec.checkpoint),
                    "checkpoint_load": load_summary,
                    "feature_shapes": [list(feature_map.shape) for feature_map in feature_maps],
                }
            )
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

        per_model_pca_maps = {
            label: pca_maps_for_features(feature_maps, args.pca_scope, args.max_pca_tokens)
            for label, feature_maps in per_model_feature_maps.items()
        }

        records = []
        for sample_idx, (sample, raw_frames, _) in enumerate(sample_frames.values()):
            model_maps = [
                (spec.label, per_model_pca_maps[spec.label][sample_idx])
                for spec in model_specs
            ]
            out_stem = safe_stem(sample.name)
            sheet_path = args.out_dir / f"{out_stem}_compare_pca_sheet.png"
            sheet = make_compare_sheet(raw_frames, model_maps, tubelet_size, args.alpha, out_stem, args.overlay)
            sheet.save(sheet_path)

            npy_paths = {}
            if args.save_npy:
                for spec in model_specs:
                    npy_path = args.out_dir / f"{out_stem}_{safe_stem(spec.label)}_features.npy"
                    np.save(npy_path, per_model_feature_maps[spec.label][sample_idx].numpy())
                    npy_paths[spec.label] = str(npy_path)

            records.append(
                {
                    "name": out_stem,
                    "source": sample.source,
                    "frame_indices": sample.frame_indices,
                    "compare_pca_sheet": str(sheet_path),
                    "features_npy": npy_paths,
                }
            )

        manifest = {
            "mode": "compare",
            "config": str(args.config),
            "device": str(device),
            "crop_size": crop_size,
            "patch_size": patch_size,
            "tubelet_size": tubelet_size,
            "overlay": args.overlay,
            "alpha": args.alpha,
            "pca_scope": args.pca_scope,
            "models": model_records,
            "samples": records,
        }
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    checkpoint = resolve_checkpoint(args, cfg)
    encoder = build_encoder(cfg, args, device)
    load_summary = load_encoder_checkpoint(encoder, checkpoint, args.checkpoint_key)

    records = []
    prepared = []
    for sample in samples:
        clip, raw_frames = frames_to_tensor(sample.frames, crop_size)
        tokens = extract_features(encoder, clip, device)
        feature_map = reshape_feature_map(tokens, tuple(clip.shape), patch_size, tubelet_size)
        prepared.append((sample, raw_frames, feature_map))

    pca_maps = pca_maps_for_features(
        [feature_map for _, _, feature_map in prepared],
        args.pca_scope,
        args.max_pca_tokens,
    )

    for (sample, raw_frames, feature_map), pca_map in zip(prepared, pca_maps):
        out_stem = safe_stem(sample.name)
        sheet_path = args.out_dir / f"{out_stem}_pca_sheet.png"
        pca_path = args.out_dir / f"{out_stem}_pca_maps.png"
        sheet = make_contact_sheet(raw_frames, pca_map, tubelet_size, args.alpha, out_stem, args.overlay)
        sheet.save(sheet_path)
        pca_sheet = make_contact_sheet(
            [Image.new("RGB", raw_frames[0].size, color=(0, 0, 0)) for _ in raw_frames],
            pca_map,
            tubelet_size,
            1.0,
            f"{out_stem} PCA",
            False,
        )
        pca_sheet.crop((raw_frames[0].width, 0, raw_frames[0].width * 2, pca_sheet.height)).save(pca_path)

        npy_path = None
        if args.save_npy:
            npy_path = args.out_dir / f"{out_stem}_features.npy"
            np.save(npy_path, feature_map.numpy())
        records.append(
            {
                "name": out_stem,
                "source": sample.source,
                "frame_indices": sample.frame_indices,
                "feature_shape": list(feature_map.shape),
                "pca_sheet": str(sheet_path),
                "pca_maps": str(pca_path),
                "features_npy": str(npy_path) if npy_path else None,
            }
        )

    manifest = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "checkpoint_load": load_summary,
        "device": str(device),
        "crop_size": crop_size,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
        "overlay": args.overlay,
        "alpha": args.alpha,
        "pca_scope": args.pca_scope,
        "samples": records,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
