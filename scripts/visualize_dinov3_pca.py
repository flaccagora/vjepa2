#!/usr/bin/env python3
"""Save PCA visualizations of DINOv3 dense feature maps using Hugging Face models.

Examples:

  # DINOv3 ViT-B on one PNG/JPEG image.
  python scripts/visualize_dinov3_pca.py \
    --model facebook/dinov3-vitb16-pretrain-lvd1689m \
    --images data/cat.jpg \
    --out-dir output/dinov3_pca_vitb

  # DINOv3 on samples from the same manifest used by V-JEPA.
  python scripts/visualize_dinov3_pca.py \
    --model vitb16 \
    --manifest ../vjepa2/surgvu_metadata_full/manifest_train.csv \
    --num-manifest-samples 4 \
    --out-dir output/dinov3_pca_surgvu

  # Compare DINOv3 against V-JEPA features saved with:
  #   python ../vjepa2/scripts/visualize_vjepa2_1_pca.py ... --save-npy
  python scripts/visualize_dinov3_pca.py \
    --model vitb16 \
    --images data/cat.jpg \
    --compare-feature-map vjepa2_1_vitb:../vjepa2/output/vjepa2_1_pca/cat_features.npy \
    --out-dir output/dinov3_vs_vjepa_pca
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
SATELLITE_MEAN = torch.tensor([0.430, 0.411, 0.296], dtype=torch.float32).view(3, 1, 1)
SATELLITE_STD = torch.tensor([0.213, 0.156, 0.143], dtype=torch.float32).view(3, 1, 1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


HF_MODEL_ALIASES = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "convnext-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext-base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext-large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "vitl16-sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "vit7b16-sat": "facebook/dinov3-vit7b16-pretrain-sat493m",
}


@dataclass
class Sample:
    name: str
    frames: list[Image.Image]
    source: str
    frame_indices: list[int] | None = None


@dataclass
class ExternalFeatureMap:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        default="vitb16",
        help="Hugging Face model id or alias. Use --list-models to see aliases.",
    )
    parser.add_argument("--list-models", action="store_true", help="List DINOv3 Hugging Face model aliases.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/dinov3_pca"))
    parser.add_argument("--images", type=Path, nargs="*", default=[], help="PNG/JPEG images to visualize.")
    parser.add_argument("--video", type=Path, default=None, help="Video file to visualize frame-by-frame.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest containing video paths in the first column.")
    parser.add_argument("--manifest-index", type=int, default=0, help="First manifest row to visualize.")
    parser.add_argument("--num-manifest-samples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=16, help="Frames sampled from each video.")
    parser.add_argument(
        "--frame-indices",
        default=None,
        help="Comma-separated video frame indices. Overrides uniform sampling.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Center-crop size fed to DINOv3. Use 384 to line up with V-JEPA 2.1 384px runs.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override model.config.patch_size when reshaping ViT patch tokens.",
    )
    parser.add_argument(
        "--normalization",
        choices=["auto", "imagenet", "satellite"],
        default="auto",
        help="Input normalization. auto uses satellite stats for SAT-493M model ids, else ImageNet stats.",
    )
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Passed to AutoModel for custom HF repos if needed.",
    )
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
    parser.add_argument("--save-npy", action="store_true", help="Also save DINOv3 feature maps as .npy files.")
    parser.add_argument(
        "--compare-feature-map",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="External feature map .npy to compare, e.g. vjepa2_1_vitb:../vjepa2/output/.../cat_features.npy.",
    )
    return parser.parse_args()


def safe_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"


def list_named_models() -> None:
    rows = [{"alias": alias, "hf_model": model_id} for alias, model_id in sorted(HF_MODEL_ALIASES.items())]
    print(json.dumps(rows, indent=2))


def resolve_model_id(value: str) -> str:
    return HF_MODEL_ALIASES.get(value, value)


def parse_compare_feature_map(value: str) -> ExternalFeatureMap:
    label, sep, path = value.partition(":")
    if not sep or not label or not path:
        raise SystemExit(f"--compare-feature-map must use LABEL:PATH, got {value!r}")
    return ExternalFeatureMap(label=safe_stem(label), path=Path(path))


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    image = image.resize(new_size, Image.Resampling.BICUBIC)
    left = max(0, (image.width - size) // 2)
    top = max(0, (image.height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def normalization_tensors(mode: str, model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "auto":
        mode = "satellite" if "sat493m" in model_id.lower() else "imagenet"
    if mode == "satellite":
        return SATELLITE_MEAN, SATELLITE_STD
    return IMAGENET_MEAN, IMAGENET_STD


def frames_to_tensor(frames: list[Image.Image], image_size: int, norm_mode: str, model_id: str) -> tuple[torch.Tensor, list[Image.Image]]:
    mean, std = normalization_tensors(norm_mode, model_id)
    processed = [center_crop_resize(frame, image_size) for frame in frames]
    arrays = [np.asarray(frame, dtype=np.float32) / 255.0 for frame in processed]
    tensor = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2)
    tensor = (tensor - mean) / std
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


def collect_samples(args: argparse.Namespace) -> list[Sample]:
    explicit_indices = parse_frame_indices(args.frame_indices)
    samples: list[Sample] = []

    for image_path in args.images:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported image suffix: {image_path}")
        samples.append(Sample(name=safe_stem(str(image_path)), frames=[Image.open(image_path).convert("RGB")], source=str(image_path)))

    if args.video is not None:
        frames, indices = read_video(args.video, args.num_frames, explicit_indices)
        samples.append(Sample(name=safe_stem(str(args.video)), frames=frames, source=str(args.video), frame_indices=indices))

    if args.manifest is not None:
        for video_path in manifest_video_paths(args.manifest, args.manifest_index, args.num_manifest_samples):
            frames, indices = read_video(video_path, args.num_frames, explicit_indices)
            samples.append(Sample(name=safe_stem(str(video_path)), frames=frames, source=str(video_path), frame_indices=indices))

    if not samples:
        raise SystemExit("Provide at least one input via --images, --video, or --manifest.")
    return samples


def dtype_from_args(value: str, device: torch.device) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32" or device.type == "cpu":
        return torch.float32
    return torch.float16


def load_hf_model(model_id: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit("DINOv3 HF loading requires transformers>=4.56.0. Install it in this environment.") from exc

    dtype = dtype_from_args(args.dtype, device)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def patch_grid_from_model(model: torch.nn.Module, pixel_values: torch.Tensor, patch_size_override: int | None) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    patch_size = patch_size_override or getattr(config, "patch_size", 16)
    if isinstance(patch_size, (list, tuple)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    _, _, height, width = pixel_values.shape
    return patch_h, height // patch_h, width // patch_w


@torch.inference_mode()
def extract_dense_features(
    model: torch.nn.Module,
    frames: torch.Tensor,
    device: torch.device,
    patch_size_override: int | None,
) -> torch.Tensor:
    pixel_values = frames.to(device=device, dtype=next(model.parameters()).dtype)
    outputs = model(pixel_values=pixel_values)
    if not hasattr(outputs, "last_hidden_state"):
        raise SystemExit(
            "This script expects a DINOv3 HF backbone exposing last_hidden_state. "
            "Use a DINOv3 ViT or ConvNeXt HF model id."
        )

    hidden = outputs.last_hidden_state
    if hidden.ndim == 4:
        return hidden.permute(0, 2, 3, 1).detach().float().cpu()
    if hidden.ndim != 3:
        raise RuntimeError(f"Unsupported last_hidden_state shape: {tuple(hidden.shape)}")

    _, h_tokens, w_tokens = patch_grid_from_model(model, pixel_values, patch_size_override)
    config = getattr(model, "config", None)
    num_register_tokens = int(getattr(config, "num_register_tokens", 0))
    tokens = hidden[:, 1 + num_register_tokens :, :].detach().float().cpu()
    expected = h_tokens * w_tokens
    if tokens.shape[1] != expected:
        raise RuntimeError(f"Cannot reshape {tokens.shape[1]} patch tokens to H,W={h_tokens},{w_tokens} ({expected}).")
    return tokens.view(frames.shape[0], h_tokens, w_tokens, -1)


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


def load_external_features(features: list[ExternalFeatureMap]) -> list[tuple[str, torch.Tensor]]:
    loaded = []
    for feature in features:
        if not feature.path.exists():
            raise SystemExit(f"External feature map not found for {feature.label!r}: {feature.path}")
        array = np.load(feature.path)
        if array.ndim != 4:
            raise SystemExit(f"Expected {feature.path} to have shape T,H,W,C, got {array.shape}.")
        loaded.append((feature.label, torch.from_numpy(array).float()))
    return loaded


def to_uint8_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")


def overlay(raw: Image.Image, pca: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(raw.convert("RGB"), pca.resize(raw.size, Image.Resampling.BILINEAR).convert("RGB"), alpha)


def make_contact_sheet(
    raw_frames: list[Image.Image],
    pca_maps: torch.Tensor,
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
        raw_idx = 0 if len(raw_frames) == 1 else min(len(raw_frames) - 1, t)
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
    alpha: float,
    title: str,
    include_overlay: bool,
) -> Image.Image:
    tile_w, tile_h = raw_frames[0].size
    label_h = 24
    rows = max(pca_maps.shape[0] for _, pca_maps in model_maps)
    cols = 1 + len(model_maps) * (2 if include_overlay else 1)
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
        raw_idx = 0 if len(raw_frames) == 1 else min(len(raw_frames) - 1, t)
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


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_named_models()
        return

    external_feature_maps = [parse_compare_feature_map(value) for value in args.compare_feature_map]
    input_count = len(args.images) + int(args.video is not None) + int(args.manifest is not None)
    manifest_sample_count = args.num_manifest_samples if args.manifest is not None else 0
    if external_feature_maps and (input_count != 1 or manifest_sample_count > 1):
        raise SystemExit("--compare-feature-map currently supports one image, one video, or one manifest sample per run.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_id = resolve_model_id(args.model)
    samples = collect_samples(args)
    model = load_hf_model(model_id, args, device)

    records = []
    prepared = []
    for sample in samples:
        frames, raw_frames = frames_to_tensor(sample.frames, args.image_size, args.normalization, model_id)
        feature_map = extract_dense_features(model, frames, device, args.patch_size)
        prepared.append((sample, raw_frames, feature_map))

    dino_pca_maps = pca_maps_for_features(
        [feature_map for _, _, feature_map in prepared],
        args.pca_scope,
        args.max_pca_tokens,
    )

    external_loaded = load_external_features(external_feature_maps)
    external_pca = [
        (label, pca_maps_for_features([feature_map], args.pca_scope, args.max_pca_tokens)[0])
        for label, feature_map in external_loaded
    ]

    for (sample, raw_frames, feature_map), dino_pca_map in zip(prepared, dino_pca_maps):
        out_stem = safe_stem(sample.name)
        model_label = safe_stem(args.model if args.model in HF_MODEL_ALIASES else model_id.split("/")[-1])
        npy_path = None
        if args.save_npy:
            npy_path = args.out_dir / f"{out_stem}_{model_label}_features.npy"
            np.save(npy_path, feature_map.numpy())

        if external_pca:
            model_maps = [(model_label, dino_pca_map), *external_pca]
            sheet_path = args.out_dir / f"{out_stem}_compare_pca_sheet.png"
            sheet = make_compare_sheet(raw_frames, model_maps, args.alpha, out_stem, args.overlay)
            sheet.save(sheet_path)
            pca_path = None
        else:
            sheet_path = args.out_dir / f"{out_stem}_pca_sheet.png"
            pca_path = args.out_dir / f"{out_stem}_pca_maps.png"
            sheet = make_contact_sheet(raw_frames, dino_pca_map, args.alpha, out_stem, args.overlay)
            sheet.save(sheet_path)
            pca_sheet = make_contact_sheet(
                [Image.new("RGB", raw_frames[0].size, color=(0, 0, 0)) for _ in raw_frames],
                dino_pca_map,
                1.0,
                f"{out_stem} PCA",
                False,
            )
            pca_sheet.crop((raw_frames[0].width, 0, raw_frames[0].width * 2, pca_sheet.height)).save(pca_path)

        records.append(
            {
                "name": out_stem,
                "source": sample.source,
                "frame_indices": sample.frame_indices,
                "feature_shape": list(feature_map.shape),
                "pca_sheet": str(sheet_path),
                "pca_maps": str(pca_path) if pca_path else None,
                "features_npy": str(npy_path) if npy_path else None,
            }
        )

    manifest = {
        "model": model_id,
        "model_alias": args.model if args.model in HF_MODEL_ALIASES else None,
        "device": str(device),
        "image_size": args.image_size,
        "patch_size": args.patch_size or getattr(getattr(model, "config", None), "patch_size", 16),
        "normalization": args.normalization,
        "overlay": args.overlay,
        "alpha": args.alpha,
        "pca_scope": args.pca_scope,
        "external_feature_maps": [{"label": item.label, "path": str(item.path)} for item in external_feature_maps],
        "samples": records,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
