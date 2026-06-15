#!/usr/bin/env python3
"""Save PCA visualizations of SigLIP 2 dense image-token maps.

This follows scripts/siglip2_pca.py for model loading: use AutoProcessor to
prepare images, then run the SigLIP 2 vision model. For dense PCA maps, this
script uses the vision model's last_hidden_state instead of the pooled global
embedding returned by get_image_features.

Examples:

  # SigLIP 2 base patch16 512px on one PNG/JPEG image.
  python scripts/visualize_siglip2_pca.py \
    --model google/siglip2-base-patch16-512 \
    --images data/cat.jpg \
    --out-dir output/siglip2_pca_base512

  # SigLIP 2 on samples from the same manifest used by V-JEPA.
  python scripts/visualize_siglip2_pca.py \
    --model google/siglip2-base-patch16-512 \
    --manifest data/surgvu/manifest_train.csv \
    --num-manifest-samples 4 \
    --out-dir output/siglip2_pca_surgvu

  # Save dense feature maps for comparison with other PCA scripts.
  python scripts/visualize_siglip2_pca.py \
    --model google/siglip2-base-patch16-512 \
    --images data/cat.jpg \
    --save-npy \
    --out-dir output/siglip2_pca_base512
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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HF_MODEL_ALIASES = {
    "base512": "google/siglip2-base-patch16-512",
    "base-p16-512": "google/siglip2-base-patch16-512",
}


@dataclass
class Sample:
    name: str
    frames: list[Image.Image]
    source: str
    frame_indices: list[int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        default="google/siglip2-base-patch16-512",
        help="Hugging Face model id or alias. Use --list-models to see aliases.",
    )
    parser.add_argument("--list-models", action="store_true", help="List SigLIP 2 model aliases.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/siglip2_pca"))
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
        "--device-map-auto",
        action="store_true",
        help="Load the model with device_map='auto', matching scripts/siglip2_pca.py.",
    )
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Passed to AutoModel/AutoProcessor for custom HF repos if needed.",
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
    parser.add_argument("--save-npy", action="store_true", help="Also save SigLIP 2 feature maps as .npy files.")
    return parser.parse_args()


def safe_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"


def list_named_models() -> None:
    rows = [{"alias": alias, "hf_model": model_id} for alias, model_id in sorted(HF_MODEL_ALIASES.items())]
    print(json.dumps(rows, indent=2))


def resolve_model_id(value: str) -> str:
    return HF_MODEL_ALIASES.get(value, value)


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


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def first_parameter_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def load_siglip2(model_id: str, args: argparse.Namespace, device: torch.device):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise SystemExit("SigLIP 2 loading requires transformers with SigLIP 2 support installed.") from exc

    dtype = dtype_from_args(args.dtype, device)
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map_auto:
        load_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(model_id, **load_kwargs)
    if not args.device_map_auto:
        model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    return model, processor


def move_processor_inputs(inputs, device: torch.device, pixel_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=pixel_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def vision_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    allowed = ("pixel_values", "pixel_attention_mask", "spatial_shapes")
    return {key: inputs[key] for key in allowed if key in inputs}


def patch_size_from_model(model: torch.nn.Module) -> int | None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", config)
    patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is None:
        return None
    if isinstance(patch_size, (list, tuple)):
        return int(patch_size[0])
    return int(patch_size)


def grid_from_spatial_shapes(
    spatial_shapes: torch.Tensor | None,
    index: int,
    token_count: int,
    patch_size: int | None,
) -> tuple[int, int]:
    if spatial_shapes is not None:
        height, width = spatial_shapes[index].detach().cpu().tolist()
        height, width = int(height), int(width)
        if height * width <= token_count:
            return height, width
        if patch_size is not None and height % patch_size == 0 and width % patch_size == 0:
            h_tokens = height // patch_size
            w_tokens = width // patch_size
            if h_tokens * w_tokens <= token_count:
                return h_tokens, w_tokens
        raise RuntimeError(
            f"Cannot map spatial_shapes[{index}]={height,width} to {token_count} vision tokens."
        )
    size = int(round(math.sqrt(token_count)))
    if size * size != token_count:
        raise RuntimeError(
            "SigLIP 2 processor did not return spatial_shapes and token count is not square: "
            f"{token_count}."
        )
    return size, size


@torch.inference_mode()
def extract_dense_features(model: torch.nn.Module, processor, frames: list[Image.Image]) -> torch.Tensor:
    inputs = processor(images=frames, return_tensors="pt")
    device = first_parameter_device(model)
    pixel_dtype = first_parameter_dtype(model)
    inputs = move_processor_inputs(inputs, device, pixel_dtype)

    vision_model = getattr(model, "vision_model", model)
    outputs = vision_model(**vision_inputs(inputs))
    if not hasattr(outputs, "last_hidden_state"):
        raise SystemExit("Expected the SigLIP 2 vision model to expose last_hidden_state.")

    hidden = outputs.last_hidden_state.detach().float().cpu()
    spatial_shapes = inputs.get("spatial_shapes")
    if spatial_shapes is not None:
        spatial_shapes = spatial_shapes.detach().cpu()

    patch_size = patch_size_from_model(model)
    maps = []
    for i in range(hidden.shape[0]):
        h_tokens, w_tokens = grid_from_spatial_shapes(spatial_shapes, i, hidden.shape[1], patch_size)
        valid_tokens = h_tokens * w_tokens
        if valid_tokens > hidden.shape[1]:
            raise RuntimeError(
                f"spatial_shapes[{i}]={h_tokens,w_tokens} requires {valid_tokens} tokens, "
                f"but last_hidden_state has {hidden.shape[1]}."
            )
        maps.append(hidden[i, :valid_tokens].view(h_tokens, w_tokens, -1))

    shapes = {tuple(feature_map.shape[:2]) for feature_map in maps}
    if len(shapes) != 1:
        raise RuntimeError(
            "All frames in one sample must produce the same SigLIP 2 patch grid. "
            f"Got grids: {sorted(shapes)}."
        )
    return torch.stack(maps, dim=0)


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


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_named_models()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_id = resolve_model_id(args.model)
    samples = collect_samples(args)
    model, processor = load_siglip2(model_id, args, device)

    records = []
    prepared = []
    for sample in samples:
        raw_frames = [frame.convert("RGB") for frame in sample.frames]
        feature_map = extract_dense_features(model, processor, raw_frames)
        prepared.append((sample, raw_frames, feature_map))

    pca_maps = pca_maps_for_features(
        [feature_map for _, _, feature_map in prepared],
        args.pca_scope,
        args.max_pca_tokens,
    )

    model_label = safe_stem(args.model if args.model in HF_MODEL_ALIASES else model_id.split("/")[-1])
    for (sample, raw_frames, feature_map), pca_map in zip(prepared, pca_maps):
        out_stem = safe_stem(sample.name)
        npy_path = None
        if args.save_npy:
            npy_path = args.out_dir / f"{out_stem}_{model_label}_features.npy"
            np.save(npy_path, feature_map.numpy())

        sheet_path = args.out_dir / f"{out_stem}_pca_sheet.png"
        pca_path = args.out_dir / f"{out_stem}_pca_maps.png"
        sheet = make_contact_sheet(raw_frames, pca_map, args.alpha, out_stem, args.overlay)
        sheet.save(sheet_path)
        pca_sheet = make_contact_sheet(
            [Image.new("RGB", raw_frames[0].size, color=(0, 0, 0)) for _ in raw_frames],
            pca_map,
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
                "pca_maps": str(pca_path),
                "features_npy": str(npy_path) if npy_path else None,
            }
        )

    manifest = {
        "model": model_id,
        "model_alias": args.model if args.model in HF_MODEL_ALIASES else None,
        "device": str(first_parameter_device(model)),
        "dtype": str(first_parameter_dtype(model)),
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
