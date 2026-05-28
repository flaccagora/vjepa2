#!/usr/bin/env python3
"""Download V-JEPA checkpoints on an internet-enabled submit/login node.

Compute nodes often cannot reach the internet. Run this script before submitting
training so configs point at local/shared filesystem checkpoints.

python scripts/cache_vjepa_models.py \
    --model vjepa2_1_vitg \
    --config configs/train_2_1/vitg16/pretrain-256px-16f.yaml \
    --write-resolved-config configs/train_2_1/vitg16/pretrain-256px-16f-resolved.yaml

python scripts/cache_vjepa_models.py \
    --model vjepa2_1_vitb \
    --config configs/train_2_1/vitb16/pretrain-256px-16f.yaml \
    --write-resolved-config configs/train_2_1/vitb16/pretrain-256px-16f-resolved.yaml

python scripts/cache_vjepa_models.py \
    --model vjepa2_1_vitG \
    --config configs/train_2_1/vitG16/pretrain-256px-16f.yaml \
    --write-resolved-config configs/train_2_1/vitG16/pretrain-256px-16f-resolved.yaml



python scripts/visualize_vjepa2_1_pca.py \
    --compare \
    --compare-pretrained vjepa2_1_vitb vjepa2_1_vitg vjepa2_1_vitG \
    --checkpoint output/surgvu_slurm/vjepa2_1_vitb_384px_16f/e14.pth.tar \
    --images data/cat.jpg \
    --out-dir output/vjepa2_1_pca_compare

"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml


BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

MODEL_REGISTRY = {
    "vjepa2_vitl": "vitl.pt",
    "vjepa2_vith": "vith.pt",
    "vjepa2_vitg": "vitg.pt",
    "vjepa2_vitg_384": "vitg-384.pt",
    "vjepa2_ac_vitg": "vjepa2-ac-vitg.pt",
    "vjepa2_1_vitb": "vjepa2_1_vitb_dist_vitG_384.pt",
    "vjepa2_1_vitl": "vjepa2_1_vitl_dist_vitG_384.pt",
    "vjepa2_1_vitg": "vjepa2_1_vitg_384.pt",
    "vjepa2_1_vitG": "vjepa2_1_vitG_384.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where checkpoints are cached.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_REGISTRY) + ["all"],
        help="Model key to cache. Can be passed multiple times. Default: infer from --config or vjepa2_1_vitb.",
    )
    parser.add_argument("--config", type=Path, help="Training config whose meta.read_checkpoint should be cached.")
    parser.add_argument(
        "--write-resolved-config",
        type=Path,
        help="Write a copy of --config with meta.read_checkpoint replaced by an absolute cached path.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path for downloaded files.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args()


def filename_to_model_key(filename: str) -> str | None:
    for key, registered_filename in MODEL_REGISTRY.items():
        if filename == registered_filename:
            return key
    return None


def checkpoint_filename(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return Path(parsed.path).name
    return Path(value).name


def run_curl(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["curl", "-fL", "--retry", "5", "--continue-at", "-", url, "-o", str(tmp_path)],
            check=True,
        )
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def cache_model(model_key: str, cache_dir: Path, force: bool = False) -> Path:
    filename = MODEL_REGISTRY[model_key]
    output_path = (cache_dir / filename).resolve()
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"cached: {model_key} -> {output_path}")
        return output_path

    url = f"{BASE_URL}/{filename}"
    print(f"downloading: {model_key} from {url}")
    run_curl(url, output_path)
    print(f"downloaded: {model_key} -> {output_path}")
    return output_path


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def resolve_local_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    parsed = urlparse(str(value))
    if parsed.scheme in {"http", "https", "s3", "gs"}:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def absolutize_config_paths(config: dict, base_dir: Path) -> dict:
    if config.get("folder"):
        config["folder"] = resolve_local_path(config["folder"], base_dir)

    data = config.get("data", {})
    if data.get("datasets"):
        data["datasets"] = [resolve_local_path(path, base_dir) for path in data["datasets"]]
    if data.get("preprocess_metadata"):
        preprocess_metadata = data["preprocess_metadata"]
        if isinstance(preprocess_metadata, (list, tuple)):
            data["preprocess_metadata"] = [
                resolve_local_path(path, base_dir) for path in preprocess_metadata
            ]
        else:
            data["preprocess_metadata"] = resolve_local_path(preprocess_metadata, base_dir)

    img_data = config.get("img_data", {})
    if img_data.get("datasets"):
        img_data["datasets"] = [resolve_local_path(path, base_dir) for path in img_data["datasets"]]

    optimization = config.get("optimization", {})
    if optimization.get("anneal_ckpt"):
        optimization["anneal_ckpt"] = resolve_local_path(optimization["anneal_ckpt"], base_dir)

    return config


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path.cwd()

    config = load_config(args.config) if args.config else None
    requested_models = args.model or []

    inferred_checkpoint_key = None
    if config is not None:
        checkpoint = config.get("meta", {}).get("read_checkpoint")
        inferred_filename = checkpoint_filename(checkpoint)
        if inferred_filename:
            inferred_checkpoint_key = filename_to_model_key(inferred_filename)
            if inferred_checkpoint_key is None:
                raise SystemExit(f"Unknown V-JEPA checkpoint filename in config: {inferred_filename}")
            if not requested_models:
                requested_models = [inferred_checkpoint_key]

    if not requested_models:
        requested_models = ["vjepa2_1_vitb"]
    if "all" in requested_models:
        requested_models = sorted(MODEL_REGISTRY)

    cached_paths = {}
    for model_key in dict.fromkeys(requested_models):
        cached_paths[model_key] = cache_model(model_key, args.cache_dir, force=args.force)

    if args.write_resolved_config:
        if config is None:
            raise SystemExit("--write-resolved-config requires --config")
        if inferred_checkpoint_key is not None:
            config["meta"]["read_checkpoint"] = str(cached_paths[inferred_checkpoint_key])
        config = absolutize_config_paths(config, base_dir)
        args.write_resolved_config.parent.mkdir(parents=True, exist_ok=True)
        with args.write_resolved_config.open("w") as f:
            yaml.dump(config, f)
        print(f"resolved config: {args.write_resolved_config}")

    manifest_path = args.manifest or (args.cache_dir / "vjepa_model_cache_manifest.json")
    with manifest_path.open("w") as f:
        json.dump({key: str(path) for key, path in cached_paths.items()}, f, indent=2)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
