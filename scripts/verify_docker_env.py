#!/usr/bin/env python3
"""Verify the mounted uv environment and external video/CUDA tools."""

from __future__ import annotations

import shutil
import subprocess
import sys


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"prefix: {sys.prefix}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg}")
    if not ffmpeg:
        print("ERROR: ffmpeg is not on PATH")
        return 1

    hwaccels = run(["ffmpeg", "-hide_banner", "-hwaccels"])
    print(hwaccels.stdout.strip())
    if "cuda" not in hwaccels.stdout:
        print("ERROR: ffmpeg does not report CUDA hardware acceleration")
        return 1

    nvidia_smi = shutil.which("nvidia-smi")
    print(f"nvidia-smi: {nvidia_smi}")
    if nvidia_smi:
        smi = run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ]
        )
        print(smi.stdout.strip())
        if smi.returncode != 0:
            print("ERROR: nvidia-smi failed")
            return smi.returncode
    else:
        print("WARNING: nvidia-smi is not available")

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: failed to import torch from mounted uv env: {exc}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch cuda wheel: {torch.version.cuda}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    if torch.version.cuda:
        try:
            cuda_major = int(str(torch.version.cuda).split(".", maxsplit=1)[0])
        except ValueError:
            cuda_major = 0
        if cuda_major >= 13:
            print(
                "ERROR: mounted torch uses CUDA 13.x. That will not run on the "
                "535.x / CUDA 12.2 driver host. Install CUDA 12.x torch wheels "
                "in the host uv env for cross-host compatibility."
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
