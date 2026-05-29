# syntax=docker/dockerfile:1.7
#
# Production V-JEPA2 environment.
#
# The default CUDA 12.2 base is deliberate: it runs on NVIDIA 535.x drivers
# that report CUDA 12.2, and also runs on newer 580.x drivers. Do not switch the
# default to CUDA 13 unless every production host has a CUDA 13-capable driver.

ARG CUDA_IMAGE_TAG=12.2.2-devel-ubuntu22.04
ARG UV_VERSION=0.11.9

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM nvidia/cuda:${CUDA_IMAGE_TAG} AS ffmpeg-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG FFMPEG_REF=n7.1.1
ARG NV_CODEC_HEADERS_REF=n12.2.72.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    make \
    nasm \
    pkg-config \
    yasm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

RUN git clone --depth 1 --branch "${NV_CODEC_HEADERS_REF}" https://github.com/FFmpeg/nv-codec-headers.git \
    && make -C nv-codec-headers -j"$(nproc)" \
    && make -C nv-codec-headers install PREFIX=/usr/local

RUN git clone --depth 1 --branch "${FFMPEG_REF}" https://github.com/FFmpeg/FFmpeg.git ffmpeg \
    && cd ffmpeg \
    && ./configure \
        --prefix=/opt/ffmpeg \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --extra-libs="-lpthread -lm" \
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-nvenc \
        --enable-libnpp \
        --enable-nonfree \
        --enable-pic \
        --disable-debug \
        --disable-doc \
        --disable-ffplay \
    && make -j"$(nproc)" \
    && make install

FROM nvidia/cuda:${CUDA_IMAGE_TAG}

ARG DEBIAN_FRONTEND=noninteractive

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/ffmpeg/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/opt/ffmpeg/lib:/usr/local/cuda/lib64 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    openssh-client \
    pkg-config \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /usr/local/bin/
COPY --from=ffmpeg-builder /opt/ffmpeg /opt/ffmpeg
COPY --from=ffmpeg-builder /usr/local/include/ffnvcodec /usr/local/include/ffnvcodec
COPY --from=ffmpeg-builder /usr/local/lib/pkgconfig/ffnvcodec.pc /usr/local/lib/pkgconfig/ffnvcodec.pc

WORKDIR /workspace

RUN ffmpeg -hide_banner -hwaccels

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
