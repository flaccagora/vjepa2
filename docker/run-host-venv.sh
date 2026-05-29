#!/usr/bin/env bash
set -euo pipefail

image="${1:-vjepa2:host-uv}"
repo_dir="$(pwd -P)"

if [[ ! -x "${repo_dir}/.venv/bin/python" ]]; then
  echo "ERROR: ${repo_dir}/.venv/bin/python is missing or not executable" >&2
  exit 1
fi

python_target="$(readlink -f "${repo_dir}/.venv/bin/python")"
uv_python_root=""
case "${python_target}" in
  "${HOME}/.local/share/uv/python/"*)
    uv_python_root="${HOME}/.local/share/uv/python"
    ;;
  *)
    echo "ERROR: .venv/bin/python does not point inside \$HOME/.local/share/uv/python:" >&2
    echo "  ${python_target}" >&2
    echo "Mount the Python root manually or recreate the venv with uv." >&2
    exit 1
    ;;
esac

docker run --rm -it --gpus all --ipc=host \
  --workdir "${repo_dir}" \
  --env VIRTUAL_ENV="${repo_dir}/.venv" \
  --env UV_PROJECT_ENVIRONMENT="${repo_dir}/.venv" \
  --env PATH="/opt/ffmpeg/bin:${repo_dir}/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
  --volume "${repo_dir}:${repo_dir}" \
  --volume "${uv_python_root}:${uv_python_root}:ro" \
  "${image}"
