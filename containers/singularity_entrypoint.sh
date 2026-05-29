#!/usr/bin/env bash
set -euo pipefail

repo_dir="${VJEPA2_REPO:-$(pwd -P)}"
venv_dir="${VIRTUAL_ENV:-${repo_dir}/.venv}"

if [[ ! -d "${repo_dir}" ]]; then
  echo "ERROR: repo directory does not exist inside container: ${repo_dir}" >&2
  exit 1
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  echo "ERROR: host uv environment is not executable inside container: ${venv_dir}/bin/python" >&2
  echo "Bind the repo and uv Python directory at the same absolute paths used on the host." >&2
  exit 1
fi

cd "${repo_dir}"

export VIRTUAL_ENV="${venv_dir}"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME

export PATH="/opt/ffmpeg/bin:${venv_dir}/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/ffmpeg/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

if [[ "${VJEPA2_ENTRYPOINT_QUIET:-0}" != "1" ]]; then
  echo "repo: ${repo_dir}"
  echo "venv: ${venv_dir}"
  echo "python: $(${venv_dir}/bin/python -c 'import sys; print(sys.executable)')"
  echo "ffmpeg: $(command -v ffmpeg || true)"
fi

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec "$@"
