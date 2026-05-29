#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd -P)"
sif="${SIF:-${repo_dir}/vjepa2.sif}"

runtime="${SINGULARITY_BIN:-}"
if [[ -z "${runtime}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    runtime="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    runtime="singularity"
  else
    echo "ERROR: neither apptainer nor singularity is on PATH" >&2
    exit 1
  fi
fi

if [[ ! -f "${sif}" ]]; then
  echo "ERROR: SIF container not found: ${sif}" >&2
  echo "Set SIF=/path/to/vjepa2.sif or place vjepa2.sif at the repo root." >&2
  exit 1
fi

if [[ ! -x "${repo_dir}/.venv/bin/python" ]]; then
  echo "ERROR: host uv env not found or not executable: ${repo_dir}/.venv/bin/python" >&2
  exit 1
fi

python_target="$(readlink -f "${repo_dir}/.venv/bin/python")"
case "${python_target}" in
  */.local/share/uv/python/cpython-*/bin/python*)
    uv_python_root="${python_target%%/cpython-*}"
    ;;
  */uv/python/cpython-*/bin/python*)
    uv_python_root="${python_target%%/cpython-*}"
    ;;
  *)
    echo "ERROR: .venv/bin/python does not look like a uv-managed CPython:" >&2
    echo "  ${python_target}" >&2
    echo "Set UV_PYTHON_ROOT=/path/to/uv/python if this is intentional." >&2
    uv_python_root="${UV_PYTHON_ROOT:-}"
    ;;
esac

if [[ -z "${uv_python_root:-}" || ! -d "${uv_python_root}" ]]; then
  echo "ERROR: uv Python root not found: ${uv_python_root:-<empty>}" >&2
  exit 1
fi

if [[ "$#" -gt 0 && "$1" == "--" ]]; then
  shift
fi

if [[ "$#" -eq 0 ]]; then
  set -- bash
fi

extra_args=()
if [[ -n "${SINGULARITY_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${SINGULARITY_EXTRA_ARGS})
fi

export SINGULARITYENV_VJEPA2_REPO="${repo_dir}"
export SINGULARITYENV_VIRTUAL_ENV="${repo_dir}/.venv"
export SINGULARITYENV_UV_PROJECT_ENVIRONMENT="${repo_dir}/.venv"
export APPTAINERENV_VJEPA2_REPO="${repo_dir}"
export APPTAINERENV_VIRTUAL_ENV="${repo_dir}/.venv"
export APPTAINERENV_UV_PROJECT_ENVIRONMENT="${repo_dir}/.venv"

exec "${runtime}" exec --nv \
  --bind "${repo_dir}:${repo_dir}" \
  --bind "${uv_python_root}:${uv_python_root}:ro" \
  --pwd "${repo_dir}" \
  "${extra_args[@]}" \
  "${sif}" \
  bash "${repo_dir}/containers/singularity_entrypoint.sh" "$@"
