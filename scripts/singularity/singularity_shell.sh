#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec scripts/singularity_exec.sh bash
