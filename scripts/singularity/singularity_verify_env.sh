#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

exec scripts/singularity/singularity_exec.sh python scripts/verify_docker_env.py
