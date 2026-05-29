#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/train_2_1/vitb16/surgvu-finetune-384px-16f-slurm_dbg.yaml}"
SIF="${SIF:-$(pwd -P)/vjepa2.sif}"
ACCOUNT="${ACCOUNT:-IscrC_FLAC}"
PARTITION="${PARTITION:-boost_usr_prod}"
QOS="${QOS:-boost_qos_dbg}"
TIME_MIN="${TIME_MIN:-30}"
CACHE_DIR="${CACHE_DIR:-checkpoints}"
RESOLVED_CONFIG="${RESOLVED_CONFIG:-output/surgvu_slurm/resolved_config.yaml}"
JOB_DIR="${JOB_DIR:-output/surgvu_slurm/singularity_job}"
JOB_SCRIPT="${JOB_SCRIPT:-${JOB_DIR}/submit.sh}"

mkdir -p "${JOB_DIR}" "$(dirname "${RESOLVED_CONFIG}")"

scripts/singularity/singularity_exec.sh python scripts/cache_vjepa_models.py \
  --config "${CONFIG}" \
  --cache-dir "${CACHE_DIR}" \
  --write-resolved-config "${RESOLVED_CONFIG}"

readarray -t cfg_values < <(
  .venv/bin/python - <<PY
import yaml
with open("${RESOLVED_CONFIG}") as f:
    cfg = yaml.safe_load(f)
print(int(cfg.get("nodes", 1)))
print(int(cfg.get("tasks_per_node", 1)))
print(int(cfg.get("cpus_per_task", 8)))
print(str(cfg.get("mem_per_gpu", "60G")))
print(str(cfg.get("exclusive", False)).lower())
print(str(cfg.get("folder", "output/surgvu_slurm/vjepa2_1_vitb_384px_16f")))
PY
)

NODES="${NODES:-${cfg_values[0]}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${cfg_values[1]}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${cfg_values[2]}}"
MEM_PER_GPU="${MEM_PER_GPU:-${cfg_values[3]}}"
EXCLUSIVE="${EXCLUSIVE:-${cfg_values[4]}}"
RUN_FOLDER="${cfg_values[5]}"
mkdir -p "${RUN_FOLDER}"

exclusive_line=""
if [[ "${EXCLUSIVE}" == "true" ]]; then
  exclusive_line="#SBATCH --exclusive"
fi

qos_line=""
if [[ -n "${QOS}" ]]; then
  qos_line="#SBATCH --qos=${QOS}"
fi

cat > "${JOB_SCRIPT}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=surgvu-vjepa2
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
${qos_line}
#SBATCH --time=${TIME_MIN}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${TASKS_PER_NODE}
#SBATCH --gres=gpu:${TASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem-per-gpu=${MEM_PER_GPU}
#SBATCH --output=${RUN_FOLDER}/slurm-%j.out
#SBATCH --error=${RUN_FOLDER}/slurm-%j.err
${exclusive_line}

set -euo pipefail
cd "$(pwd -P)"

export SIF="${SIF}"
export VJEPA2_ENTRYPOINT_QUIET=1

srun scripts/singularity/singularity_exec.sh python scripts/run_training_config.py --fname "${RESOLVED_CONFIG}"
SBATCH

echo "Wrote ${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"
