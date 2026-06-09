#!/bin/bash
#SBATCH -J streamvln-nofuture-vu-8g
#SBATCH -p eailab_os
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=reserved
#SBATCH --time=24:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
RESULT_ROOT=/mnt/inspurfs/evla2_t/lizhen/results/StreamVLN/future_visual_tokens
CHECKPOINT_ROOT=/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-no-future-cont-48g-alloc6367298-20260609-191202
RUN_NAME="${RUN_NAME:-no-future-cont-r2r-val-unseen-8g-${SLURM_JOB_ID}}"
LOG_DIR="${EXP_DIR}/logs/${RUN_NAME}"
OUT_ROOT="${RESULT_ROOT}/${RUN_NAME}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
cd "${REPO_DIR}"

CHECKPOINTS=(
  checkpoint-3000
  checkpoint-4000
  checkpoint-4988
)

cat > "${OUT_ROOT}/manifest.tsv" <<EOF
name	checkpoint
EOF
for ckpt in "${CHECKPOINTS[@]}"; do
  path="${CHECKPOINT_ROOT}/${ckpt}"
  if [ ! -f "${path}/config.json" ] || [ ! -f "${path}/adapter_model.safetensors" ] || [ ! -f "${path}/non_lora_trainables.bin" ]; then
    echo "Invalid checkpoint: ${path}" >&2
    exit 1
  fi
  printf "%s\t%s\n" "${ckpt}" "${path}" >> "${OUT_ROOT}/manifest.tsv"
done

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "job_name=${SLURM_JOB_NAME}"
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "git_head=$(git rev-parse --short HEAD)"
  echo "repo=${REPO_DIR}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "log_dir=${LOG_DIR}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  cat "${OUT_ROOT}/manifest.tsv"
} | tee "${LOG_DIR}/launcher.log"

nvidia-smi | tee "${LOG_DIR}/nvidia-smi-at-start.log"

export STREAMVLN_ATTENTION=sdpa
unset STREAMVLN_MAX_EPISODES
unset STREAMVLN_HABITAT_GPU_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

idx=0
for ckpt in "${CHECKPOINTS[@]}"; do
  idx=$((idx + 1))
  checkpoint="${CHECKPOINT_ROOT}/${ckpt}"
  out_dir="${OUT_ROOT}/${ckpt}"
  port=$((24200 + idx))
  mkdir -p "${out_dir}"

  {
    echo "name=${ckpt}"
    echo "checkpoint=${checkpoint}"
    echo "out_dir=${out_dir}"
    echo "master_port=${port}"
    echo "date=$(date -Is)"
  } | tee "${LOG_DIR}/${ckpt}.launcher.log"

  "${REPO_DIR}/scripts/streamvln_container.sh" \
    torchrun --nproc_per_node=8 --master_port="${port}" \
    streamvln/streamvln_eval.py \
    --model_path "${checkpoint}" \
    --habitat_config_path config/vln_r2r.yaml \
    --eval_split val_unseen \
    --output_path "${out_dir}" \
    --num_future_steps 4 \
    --num_frames 32 \
    --num_history 8 \
    > "${LOG_DIR}/${ckpt}.out" \
    2> "${LOG_DIR}/${ckpt}.err"

  echo "${ckpt}: completed at $(date -Is)" | tee -a "${LOG_DIR}/launcher.log"
done

python - <<'PY' "${OUT_ROOT}" > "${OUT_ROOT}/summary.tsv"
import json
import os
import sys

root = sys.argv[1]
names = ["checkpoint-3000", "checkpoint-4000", "checkpoint-4988"]
print("name\tSR\tSPL\tOS\tNE\tlength")
for name in names:
    result_path = os.path.join(root, name, "result.json")
    final = None
    if os.path.exists(result_path):
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "sucs_all" in obj:
                    final = obj
    if final is None:
        print(f"{name}\tNA\tNA\tNA\tNA\t0")
    else:
        print(
            f"{name}\t{final['sucs_all']:.6f}\t{final['spls_all']:.6f}\t"
            f"{final['oss_all']:.6f}\t{final['ones_all']:.6f}\t{final['length']}"
        )
PY

cat "${OUT_ROOT}/summary.tsv" | tee -a "${LOG_DIR}/launcher.log"
