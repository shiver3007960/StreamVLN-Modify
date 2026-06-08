#!/bin/bash
#SBATCH -J streamvln-future-base-smoke
#SBATCH -p eailab_os
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --quotatype=reserved
#SBATCH --time=02:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
RUN_DIR="${EXP_DIR}/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
CHECKPOINT="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
MASTER_PORT=$((RANDOM % 101 + 21000))

mkdir -p "${RUN_DIR}"

cd "${REPO_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "repo=${REPO_DIR}"
echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

export STREAMVLN_ATTENTION=sdpa
export STREAMVLN_MAX_EPISODES=1
export STREAMVLN_HABITAT_GPU_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface

srun -n1 --cpu-bind=none "${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  streamvln/streamvln_eval.py \
  --model_path "${CHECKPOINT}" \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split val_seen \
  --output_path "${RUN_DIR}" \
  --num_future_steps 4 \
  --num_frames 32 \
  --num_history 8
