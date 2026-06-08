#!/bin/bash
#SBATCH -J streamvln-future-base-vu-8g
#SBATCH -p eailab_os
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=reserved
#SBATCH --time=12:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
RUN_DIR="${EXP_DIR}/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
CHECKPOINT="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
MASTER_PORT=$((RANDOM % 101 + 22000))

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
echo "master_port=${MASTER_PORT}"
nvidia-smi

export STREAMVLN_ATTENTION=sdpa
unset STREAMVLN_MAX_EPISODES
unset STREAMVLN_HABITAT_GPU_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

"${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  streamvln/streamvln_eval.py \
  --model_path "${CHECKPOINT}" \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split val_unseen \
  --output_path "${RUN_DIR}" \
  --num_future_steps 4 \
  --num_frames 32 \
  --num_history 8
