#!/bin/bash
#SBATCH -J streamvln-future-s1-48c
#SBATCH -p eailab_system
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=reserved
#SBATCH --time=48:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
TARGET_NODES=6
RUN_GROUP="${STREAMVLN_FUTURE_RUN_ID:?Set STREAMVLN_FUTURE_RUN_ID for stitched candidates}"
RUN_NAME="streamvln-future-s1-48g-${RUN_GROUP}"
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
LOG_DIR="${EXP_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT_DIR="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/${RUN_NAME}"
LOCK_DIR="${EXP_DIR}/locks/stage1_predictor_pretrain"
RDZV_DIR="${EXP_DIR}/rdzv/${RUN_GROUP}"
MODEL_PATH="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
VISION_TOWER="/mnt/inspurfs/evla1_t/zhangpingrui/siglip-so400m-patch14-384"
VIDEO_FOLDER="data/trajectory_data/R2R,data/trajectory_data/RxR,data/trajectory_data/EnvDrop"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "$(dirname "${LOCK_DIR}")" "${RDZV_DIR}"
cd "${REPO_DIR}"

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "${SLURM_JOB_ID}" > "${LOCK_DIR}/job_id"
  echo "${RUN_NAME}" > "${LOCK_DIR}/run_name"
  echo "${OUT_DIR}" > "${LOCK_DIR}/output_dir"
else
  if [[ -f "${LOCK_DIR}/done" ]]; then
    echo "Stage1 predictor run is already marked done. Exiting."
    exit 0
  fi
  owner_job="$(cat "${LOCK_DIR}/job_id" 2>/dev/null || true)"
  if [[ -n "${owner_job}" ]] && [[ -z "$(squeue -h -j "${owner_job}" 2>/dev/null)" ]]; then
    echo "Removing stale stage1 predictor lock owned by ${owner_job}."
    rm -rf "${LOCK_DIR}"
    mkdir "${LOCK_DIR}"
    echo "${SLURM_JOB_ID}" > "${LOCK_DIR}/job_id"
    echo "${RUN_NAME}" > "${LOCK_DIR}/run_name"
    echo "${OUT_DIR}" > "${LOCK_DIR}/output_dir"
  fi
fi

NODE_RANK=""
for rank in $(seq 0 $((TARGET_NODES - 1))); do
  if mkdir "${RDZV_DIR}/rank_${rank}" 2>/dev/null; then
    NODE_RANK="${rank}"
    hostname > "${RDZV_DIR}/rank_${rank}/host"
    echo "${SLURM_JOB_ID}" > "${RDZV_DIR}/rank_${rank}/job_id"
    break
  fi
done

if [[ -z "${NODE_RANK}" ]]; then
  echo "All ${TARGET_NODES} ranks are already claimed. Exiting surplus candidate."
  exit 0
fi

cleanup_rank_on_failure() {
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    rm -rf "${RDZV_DIR}/rank_${NODE_RANK}"
    if [[ "${NODE_RANK}" == "0" ]]; then
      rm -f "${RDZV_DIR}/master"
      rm -rf "${LOCK_DIR}"
    fi
  elif [[ "${NODE_RANK}" == "0" ]]; then
    touch "${LOCK_DIR}/done"
  fi
  exit "${rc}"
}
trap cleanup_rank_on_failure EXIT

if [[ "${NODE_RANK}" == "0" ]]; then
  MASTER_ADDR="$(hostname)"
  MASTER_PORT=$((RANDOM % 101 + 27000))
  {
    echo "MASTER_ADDR=${MASTER_ADDR}"
    echo "MASTER_PORT=${MASTER_PORT}"
  } > "${RDZV_DIR}/master"
fi

echo "job_id=${SLURM_JOB_ID}"
echo "run_group=${RUN_GROUP}"
echo "run_name=${RUN_NAME}"
echo "node_rank=${NODE_RANK}"
echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "output_dir=${OUT_DIR}"
nvidia-smi

while [[ "$(find "${RDZV_DIR}" -maxdepth 1 -type d -name 'rank_*' | wc -l)" -lt "${TARGET_NODES}" ]]; do
  echo "Waiting for ${TARGET_NODES} ranks; current=$(find "${RDZV_DIR}" -maxdepth 1 -type d -name 'rank_*' | wc -l)"
  sleep 30
done

while [[ ! -f "${RDZV_DIR}/master" ]]; do
  echo "Waiting for master file..."
  sleep 5
done

source "${RDZV_DIR}/master"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"

export STREAMVLN_ATTENTION=sdpa
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=streamvln-future
export WANDB_NAME="${RUN_NAME}"

"${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nnodes="${TARGET_NODES}" --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  streamvln/streamvln_train.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path "${MODEL_PATH}" \
  --version qwen_1_5 \
  --video_folder "${VIDEO_FOLDER}" \
  --group_by_task False \
  --num_history 8 \
  --num_future_steps 4 \
  --num_frames 32 \
  --future_target_offset 4 \
  --data_augmentation True \
  --use_future_tokens True \
  --future_token_count 196 \
  --future_qformer_depth 2 \
  --future_qformer_heads 8 \
  --future_pretrain_only True \
  --future_fusion False \
  --future_loss_weight 1.0 \
  --mm_tunable_parts future_predictor \
  --vision_tower "${VISION_TOWER}" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints "(1x1),...,(6x6)" \
  --bf16 True \
  --tf32 True \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.075 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --model_max_length 32768 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --dataloader_drop_last True \
  --report_to wandb \
  --attn_implementation sdpa
