#!/bin/bash
#SBATCH -J streamvln-future-s1-48g
#SBATCH -p eailab_system
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=reserved
#SBATCH --time=48:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
LOG_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens/logs/${RUN_NAME}"
OUT_DIR="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/${RUN_NAME}"
LOCK_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens/locks/stage1_predictor_pretrain"
MODEL_PATH="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
VISION_TOWER="/mnt/inspurfs/evla1_t/zhangpingrui/siglip-so400m-patch14-384"
VIDEO_FOLDER="data/trajectory_data/R2R,data/trajectory_data/RxR,data/trajectory_data/EnvDrop"
MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)"
MASTER_PORT=$((RANDOM % 101 + 26000))

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "$(dirname "${LOCK_DIR}")"
cd "${REPO_DIR}"

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "${SLURM_JOB_ID}" > "${LOCK_DIR}/job_id"
  echo "${RUN_NAME}" > "${LOCK_DIR}/run_name"
  echo "${OUT_DIR}" > "${LOCK_DIR}/output_dir"
else
  owner_job="$(cat "${LOCK_DIR}/job_id" 2>/dev/null || true)"
  if [[ -n "${owner_job}" ]] && [[ -n "$(squeue -h -j "${owner_job}" 2>/dev/null)" ]]; then
    echo "Another stage1 predictor job already owns the run lock: ${owner_job}. Exiting."
    exit 0
  fi
  if [[ -f "${LOCK_DIR}/done" ]]; then
    echo "Stage1 predictor run is already marked done. Exiting."
    exit 0
  fi
  echo "Removing stale stage1 predictor lock owned by ${owner_job:-unknown}."
  rm -rf "${LOCK_DIR}"
  mkdir "${LOCK_DIR}"
  echo "${SLURM_JOB_ID}" > "${LOCK_DIR}/job_id"
  echo "${RUN_NAME}" > "${LOCK_DIR}/run_name"
  echo "${OUT_DIR}" > "${LOCK_DIR}/output_dir"
fi

cleanup_lock_on_failure() {
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    rm -rf "${LOCK_DIR}"
  else
    touch "${LOCK_DIR}/done"
  fi
  exit "${rc}"
}
trap cleanup_lock_on_failure EXIT

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "output_dir=${OUT_DIR}"
echo "model_path=${MODEL_PATH}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "slurm_nnodes=${SLURM_NNODES}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
nvidia-smi

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
export WANDB_MODE="${WANDB_MODE:-offline}"
export MAX_JOBS=8

srun -N "${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=none bash -lc "
  cd '${REPO_DIR}' &&
  export TORCH_EXTENSIONS_DIR=/tmp/streamvln_torch_extensions_${SLURM_JOB_ID}_\${SLURM_NODEID} &&
  mkdir -p \${TORCH_EXTENSIONS_DIR} &&
  '${REPO_DIR}/scripts/streamvln_container.sh' \
    torchrun --nnodes='${SLURM_NNODES}' --nproc_per_node=8 \
    --node_rank=\"\${SLURM_NODEID}\" \
    --master_addr='${MASTER_ADDR}' --master_port='${MASTER_PORT}' \
    streamvln/streamvln_train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path '${MODEL_PATH}' \
    --version qwen_1_5 \
    --video_folder '${VIDEO_FOLDER}' \
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
    --future_eval_size 1024 \
    --mm_tunable_parts future_predictor \
    --vision_tower '${VISION_TOWER}' \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints '(1x1),...,(6x6)' \
    --bf16 True \
    --tf32 True \
    --run_name '${RUN_NAME}' \
    --output_dir '${OUT_DIR}' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
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
"
