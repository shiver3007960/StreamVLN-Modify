#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?Usage: $0 <allocation-job-id>}"

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="streamvln-future-s1-48g-alloc${ALLOC_JOB_ID}-${RUN_STAMP}"
LOG_DIR="${EXP_DIR}/logs/${RUN_NAME}"
OUT_DIR="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/${RUN_NAME}"
MODEL_PATH="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
VISION_TOWER="/mnt/inspurfs/evla1_t/zhangpingrui/siglip-so400m-patch14-384"
VIDEO_FOLDER="data/trajectory_data/R2R,data/trajectory_data/RxR,data/trajectory_data/EnvDrop"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

NODELIST="$(scontrol show job "${ALLOC_JOB_ID}" | awk -F= '/ NodeList=/{print $2}' | awk '{print $1}')"
NNODES="$(scontrol show job "${ALLOC_JOB_ID}" | awk -F= '/ NumNodes=/{print $2}' | awk '{print $1}')"
MASTER_ADDR="$(scontrol show hostname "${NODELIST}" | head -n1)"
MASTER_PORT=$((RANDOM % 101 + 28000))

{
  echo "job_id=${ALLOC_JOB_ID}"
  echo "run_name=${RUN_NAME}"
  echo "date=$(date -Is)"
  echo "git_head=$(git rev-parse --short HEAD)"
  echo "node_list=${NODELIST}"
  echo "nnodes=${NNODES}"
  echo "master_addr=${MASTER_ADDR}"
  echo "master_port=${MASTER_PORT}"
  echo "output_dir=${OUT_DIR}"
  echo "log_dir=${LOG_DIR}"
  echo "deepspeed=scripts/zero2_future_predictor.json"
  echo "per_device_train_batch_size=1"
  echo "gradient_accumulation_steps=2"
  echo "eval_steps=500"
  echo "save_steps=1000"
} | tee "${LOG_DIR}/launcher.log"

export STREAMVLN_ATTENTION=sdpa
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=streamvln-future
export WANDB_NAME="${RUN_NAME}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUT_DIR}/wandb"
export MAX_JOBS=8

srun --jobid="${ALLOC_JOB_ID}" --overlap -N "${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --cpu-bind=none bash -lc "
  cd '${REPO_DIR}' &&
  export TORCH_EXTENSIONS_DIR=/tmp/streamvln_torch_extensions_alloc${ALLOC_JOB_ID}_\${SLURM_NODEID} &&
  mkdir -p \${TORCH_EXTENSIONS_DIR} &&
  '${REPO_DIR}/scripts/streamvln_container.sh' \
    torchrun --nnodes='${NNODES}' --nproc_per_node=8 \
    --node_rank=\"\${SLURM_NODEID}\" \
    --master_addr='${MASTER_ADDR}' --master_port='${MASTER_PORT}' \
    streamvln/streamvln_train.py \
    --deepspeed scripts/zero2_future_predictor.json \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --report_to wandb \
    --attn_implementation sdpa
" > "${LOG_DIR}/train.out" 2> "${LOG_DIR}/train.err"
