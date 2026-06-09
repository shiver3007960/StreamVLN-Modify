#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?Usage: $0 <allocation-job-id>}"

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
STAGE1_CKPT="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/checkpoint-4000"
VISION_TOWER="/mnt/inspurfs/evla1_t/zhangpingrui/siglip-so400m-patch14-384"
VIDEO_FOLDER="data/trajectory_data/R2R,data/trajectory_data/RxR,data/trajectory_data/EnvDrop"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="streamvln-future-cat-48g-alloc${ALLOC_JOB_ID}-${RUN_STAMP}"
LOG_DIR="${EXP_DIR}/logs/${RUN_NAME}"
OUT_DIR="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/${RUN_NAME}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

retry_scontrol() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$@"; then
      return 0
    fi
    echo "scontrol command failed, retry ${attempt}/5: $*" >&2
    sleep 5
  done
  "$@"
}

JOB_INFO="$(retry_scontrol scontrol show job "${ALLOC_JOB_ID}")"
NODELIST="$(awk -F= '/ NodeList=/{print $2}' <<<"${JOB_INFO}" | awk '{print $1}')"
NNODES="$(awk -F= '/ NumNodes=/{print $2}' <<<"${JOB_INFO}" | awk '{print $1}')"
MASTER_ADDR="$(retry_scontrol scontrol show hostname "${NODELIST}" | head -n1)"
MASTER_PORT=$((RANDOM % 101 + 28750))
GPUS_PER_NODE=8
PER_DEVICE_BATCH=1
GRAD_ACCUM=2
GLOBAL_BATCH=$((NNODES * GPUS_PER_NODE * PER_DEVICE_BATCH * GRAD_ACCUM))

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
  echo "stage1_checkpoint=${STAGE1_CKPT}"
  echo "injection=concat"
  echo "trainable=future_predictor,future_connector,mm_mlp_adapter,mm_lora_layer"
  echo "per_device_train_batch_size=${PER_DEVICE_BATCH}"
  echo "gradient_accumulation_steps=${GRAD_ACCUM}"
  echo "global_batch=${GLOBAL_BATCH}"
  echo "target_control_global_batch=96"
  echo "save_steps=1000"
} | tee "${LOG_DIR}/launcher.log"

if [ "${GLOBAL_BATCH}" -ne 96 ]; then
  echo "Unexpected global batch ${GLOBAL_BATCH}; expected 96 for control-variable match." >&2
  exit 2
fi

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

srun --jobid="${ALLOC_JOB_ID}" --overlap \
  -N "${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --cpu-bind=none \
  -o "${LOG_DIR}/train.out" -e "${LOG_DIR}/train.err" \
  bash -lc "
  cd '${REPO_DIR}' &&
  '${REPO_DIR}/scripts/streamvln_container.sh' \
    torchrun --nnodes='${NNODES}' --nproc_per_node='${GPUS_PER_NODE}' \
    --node_rank=\"\${SLURM_NODEID}\" \
    --master_addr='${MASTER_ADDR}' --master_port='${MASTER_PORT}' \
    streamvln/streamvln_train.py \
    --model_name_or_path '${STAGE1_CKPT}' \
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
    --future_pretrain_only False \
    --future_fusion False \
    --future_injection_mode concat \
    --future_loss_weight 0.1 \
    --mm_tunable_parts future_predictor,future_connector,mm_mlp_adapter,mm_lora_layer \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,v_proj \
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
    --per_device_train_batch_size '${PER_DEVICE_BATCH}' \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps '${GRAD_ACCUM}' \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.075 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 32768 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --ddp_find_unused_parameters True \
    --ddp_timeout 7200 \
    --report_to wandb \
    --attn_implementation sdpa
"
