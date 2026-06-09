#!/bin/bash
#SBATCH -J streamvln-future-cat-smoke
#SBATCH -p eailab_system
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=reserved
#SBATCH --time=06:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
STAGE1_CKPT="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/checkpoint-4000"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_NAME:-streamvln-future-cat-smoke}-${SLURM_JOB_ID:-manual}}"
LOG_DIR="${EXP_DIR}/logs/${RUN_NAME}"
OUT_DIR="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/${RUN_NAME}"
VISION_TOWER="/mnt/inspurfs/evla1_t/zhangpingrui/siglip-so400m-patch14-384"
VIDEO_FOLDER="data/trajectory_data/R2R,data/trajectory_data/RxR,data/trajectory_data/EnvDrop"
MASTER_PORT=$((RANDOM % 101 + 25200))

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "run_name=${RUN_NAME}"
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "git_head=$(git rev-parse --short HEAD)"
  echo "output_dir=${OUT_DIR}"
  echo "stage1_checkpoint=${STAGE1_CKPT}"
  echo "injection=concat"
  echo "trainable=future_predictor,future_connector,mm_mlp_adapter,mm_lora_layer"
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

COMMON_ARGS=(
  streamvln/streamvln_train.py
  --model_name_or_path "${STAGE1_CKPT}"
  --version qwen_1_5
  --video_folder "${VIDEO_FOLDER}"
  --group_by_task False
  --num_history 8
  --num_future_steps 4
  --num_frames 32
  --future_target_offset 4
  --data_augmentation True
  --use_future_tokens True
  --future_token_count 196
  --future_qformer_depth 2
  --future_qformer_heads 8
  --future_pretrain_only False
  --future_fusion False
  --future_injection_mode concat
  --future_loss_weight 0.1
  --mm_tunable_parts "future_predictor,future_connector,mm_mlp_adapter,mm_lora_layer"
  --lora_enable True
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05
  --lora_target_modules q_proj,v_proj
  --vision_tower "${VISION_TOWER}"
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --image_aspect_ratio anyres_max_9
  --image_grid_pinpoints "(1x1),...,(6x6)"
  --bf16 True
  --tf32 True
  --run_name "${RUN_NAME}"
  --output_dir "${OUT_DIR}"
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 2
  --evaluation_strategy no
  --save_strategy steps
  --save_total_limit 2
  --learning_rate 2e-5
  --mm_projector_lr 2e-5
  --weight_decay 0.
  --warmup_ratio 0.075
  --lr_scheduler_type cosine
  --logging_steps 1
  --model_max_length 32768
  --gradient_checkpointing False
  --dataloader_num_workers 8
  --lazy_preprocess True
  --dataloader_drop_last True
  --ddp_find_unused_parameters True
  --ddp_timeout 7200
  --report_to wandb
  --attn_implementation sdpa
)

echo "phase=initial_train_save" | tee -a "${LOG_DIR}/launcher.log"
"${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  "${COMMON_ARGS[@]}" \
  --max_steps 10 \
  --save_steps 10

test -f "${OUT_DIR}/checkpoint-10/trainer_state.json"
test -f "${OUT_DIR}/checkpoint-10/non_lora_trainables.bin"

echo "phase=resume_train" | tee -a "${LOG_DIR}/launcher.log"
"${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nproc_per_node=8 --master_port="$((MASTER_PORT + 1))" \
  "${COMMON_ARGS[@]}" \
  --max_steps 12 \
  --save_steps 2

test -f "${OUT_DIR}/checkpoint-12/trainer_state.json"
test -f "${OUT_DIR}/checkpoint-12/non_lora_trainables.bin"

python - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR}")
s10 = json.load(open(out / "checkpoint-10" / "trainer_state.json"))
s12 = json.load(open(out / "checkpoint-12" / "trainer_state.json"))
print("checkpoint-10 global_step", s10.get("global_step"))
print("checkpoint-12 global_step", s12.get("global_step"))
assert s10.get("global_step") == 10
assert s12.get("global_step") == 12
PY

echo "direct_cat_smoke=OK" | tee -a "${LOG_DIR}/launcher.log"
