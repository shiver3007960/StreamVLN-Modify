#!/bin/bash
#SBATCH -J streamvln-future-base-train-smoke-2g
#SBATCH -p eailab_os
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --quotatype=reserved
#SBATCH --time=03:00:00
#SBATCH -o /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.out
#SBATCH -e /mnt/hwfile/lizhen/StreamVLN/experiments/streamvln/future_visual_tokens/logs/%x-%j.err

set -euo pipefail

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
LOG_DIR="${EXP_DIR}/logs/${RUN_NAME}"
OUT_DIR="/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/${RUN_NAME}"
MODEL_PATH="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
VISION_TOWER="/mnt/inspurfs/evla1_t/zhangpingrui/siglip-so400m-patch14-384"
VIDEO_FOLDER="data/trajectory_data/R2R,data/trajectory_data/RxR,data/trajectory_data/EnvDrop"
MASTER_PORT=$((RANDOM % 101 + 23000))

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "repo=${REPO_DIR}"
echo "log_dir=${LOG_DIR}"
echo "output_dir=${OUT_DIR}"
echo "model_path=${MODEL_PATH}"
echo "vision_tower=${VISION_TOWER}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "master_port=${MASTER_PORT}"
nvidia-smi

export STREAMVLN_ATTENTION=sdpa
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  streamvln/streamvln_train.py
  --deepspeed scripts/zero2.json
  --model_name_or_path "${MODEL_PATH}"
  --version qwen_1_5
  --video_folder "${VIDEO_FOLDER}"
  --group_by_task False
  --num_history 2
  --num_future_steps 4
  --num_frames 4
  --data_augmentation False
  --mm_tunable_parts "mm_mlp_adapter,mm_lora_layer"
  --lora_enable True
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05
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
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 1
  --evaluation_strategy no
  --save_strategy steps
  --save_total_limit 2
  --learning_rate 1e-6
  --mm_vision_tower_lr 1e-6
  --weight_decay 0.
  --warmup_steps 0
  --lr_scheduler_type constant
  --logging_steps 1
  --model_max_length 4096
  --gradient_checkpointing True
  --dataloader_num_workers 2
  --lazy_preprocess True
  --dataloader_drop_last True
  --report_to none
  --attn_implementation sdpa
)

echo "phase=initial_train_save"
"${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  "${COMMON_ARGS[@]}" \
  --max_steps 2 \
  --save_steps 2

echo "checkpoint_after_phase1"
find "${OUT_DIR}" -maxdepth 2 -type f \( -name trainer_state.json -o -name model.safetensors.index.json -o -name "*.bin" \) -print | sort
test -f "${OUT_DIR}/checkpoint-2/trainer_state.json"

echo "phase=resume_train"
"${REPO_DIR}/scripts/streamvln_container.sh" \
  torchrun --nproc_per_node=2 --master_port="$((MASTER_PORT + 1))" \
  "${COMMON_ARGS[@]}" \
  --max_steps 3 \
  --save_steps 1

echo "checkpoint_after_resume"
find "${OUT_DIR}" -maxdepth 2 -type f \( -name trainer_state.json -o -name model.safetensors.index.json -o -name "*.bin" \) -print | sort
test -f "${OUT_DIR}/checkpoint-3/trainer_state.json"
python - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR}")
s2 = json.load(open(out / "checkpoint-2" / "trainer_state.json"))
s3 = json.load(open(out / "checkpoint-3" / "trainer_state.json"))
print("checkpoint-2 global_step", s2.get("global_step"))
print("checkpoint-3 global_step", s3.get("global_step"))
assert s2.get("global_step") == 2
assert s3.get("global_step") == 3
PY

echo "smoke_train_save_resume=OK"
