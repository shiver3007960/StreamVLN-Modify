#!/usr/bin/env bash
set -euo pipefail

ALLOC_JOB_ID="${1:?Usage: $0 <allocation-job-id>}"

REPO_DIR=/mnt/hwfile/lizhen/StreamVLN
EXP_DIR="${REPO_DIR}/experiments/streamvln/future_visual_tokens"
RESULT_ROOT=/mnt/inspurfs/evla2_t/lizhen/results/StreamVLN/future_visual_tokens
STAGE2_RUN=/mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618
OFFICIAL_CKPT="${REPO_DIR}/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-r2r-val-unseen-eval-alloc${ALLOC_JOB_ID}-${RUN_STAMP}}"
LOG_DIR="${EXP_DIR}/logs/${RUN_NAME}"
OUT_ROOT="${RESULT_ROOT}/${RUN_NAME}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
cd "${REPO_DIR}"

NODELIST="$(scontrol show job "${ALLOC_JOB_ID}" | awk -F= '/ NodeList=/{print $2}' | awk '{print $1}')"
mapfile -t NODES < <(scontrol show hostname "${NODELIST}")

if [ "${#NODES[@]}" -lt 4 ]; then
  echo "Need at least 4 nodes in allocation, got ${#NODES[@]}." >&2
  exit 1
fi

cat > "${OUT_ROOT}/manifest.tsv" <<EOF
name	checkpoint	node
official	${OFFICIAL_CKPT}	${NODES[0]}
future_ckpt3000	${STAGE2_RUN}/checkpoint-3000	${NODES[1]}
future_ckpt4000	${STAGE2_RUN}/checkpoint-4000	${NODES[2]}
future_ckpt4978	${STAGE2_RUN}/checkpoint-4978	${NODES[3]}
EOF

{
  echo "job_id=${ALLOC_JOB_ID}"
  echo "run_name=${RUN_NAME}"
  echo "date=$(date -Is)"
  echo "git_head=$(git rev-parse --short HEAD)"
  echo "node_list=${NODELIST}"
  echo "out_root=${OUT_ROOT}"
  echo "log_dir=${LOG_DIR}"
  cat "${OUT_ROOT}/manifest.tsv"
} | tee "${LOG_DIR}/launcher.log"

export STREAMVLN_ATTENTION=sdpa
unset STREAMVLN_MAX_EPISODES
unset STREAMVLN_HABITAT_GPU_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

run_eval() {
  local name="$1"
  local checkpoint="$2"
  local node="$3"
  local port="$4"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  {
    echo "name=${name}"
    echo "checkpoint=${checkpoint}"
    echo "node=${node}"
    echo "out_dir=${out_dir}"
    echo "master_port=${port}"
    echo "date=$(date -Is)"
  } > "${LOG_DIR}/${name}.launcher.log"

  srun --jobid="${ALLOC_JOB_ID}" --overlap \
    --nodelist="${node}" -N1 --ntasks=1 --gres=gpu:8 --cpu-bind=none \
    -o "${LOG_DIR}/${name}.out" -e "${LOG_DIR}/${name}.err" \
    bash -lc "
      cd '${REPO_DIR}' &&
      export STREAMVLN_ATTENTION=sdpa &&
      unset STREAMVLN_MAX_EPISODES &&
      unset STREAMVLN_HABITAT_GPU_DEVICE_ID &&
      export TOKENIZERS_PARALLELISM=false &&
      export HF_HOME=/mnt/inspurfs/evla2_t/lizhen/cache/huggingface &&
      export CUDA_DEVICE_ORDER=PCI_BUS_ID &&
      export NCCL_DEBUG=WARN &&
      export NCCL_NVLS_ENABLE=0 &&
      export NCCL_IB_DISABLE=1 &&
      '${REPO_DIR}/scripts/streamvln_container.sh' \
        torchrun --nproc_per_node=8 --master_port='${port}' \
        streamvln/streamvln_eval.py \
        --model_path '${checkpoint}' \
        --habitat_config_path config/vln_r2r.yaml \
        --eval_split val_unseen \
        --output_path '${out_dir}' \
        --num_future_steps 4 \
        --num_frames 32 \
        --num_history 8
    " &
  echo "$!" > "${LOG_DIR}/${name}.pid"
}

run_eval official "${OFFICIAL_CKPT}" "${NODES[0]}" 24101
run_eval future_ckpt3000 "${STAGE2_RUN}/checkpoint-3000" "${NODES[1]}" 24102
run_eval future_ckpt4000 "${STAGE2_RUN}/checkpoint-4000" "${NODES[2]}" 24103
run_eval future_ckpt4978 "${STAGE2_RUN}/checkpoint-4978" "${NODES[3]}" 24104

status=0
for pid_file in "${LOG_DIR}"/*.pid; do
  pid="$(cat "${pid_file}")"
  name="$(basename "${pid_file}" .pid)"
  if wait "${pid}"; then
    echo "${name}: completed" | tee -a "${LOG_DIR}/launcher.log"
  else
    echo "${name}: failed" | tee -a "${LOG_DIR}/launcher.log"
    status=1
  fi
done

python - <<'PY' "${OUT_ROOT}" > "${OUT_ROOT}/summary.tsv"
import json
import os
import sys

root = sys.argv[1]
print("name\tSR\tSPL\tOS\tNE\tlength")
for name in ["official", "future_ckpt3000", "future_ckpt4000", "future_ckpt4978"]:
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
exit "${status}"
