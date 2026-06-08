#!/usr/bin/env bash
set -euo pipefail

IMAGE="${STREAMVLN_APPTAINER_IMAGE:-/mnt/hwfile/lizhen/apptainer/vulkan_ros.sif}"
CONDA_PREFIX_DIR="${STREAMVLN_CONDA_PREFIX:-/mnt/petrelfs/lizhen/miniconda3/envs/streamvln}"
REPO_DIR="${STREAMVLN_REPO_DIR:-/mnt/hwfile/lizhen/StreamVLN}"

if [[ ! -d "${CONDA_PREFIX_DIR}" ]]; then
  echo "Missing conda env: ${CONDA_PREFIX_DIR}" >&2
  echo "Run scripts/install_streamvln_env.sh first." >&2
  exit 2
fi

APPTAINER_BIND_ARGS=(--bind /mnt:/mnt)
NVIDIA_LIB_DIR="${REPO_DIR}/third_party/nvidia_libs"

bind_host_lib() {
  local host_path="$1"
  local target_name="$2"
  if [[ -n "${host_path}" && -s "${host_path}" ]]; then
    mkdir -p "${NVIDIA_LIB_DIR}"
    touch "${NVIDIA_LIB_DIR}/${target_name}"
    APPTAINER_BIND_ARGS+=(--bind "${host_path}:${NVIDIA_LIB_DIR}/${target_name}")
  fi
}

CUDA_LIB_HOST=""
if [[ -e /lib64/libcuda.so.1 ]]; then
  CUDA_LIB_HOST="$(readlink -f /lib64/libcuda.so.1)"
elif [[ -e /lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
  CUDA_LIB_HOST="$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)"
fi

bind_host_lib "${CUDA_LIB_HOST}" libcuda.so.1
bind_host_lib "${CUDA_LIB_HOST}" libcuda.so

for lib_path in \
  /lib64/libEGL.so.1 \
  /lib64/libEGL_nvidia.so.0 \
  /lib64/libGLX_nvidia.so.0 \
  /lib64/libnvidia-eglcore.so.* \
  /lib64/libnvidia-glsi.so.* \
  /lib64/libnvidia-glcore.so.* \
  /lib64/libnvidia-gpucomp.so.* \
  /lib64/libnvidia-tls.so.*; do
  if [[ -e "${lib_path}" ]]; then
    bind_host_lib "$(readlink -f "${lib_path}")" "$(basename "$(readlink -f "${lib_path}")")"
  fi
done

if [[ -e /lib64/libEGL.so.1 ]]; then
  bind_host_lib "$(readlink -f /lib64/libEGL.so.1)" libEGL.so.1
  bind_host_lib "$(readlink -f /lib64/libEGL.so.1)" libEGL.so
fi
if [[ -e /lib64/libEGL_nvidia.so.0 ]]; then
  bind_host_lib "$(readlink -f /lib64/libEGL_nvidia.so.0)" libEGL_nvidia.so.0
fi
if [[ -e /lib64/libGLX_nvidia.so.0 ]]; then
  bind_host_lib "$(readlink -f /lib64/libGLX_nvidia.so.0)" libGLX_nvidia.so.0
fi
if [[ -e /lib64/libnvidia-ml.so.1 ]]; then
  NVIDIA_ML_HOST="$(readlink -f /lib64/libnvidia-ml.so.1)"
  bind_host_lib "${NVIDIA_ML_HOST}" "$(basename "${NVIDIA_ML_HOST}")"
  bind_host_lib "${NVIDIA_ML_HOST}" libnvidia-ml.so.1
  bind_host_lib "${NVIDIA_ML_HOST}" libnvidia-ml.so
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

for env_name in \
  STREAMVLN_ATTENTION \
  STREAMVLN_MAX_EPISODES \
  STREAMVLN_HABITAT_GPU_DEVICE_ID \
  TOKENIZERS_PARALLELISM \
  PYTORCH_CUDA_ALLOC_CONF \
  NCCL_DEBUG \
  NCCL_NVLS_ENABLE \
  NCCL_P2P_DISABLE \
  NCCL_IB_DISABLE \
  NCCL_SOCKET_IFNAME \
  CUDA_DEVICE_ORDER \
  WANDB_API_KEY \
  WANDB_ENTITY \
  WANDB_PROJECT \
  WANDB_NAME \
  WANDB_MODE \
  WANDB_DIR; do
  if [[ -n "${!env_name:-}" ]]; then
    export "APPTAINERENV_${env_name}=${!env_name}"
  fi
done

exec /usr/bin/apptainer exec --nv --cleanenv "${APPTAINER_BIND_ARGS[@]}" "${IMAGE}" bash -lc "
  source /mnt/petrelfs/lizhen/miniconda3/etc/profile.d/conda.sh
  conda activate '${CONDA_PREFIX_DIR}'
  cd '${REPO_DIR}'
  export PYTHONPATH='${REPO_DIR}:${REPO_DIR}/streamvln'
  export LD_LIBRARY_PATH='${NVIDIA_LIB_DIR}':'${REPO_DIR}/third_party/libcuda':'${REPO_DIR}/third_party/libegl':\${LD_LIBRARY_PATH:-}
  export HF_HOME=\${HF_HOME:-/mnt/inspurfs/evla2_t/lizhen/cache/huggingface}
  export MAGNUM_LOG=quiet
  export HABITAT_SIM_LOG=quiet
  unset HTTPS_CERT HTTPS_CERT_KEY SSL_CERT SSL_CERT_FILE
  exec \"\$@\"
" -- "$@"
