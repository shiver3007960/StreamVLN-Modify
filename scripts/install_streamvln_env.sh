#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/mnt/petrelfs/lizhen/miniconda3/bin/conda}"
ENV_NAME="${STREAMVLN_ENV_NAME:-streamvln}"
ENV_PREFIX="${STREAMVLN_CONDA_PREFIX:-/mnt/petrelfs/lizhen/miniconda3/envs/${ENV_NAME}}"
HABITAT_ROOT="${STREAMVLN_HABITAT_ROOT:-/mnt/hwfile/lizhen/third_party/habitat-lab-v0.2.4-full}"
REPO_DIR="${STREAMVLN_REPO_DIR:-/mnt/hwfile/lizhen/StreamVLN}"
EGL_DIR="${REPO_DIR}/third_party/libegl"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.9
else
  echo "Reusing existing conda env: ${ENV_PREFIX}"
fi

if ! "${CONDA_BIN}" list -p "${ENV_PREFIX}" habitat-sim | grep -q '^habitat-sim '; then
  "${CONDA_BIN}" install -y -p "${ENV_PREFIX}" habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
else
  echo "habitat-sim already installed in ${ENV_PREFIX}"
fi

if [[ ! -d "${HABITAT_ROOT}/habitat-lab" || ! -d "${HABITAT_ROOT}/habitat-baselines" ]]; then
  mkdir -p "$(dirname "${HABITAT_ROOT}")"
  git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git "${HABITAT_ROOT}"
else
  echo "Reusing Habitat-Lab source: ${HABITAT_ROOT}"
fi

mkdir -p "${EGL_DIR}"
ln -sfn /usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0 "${EGL_DIR}/libEGL.so.1"

source /mnt/petrelfs/lizhen/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"
export LD_LIBRARY_PATH="${EGL_DIR}:${LD_LIBRARY_PATH:-}"
pip install -e "${HABITAT_ROOT}/habitat-lab"
pip install -e "${HABITAT_ROOT}/habitat-baselines"
pip install --no-build-isolation -r <(grep -v '^av==' "${REPO_DIR}/requirements.txt" | grep -v '^wavedrom==')

python - <<'PY'
import sys
print("streamvln env python", sys.version)
PY
