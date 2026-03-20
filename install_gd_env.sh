#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-gd_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.20}"
TORCH_VERSION="${TORCH_VERSION:-1.13.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.14.1}"
PYTORCH_CUDA_VERSION="${PYTORCH_CUDA_VERSION:-11.7}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-1}"
BUILD_MASK2FORMER_OPS="${BUILD_MASK2FORMER_OPS:-1}"

usage() {
  cat <<EOF
Usage:
  bash install_gd_env.sh

Optional env vars:
  ENV_NAME=gd_env
  PYTHON_VERSION=3.10.20
  TORCH_VERSION=1.13.1
  TORCHVISION_VERSION=0.14.1
  PYTORCH_CUDA_VERSION=11.7
  INSTALL_DETECTRON2=1
  BUILD_MASK2FORMER_OPS=1

Example:
  ENV_NAME=gd_env bash install_gd_env.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Error: required path not found: $1" >&2
    exit 1
  fi
}

run_in_env() {
  conda run -n "${ENV_NAME}" "$@"
}

echo "Project dir: ${PROJECT_DIR}"
echo "Target env: ${ENV_NAME}"

require_cmd conda
require_path "${PROJECT_DIR}/detectron2-main"
require_path "${PROJECT_DIR}/mask2former/modeling/pixel_decoder/ops/setup.py"
require_path "${PROJECT_DIR}/requirements.txt"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[1/6] Creating conda environment ${ENV_NAME}..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "[1/6] Conda environment ${ENV_NAME} already exists, reusing it."
fi

echo "[2/6] Installing PyTorch stack..."
conda install -y -n "${ENV_NAME}" \
  "pytorch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "pytorch-cuda=${PYTORCH_CUDA_VERSION}" \
  -c pytorch -c nvidia

echo "[3/6] Installing base Python dependencies..."
run_in_env python -m pip install --upgrade pip setuptools wheel
run_in_env python -m pip install -r "${PROJECT_DIR}/requirements.txt"
run_in_env python -m pip install \
  "fvcore==0.1.5.post20221221" \
  "iopath==0.1.9" \
  "pycocotools==2.0.11" \
  "yacs==0.1.8" \
  "omegaconf==2.3.0" \
  "hydra-core==1.3.2" \
  "timm==1.0.25" \
  "tensorboard==2.20.0" \
  "termcolor==3.3.0" \
  "cloudpickle==3.1.2" \
  "tabulate==0.10.0" \
  "tqdm==4.67.3" \
  "matplotlib==3.10.8" \
  "scipy==1.15.3" \
  "black==26.3.1"

if [[ "${INSTALL_DETECTRON2}" == "1" ]]; then
  echo "[4/6] Installing local detectron2 from detectron2-main..."
  run_in_env python -m pip install -e "${PROJECT_DIR}/detectron2-main"
else
  echo "[4/6] Skipping local detectron2 install."
fi

if [[ "${BUILD_MASK2FORMER_OPS}" == "1" ]]; then
  echo "[5/6] Building Mask2Former custom ops..."
  (
    cd "${PROJECT_DIR}/mask2former/modeling/pixel_decoder/ops"
    run_in_env python setup.py build install
  )
else
  echo "[5/6] Skipping Mask2Former custom ops build."
fi

echo "[6/6] Verifying core imports..."
run_in_env python - <<'PY'
import torch
import torchvision
import detectron2
import cv2
import timm

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("detectron2:", detectron2.__version__)
print("opencv:", cv2.__version__)
print("timm:", timm.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

echo
echo "Environment setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
