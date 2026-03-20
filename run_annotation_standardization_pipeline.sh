#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/tianruiliu/codespace/data_process"
PROJECT_DIR="${ROOT_DIR}/SemiT-SAM"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
STANDARDIZED_DIR="${ROOT_DIR}/standardized_annotations"
CONDA_ENV="${CONDA_ENV:-gd_env}"
DEVICE="${DEVICE:-cpu}"

cd "${PROJECT_DIR}"

echo "[1/3] 标准化两位医生标注..."
if [ "${DEVICE}" = "cuda" ]; then
  conda run -n "${CONDA_ENV}" python -c "import runpy, sys, torch; _ = torch.tensor([0.0], device='cuda'); sys.argv=['normalize_doctor_annotations.py', '--output-dir', '${STANDARDIZED_DIR}', '--device', 'cuda']; runpy.run_path('normalize_doctor_annotations.py', run_name='__main__')"
else
  conda run -n "${CONDA_ENV}" python normalize_doctor_annotations.py --output-dir "${STANDARDIZED_DIR}" --device "${DEVICE}"
fi

echo "[2/3] 生成双医生原始标注叠加图..."
TMP_PILLOW_DIR="/tmp/codex_pillow"
TMP_PILLOW_BAK="/tmp/codex_pillow.__codex_bak__"
RESTORE_TMP_PILLOW=0
if [ -d "${TMP_PILLOW_DIR}" ]; then
  mv "${TMP_PILLOW_DIR}" "${TMP_PILLOW_BAK}"
  RESTORE_TMP_PILLOW=1
fi

cleanup() {
  if [ "${RESTORE_TMP_PILLOW}" -eq 1 ] && [ -d "${TMP_PILLOW_BAK}" ]; then
    mv "${TMP_PILLOW_BAK}" "${TMP_PILLOW_DIR}"
  fi
}
trap cleanup EXIT

conda run -n "${CONDA_ENV}" python "${SCRIPTS_DIR}/render_dual_json_overlays.py"

echo "[3/3] 生成标准化标注叠加图..."
conda run -n "${CONDA_ENV}" python render_standardized_overlays.py

echo
echo "流程完成。"
echo "标准化结果:"
echo "  ${STANDARDIZED_DIR}/normalized_annotations.json"
echo "  ${STANDARDIZED_DIR}/summary.json"
echo "  ${STANDARDIZED_DIR}/ignored_annotations.json"
echo
echo "双医生原始叠加图:"
echo "  ${ROOT_DIR}/586份数据20260116/586张原图/双json叠加标注/index.html"
echo
echo "标准化结果叠加图:"
echo "  ${STANDARDIZED_DIR}/overlays/index.html"
