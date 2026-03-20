#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-gd_env}"
DEVICE="${DEVICE:-cpu}"

DATA_ROOT=""
ANNOTATION_DIR=""
IMAGE_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_per_doctor_annotation_standardization_pipeline.sh \
    --data-root /path/to/data_root \
    --image-dir /path/to/images \
    [--annotation-dir /path/to/annotations] \
    [--output-dir /path/to/output] \
    [--conda-env gd_env] \
    [--device cpu|cuda] \
    [extra normalize args...]

Behavior:
  - Only runs doctor-annotation standardization.
  - Processes every json file under annotation-dir independently.
  - Writes outputs under data-root by default:
      <data-root>/standardized_annotations_per_doctor/
  - Relative paths are resolved from this script's project directory.

Examples:
  bash run_per_doctor_annotation_standardization_pipeline.sh \
    --data-root /data/case_001 \
    --annotation-dir /data/case_001/annotations \
    --image-dir /data/case_001/images

  bash run_per_doctor_annotation_standardization_pipeline.sh \
    --data-root /data/case_001 \
    --image-dir /data/case_001/images \
    --device cuda \
    --single-min-match-score 0.40
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --annotation-dir)
      ANNOTATION_DIR="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Error: --data-root is required." >&2
  usage
  exit 1
fi

if [[ -z "${IMAGE_DIR}" ]]; then
  echo "Error: --image-dir is required." >&2
  usage
  exit 1
fi

if [[ -z "${ANNOTATION_DIR}" ]]; then
  ANNOTATION_DIR="${DATA_ROOT}/annotations"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${DATA_ROOT}/standardized_annotations_per_doctor"
fi

cd "${PROJECT_DIR}"

echo "Running per-doctor annotation standardization..."
echo "  data root: ${DATA_ROOT}"
echo "  annotation dir: ${ANNOTATION_DIR}"
echo "  image dir: ${IMAGE_DIR}"
echo "  output dir: ${OUTPUT_DIR}"
echo "  conda env: ${CONDA_ENV}"
echo "  device: ${DEVICE}"

conda run -n "${CONDA_ENV}" python normalize_doctor_annotations_individually.py \
  --data-root "${DATA_ROOT}" \
  --annotation-dir "${ANNOTATION_DIR}" \
  --image-dir "${IMAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
