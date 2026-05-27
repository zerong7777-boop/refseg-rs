#!/usr/bin/env bash
# Purpose: continue RSRefSegRS training from a packaged metric-specific checkpoint.
# Expected env: set REFSEG_RSREFSEGRS_DATA_ROOT; optionally source examples/env.sh first.
# Output: ${REFSEG_WORK_DIR:-${REFSEG_OUTPUT_ROOT}/resume_rsrefsegrs_<kind>_<timestamp>}.
# Metric target: defaults to test_miou_best; set REFSEG_RSREFSEGRS_CHECKPOINT_KIND=test_oiou_best to switch.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${REFSEG_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CHECKPOINT_ROOT="${REFSEG_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
OUTPUT_ROOT="${REFSEG_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
RUN_TAG="${REFSEG_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${REFSEG_DEVICE:-cuda:0}"
DATA_ROOT="${REFSEG_RSREFSEGRS_DATA_ROOT:-}"
CHECKPOINT_KIND="${REFSEG_RSREFSEGRS_CHECKPOINT_KIND:-test_miou_best}"

case "${CHECKPOINT_KIND}" in
  test_miou_best)
    DEFAULT_CHECKPOINT="${CHECKPOINT_ROOT}/rsrefsegrs/rsrefsegrs_test_miou_best.state_dict.pth"
    ;;
  test_oiou_best)
    DEFAULT_CHECKPOINT="${CHECKPOINT_ROOT}/rsrefsegrs/rsrefsegrs_test_oiou_best.state_dict.pth"
    ;;
  *)
    echo "FAIL: unsupported REFSEG_RSREFSEGRS_CHECKPOINT_KIND=${CHECKPOINT_KIND}; use test_miou_best or test_oiou_best" >&2
    exit 1
    ;;
esac

CHECKPOINT="${REFSEG_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
WORK_DIR="${REFSEG_WORK_DIR:-${OUTPUT_ROOT}/resume_rsrefsegrs_${CHECKPOINT_KIND}_${RUN_TAG}}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "FAIL: REFSEG_RSREFSEGRS_DATA_ROOT is not set" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "FAIL: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FAIL: python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -n "${REFSEG_RUNTIME_SITE_PACKAGES:-}" ]]; then
  if [[ ! -d "${REFSEG_RUNTIME_SITE_PACKAGES}" ]]; then
    echo "FAIL: REFSEG_RUNTIME_SITE_PACKAGES does not exist: ${REFSEG_RUNTIME_SITE_PACKAGES}" >&2
    exit 1
  fi
  export PYTHONPATH="${REFSEG_RUNTIME_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${WORK_DIR}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m refseg_runtime.train \
  --ann-path "${DATA_ROOT}/annotations_train_segmentation_RSRefSegRS.txt" \
  --data-root "${DATA_ROOT}" \
  --img-prefix . \
  --mask-prefix . \
  --val-ann-path "${DATA_ROOT}/annotations_val_segmentation_RSRefSegRS.txt" \
  --val-data-root "${DATA_ROOT}" \
  --val-img-prefix . \
  --val-mask-prefix . \
  --val-max-samples "${REFSEG_VAL_MAX_SAMPLES:-0}" \
  --batch-size "${REFSEG_BATCH_SIZE:-1}" \
  --num-workers "${REFSEG_NUM_WORKERS:-0}" \
  --max-samples "${REFSEG_MAX_SAMPLES:-0}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --lr "${REFSEG_LR:-1e-5}" \
  --max-steps "${REFSEG_MAX_STEPS:-1000000}" \
  --max-epochs "${REFSEG_MAX_EPOCHS:-20}" \
  --save-every "${REFSEG_SAVE_EVERY:-0}" \
  --grad-clip-norm "${REFSEG_GRAD_CLIP_NORM:-0.1}" \
  --amp \
  --val-interval "${REFSEG_VAL_INTERVAL:-1}" \
  --val-pred-threshold "${REFSEG_VAL_PRED_THRESHOLD:-0.5}" \
  --val-gt-mode gt_positive \
  --val-iou-thresholds 0.5,0.6,0.7,0.8,0.9 \
  --best-metric "${REFSEG_BEST_METRIC:-miou}" \
  --early-stop-patience "${REFSEG_EARLY_STOP_PATIENCE:-8}" \
  --early-stop-min-delta "${REFSEG_EARLY_STOP_MIN_DELTA:-0.05}" \
  --log-every "${REFSEG_LOG_EVERY:-100}" \
  --work-dir "${WORK_DIR}"
