#!/usr/bin/env bash
# Purpose: evaluate the RSRefSegRS test oIoU-best packaged checkpoint.
# Expected env: set REFSEG_RSREFSEGRS_DATA_ROOT; optionally source examples/env.sh first.
# Output: ${REFSEG_REPORT_JSON:-${REFSEG_OUTPUT_ROOT}/eval_rsrefsegrs_test_oiou/report_t05.json}.
# Metric target: RSRefSegRS test oIoU-best checkpoint, evaluated at pred-threshold 0.5 by default.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${REFSEG_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CHECKPOINT_ROOT="${REFSEG_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
OUTPUT_ROOT="${REFSEG_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${REFSEG_DEVICE:-cuda:0}"
DATA_ROOT="${REFSEG_RSREFSEGRS_DATA_ROOT:-}"
CHECKPOINT="${REFSEG_CHECKPOINT:-${CHECKPOINT_ROOT}/rsrefsegrs/rsrefsegrs_test_oiou_best.state_dict.pth}"
THRESHOLD="${REFSEG_PRED_THRESHOLD:-0.5}"
MAX_SAMPLES="${REFSEG_MAX_SAMPLES:-0}"
REPORT_JSON="${REFSEG_REPORT_JSON:-${OUTPUT_ROOT}/eval_rsrefsegrs_test_oiou/report_t05.json}"

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

mkdir -p "$(dirname "${REPORT_JSON}")"
PREDICTOR_CONFIG=$(printf '{"device":"%s"}' "${DEVICE}")

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m refseg_runtime.eval \
  --ann-path "${DATA_ROOT}/annotations_test_segmentation_RSRefSegRS.txt" \
  --data-root "${DATA_ROOT}" \
  --img-prefix . \
  --mask-prefix . \
  --checkpoint "${CHECKPOINT}" \
  --predictor-config-json "${PREDICTOR_CONFIG}" \
  --pred-threshold "${THRESHOLD}" \
  --gt-mode gt_positive \
  --max-samples "${MAX_SAMPLES}" \
  --report-json "${REPORT_JSON}"
