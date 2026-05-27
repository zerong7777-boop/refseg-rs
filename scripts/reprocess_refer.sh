#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${REFSEG_REFER_DATA_ROOT:-}"
OUT_DIR="${REFSEG_REPROCESS_OUT_DIR:-${PWD}/refseg_reprocess_refer_out}"
ANN_FILE="${REFSEG_REFER_ANN_FILE:-}"
MASK_ROOT="${REFSEG_REFER_MASK_ROOT:-}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "FAIL: REFSEG_REFER_DATA_ROOT is not set" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "FAIL: REFSEG_REFER_DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FAIL: python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${ANN_FILE}" ]]; then
  if [[ -f "${DATA_ROOT}/en_txt/merged_train_trimmed.txt" ]]; then
    ANN_FILE="${DATA_ROOT}/en_txt/merged_train_trimmed.txt"
  elif [[ -f "${DATA_ROOT}/merged_train_trimmed.txt" ]]; then
    ANN_FILE="${DATA_ROOT}/merged_train_trimmed.txt"
  else
    echo "FAIL: could not find a canonical refer train annotation under ${DATA_ROOT}" >&2
    exit 1
  fi
fi
if [[ -z "${MASK_ROOT}" ]]; then
  if [[ -d "${DATA_ROOT}/masked" ]]; then
    MASK_ROOT="${DATA_ROOT}/masked"
  else
    echo "FAIL: could not find a masked/ directory under ${DATA_ROOT}" >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"
OUT_ANN="${OUT_DIR}/merged_train_resampled.txt"
SUMMARY_JSON="${OUT_DIR}/merged_train_resampled.summary.json"
if [[ -e "${OUT_ANN}" || -e "${SUMMARY_JSON}" ]]; then
  echo "FAIL: output files already exist in ${OUT_DIR}; choose a new output directory" >&2
  exit 1
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/refseg_build_resampled_ann.py"   --ann-file "${ANN_FILE}"   --mask-root "${MASK_ROOT}"   --out-ann "${OUT_ANN}"   --summary-json "${SUMMARY_JSON}"

echo "Generated refer reprocess outputs:"
echo "- ${OUT_ANN}"
echo "- ${SUMMARY_JSON}"
echo "Scope: hard-example resampling only; source annotation and masks are never modified."
