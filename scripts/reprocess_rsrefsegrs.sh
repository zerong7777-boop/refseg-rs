#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${REFSEG_RSREFSEGRS_DATA_ROOT:-}"
OUT_DIR="${REFSEG_REPROCESS_OUT_DIR:-${PWD}/refseg_reprocess_rsrefsegrs_out}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "FAIL: REFSEG_RSREFSEGRS_DATA_ROOT is not set" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "FAIL: REFSEG_RSREFSEGRS_DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FAIL: python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export REFSEG_REPROCESS_OUT_DIR="${OUT_DIR}"
for name in \
  annotations_train_segmentation_RSRefSegRS.txt \
  annotations_val_segmentation_RSRefSegRS.txt \
  annotations_test_segmentation_RSRefSegRS.txt; do
  if [[ -e "${OUT_DIR}/${name}" ]]; then
    echo "FAIL: output file already exists: ${OUT_DIR}/${name}" >&2
    exit 1
  fi
done

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -B - <<'PY'
import json
import os
from pathlib import Path

from refseg_runtime.annotations import load_records

DATA_ROOT = Path(os.environ["REFSEG_RSREFSEGRS_DATA_ROOT"])
OUT_DIR = Path(os.environ["REFSEG_REPROCESS_OUT_DIR"])
SPLITS = {
    "train": "annotations_train_segmentation_RSRefSegRS.txt",
    "val": "annotations_val_segmentation_RSRefSegRS.txt",
    "test": "annotations_test_segmentation_RSRefSegRS.txt",
}

summary = {
    "source_root": str(DATA_ROOT),
    "output_root": str(OUT_DIR),
    "mode": "normalize_copy_only",
    "note": "This wrapper does not rebuild annotations from raw imagery; it normalizes the canonical txt files into a separate output directory.",
    "files": [],
}
for split, name in SPLITS.items():
    src = DATA_ROOT / name
    if not src.exists():
        raise SystemExit(f"missing source annotation file: {src}")
    records = load_records(str(src), ann_format='txt')
    dst = OUT_DIR / name
    with src.open('r', encoding='utf-8-sig') as reader, dst.open('w', encoding='utf-8') as writer:
        for line in reader:
            line = line.rstrip('\r\n')
            if line.strip():
                writer.write(line.rstrip() + '\n')
    normalized = load_records(str(dst), ann_format='txt')
    if len(normalized) != len(records):
        raise SystemExit(f"row count changed during normalization for {split}: {len(records)} -> {len(normalized)}")
    summary['files'].append({
        'split': split,
        'source': str(src),
        'output': str(dst),
        'rows': len(normalized),
    })
summary_path = OUT_DIR / 'reprocess_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"[INFO] wrote normalized annotation files into {OUT_DIR}")
print(f"[INFO] summary_json={summary_path}")
for item in summary['files']:
    print(f"[INFO] {item['split']}: rows={item['rows']} -> {item['output']}")
print("Scope: normalization/canonicalization only; raw-to-annotation conversion is not shipped in this release.")
PY
