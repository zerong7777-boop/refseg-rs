#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${REFSEG_REFER_DATA_ROOT:-}"

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

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -B - <<'PY'
import os
from pathlib import Path

from refseg_runtime.annotations import build_samples, load_records, resolve_path

DATA_ROOT = Path(os.environ["REFSEG_REFER_DATA_ROOT"])
EXPECTED = {
    "train": 103945,
    "val": 23267,
}
ANN_CANDIDATES = {
    "train": [DATA_ROOT / "en_txt" / "merged_train_trimmed.txt", DATA_ROOT / "merged_train_trimmed.txt"],
    "val": [DATA_ROOT / "en_txt" / "merged_val_trimmed.txt", DATA_ROOT / "merged_val_trimmed.txt"],
}


def pick_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob('*') if item.is_file())


def collect_missing(records, data_root: Path, img_prefix: str, mask_prefix: str, limit: int = 5):
    missing_img = []
    missing_gt = []
    for rec in records:
        img_path = Path(resolve_path(rec["img_rel"], str(data_root), img_prefix))
        gt_path = Path(resolve_path(rec["gt_rel"], str(data_root), mask_prefix))
        if len(missing_img) < limit and not img_path.exists():
            missing_img.append(str(img_path))
        if len(missing_gt) < limit and not gt_path.exists():
            missing_gt.append(str(gt_path))
        if len(missing_img) >= limit and len(missing_gt) >= limit:
            break
    return missing_img, missing_gt


print(f"DATA_ROOT: {DATA_ROOT}")
problems = []
summary_rows = []
for split, expected_rows in EXPECTED.items():
    ann_path = pick_existing(ANN_CANDIDATES[split])
    if ann_path is None:
        problems.append(f"missing annotation file for {split}: {', '.join(str(p) for p in ANN_CANDIDATES[split])}")
        summary_rows.append((split, 0, expected_rows, 0, 0, 0, 0, 0, 0, None))
        continue

    records = load_records(str(ann_path), ann_format='txt')
    samples, missing_img_count, missing_gt_count = build_samples(
        records=records,
        data_root=str(DATA_ROOT),
        img_prefix='images',
        mask_prefix='masked',
    )
    unique_img_refs = len({rec['img_rel'] for rec in records})
    unique_gt_refs = len({rec['gt_rel'] for rec in records})
    img_file_count = count_files(DATA_ROOT / 'images')
    gt_file_count = count_files(DATA_ROOT / 'masked')
    missing_img_list, missing_gt_list = collect_missing(records, DATA_ROOT, 'images', 'masked')
    row_count = len(records)
    line_status = 'PASS' if row_count == expected_rows and missing_img_count == 0 and missing_gt_count == 0 else 'FAIL'
    if row_count != expected_rows:
        problems.append(f"{split}: expected {expected_rows} rows, found {row_count}")
    if missing_img_count:
        problems.append(f"{split}: missing images={missing_img_count}")
    if missing_gt_count:
        problems.append(f"{split}: missing masks={missing_gt_count}")
    if img_file_count < unique_img_refs:
        problems.append(f"{split}: image file count {img_file_count} is below unique referenced images {unique_img_refs}")
    if gt_file_count < unique_gt_refs:
        problems.append(f"{split}: mask file count {gt_file_count} is below unique referenced masks {unique_gt_refs}")
    summary_rows.append((split, row_count, expected_rows, missing_img_count, missing_gt_count, unique_img_refs, unique_gt_refs, img_file_count, gt_file_count, ann_path))
    print(f"[{line_status}] {split}: ann={ann_path}")
    print(f"  rows={row_count} expected={expected_rows} missing_img={missing_img_count} missing_gt={missing_gt_count}")
    print(f"  unique_img_refs={unique_img_refs} unique_gt_refs={unique_gt_refs}")
    print(f"  image_files={img_file_count} mask_files={gt_file_count}")
    if missing_img_list:
        print(f"  missing_img_sample={missing_img_list}")
    if missing_gt_list:
        print(f"  missing_gt_sample={missing_gt_list}")

print('SUMMARY:')
for split, row_count, expected_rows, missing_img_count, missing_gt_count, unique_img_refs, unique_gt_refs, img_file_count, gt_file_count, ann_path in summary_rows:
    print(
        f"- {split}: rows={row_count}/{expected_rows}, missing_img={missing_img_count}, missing_gt={missing_gt_count}, "
        f"unique_img_refs={unique_img_refs}, unique_gt_refs={unique_gt_refs}, image_files={img_file_count}, mask_files={gt_file_count}"
    )

if problems:
    print('FAIL: refer data check found issues:')
    for item in problems:
        print(f"- {item}")
    print('NEXT: verify the annotation files and the images/ and masked/ trees under REFSEG_REFER_DATA_ROOT, then rerun this check.')
    raise SystemExit(1)

print('PASS: refer data layout and annotation counts match the packaged runtime expectations.')
print('NEXT: run scripts/reprocess_refer.sh if you want a resampled train annotation for experimentation.')
PY
