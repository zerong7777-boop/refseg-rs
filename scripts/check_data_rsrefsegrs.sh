#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${REFSEG_RSREFSEGRS_DATA_ROOT:-}"

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

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -B - <<'PY'
import os
from pathlib import Path

from refseg_runtime.annotations import build_samples, load_records, resolve_path

DATA_ROOT = Path(os.environ["REFSEG_RSREFSEGRS_DATA_ROOT"])
SPLITS = {
    'train': {
        'ann': DATA_ROOT / 'annotations_train_segmentation_RSRefSegRS.txt',
        'expected_rows': 2264,
        'img_prefix': '',
        'mask_prefix': '',
        'image_dir': DATA_ROOT / 'RefSegRS' / 'train' / 'images',
        'mask_dir': DATA_ROOT / 'assets_RefSegRS' / 'RefSeg_train',
    },
    'val': {
        'ann': DATA_ROOT / 'annotations_val_segmentation_RSRefSegRS.txt',
        'expected_rows': 431,
        'img_prefix': '',
        'mask_prefix': '',
        'image_dir': DATA_ROOT / 'RefSegRS' / 'val' / 'images',
        'mask_dir': DATA_ROOT / 'assets_RefSegRS' / 'RefSeg_val',
    },
    'test': {
        'ann': DATA_ROOT / 'annotations_test_segmentation_RSRefSegRS.txt',
        'expected_rows': 1817,
        'img_prefix': '',
        'mask_prefix': '',
        'image_dir': DATA_ROOT / 'RefSegRS' / 'test' / 'images',
        'mask_dir': DATA_ROOT / 'assets_RefSegRS' / 'RefSeg_test',
    },
}


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob('*') if item.is_file())


def collect_missing(records, data_root: Path, img_prefix: str, mask_prefix: str, limit: int = 5):
    missing_img = []
    missing_gt = []
    for rec in records:
        img_path = Path(resolve_path(rec['img_rel'], str(data_root), img_prefix))
        gt_path = Path(resolve_path(rec['gt_rel'], str(data_root), mask_prefix))
        if len(missing_img) < limit and not img_path.exists():
            missing_img.append(str(img_path))
        if len(missing_gt) < limit and not gt_path.exists():
            missing_gt.append(str(gt_path))
        if len(missing_img) >= limit and len(missing_gt) >= limit:
            break
    return missing_img, missing_gt


print(f'DATA_ROOT: {DATA_ROOT}')
problems = []
summary_stats = {}
for split, spec in SPLITS.items():
    ann_path = spec['ann']
    expected_rows = spec['expected_rows']
    if not ann_path.exists():
        problems.append(f"missing annotation file: {ann_path}")
        print(f"[FAIL] {split}: missing annotation file {ann_path}")
        continue

    records = load_records(str(ann_path), ann_format='txt')
    samples, missing_img_count, missing_gt_count = build_samples(
        records=records,
        data_root=str(DATA_ROOT),
        img_prefix=spec['img_prefix'],
        mask_prefix=spec['mask_prefix'],
    )
    row_count = len(records)
    unique_img_refs = len({rec['img_rel'] for rec in records})
    unique_gt_refs = len({rec['gt_rel'] for rec in records})
    image_file_count = count_files(spec['image_dir'])
    mask_file_count = count_files(spec['mask_dir'])
    missing_img_list, missing_gt_list = collect_missing(records, DATA_ROOT, spec['img_prefix'], spec['mask_prefix'])
    status = 'PASS' if row_count == expected_rows and missing_img_count == 0 and missing_gt_count == 0 and image_file_count >= unique_img_refs and mask_file_count >= unique_gt_refs else 'FAIL'
    if row_count != expected_rows:
        problems.append(f"{split}: expected {expected_rows} rows, found {row_count}")
    if missing_img_count:
        problems.append(f"{split}: missing images={missing_img_count}")
    if missing_gt_count:
        problems.append(f"{split}: missing masks={missing_gt_count}")
    if image_file_count < unique_img_refs:
        problems.append(f"{split}: image file count {image_file_count} is below unique referenced images {unique_img_refs}")
    if mask_file_count < unique_gt_refs:
        problems.append(f"{split}: mask file count {mask_file_count} is below unique referenced masks {unique_gt_refs}")
    print(f"[{status}] {split}: ann={ann_path}")
    print(f"  rows={row_count} expected={expected_rows} missing_img={missing_img_count} missing_gt={missing_gt_count}")
    print(f"  unique_img_refs={unique_img_refs} unique_gt_refs={unique_gt_refs}")
    print(f"  image_dir={spec['image_dir']} files={image_file_count}")
    print(f"  mask_dir={spec['mask_dir']} files={mask_file_count}")
    summary_stats[split] = (unique_img_refs, unique_gt_refs)
    if missing_img_list:
        print(f"  missing_img_sample={missing_img_list}")
    if missing_gt_list:
        print(f"  missing_gt_sample={missing_gt_list}")

print('SUMMARY:')
for split, spec in SPLITS.items():
    uniq_img, uniq_gt = summary_stats.get(split, ('n/a', 'n/a'))
    print(f"- {split}: expected_rows={spec['expected_rows']}, unique_img_refs={uniq_img}, unique_gt_refs={uniq_gt}, image_dir={spec['image_dir']}, mask_dir={spec['mask_dir']}")

if problems:
    print('FAIL: RSRefSegRS data check found issues:')
    for item in problems:
        print(f'- {item}')
    print('NEXT: verify the canonical annotation txt files and the RefSegRS/ and assets_RefSegRS/ trees under REFSEG_RSREFSEGRS_DATA_ROOT, then rerun this check.')
    raise SystemExit(1)

print('PASS: RSRefSegRS annotations and file layout match the packaged runtime expectations.')
print('NEXT: run scripts/reprocess_rsrefsegrs.sh to normalize the canonical annotation txt files into a fresh output directory if needed.')
PY
