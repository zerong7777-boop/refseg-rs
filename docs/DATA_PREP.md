# RefSeg Data Preparation

This release provides two release-ready data checks and two reprocessing wrappers.

## `refer_data_20250908`

Example data root:
- `/path/to/refer_data_20250908`

Expected layout:
- `en_txt/merged_train_trimmed.txt` with 103945 non-empty rows
- `en_txt/merged_val_trimmed.txt` with 23267 non-empty rows
- `images/`
- `masked/`

Reference expectations:
- `REFSEG_REFER_DATA_ROOT` should point at the dataset root.
- The packaged runtime resolves valid samples with `missing_img=0` and `missing_gt=0`.
- The loader accepts both the canonical `en_txt/...` layout and a flat root-level copy of the same annotation files when present.

### `check_data_refer.sh`

Validates:
- `REFSEG_REFER_DATA_ROOT` exists.
- The train and val annotation files are present and parse successfully.
- Non-empty row counts match the expected 103945 and 23267 values.
- `images/` and `masked/` contain enough files for the referenced samples.
- The runtime resolver finds no missing image or mask paths.
- A short sample of missing paths is printed when anything is missing.

Recommended next action if the check fails:
- Fix the annotation file location, `images/`, or `masked/` tree under `REFSEG_REFER_DATA_ROOT`, then rerun the check.

### `reprocess_refer.sh`

Wraps the shipped `tools/refseg_build_resampled_ann.py` helper.

What it does:
- Reads the canonical refer train annotation file.
- Uses the `masked/` directory to score hard examples.
- Writes a resampled annotation txt file and a summary JSON into a fresh output directory.
- Never mutates the source dataset.

What it generates by default:
- `merged_train_resampled.txt`
- `merged_train_resampled.summary.json`

## `RSRefSegRS`

Example data root:
- `/path/to/RSRefSegRS/RSRefSeg`

Expected annotation files:
- `annotations_train_segmentation_RSRefSegRS.txt` with 2264 rows
- `annotations_val_segmentation_RSRefSegRS.txt` with 431 rows
- `annotations_test_segmentation_RSRefSegRS.txt` with 1817 rows

Path conventions in the annotation rows:
- Images use relative paths like `RefSegRS/train/images/...`
- Masks use relative paths like `assets_RefSegRS/RefSeg_train/...`
- The check script resolves these relative paths against `REFSEG_RSREFSEGRS_DATA_ROOT`

Reference expectations:
- The packaged runtime resolves valid samples with `missing_img=0` and `missing_gt=0`.
- Train, val, and test row counts should match the values above.

### `check_data_rsrefsegrs.sh`

Validates:
- `REFSEG_RSREFSEGRS_DATA_ROOT` exists.
- All three canonical annotation files are present and parse successfully.
- Row counts match the expected 2264, 431, and 1817 values.
- The resolved image and mask paths exist.
- The file counts under `RefSegRS/{train,val,test}/images` and `assets_RefSegRS/RefSeg_{train,val,test}` cover the unique referenced image and mask paths, and the script reports a failure if either tree is smaller than the referenced set.
- A short sample of missing paths is printed when anything is missing.

Recommended next action if the check fails:
- Fix the canonical annotation files or the `RefSegRS/` and `assets_RefSegRS/` trees under `REFSEG_RSREFSEGRS_DATA_ROOT`, then rerun the check.

### `reprocess_rsrefsegrs.sh`

This release does not provide a raw-to-annotation builder for RSRefSegRS.

What it does:
- Normalizes and copies the canonical annotation txt files into a fresh output directory.
- Strips BOMs, removes blank lines, and preserves the row count.
- Writes a small summary JSON alongside the normalized files.

What it does not do:
- It does not synthesize annotations from raw imagery or masks.
- It does not change the dataset semantics.

Default outputs:
- `annotations_train_segmentation_RSRefSegRS.txt`
- `annotations_val_segmentation_RSRefSegRS.txt`
- `annotations_test_segmentation_RSRefSegRS.txt`
- `reprocess_summary.json`

## Environment variables

The scripts use these release-ready variables when set:
- `REFSEG_REFER_DATA_ROOT`
- `REFSEG_RSREFSEGRS_DATA_ROOT`
- `REFSEG_REPROCESS_OUT_DIR`
- `PYTHON_BIN`
- `PYTHONNOUSERSITE`
