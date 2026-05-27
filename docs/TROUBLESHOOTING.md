# Troubleshooting

## Path Mismatch

Symptom:

- `FAIL: REFSEG_REFER_DATA_ROOT is not set`
- `FAIL: REFSEG_RSREFSEGRS_DATA_ROOT does not exist`
- data checks report missing images or masks

Action:

- Edit the file sourced by `REFSEG_PATHS_FILE`.
- Rerun `source examples/env.sh`.
- Rerun the matching data check script.
- Check `docs/DATA_PREP.md` for the expected layout and annotation filenames.

## Missing Modules

Symptom:

- `FAIL import torch`
- `FAIL import transformers`
- `ModuleNotFoundError`

Action:

- Activate the intended Python environment.
- Install PyTorch separately using the command in `ENVIRONMENT.md`.
- Run `python -m pip install -r requirements/runtime.txt`.
- Rerun `scripts/check_env.sh`.

## User-Site Package Contamination

Symptom:

- A clean environment unexpectedly imports packages from `~/.local`.
- Torch fails with CUDA library errors even though the env has the correct wheel.

Action:

- Keep `PYTHONNOUSERSITE=1` from `examples/env.sh`.
- Avoid manually prepending unrelated directories to `PYTHONPATH`.
- If dependencies live in a separate site-packages tree, set `REFSEG_RUNTIME_SITE_PACKAGES` to that single directory.

## CUDA Or GPU Selection

Symptom:

- `torch.cuda.is_available: False`
- `CUDA requested by REFSEG_DEVICE='cuda:0' but no usable CUDA device is available`
- out-of-memory during eval or training

Action:

- Confirm the NVIDIA driver supports the installed PyTorch CUDA build.
- Set `REFSEG_DEVICE=cpu` only for non-performance smoke checks.
- Select a GPU with `CUDA_VISIBLE_DEVICES` or set `REFSEG_DEVICE=cuda:<index>`.
- Reduce smoke size with `REFSEG_MAX_SAMPLES=1`.

## Hugging Face Timeouts

Symptom:

- Repeated retries against `huggingface.co` for `bert-base-uncased`.

Action:

- Let the first run populate the Hugging Face cache when internet is available.
- After the cache is populated, use:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Checkpoint Form

Symptom:

- `Checkpoint must be a sanitized .state_dict.pth file`

Action:

- Use only the packaged files under `checkpoints/`.
- Do not pass raw legacy `.pth` checkpoints to the user runtime.

## Data Format Mismatch

Symptom:

- data checks pass root existence but fail row counts or missing referenced files

Action:

- For refer data, confirm `en_txt/merged_train_trimmed.txt`, `en_txt/merged_val_trimmed.txt`, `images/`, and `masked/`.
- For RSRefSegRS, confirm `annotations_*_segmentation_RSRefSegRS.txt`, `RefSegRS/{train,val,test}/images`, and `assets_RefSegRS/RefSeg_{train,val,test}`.
- If the user only has raw, unprocessed data, keep the package in place and adapt preprocessing on the user machine during remote deployment.
