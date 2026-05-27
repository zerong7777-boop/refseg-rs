# refseg_runtime

Runtime package for RefSeg inference, evaluation, checkpoint inspection, and lightweight training bring-up.

## Scope

- txt/json annotation parsing
- dataset loading and preprocessing
- metric evaluation
- checkpoint inspection
- portable runtime model factories
- one release-safe reprocessing helper: `tools/refseg_build_resampled_ann.py`

## Canonical checkpoint form

Use sanitized checkpoints ending in `.state_dict.pth`. The runtime accepts only that form.

## Example usage

```bash
python -m refseg_runtime.eval \
  --ann-path <annotation_file> \
  --data-root <dataset_root> \
  --checkpoint <model.state_dict.pth> \
  --max-samples 32
```

```bash
python -m refseg_runtime.train \
  --ann-path <train_annotation_file> \
  --data-root <dataset_root> \
  --checkpoint <model.state_dict.pth> \
  --work-dir <work_dir>
```

```bash
python -m refseg_runtime.inspect \
  --checkpoint <model.state_dict.pth>
```

If the environment needs an explicit auxiliary `site-packages` path for the text backend, set `REFSEG_RUNTIME_SITE_PACKAGES` before running the runtime commands.
