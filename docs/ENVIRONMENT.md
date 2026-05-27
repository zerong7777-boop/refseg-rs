# RefSeg Environment

This release targets Linux with Python 3.10. The verified reference environment is:
- Python 3.10.16
- torch 2.1.0+cu121
- Linux x86_64

## Recommended install order

1. Create or activate a clean Python 3.10 environment.
2. Install PyTorch 2.1.0+cu121 from the official PyTorch wheel index that matches CUDA 12.1.
3. Install the runtime requirements from `requirements/runtime.txt`.
4. Set local paths with `examples/paths.env.example` and source `examples/env.sh`.
5. Run `scripts/check_env.sh` before the first package command.

Example:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0+cu121 torchvision==0.16.0+cu121
python -m pip install -r requirements/runtime.txt
source examples/env.sh
scripts/check_env.sh
```

## Internet-required dependencies

The runtime needs internet access the first time you install or refresh these packages:
- The PyTorch 2.1.0+cu121 wheel from the official PyTorch index.
- `transformers`, `einops`, `Pillow`, `opencv-python-headless`, and `typing_extensions` from PyPI if they are not already present.
- Hugging Face model files for the text backend the first time `transformers` initializes `bert-base-uncased`, unless you already have that cache populated.

If you run fully offline, pre-populate the Python environment and Hugging Face cache before using the runtime.

## Runtime notes

- The runtime accepts only sanitized checkpoints ending in `.state_dict.pth`.
- `examples/env.sh` exports `PYTHONNOUSERSITE=1` so user-site packages do not leak into the runtime.
- `REFSEG_RUNTIME_SITE_PACKAGES` is optional and can be used when the runtime dependencies live in a separate site-packages directory.
- When the shell bootstrap scripts point `REFSEG_RUNTIME_SITE_PACKAGES` at a standard `.../lib/python3.10/site-packages` tree, they derive the sibling `lib` directory and prepend it to `LD_LIBRARY_PATH`.
- After `bert-base-uncased` is downloaded into the Hugging Face cache, you may set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to avoid repeated online metadata checks in slow network environments.
- The default execution path is single-GPU. For multi-GPU use, set `CUDA_VISIBLE_DEVICES` or launch one process per GPU; the runtime scripts do not require a distributed launcher.
