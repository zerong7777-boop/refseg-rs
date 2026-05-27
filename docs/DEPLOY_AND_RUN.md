# Deploy And Run

Run these commands from the repository root.

## 1. Install Environment

Follow `docs/ENVIRONMENT.md`. The reference baseline is Python 3.10 with `torch==2.1.0+cu121`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0+cu121 torchvision==0.16.0+cu121
python -m pip install -r requirements/runtime.txt
```

## 2. Set Local Paths

Copy the template and edit it:

```bash
cp examples/paths.env.example examples/paths.env
${EDITOR:-vi} examples/paths.env
export REFSEG_PATHS_FILE="$PWD/examples/paths.env"
source examples/env.sh
```

At minimum set:

```bash
export REFSEG_PROJECT_ROOT="/absolute/path/to/refseg-rs"
export REFSEG_REFER_DATA_ROOT="/absolute/path/to/refer_data_20250908"
export REFSEG_RSREFSEGRS_DATA_ROOT="/absolute/path/to/RSRefSegRS/RSRefSeg"
export REFSEG_DEVICE="cuda:0"
```

## 3. Check Environment

```bash
scripts/check_env.sh
```

If Hugging Face model files are already cached and the network is slow, you can avoid repeated online metadata checks:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 4. Check Data

```bash
scripts/check_data_refer.sh
scripts/check_data_rsrefsegrs.sh
```

Both scripts should end with `PASS` before eval or training.

## 5. Reproduce Evaluation

```bash
scripts/eval_refer_miou.sh
scripts/eval_refer_oiou.sh
scripts/eval_rsrefsegrs_test_miou.sh
scripts/eval_rsrefsegrs_test_oiou.sh
```

Reports are written under `${REFSEG_OUTPUT_ROOT:-$REFSEG_PROJECT_ROOT/outputs}` unless `REFSEG_REPORT_JSON` is set.

For a quick smoke run:

```bash
export REFSEG_MAX_SAMPLES=1
scripts/eval_refer_miou.sh
unset REFSEG_MAX_SAMPLES
```

## 6. Continue Training

Refer-data metric-specific resumes:

```bash
scripts/resume_refer_miou.sh
scripts/resume_refer_oiou.sh
```

RSRefSegRS defaults to the test-mIoU checkpoint:

```bash
scripts/resume_rsrefsegrs.sh
```

To continue from the test-oIoU checkpoint instead:

```bash
REFSEG_RSREFSEGRS_CHECKPOINT_KIND=test_oiou_best scripts/resume_rsrefsegrs.sh
```

Useful overrides:

```bash
export REFSEG_OUTPUT_ROOT="$REFSEG_PROJECT_ROOT/outputs"
export REFSEG_LR=1e-5
export REFSEG_MAX_STEPS=1000
export REFSEG_MAX_EPOCHS=20
export REFSEG_VAL_MAX_SAMPLES=4000
```

For a one-step training smoke test:

```bash
REFSEG_MAX_SAMPLES=1 REFSEG_VAL_MAX_SAMPLES=1 REFSEG_MAX_STEPS=1 REFSEG_MAX_EPOCHS=1 \
  scripts/resume_rsrefsegrs.sh
```
