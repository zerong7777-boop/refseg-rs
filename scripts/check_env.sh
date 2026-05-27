#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FAIL: python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
if [[ -n "${REFSEG_RUNTIME_SITE_PACKAGES:-}" ]]; then
  if [[ ! -d "${REFSEG_RUNTIME_SITE_PACKAGES}" ]]; then
    echo "FAIL: REFSEG_RUNTIME_SITE_PACKAGES does not exist: ${REFSEG_RUNTIME_SITE_PACKAGES}" >&2
    exit 1
  fi
  export PYTHONPATH="${REFSEG_RUNTIME_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
  runtime_lib_dir=$(cd "$(dirname "$(dirname "${REFSEG_RUNTIME_SITE_PACKAGES}")")" && pwd)
  if [[ -d "${runtime_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${runtime_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"
echo "REFSEG_DEVICE: ${REFSEG_DEVICE:-<unset>}"
echo "REFSEG_RUNTIME_SITE_PACKAGES: ${REFSEG_RUNTIME_SITE_PACKAGES:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"

"${PYTHON_BIN}" -B - <<'PY'
import importlib
import os

failures = []

device = os.environ.get("REFSEG_DEVICE", "").strip()

for name in ("numpy", "PIL", "cv2", "typing_extensions", "transformers", "einops"):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "n/a")
        print(f"PASS import {name}: {version}")
    except Exception as exc:
        failures.append((name, exc))
        print(f"FAIL import {name}: {type(exc).__name__}: {exc}")

try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    print(f"PASS import torch: {torch.__version__}")
    print(f"INFO torch.cuda.is_available: {cuda_available}")
    print(f"INFO torch.cuda.device_count: {cuda_count}")
    print(f"INFO torch.version.cuda: {torch.version.cuda}")
    if device.startswith("cuda") and (not cuda_available or cuda_count < 1):
        failures.append(("torch.cuda", RuntimeError(f"CUDA requested by REFSEG_DEVICE={device!r} but no usable CUDA device is available")))
        print(f"FAIL CUDA device check: requested {device!r} but torch CUDA is unavailable")
except Exception as exc:
    failures.append(("torch", exc))
    print(f"FAIL import torch: {type(exc).__name__}: {exc}")
    if device.startswith("cuda"):
        failures.append(("torch.cuda", RuntimeError(f"CUDA requested by REFSEG_DEVICE={device!r} but torch could not initialize")))
        print(f"FAIL CUDA device check: requested {device!r} but torch could not initialize")

try:
    from refseg_runtime.runtime_env import ensure_transformers_backend
    resolved = ensure_transformers_backend(os.environ.get("REFSEG_RUNTIME_SITE_PACKAGES", ""))
    print(f"PASS transformers backend init: {resolved or 'current-env'}")
except Exception as exc:
    failures.append(("refseg_runtime.runtime_env.ensure_transformers_backend", exc))
    print(f"FAIL transformers backend init: {type(exc).__name__}: {exc}")

if failures:
    print("ENV CHECK FAILED")
    for name, exc in failures:
        print(f" - {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print("ENV CHECK PASSED")
PY
