#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATHS_FILE="${REFSEG_PATHS_FILE:-${SCRIPT_DIR}/paths.env.example}"

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "ERROR: missing local paths file: ${PATHS_FILE}" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${PATHS_FILE}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${REFSEG_RUNTIME_SITE_PACKAGES:-}" ]]; then
  export PYTHONPATH="${REFSEG_RUNTIME_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
  runtime_lib_dir=$(cd "$(dirname "$(dirname "${REFSEG_RUNTIME_SITE_PACKAGES}")")" && pwd)
  if [[ -d "${runtime_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${runtime_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi
if [[ -n "${REFSEG_PROJECT_ROOT:-}" ]]; then
  export PYTHONPATH="${REFSEG_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
