#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${ROOT_DIR}/method/run.sh" --help
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 1
}

printf 'Checking Python dependencies...\n'
"${PYTHON_BIN}" - <<'PY'
import numpy
import pandas
import sklearn
import tqdm
import xgboost

print(f'  numpy={numpy.__version__}')
print(f'  pandas={pandas.__version__}')
print(f'  scikit-learn={sklearn.__version__}')
print(f'  tqdm={tqdm.__version__}')
print(f'  xgboost={xgboost.__version__}')
PY

printf 'Checking dataset files...\n'
(
  cd "${ROOT_DIR}"
  sha256sum --check checksums.sha256
)

export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ $# -eq 0 ]]; then
  set -- --dataset all
fi

printf 'Starting reproduction with CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
exec "${ROOT_DIR}/method/run.sh" "$@"
