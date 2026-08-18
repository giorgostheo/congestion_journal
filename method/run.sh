#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"
DATASET="${DATASET:-pems}"

usage() {
  printf 'Usage: %s [--dataset DATASET] [--run-timestamp TIMESTAMP]\n' "$(basename "$0")"
  printf '\n'
  printf 'Runs the full experiment sequence:\n'
  printf '  1. Naive congestion baselines\n'
  printf '  2. Global model (upstream transitions + congestion + pre-filter)\n'
  printf '     Saves the artifacts required by the transfer model.\n'
  printf '  3. Local model (transition weights 1 and 3)\n'
  printf '  4. Transfer model (transition weights 1 and 3)\n'
  printf '\n'
  printf 'Dataset options (--dataset):\n'
  printf '  pems   PEMS-BAY.csv  (default)\n'
  printf '  metr   METR-LA.csv\n'
  printf '  all    Run both datasets in sequence\n'
  printf '\n'
  printf 'Environment overrides:\n'
  printf '  DATASET=metr %s\n' "$(basename "$0")"
  printf '  PYTHON_BIN=/path/to/python %s\n' "$(basename "$0")"
  printf '  CUDA_VISIBLE_DEVICES=-1 %s\n' "$(basename "$0")"
  printf '  RUN_TIMESTAMP=20260515_120000_000000 %s\n' "$(basename "$0")"
  printf '  XGB_SELECTION_METRIC=trsp XGB_TRANSITION_WEIGHTS=1,2,3,5,8 %s\n' "$(basename "$0")"
  printf '  TRSP_WINDOW=1 TRSP_TRANSITION_WEIGHT=0.5 %s\n' "$(basename "$0")"
  printf '\n'
  printf 'Feature ablation flags (all default on; set to 0 to disable):\n'
  printf '  XGB_FEAT_PROFILE     time-of-week congestion probability (now and at t+h)\n'
  printf '  XGB_FEAT_STATE       time-in-state / episode-duration counters\n'
  printf '  XGB_FEAT_MARGIN      speed margin to the congestion threshold\n'
  printf '  XGB_FEAT_WAVE        wave-aligned upstream spatial lags (global model only)\n'
  printf '  XGB_FEAT_PROF_RESID  speed residual vs time-of-week speed profile\n'
  printf '  XGB_FEAT_NAIVE       yesterday/last-week congestion labels as features\n'
  printf '  XGB_FEAT_EMA         exponential moving averages of speed\n'
  printf '  XGB_FEAT_HOLIDAY     public-holiday calendar flag\n'
  printf '  XGB_WAVE_STEPS_PER_HOP=1  sampling steps of lag per upstream hop\n'
  printf '  e.g. XGB_FEAT_NAIVE=0 XGB_FEAT_EMA=0 %s\n' "$(basename "$0")"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --dataset\n' >&2
        exit 2
      fi
      DATASET="$2"
      shift 2
      ;;
    --run-timestamp)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --run-timestamp\n' >&2
        exit 2
      fi
      RUN_TIMESTAMP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${DATASET}" in
  pems|metr|all) ;;
  *)
    printf 'Unknown dataset: %s. Valid options: pems metr all\n' "${DATASET}" >&2
    exit 2
    ;;
esac

if [[ -z "${RUN_TIMESTAMP}" ]]; then
  RUN_TIMESTAMP="$(${PYTHON_BIN} - <<'PY'
from datetime import datetime
print(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
PY
)"
fi

export RUN_TIMESTAMP="${RUN_TIMESTAMP}"
OUTPUT_DIR="${SCRIPT_DIR}/../results"
mkdir -p "${OUTPUT_DIR}"
export XGB_SUMMARY_RESULTS_PATH="${XGB_SUMMARY_RESULTS_PATH:-${OUTPUT_DIR}/results_${RUN_TIMESTAMP}.csv}"
export XGB_SUMMARY_TT_PATH="${XGB_SUMMARY_TT_PATH:-${OUTPUT_DIR}/training_times_${RUN_TIMESTAMP}.csv}"

printf 'Run timestamp : %s\n' "${RUN_TIMESTAMP}"
printf 'Dataset       : %s\n' "${DATASET}"
printf 'Summary CSV   : %s\n' "${XGB_SUMMARY_RESULTS_PATH}"
printf 'Timing CSV    : %s\n' "${XGB_SUMMARY_TT_PATH}"

run_dataset() {
  local ds="$1"
  export DATASET="${ds}"

  local ds_results_dir
  ds_results_dir="$(${PYTHON_BIN} - "${SCRIPT_DIR}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from shared import RESULTS_DIR
print(RESULTS_DIR)
PY
)"
  local ds_target_sensors
  ds_target_sensors="$(${PYTHON_BIN} - "${SCRIPT_DIR}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from shared import LOCATIONS
print(', '.join(item for sublist in LOCATIONS for item in sublist))
PY
)"

  printf '\n=== Dataset: %s | Results: %s ===\n' "${ds}" "${ds_results_dir}"
  printf 'Target sensors: %s\n' "${ds_target_sensors}"

  printf '[%s][1/4] Running naive congestion baselines...\n' "${ds}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/naive.py"

  printf '[%s][2/4] Running global model and saving transfer artifacts...\n' "${ds}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/global_model.py" --save-transfer-artifacts

  if [[ ! -f "${ds_results_dir}/global_artifacts.pkl" || ! -f "${ds_results_dir}/global_horizon_meta.pkl" ]]; then
    printf 'global_model.py completed but transfer artifacts are missing. Expected:\n' >&2
    printf '  %s\n' "${ds_results_dir}/global_artifacts.pkl" >&2
    printf '  %s\n' "${ds_results_dir}/global_horizon_meta.pkl" >&2
    exit 1
  fi

  printf '[%s][3/4] Running local model...\n' "${ds}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/local.py"

  printf '[%s][4/4] Running transfer model...\n' "${ds}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/transfer.py"

  printf '[%s] All runs completed. Outputs are in %s\n' "${ds}" "${ds_results_dir}"
}

if [[ "${DATASET}" == "all" ]]; then
  for ds in pems metr; do
    run_dataset "${ds}"
  done
else
  run_dataset "${DATASET}"
fi
