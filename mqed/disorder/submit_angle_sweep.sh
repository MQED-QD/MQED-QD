#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PARAM_FILE="${PARAM_FILE:-${REPO_ROOT}/mqed/disorder/angle_sweep_params.tsv}"
JOBARRAY_SCRIPT="${JOBARRAY_SCRIPT:-${REPO_ROOT}/mqed/disorder/angle_disorder_jobarray.sh}"
QSUB_BIN="${QSUB_BIN:-qsub}"

if [ ! -f "${PARAM_FILE}" ]; then
  echo "Parameter file not found: ${PARAM_FILE}" >&2
  exit 1
fi

if [ ! -f "${JOBARRAY_SCRIPT}" ]; then
  echo "Job-array script not found: ${JOBARRAY_SCRIPT}" >&2
  exit 1
fi

if ! command -v "${QSUB_BIN}" >/dev/null 2>&1; then
  echo "qsub executable not found: ${QSUB_BIN}" >&2
  exit 1
fi

TASK_COUNT="$(awk 'NR > 1 && NF {count++} END {print count + 0}' "${PARAM_FILE}")"

if [ "${TASK_COUNT}" -le 0 ]; then
  echo "No sweep rows found in ${PARAM_FILE}" >&2
  exit 1
fi

echo "Submitting ${TASK_COUNT} angle-disorder sweep tasks using ${PARAM_FILE}"
echo "Command: ${QSUB_BIN} -cwd -v MQED_REPO_ROOT=${REPO_ROOT} -t 1-${TASK_COUNT} $* ${JOBARRAY_SCRIPT}"

"${QSUB_BIN}" -cwd -v "MQED_REPO_ROOT=${REPO_ROOT}" -t "1-${TASK_COUNT}" "$@" "${JOBARRAY_SCRIPT}"
