#!/bin/bash
#$ -N nn_disorder_array
#$ -t 1-5
#$ -M gliu8@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q long@@theta_lab
#$ -cwd

# SGE array launcher for NN-disorder parameter sweeps.
#
# The sweep points are read from a TSV file (default:
# `mqed/disorder/nn_sweep_params.tsv`), one row per array task after the header.
# Each row must contain seven tab-separated fields:
#   label    sigma_J_eV    sigma_eps_eV    J_0_eV    eps_0_eV    k_parallel    sigma_sites
#
# Environment variables:
#   NN_CONFIG_NAME        Hydra config name (default: my_nn_chain)
#   NN_SWEEP_PARAM_FILE   Path to the TSV parameter file
#
# Typical HPC submission:
#   qsub -t 1-4 mqed/disorder/jobarray.sh
#   qsub -v NN_CONFIG_NAME=nn_chain,NN_SWEEP_PARAM_FILE=mqed/disorder/nn_sweep_params_figure_3.tsv -t 1-4 mqed/disorder/jobarray.sh
#
# Local single-task test without SGE:
#   bash mqed/disorder/jobarray.sh 1
#   bash mqed/disorder/jobarray.sh 1 mqed/disorder/nn_sweep_params_figure_3.tsv
#   NN_CONFIG_NAME=nn_chain bash mqed/disorder/jobarray.sh 1
#
# Keep the `-t` range aligned with the number of data rows in the TSV file.

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mqed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MQED_REPO_ROOT:-}" ]; then
  REPO_ROOT="${MQED_REPO_ROOT}"
elif [ -n "${SGE_O_WORKDIR:-}" ] && [ -f "${SGE_O_WORKDIR}/mqed/disorder/nn_sweep_params.tsv" ]; then
  REPO_ROOT="${SGE_O_WORKDIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

TASK_INDEX="${SGE_TASK_ID:-${1:-1}}"
MPI_NPROC="${NSLOTS:-24}"
CONFIG_NAME="${NN_CONFIG_NAME:-my_nn_chain}"

DEFAULT_PARAM_FILE="mqed/disorder/nn_sweep_params.tsv"
PARAM_FILE_INPUT="${NN_SWEEP_PARAM_FILE:-${MQED_SWEEP_PARAM_FILE:-${2:-${DEFAULT_PARAM_FILE}}}}"

if [[ "${PARAM_FILE_INPUT}" = /* ]]; then
  PARAM_FILE="${PARAM_FILE_INPUT}"
else
  PARAM_FILE="${REPO_ROOT}/${PARAM_FILE_INPUT}"
fi

if [ ! -f "${PARAM_FILE}" ]; then
  echo "Parameter file not found: ${PARAM_FILE}" >&2
  echo "Set NN_SWEEP_PARAM_FILE or pass the TSV path as the second argument." >&2
  exit 1
fi

PARAM_LINE="$(awk -v FS='[[:space:]]+' -v OFS='\t' -v idx="${TASK_INDEX}" '
  NR > 1 && NF {
    count++
    if (count == idx) {
      if (NF != 7) {
        print "__MALFORMED__", NF, $0
        exit
      }
      print $1, $2, $3, $4, $5, $6, $7
      exit
    }
  }
' "${PARAM_FILE}")"

if [ -z "${PARAM_LINE}" ]; then
  echo "No parameter row found for task index ${TASK_INDEX} in ${PARAM_FILE}" >&2
  exit 1
fi

if [[ "${PARAM_LINE}" == __MALFORMED__* ]]; then
  BAD_FIELD_COUNT="$(printf '%s' "${PARAM_LINE}" | cut -f2)"
  BAD_LINE="$(printf '%s' "${PARAM_LINE}" | cut -f3-)"
  echo "Malformed parameter row for task index ${TASK_INDEX}: expected 7 fields, found ${BAD_FIELD_COUNT}" >&2
  echo "Row content: ${BAD_LINE}" >&2
  exit 1
fi

IFS=$'\t' read -r LABEL SIGMA_J SIGMA_EPS J0 EPS0 KPARALLEL SIGMA_SITES <<< "${PARAM_LINE}"

if [ -z "${LABEL}" ] || [ -z "${SIGMA_J}" ] || [ -z "${SIGMA_EPS}" ] || [ -z "${J0}" ] || [ -z "${EPS0}" ] || [ -z "${KPARALLEL}" ] || [ -z "${SIGMA_SITES}" ]; then
  echo "Malformed parameter row for task index ${TASK_INDEX}: ${PARAM_LINE}" >&2
  exit 1
fi

echo "Running task ${TASK_INDEX} with label=${LABEL}"
echo "Parameter file: ${PARAM_FILE}"
echo "Config name: ${CONFIG_NAME}"
echo "sigma_J_eV=${SIGMA_J}, sigma_eps_eV=${SIGMA_EPS}, J_0_eV=${J0}, eps_0_eV=${EPS0}"
echo "k_parallel=${KPARALLEL}, sigma_sites=${SIGMA_SITES}"
echo "Using ${MPI_NPROC} MPI ranks"
echo "start time: $(date)"

mpirun -np "${MPI_NPROC}" mqed_nn_disorder \
  --config-name "${CONFIG_NAME}" \
  disorder.backend=mpi \
  disorder.mpi_auto_launch=false \
  disorder.mpi_nproc="${MPI_NPROC}" \
  disorder.sigma_J_eV="${SIGMA_J}" \
  disorder.sigma_eps_eV="${SIGMA_EPS}" \
  chain.J_0_eV="${J0}" \
  chain.eps_0_eV="${EPS0}" \
  initial_state.k_parallel="${KPARALLEL}" \
  initial_state.sigma_sites="${SIGMA_SITES}" \
  hydra.run.dir="outputs/nn_sweep/${LABEL}"

echo "end time: $(date)"
