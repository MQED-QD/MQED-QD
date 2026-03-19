#!/bin/bash

##$ -M gliu8@nd.edu
##$ -m abe
#$ -pe smp 24
#$ -q long@@theta_lab
#$ -N nn_disorder
#$ -cwd

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mqed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MQED_REPO_ROOT:-}" ]; then
  REPO_ROOT="${MQED_REPO_ROOT}"
elif [ -n "${SGE_O_WORKDIR:-}" ] && [ -d "${SGE_O_WORKDIR}/mqed" ]; then
  REPO_ROOT="${SGE_O_WORKDIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

MPI_NPROC="${NSLOTS:-24}"

echo "Running NN disorder simulation with ${MPI_NPROC} MPI ranks"
echo "start time: $(date)"

mpirun -np "${MPI_NPROC}" mqed_nn_disorder \
  --config-name my_nn_chain \
  disorder.backend=mpi \
  disorder.mpi_auto_launch=false \
  disorder.mpi_nproc="${MPI_NPROC}"

echo "end time: $(date)"
