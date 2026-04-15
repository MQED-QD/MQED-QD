#!/bin/bash
#$ -N gf_sommerfeld_single
#$ -M gliu8@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q long
#$ -cwd

###############################################################################
#  SGE single-job Launcher for Dyadic Green's Function (Sommerfeld Integration)
###############################################################################

set -euo pipefail

# ── Conda environment ────────────────────────────────────────────────────────
# On HPC the conda module must be loaded first.  On a local machine the
# `module` command will not exist — the "|| true" lets the script continue.
module load conda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mqed

# ── Repository root detection ────────────────────────────────────────────────
# Priority: MQED_REPO_ROOT  >  SGE_O_WORKDIR (if TSV found)  >  script/../..
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MQED_REPO_ROOT:-}" ]; then
  REPO_ROOT="${MQED_REPO_ROOT}"
elif [ -n "${SGE_O_WORKDIR:-}" ] && [ -f "${SGE_O_WORKDIR}/mqed/Dyadic_GF/gf_sommerfeld_params.tsv" ]; then
  REPO_ROOT="${SGE_O_WORKDIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

# NSLOTS is set by SGE's -pe directive.  On a laptop default to 4 cores so
# MPI does not over-subscribe a small machine.
MPI_NPROC="${NSLOTS:-8}"

# Hydra config name (no .yaml extension).  Override via GF_CONFIG_NAME.
CONFIG_NAME="${GF_CONFIG_NAME:-GF_Sommerfeld_customed}"

# ── Launch ───────────────────────────────────────────────────────────────────
# We pass Hydra config overrides on the command line.  The key points:
#   • parallel.backend=mpi          — distribute energy points across ranks
#   • parallel.mpi_auto_launch=false — we already launched via mpirun
#   • parallel.mpi_nproc            — tell the program how many ranks exist
#   • simulation.energy_eV.*        — override the energy sweep range
#   • simulation.position.zD_nm     — donor  height  (also sets zD in metres)
#   • simulation.position.zA_nm     — acceptor height (also sets zA in metres)
#   • simulation.material           — material key

mpirun -np "${MPI_NPROC}" mqed_GF_Sommerfeld \
  --config-name "${CONFIG_NAME}" \
  parallel.backend=mpi \
  parallel.mpi_auto_launch=false \
  parallel.mpi_nproc="${MPI_NPROC}" \
  simulation.energy_eV.min="3.30" \
  simulation.energy_eV.max="3.70" \
  simulation.energy_eV.points="100" \
  simulation.position.zD="1.0e-9" \
  simulation.position.zD_nm="1" \
  simulation.position.zA="1.0e-9" \
  simulation.material="Ag" \


