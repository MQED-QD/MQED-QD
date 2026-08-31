#!/bin/bash
#$ -N gf_nlayer_single
#$ -M gliu8@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q long
#$ -cwd

###############################################################################
#  SGE Single-Job Launcher for N-Layer Dyadic Green's Function
###############################################################################
#
#  PURPOSE
#  -------
#  This script launches one SGE job for the N-layer dyadic Green's function
#  solver.  The job computes a configured photon-energy/Rx grid for one
#  multilayer stack. MPI normally distributes energy rows and can split Rx
#  chunks when too few energies are available to occupy all ranks.
#
#  This script launches `mqed_GF_NLayer` under `mpirun` and tells Hydra not to
#  auto-launch MPI a second time:
#
#    parallel.backend=mpi
#    parallel.mpi_auto_launch=false
#    parallel.mpi_nproc=${NSLOTS}
#
#  The layer stack, material models, source-layer index, integration settings,
#  energy grid, and output prefix should normally live in the Hydra config file.
#  Use command-line overrides below only for run-specific changes.
#
#  ENVIRONMENT VARIABLES (all optional)
#  ------------------------------------
#    GF_CONFIG_NAME       Hydra config name, without .yaml extension
#                         (default: GF_NLayer_five_layer)
#    MQED_REPO_ROOT       Repository root directory (default: auto-detected)
#    NSLOTS               Number of MPI ranks (set by SGE; local default: 4)
#
#  -----------------------------------------------------------------------------
#  USAGE EXAMPLES
#  -----------------------------------------------------------------------------
#
#  HPC (SGE) - submit with the default config:
#
#      qsub mqed/Dyadic_GF/gf_nlayer_single_job.sh
#
#    Submit with your own shared or local Hydra config:
#
#      qsub -v GF_CONFIG_NAME=GF_NLayer_my_stack \
#           mqed/Dyadic_GF/gf_nlayer_single_job.sh
#
#    Personal configs can live under `local/configs/Dyadic_GF/` if local Hydra
#    config support is enabled in this checkout.
#
#  Local desktop / laptop - run without SGE:
#
#      bash mqed/Dyadic_GF/gf_nlayer_single_job.sh
#
#      GF_CONFIG_NAME=GF_NLayer_my_stack \
#        NSLOTS=4 bash mqed/Dyadic_GF/gf_nlayer_single_job.sh
#
#  Quick smoke test, bypassing this script:
#
#      mqed_GF_NLayer \
#        --config-name GF_NLayer_five_layer \
#        parallel.backend=sequential \
#        simulation.energy_eV.points=1 \
#        simulation.position.Rx_nm.points=1 \
#        simulation.position.Rx_nm.stop=0
#
#  IMPORTANT
#  ---------
#  - Keep stack-specific quantities in the Hydra config when possible:
#    `stack.layers`, `stack.source_layer`, `materials`, and integration knobs.
#  - The source and observer positions are inside the finite source layer and
#    can be supplied as `simulation.position.zD_nm` and `simulation.position.zA_nm`.
#  - The N-layer runner supports `parallel.backend=sequential`, `joblib`, and
#    `mpi`. This launcher uses MPI because SGE may distribute slots across nodes.
#  - Output files are written under Hydra's run directory.  This script sets a
#    stable run directory under `outputs/gf_nlayer/single_job` for convenience.
#
###############################################################################

set -euo pipefail

# -- Conda environment ---------------------------------------------------------
# On HPC the conda module must be loaded first.  On a local machine the
# `module` command will not exist; the "|| true" lets the script continue.
module load conda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mqed

# -- Repository root detection -------------------------------------------------
# Priority: MQED_REPO_ROOT  >  SGE_O_WORKDIR  >  script/../..
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MQED_REPO_ROOT:-}" ]; then
  REPO_ROOT="${MQED_REPO_ROOT}"
elif [ -n "${SGE_O_WORKDIR:-}" ] && [ -d "${SGE_O_WORKDIR}/mqed/Dyadic_GF" ]; then
  REPO_ROOT="${SGE_O_WORKDIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

# NSLOTS is set by SGE's -pe directive.  On a laptop default to 4 ranks so MPI
# does not over-subscribe a small machine.
MPI_NPROC="${NSLOTS:-4}"

# Hydra config name (no .yaml extension).  Override via GF_CONFIG_NAME.
# Example: GF_CONFIG_NAME=GF_NLayer_my_stack qsub mqed/Dyadic_GF/gf_nlayer_single_job.sh
CONFIG_NAME="${GF_CONFIG_NAME:-GF_NLayer_five_layer}"

# -- Summary -------------------------------------------------------------------
echo "========================================================================"
echo "  N-Layer Dyadic Green's Function"
echo "========================================================================"
echo "  Config name   : ${CONFIG_NAME}"
echo "  Repository    : ${REPO_ROOT}"
echo "  MPI ranks     : ${MPI_NPROC}"
echo "  Start time    : $(date)"
echo "========================================================================"

# -- Launch --------------------------------------------------------------------
# We pass only run-control Hydra overrides on the command line.  The key points:
#   - parallel.backend=mpi              - distribute energy rows or scarce-energy Rx chunks
#   - parallel.mpi_auto_launch=false    - mpirun already launched the ranks
#   - parallel.mpi_nproc                - record the allocated MPI rank count
#   - hydra.run.dir                     - stable output directory for this job
#
# To make this a production run, prefer editing or selecting your Hydra config
# through GF_CONFIG_NAME instead of hardcoding stack-specific overrides here.

mpirun -np "${MPI_NPROC}" mqed_GF_NLayer \
  --config-name "${CONFIG_NAME}" \
  parallel.backend=mpi \
  parallel.mpi_auto_launch=false \
  parallel.mpi_nproc="${MPI_NPROC}" \
  hydra.run.dir="outputs/gf_nlayer/single_job"

echo "========================================================================"
echo "  Finished N-layer Green's function job"
echo "  End time: $(date)"
echo "========================================================================"
