#!/bin/bash
#$ -N gf_mie_single
#$ -M gliu8@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q long
#$ -cwd

###############################################################################
#  SGE Single-Job Launcher for Mie Dyadic Green's Function
###############################################################################
#
#  PURPOSE
#  -------
#  This script launches one SGE job for the generalized Mie dyadic Green's
#  function solver.  It is intended for many-frequency core-shell/sphere runs,
#  for example reproducing spectral-density curves in a spherical cavity.
#
#  This script launches `mqed_GF_Mie` under `mpirun` and tells Hydra not to
#  auto-launch MPI a second time:
#
#    parallel.backend=mpi
#    parallel.mpi_auto_launch=false
#    parallel.mpi_nproc=${NSLOTS}
#
#  Geometry, materials, source/observer positions, spectral segments, nmax, and
#  output prefix should normally live in the Hydra YAML config.  Use command-line
#  overrides below only for run-control settings such as MPI and Hydra run dir.
#
#  ENVIRONMENT VARIABLES (all optional)
#  ------------------------------------
#    GF_CONFIG_NAME       Hydra config name, without .yaml extension
#                         (default: GF_Mie)
#    MQED_REPO_ROOT       Repository root directory (default: auto-detected)
#    NSLOTS               Number of MPI ranks (set by SGE; local default: 4)
#    DRY_RUN              If set to 1, print the command and exit without running
#
#  -----------------------------------------------------------------------------
#  USAGE EXAMPLES
#  -----------------------------------------------------------------------------
#
#  HPC (SGE) - submit with the default Mie config:
#
#      qsub mqed/Dyadic_GF/gf_mie_single_job.sh
#
#    Submit with your own shared or local Hydra config:
#
#      qsub -v GF_CONFIG_NAME=GF_Mie_core_shell_literature \
#           mqed/Dyadic_GF/gf_mie_single_job.sh
#
#    Personal configs can live under `local/configs/Dyadic_GF/` if local Hydra
#    config support is enabled in this checkout.
#
#  Local desktop / laptop - run without SGE:
#
#      bash mqed/Dyadic_GF/gf_mie_single_job.sh
#
#      GF_CONFIG_NAME=GF_Mie NSLOTS=4 \
#        bash mqed/Dyadic_GF/gf_mie_single_job.sh
#
#  Dry-run the launcher command without starting the Mie calculation:
#
#      DRY_RUN=1 GF_CONFIG_NAME=GF_Mie \
#        bash mqed/Dyadic_GF/gf_mie_single_job.sh
#
#  Quick smoke test, bypassing this script:
#
#      mqed_GF_Mie \
#        --config-name GF_Mie \
#        parallel.backend=sequential \
#        simulation.energy_eV='[1.8]' \
#        simulation.position.Rx_nm='[0.0,2.0,20.0]'
#
#  IMPORTANT FOR THE CURRENT CORE-SHELL CAVITY CONFIG
#  --------------------------------------------------
#  - `simulation.source_position_nm` is the fixed source/donor position.
#  - `simulation.position.Rx_nm` gives observer/acceptor x-offsets from that
#    source in scan layout.  With source [0,0,0] and Rx [0,2,20] nm, the driver
#    computes G([0,0,0],[0,0,0]), G([2,0,0],[0,0,0]), and
#    G([20,0,0],[0,0,0]) for each energy.
#  - The current core-shell example has radii [160, 60] nm and supports the
#    core/cavity scan path for observer radii below 60 nm.  Keep scan observer
#    points inside that core/cavity unless you intentionally want shell-region
#    zero-structure placeholders and warnings.
#  - `output.layout: scan` writes a fixed-source observer scan.  For many-emitter
#    quantum dynamics in a non-translational spherical geometry, use a separate
#    pair-layout config with `simulation.emitter_positions_nm`.
#  - Output files are written under Hydra's run directory.  This script sets a
#    stable run directory under `outputs/gf_mie/single_job` for convenience.
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
# Example: GF_CONFIG_NAME=GF_Mie_core_shell_literature qsub mqed/Dyadic_GF/gf_mie_single_job.sh
CONFIG_NAME="${GF_CONFIG_NAME:-GF_Mie}"

CMD=(
  mpirun -np "${MPI_NPROC}" mqed_GF_Mie
  --config-name "${CONFIG_NAME}"
  parallel.backend=mpi
  parallel.mpi_auto_launch=false
  parallel.mpi_nproc="${MPI_NPROC}"
  hydra.run.dir="outputs/gf_mie/single_job"
)

# -- Summary -------------------------------------------------------------------
echo "========================================================================"
echo "  Mie Dyadic Green's Function"
echo "========================================================================"
echo "  Config name   : ${CONFIG_NAME}"
echo "  Repository    : ${REPO_ROOT}"
echo "  MPI ranks     : ${MPI_NPROC}"
echo "  Start time    : $(date)"
echo "========================================================================"

# -- Launch --------------------------------------------------------------------
# We pass only run-control Hydra overrides on the command line.  The key points:
#   - parallel.backend=mpi              - distribute energy points across ranks
#   - parallel.mpi_auto_launch=false    - mpirun already launched the ranks
#   - parallel.mpi_nproc                - record the allocated MPI rank count
#   - hydra.run.dir                     - stable output/log directory for this job
#
# To make this a production run, prefer editing or selecting your Hydra config
# through GF_CONFIG_NAME instead of hardcoding geometry/material overrides here.

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'Dry run command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"

echo "========================================================================"
echo "  Finished Mie Green's function job"
echo "  End time: $(date)"
echo "========================================================================"
