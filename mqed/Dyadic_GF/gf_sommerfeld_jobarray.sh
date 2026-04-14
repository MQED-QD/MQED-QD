#!/bin/bash
#$ -N gf_sommerfeld_array
#$ -t 1-4
#$ -M gliu8@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q long
#$ -cwd

###############################################################################
#  SGE Job-Array Launcher for Dyadic Green's Function (Sommerfeld Integration)
###############################################################################
#
#  PURPOSE
#  -------
#  This script launches one SGE array task per row in a TSV parameter file.
#  Each task computes the dyadic Green's function over a range of photon
#  energies for a specific geometry (donor/acceptor height, material, etc.)
#  using MPI parallelism across the energy axis.
#
#  The sweep parameters are read from a TSV file (default:
#  `mqed/Dyadic_GF/gf_sommerfeld_params.tsv`), one data row per array task
#  after the header.  Each row must contain seven tab-separated fields:
#
#    label  energy_min_eV  energy_max_eV  energy_points  zD_nm  zA_nm  material
#
#  Columns
#  -------
#    label          : Human-readable tag; used as the output sub-directory name.
#    energy_min_eV  : Lower bound of the energy sweep  (eV).
#    energy_max_eV  : Upper bound of the energy sweep  (eV).
#    energy_points  : Number of energy sample points in [min, max].
#    zD_nm          : Donor   height above the surface  (nm).
#    zA_nm          : Acceptor height above the surface (nm).
#    material       : Material key understood by MQED   (e.g. Ag, Au).
#
#  ENVIRONMENT VARIABLES (all optional)
#  ------------------------------------
#    GF_CONFIG_NAME          Hydra config name          (default: GF_Sommerfeld)
#    GF_SWEEP_PARAM_FILE     Path to the TSV file       (default: auto-detected)
#    MQED_REPO_ROOT          Repository root directory   (default: auto-detected)
#
#  ─────────────────────────────────────────────────────────────────────────────
#  USAGE EXAMPLES
#  ─────────────────────────────────────────────────────────────────────────────
#
#  ▸ HPC (SGE) — submit the full array (4 tasks, matching 4 data rows):
#
#      qsub -t 1-4 mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh
#
#    Override the parameter file or config:
#
#      qsub -v GF_CONFIG_NAME=GF_Sommerfeld,GF_SWEEP_PARAM_FILE=my_params.tsv \
#           -t 1-2 mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh
#
#  ▸ Local desktop / laptop — run a single task without SGE:
#
#      # Task 1, default TSV:
#      bash mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh 1
#
#      # Task 2, custom TSV:
#      bash mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh 2 my_params.tsv
#
#      # Task 1, custom Hydra config:
#      GF_CONFIG_NAME=GF_Sommerfeld_custom bash mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh 1
#
#    TIP:  On a laptop with limited cores, override MPI_NPROC:
#
#      NSLOTS=4 bash mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh 1
#
#  ▸ Quick smoke test (no real MPI, sequential backend on a laptop):
#
#      You can bypass MPI entirely for debugging by running the command
#      directly with the sequential backend:
#
#        mqed_GF_Sommerfeld \
#          simulation.energy_eV.min=2.0 simulation.energy_eV.max=3.0 \
#          simulation.energy_eV.points=5 \
#          simulation.position.zD_nm=10 \
#          parallel.backend=sequential
#
#  IMPORTANT
#  ---------
#  • Keep the SGE `-t 1-N` range aligned with the number of data rows in
#    the TSV file (excluding the header).
#  • When launched via `mpirun` (this script), set
#    `parallel.mpi_auto_launch=false` so the program does not try to
#    re-launch itself.
#  • Output directories are created under  outputs/gf_sommerfeld/<label>/
#    relative to hydra.run.dir.
#
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

# ── Task index & resource detection ─────────────────────────────────────────
# SGE_TASK_ID is set automatically by SGE.  For local runs pass it as $1.
TASK_INDEX="${SGE_TASK_ID:-${1:-1}}"

# NSLOTS is set by SGE's -pe directive.  On a laptop default to 4 cores so
# MPI does not over-subscribe a small machine.
MPI_NPROC="${NSLOTS:-4}"

# Hydra config name (no .yaml extension).  Override via GF_CONFIG_NAME.
CONFIG_NAME="${GF_CONFIG_NAME:-GF_Sommerfeld}"

# ── Locate the parameter file ───────────────────────────────────────────────
DEFAULT_PARAM_FILE="mqed/Dyadic_GF/gf_sommerfeld_params.tsv"
PARAM_FILE_INPUT="${GF_SWEEP_PARAM_FILE:-${MQED_SWEEP_PARAM_FILE:-${2:-${DEFAULT_PARAM_FILE}}}}"

# Accept both absolute and relative paths.
if [[ "${PARAM_FILE_INPUT}" = /* ]]; then
  PARAM_FILE="${PARAM_FILE_INPUT}"
else
  PARAM_FILE="${REPO_ROOT}/${PARAM_FILE_INPUT}"
fi

if [ ! -f "${PARAM_FILE}" ]; then
  echo "ERROR: Parameter file not found: ${PARAM_FILE}" >&2
  echo "Set GF_SWEEP_PARAM_FILE or pass the TSV path as the second argument." >&2
  exit 1
fi

# ── Parse the TSV row for this task ──────────────────────────────────────────
# Skip the header (NR > 1), count non-empty rows, select the row matching
# TASK_INDEX.  Expect exactly 7 tab-separated fields per row.
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
  echo "ERROR: No parameter row found for task index ${TASK_INDEX} in ${PARAM_FILE}" >&2
  exit 1
fi

if [[ "${PARAM_LINE}" == __MALFORMED__* ]]; then
  BAD_FIELD_COUNT="$(printf '%s' "${PARAM_LINE}" | cut -f2)"
  BAD_LINE="$(printf '%s' "${PARAM_LINE}" | cut -f3-)"
  echo "ERROR: Malformed parameter row for task index ${TASK_INDEX}: expected 7 fields, found ${BAD_FIELD_COUNT}" >&2
  echo "Row content: ${BAD_LINE}" >&2
  exit 1
fi

# Unpack the seven fields.
IFS=$'\t' read -r LABEL ENERGY_MIN ENERGY_MAX ENERGY_PTS ZD_NM ZA_NM MATERIAL <<< "${PARAM_LINE}"

# Validate that none of the fields are empty.
for var in LABEL ENERGY_MIN ENERGY_MAX ENERGY_PTS ZD_NM ZA_NM MATERIAL; do
  if [ -z "${!var}" ]; then
    echo "ERROR: Field '${var}' is empty for task index ${TASK_INDEX}." >&2
    echo "Row: ${PARAM_LINE}" >&2
    exit 1
  fi
done

# ── Convert nm → metres for the physics keys ─────────────────────────────────
# The config has two separate keys:
#   simulation.position.zD_nm  → used only in the output filename template
#   simulation.position.zD     → the actual height in metres used by the solver
# We derive the metre values from the TSV's nm values so the user only has to
# think in nanometres.
ZD_M="$(awk "BEGIN {printf \"%.10e\", ${ZD_NM} * 1.0e-9}")"
ZA_M="$(awk "BEGIN {printf \"%.10e\", ${ZA_NM} * 1.0e-9}")"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "  Dyadic Green's Function — Sommerfeld Integration"
echo "  Task ${TASK_INDEX}  |  label = ${LABEL}"
echo "========================================================================"
echo "  Config name    : ${CONFIG_NAME}"
echo "  Parameter file : ${PARAM_FILE}"
echo "  Energy range   : ${ENERGY_MIN} – ${ENERGY_MAX} eV  (${ENERGY_PTS} points)"
echo "  Donor height   : ${ZD_NM} nm  (${ZD_M} m)"
echo "  Acceptor height: ${ZA_NM} nm  (${ZA_M} m)"
echo "  Material       : ${MATERIAL}"
echo "  MPI ranks      : ${MPI_NPROC}"
echo "  Start time     : $(date)"
echo "========================================================================"

# ── Launch ───────────────────────────────────────────────────────────────────
# We pass Hydra config overrides on the command line.  The key points:
#   • parallel.backend=mpi          — distribute energy points across ranks
#   • parallel.mpi_auto_launch=false — we already launched via mpirun
#   • parallel.mpi_nproc            — tell the program how many ranks exist
#   • simulation.energy_eV.*        — override the energy sweep range
#   • simulation.position.zD_nm     — donor  height  (also sets zD in metres)
#   • simulation.position.zA_nm     — acceptor height (also sets zA in metres)
#   • simulation.material           — material key
#   • hydra.run.dir                 — per-task output directory

mpirun -np "${MPI_NPROC}" mqed_GF_Sommerfeld \
  --config-name "${CONFIG_NAME}" \
  parallel.backend=mpi \
  parallel.mpi_auto_launch=false \
  parallel.mpi_nproc="${MPI_NPROC}" \
  simulation.energy_eV.min="${ENERGY_MIN}" \
  simulation.energy_eV.max="${ENERGY_MAX}" \
  simulation.energy_eV.points="${ENERGY_PTS}" \
  simulation.position.zD="${ZD_M}" \
  simulation.position.zD_nm="${ZD_NM}" \
  simulation.position.zA="${ZA_M}" \
  simulation.material="${MATERIAL}" \
  hydra.run.dir="outputs/gf_sommerfeld/${LABEL}"

echo "========================================================================"
echo "  Finished task ${TASK_INDEX}  (${LABEL})"
echo "  End time: $(date)"
echo "========================================================================"
