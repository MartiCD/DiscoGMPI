#!/bin/bash
#SBATCH --account=bsc21
#SBATCH --qos=gpp_data
#SBATCH --job-name=nasa_almond_rcs
#SBATCH --output=nasa_almond_rcs_%j.out
#SBATCH --error=nasa_almond_rcs_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

set -euo pipefail

# Run this script from the DiscoGMPI repository root with:
#   sbatch examples/run_nasa_almond_rcs.slurm
#
# The defaults target the Austin CEM Problem IIIA smallest PEC almond at fx1
# (3.5 GHz) using the STEP-derived EPW4 mesh generated from
# NasaAlmond_scale_120cm.step, with the RCS sampled on the Austin-style
# theta=90 deg, 0<=phi<=180 deg cut. Override parameters at submission time,
# for example:
#
#   sbatch --export=ALL,ORDER=4,FINAL_TIME=1.03,RCS_START_TIME=0.43 \
#     examples/run_nasa_almond_rcs.slurm
#
#   sbatch --export=ALL,MESH=examples/meshes/custom_almond.vtk \
#     examples/run_nasa_almond_rcs.slurm

module purge
module load intel/2023.0
module load impi/2021.10.0
module load julia/1.10.0

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SLURM_NTASKS="${SLURM_NTASKS:-8}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

MPIEXEC="${MPIEXEC:-mpiexec}"
PYTHON="${PYTHON:-python3}"

DEFAULT_STEP_MESH="examples/meshes/pec_almond_scattering_step_fx1_epw4.vtk"
STEP_MESH_GENERATOR="examples/meshes/generate_pec_almond_scattering_step_mesh.sh"
MESH="${MESH:-${DEFAULT_STEP_MESH}}"
FREQUENCY_HZ="${FREQUENCY_HZ:-3.5e9}"
FREQUENCY_KEY="${FREQUENCY_KEY:-fx1}"
REFERENCE_KIND="${REFERENCE_KIND:-ref}"
ORDER="${ORDER:-2}"
ESPRK_ORDER="${ESPRK_ORDER:-$((ORDER + 1))}"
CFL="${CFL:-0.15}"
FLUX="${FLUX:-centered}"
SIGMA_MAX="${SIGMA_MAX:-12.0}"
SIGMA_DEGREE="${SIGMA_DEGREE:-2}"
RCS_EVERY="${RCS_EVERY:-5}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-250}"
PARAVIEW_EVERY="${PARAVIEW_EVERY:-0}"

PERIOD="$(
  FREQUENCY_HZ="${FREQUENCY_HZ}" "${PYTHON}" -c \
    'import os; print(299792458.0 / float(os.environ["FREQUENCY_HZ"]))'
)"
PML_WIDTH="${PML_WIDTH:-${PERIOD}}"
FINAL_TIME="$(
  PERIOD="${PERIOD}" FINAL_TIME="${FINAL_TIME:-}" "${PYTHON}" -c \
    'import os; print(os.environ["FINAL_TIME"] or 6.0 * float(os.environ["PERIOD"]))'
)"
RCS_START_TIME="$(
  PERIOD="${PERIOD}" RCS_START_TIME="${RCS_START_TIME:-}" "${PYTHON}" -c \
    'import os; print(os.environ["RCS_START_TIME"] or 3.0 * float(os.environ["PERIOD"]))'
)"
PHI_DEGREES="${PHI_DEGREES:-$("${PYTHON}" -c 'print(",".join(f"{0.5*i:.1f}" for i in range(361)))')}"

RUN_LABEL="${RUN_LABEL:-pec_almond_step_rcs_${FREQUENCY_KEY}_p${ORDER}_theta90_phi0_180}"
OUTPUT_DIR="${OUTPUT_DIR:-output/validation_sequence/${RUN_LABEL}}"

if [[ ! -f "${MESH}" ]]; then
  cat >&2 <<EOF
Almond mesh not found:
  ${MESH}

The default mesh is the STEP-derived Austin IIIA almond mesh:
  ${DEFAULT_STEP_MESH}

Generate it from the STEP CAD file with:
  bash ${STEP_MESH_GENERATOR}

Or submit this SLURM job with an explicit VTK mesh:
  sbatch --export=ALL,MESH=/path/to/almond_mesh.vtk examples/run_nasa_almond_rcs.slurm
EOF
  exit 1
fi

echo "SLURM_JOB_ID           = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST         = ${SLURM_NODELIST:-local}"
echo "SLURM_NTASKS           = ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK    = ${SLURM_CPUS_PER_TASK}"
echo "JULIA_DEPOT_PATH       = ${JULIA_DEPOT_PATH}"
echo "JULIA_NUM_THREADS      = ${JULIA_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS   = ${OPENBLAS_NUM_THREADS}"
echo "OMP_NUM_THREADS        = ${OMP_NUM_THREADS}"
echo "MPIEXEC                = ${MPIEXEC}"
echo "Julia:"
julia --version
echo "Python:"
"${PYTHON}" --version
echo "MPI launcher:"
which "${MPIEXEC}"
"${MPIEXEC}" --version || true
echo "Run parameters:"
echo "  mesh                 = ${MESH}"
echo "  default_step_mesh    = ${DEFAULT_STEP_MESH}"
echo "  step_mesh_generator  = ${STEP_MESH_GENERATOR}"
echo "  output_dir           = ${OUTPUT_DIR}"
echo "  frequency_hz         = ${FREQUENCY_HZ}"
echo "  frequency_key        = ${FREQUENCY_KEY}"
echo "  wavelength/period    = ${PERIOD}"
echo "  final_time           = ${FINAL_TIME}"
echo "  rcs_start_time       = ${RCS_START_TIME}"
echo "  order/esprk_order    = ${ORDER}/${ESPRK_ORDER}"
echo "  ntasks               = ${SLURM_NTASKS}"

"${MPIEXEC}" -n "${SLURM_NTASKS}" -genv OPENBLAS_NUM_THREADS 1 \
  julia --project=. \
  examples/distributed_pec_almond_scattering.jl \
  --mesh="${MESH}" \
  --frequency-hz="${FREQUENCY_HZ}" \
  --order="${ORDER}" \
  --esprk-order="${ESPRK_ORDER}" \
  --final-time="${FINAL_TIME}" \
  --cfl="${CFL}" \
  --flux="${FLUX}" \
  --pml-width="${PML_WIDTH}" \
  --sigma-max="${SIGMA_MAX}" \
  --sigma-degree="${SIGMA_DEGREE}" \
  --enable-rcs \
  --rcs-start-time="${RCS_START_TIME}" \
  --rcs-every="${RCS_EVERY}" \
  --rcs-theta-count=1 \
  --rcs-theta-min-degrees=90 \
  --rcs-theta-max-degrees=90 \
  --rcs-phi-degrees="${PHI_DEGREES}" \
  --diagnostics-every="${DIAGNOSTICS_EVERY}" \
  --paraview-every="${PARAVIEW_EVERY}" \
  --output-dir="${OUTPUT_DIR}"

"${PYTHON}" examples/postprocess_pec_almond_rcs.py \
  --run-dir="${OUTPUT_DIR}" \
  --frequency-hz="${FREQUENCY_HZ}" \
  --frequency-key="${FREQUENCY_KEY}" \
  --reference-kind="${REFERENCE_KIND}" \
  --polarization=both \
  --dg-angle=phi \
  --theta-filter=90 \
  --write-png \
  --title="NASA PEC almond RCS: ${FREQUENCY_KEY}, p=${ORDER}"

echo "NASA almond RCS workflow complete."
echo "Run directory: ${OUTPUT_DIR}"
echo "RCS CSV:       ${OUTPUT_DIR}/diagnostics/rcs.csv"
echo "Plots:         ${OUTPUT_DIR}/plots"
