#!/bin/bash
#SBATCH --account=bsc21
#SBATCH --qos=gpp_data
#SBATCH --job-name=sphere_bistatic_rcs
#SBATCH --output=pec_sphere_bistatic_rcs_%j.out
#SBATCH --error=pec_sphere_bistatic_rcs_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=224
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1

set -euo pipefail

# Larger bistatic PEC-sphere RCS validation against the analytical Mie series.
#
# Intended scale:
#   mesh:        approximately 50,000 tetrahedra
#   MPI layout:  2 nodes, 224 ranks
#   load:        about 200--250 tetrahedra per rank
#
# Generate the default mesh before submitting:
#   bash examples/meshes/generate_metallic_sphere_scattering_50k_mesh.sh
#
# Submit:
#   sbatch examples/run_pec_sphere_bistatic_rcs_50k.slurm
#
# Useful overrides:
#   sbatch --export=ALL,MESH=examples/meshes/custom_sphere.vtk \\
#     examples/run_pec_sphere_bistatic_rcs_50k.slurm
#
#   sbatch --export=ALL,THETA_COUNT=181,PHI_DEGREES=0,90,FINAL_TIME=12 \\
#     examples/run_pec_sphere_bistatic_rcs_50k.slurm

module purge
module load intel/2023.0
module load impi/2021.10.0
module load julia/1.10.0

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SLURM_NTASKS="${SLURM_NTASKS:-224}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.matplotlib}"

MPIEXEC="${MPIEXEC:-mpiexec}"
PYTHON="${PYTHON:-python3}"

MESH="${MESH:-examples/meshes/metallic_sphere_scattering_epw5p5_50k.vtk}"
MESH_GENERATOR="${MESH_GENERATOR:-examples/meshes/generate_metallic_sphere_scattering_50k_mesh.sh}"

RADIUS="${RADIUS:-1.0}"
WAVELENGTH="${WAVELENGTH:-1.0}"
FREQUENCY="${FREQUENCY:-1.0}"
AMPLITUDE="${AMPLITUDE:-1.0}"
ORDER="${ORDER:-4}"
ESPRK_ORDER="${ESPRK_ORDER:-$((ORDER + 1))}"
CFL="${CFL:-0.8}"
FLUX="${FLUX:-centered}"
PML_WIDTH="${PML_WIDTH:-1.0}"
SIGMA_MAX="${SIGMA_MAX:-12.0}"
SIGMA_DEGREE="${SIGMA_DEGREE:-2}"
PML_A="${PML_A:-0.5}"

FINAL_TIME="${FINAL_TIME:-12.0}"
RCS_START_TIME="${RCS_START_TIME:-5.0}"
RCS_EVERY="${RCS_EVERY:-5}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-250}"
PARAVIEW_EVERY="${PARAVIEW_EVERY:-0}"

# Bistatic observation grid.  With the current incident-wave convention
# (propagation +z, electric polarization +x), phi=0 and phi=90 give the two
# principal scattering planes for the PEC sphere.  Add more comma-separated phi
# cuts for a denser angular map.
THETA_COUNT="${THETA_COUNT:-361}"
THETA_MIN_DEGREES="${THETA_MIN_DEGREES:-0}"
THETA_MAX_DEGREES="${THETA_MAX_DEGREES:-180}"
PHI_DEGREES="${PHI_DEGREES:-0,90}"

MIE_TERMS="${MIE_TERMS:-0}"
MIN_MAX_RCS="${MIN_MAX_RCS:-0}"
MAX_ABSOLUTE_NORMALIZED_ERROR="${MAX_ABSOLUTE_NORMALIZED_ERROR:-Inf}"
MAX_RMS_RELATIVE_ERROR="${MAX_RMS_RELATIVE_ERROR:-Inf}"
MAX_DB_ERROR="${MAX_DB_ERROR:-Inf}"
POSTPROCESS="${POSTPROCESS:-1}"
WRITE_PNG="${WRITE_PNG:-1}"

RUN_LABEL="${RUN_LABEL:-pec_sphere_bistatic_rcs_50k_p${ORDER}_theta${THETA_COUNT}_phi0_90}"
OUTPUT_DIR="${OUTPUT_DIR:-output/validation_sequence/${RUN_LABEL}}"

if [[ ! -f "${MESH}" ]]; then
  cat >&2 <<EOF
Sphere mesh not found:
  ${MESH}

Generate the default approximately 50k-tet mesh with:
  bash ${MESH_GENERATOR}

Or submit this job with an explicit VTK mesh:
  sbatch --export=ALL,MESH=/path/to/sphere_mesh.vtk \\
    examples/run_pec_sphere_bistatic_rcs_50k.slurm
EOF
  exit 1
fi

echo "SLURM_JOB_ID                 = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST               = ${SLURM_NODELIST:-local}"
echo "SLURM_NNODES                 = ${SLURM_NNODES:-unknown}"
echo "SLURM_NTASKS                 = ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE        = ${SLURM_NTASKS_PER_NODE:-unknown}"
echo "SLURM_CPUS_PER_TASK          = ${SLURM_CPUS_PER_TASK}"
echo "JULIA_DEPOT_PATH             = ${JULIA_DEPOT_PATH}"
echo "JULIA_NUM_THREADS            = ${JULIA_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS         = ${OPENBLAS_NUM_THREADS}"
echo "OMP_NUM_THREADS              = ${OMP_NUM_THREADS}"
echo "MPIEXEC                      = ${MPIEXEC}"
echo "Julia:"
julia --version
echo "Python:"
"${PYTHON}" --version
echo "MPI launcher:"
which "${MPIEXEC}"
"${MPIEXEC}" --version || true
echo "Run parameters:"
echo "  mesh                       = ${MESH}"
echo "  output_dir                 = ${OUTPUT_DIR}"
echo "  radius                     = ${RADIUS}"
echo "  wavelength/frequency       = ${WAVELENGTH}/${FREQUENCY}"
echo "  amplitude                  = ${AMPLITUDE}"
echo "  order/esprk_order          = ${ORDER}/${ESPRK_ORDER}"
echo "  cfl                        = ${CFL}"
echo "  flux                       = ${FLUX}"
echo "  pml_width                  = ${PML_WIDTH}"
echo "  sigma_max/degree           = ${SIGMA_MAX}/${SIGMA_DEGREE}"
echo "  pml_a                      = ${PML_A}"
echo "  final_time                 = ${FINAL_TIME}"
echo "  rcs_start_time             = ${RCS_START_TIME}"
echo "  rcs_every                  = ${RCS_EVERY}"
echo "  diagnostics_every          = ${DIAGNOSTICS_EVERY}"
echo "  paraview_every             = ${PARAVIEW_EVERY}"
echo "  theta_count                = ${THETA_COUNT}"
echo "  theta_min/max              = ${THETA_MIN_DEGREES}/${THETA_MAX_DEGREES}"
echo "  phi_degrees                = ${PHI_DEGREES}"
echo "  expected tets/rank         = about 50000/${SLURM_NTASKS}"

"${MPIEXEC}" -n "${SLURM_NTASKS}" -genv OPENBLAS_NUM_THREADS 1 \
  julia --project=. \
  examples/validate_coarse_sphere_rcs.jl \
  --mesh="${MESH}" \
  --radius="${RADIUS}" \
  --wavelength="${WAVELENGTH}" \
  --frequency="${FREQUENCY}" \
  --amplitude="${AMPLITUDE}" \
  --order="${ORDER}" \
  --esprk-order="${ESPRK_ORDER}" \
  --final-time="${FINAL_TIME}" \
  --cfl="${CFL}" \
  --flux="${FLUX}" \
  --pml-width="${PML_WIDTH}" \
  --sigma-max="${SIGMA_MAX}" \
  --sigma-degree="${SIGMA_DEGREE}" \
  --pml-a="${PML_A}" \
  --enable-rcs \
  --rcs-start-time="${RCS_START_TIME}" \
  --rcs-every="${RCS_EVERY}" \
  --rcs-theta-count="${THETA_COUNT}" \
  --rcs-theta-min-degrees="${THETA_MIN_DEGREES}" \
  --rcs-theta-max-degrees="${THETA_MAX_DEGREES}" \
  --rcs-phi-degrees="${PHI_DEGREES}" \
  --diagnostics-every="${DIAGNOSTICS_EVERY}" \
  --paraview-every="${PARAVIEW_EVERY}" \
  --min-max-rcs="${MIN_MAX_RCS}" \
  --max-absolute-normalized-error="${MAX_ABSOLUTE_NORMALIZED_ERROR}" \
  --max-rms-relative-error="${MAX_RMS_RELATIVE_ERROR}" \
  --max-db-error="${MAX_DB_ERROR}" \
  --mie-terms="${MIE_TERMS}" \
  --output-dir="${OUTPUT_DIR}"

if [[ "${POSTPROCESS}" == "1" ||
      "${POSTPROCESS}" == "true" ||
      "${POSTPROCESS}" == "TRUE" ]]; then
  POSTPROCESS_ARGS=(
    --run-dir="${OUTPUT_DIR}"
    --radius="${RADIUS}"
    --wavelength="${WAVELENGTH}"
    --terms="${MIE_TERMS}"
    --title="PEC sphere bistatic RCS: 50k mesh, p=${ORDER}"
  )
  if [[ "${WRITE_PNG}" == "1" ||
        "${WRITE_PNG}" == "true" ||
        "${WRITE_PNG}" == "TRUE" ]]; then
    POSTPROCESS_ARGS+=(--write-png)
  fi
  "${PYTHON}" examples/postprocess_metallic_sphere_mie_rcs.py \
    "${POSTPROCESS_ARGS[@]}"
fi

echo "PEC sphere bistatic RCS workflow complete."
echo "Run directory:       ${OUTPUT_DIR}"
echo "RCS CSV:             ${OUTPUT_DIR}/diagnostics/rcs.csv"
echo "Mie comparison CSV:  ${OUTPUT_DIR}/diagnostics/mie_rcs_comparison.csv"
echo "Plots:               ${OUTPUT_DIR}/plots"
