#!/bin/bash
#SBATCH --account=bsc21
#SBATCH --qos=gpp_data
#SBATCH --job-name=almond_austin_mono
#SBATCH --output=nasa_almond_austin_monostatic_rcs_%j.out
#SBATCH --error=nasa_almond_austin_monostatic_rcs_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

set -euo pipefail

# Austin-specific monostatic PEC almond RCS sweep.
#
# Run from the DiscoGMPI repository root with:
#   sbatch examples/run_nasa_almond_austin_monostatic_rcs.slurm
#
# The default sweep is intentionally small:
#   PHI_DEGREES=0:30:180, POLARIZATION=both
#
# The full Austin IIIA azimuth sweep is much more expensive because this driver
# launches one complete time-domain scattering run per (phi_i, polarization):
#   sbatch --export=ALL,PHI_DEGREES=0:0.5:180,POLARIZATION=both \
#     examples/run_nasa_almond_austin_monostatic_rcs.slurm
#
# Typical overrides:
#   sbatch --export=ALL,ORDER=4,FINAL_TIME=1.03,RCS_START_TIME=0.43 \
#     examples/run_nasa_almond_austin_monostatic_rcs.slurm
#
#   sbatch --export=ALL,MESH=examples/meshes/custom_almond.vtk \
#     examples/run_nasa_almond_austin_monostatic_rcs.slurm

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
REFERENCE_DIR="${REFERENCE_DIR:-benchmark/AustinCEMBenchmarks/Problem_III-Almonds/IIIA-Reference Data}"

ORDER="${ORDER:-2}"
ESPRK_ORDER="${ESPRK_ORDER:-$((ORDER + 1))}"
CFL="${CFL:-0.15}"
FLUX="${FLUX:-centered}"
SIGMA_MAX="${SIGMA_MAX:-12.0}"
SIGMA_DEGREE="${SIGMA_DEGREE:-2}"
RCS_EVERY="${RCS_EVERY:-5}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-250}"
PARAVIEW_EVERY="${PARAVIEW_EVERY:-0}"
POSTPROCESS="${POSTPROCESS:-1}"
WRITE_PNG="${WRITE_PNG:-1}"
SHOW_FAILED="${SHOW_FAILED:-0}"
CASE_ONLY="${CASE_ONLY:-0}"

PHI_DEGREES="${PHI_DEGREES:-0:0.5:180}"
POLARIZATION="${POLARIZATION:-both}"
MAX_FINAL_RELATIVE_RMS="${MAX_FINAL_RELATIVE_RMS:-1e-1}"
MAX_SAMPLED_RELATIVE_RMS="${MAX_SAMPLED_RELATIVE_RMS:-1e-1}"
FAIL_ON_RESIDUAL_FAILURE="${FAIL_ON_RESIDUAL_FAILURE:-0}"

# Optional Slurm-array mapping. The array index enumerates phi first and
# polarization second, e.g. 0_H, 0_V, 30_H, 30_V.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${AUSTIN_ARRAY_PHI_DEGREES:-}" ]]; then
  array_phi_list="${AUSTIN_ARRAY_PHI_DEGREES// /}"
  array_polarization_list="${AUSTIN_ARRAY_POLARIZATIONS:-H;V}"
  # ';' is used for Slurm-exported lists because ',' separates --export fields.
  array_phi_list="${array_phi_list//,/;}"
  array_polarization_list="${array_polarization_list//,/;}"
  IFS=';' read -r -a array_phi_values <<< "${array_phi_list}"
  IFS=';' read -r -a array_polarizations <<< "${array_polarization_list}"
  array_pol_count="${#array_polarizations[@]}"
  array_case_count=$(( ${#array_phi_values[@]} * array_pol_count ))
  (( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < array_case_count )) || {
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside 0:$((array_case_count - 1))." >&2
    exit 1
  }
  array_phi_index=$(( SLURM_ARRAY_TASK_ID / array_pol_count ))
  array_pol_index=$(( SLURM_ARRAY_TASK_ID % array_pol_count ))
  PHI_DEGREES="${array_phi_values[array_phi_index]}"
  POLARIZATION="${array_polarizations[array_pol_index]}"
  CASE_ONLY=1
  POSTPROCESS=0
fi

WAVELENGTH="$(
  FREQUENCY_HZ="${FREQUENCY_HZ}" "${PYTHON}" -c \
    'import os; print(299792458.0 / float(os.environ["FREQUENCY_HZ"]))'
)"
PML_WIDTH="${PML_WIDTH:-${WAVELENGTH}}"
FINAL_TIME="$(
  WAVELENGTH="${WAVELENGTH}" FINAL_TIME="${FINAL_TIME:-}" "${PYTHON}" -c \
    'import os; print(os.environ["FINAL_TIME"] or 6.0 * float(os.environ["WAVELENGTH"]))'
)"
RCS_START_TIME="$(
  WAVELENGTH="${WAVELENGTH}" RCS_START_TIME="${RCS_START_TIME:-}" "${PYTHON}" -c \
    'import os; print(os.environ["RCS_START_TIME"] or 3.0 * float(os.environ["WAVELENGTH"]))'
)"

RUN_LABEL="${RUN_LABEL:-pec_almond_austin_monostatic_${FREQUENCY_KEY}_p${ORDER}}"
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
  sbatch --export=ALL,MESH=/path/to/almond_mesh.vtk \
    examples/run_nasa_almond_austin_monostatic_rcs.slurm
EOF
  exit 1
fi

EXTRA_AUSTIN_ARGS=()
if [[ "${FAIL_ON_RESIDUAL_FAILURE}" == "1" ||
      "${FAIL_ON_RESIDUAL_FAILURE}" == "true" ||
      "${FAIL_ON_RESIDUAL_FAILURE}" == "TRUE" ]]; then
  EXTRA_AUSTIN_ARGS+=(--fail-on-residual-failure)
fi
if [[ "${CASE_ONLY}" == "1" ||
      "${CASE_ONLY}" == "true" ||
      "${CASE_ONLY}" == "TRUE" ]]; then
  EXTRA_AUSTIN_ARGS+=(--case-only)
fi

echo "SLURM_JOB_ID             = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST           = ${SLURM_NODELIST:-local}"
echo "SLURM_NTASKS             = ${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK      = ${SLURM_CPUS_PER_TASK}"
echo "JULIA_DEPOT_PATH         = ${JULIA_DEPOT_PATH}"
echo "JULIA_NUM_THREADS        = ${JULIA_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS     = ${OPENBLAS_NUM_THREADS}"
echo "OMP_NUM_THREADS          = ${OMP_NUM_THREADS}"
echo "MPIEXEC                  = ${MPIEXEC}"
echo "Julia:"
julia --version
echo "Python:"
"${PYTHON}" --version
echo "MPI launcher:"
which "${MPIEXEC}"
"${MPIEXEC}" --version || true
echo "Run parameters:"
echo "  mesh                   = ${MESH}"
echo "  output_dir             = ${OUTPUT_DIR}"
echo "  frequency_hz           = ${FREQUENCY_HZ}"
echo "  frequency_key          = ${FREQUENCY_KEY}"
echo "  reference_kind         = ${REFERENCE_KIND}"
echo "  reference_dir          = ${REFERENCE_DIR}"
echo "  wavelength             = ${WAVELENGTH}"
echo "  final_time             = ${FINAL_TIME}"
echo "  rcs_start_time         = ${RCS_START_TIME}"
echo "  phi_degrees            = ${PHI_DEGREES}"
echo "  polarization           = ${POLARIZATION}"
echo "  order/esprk_order      = ${ORDER}/${ESPRK_ORDER}"
echo "  cfl                    = ${CFL}"
echo "  flux                   = ${FLUX}"
echo "  pml_width              = ${PML_WIDTH}"
echo "  sigma_max/degree       = ${SIGMA_MAX}/${SIGMA_DEGREE}"
echo "  residual gate final    = ${MAX_FINAL_RELATIVE_RMS}"
echo "  residual gate sampled  = ${MAX_SAMPLED_RELATIVE_RMS}"
echo "  fail on residual       = ${FAIL_ON_RESIDUAL_FAILURE}"
echo "  case only             = ${CASE_ONLY}"
echo "  postprocess            = ${POSTPROCESS}"
echo "  write png              = ${WRITE_PNG}"
echo "  show failed in plots   = ${SHOW_FAILED}"
echo "  ntasks                 = ${SLURM_NTASKS}"

"${MPIEXEC}" -n "${SLURM_NTASKS}" -genv OPENBLAS_NUM_THREADS 1 \
  julia --project=. \
  examples/distributed_pec_almond_austin_monostatic_rcs.jl \
  --mesh="${MESH}" \
  --frequency-hz="${FREQUENCY_HZ}" \
  --frequency-key="${FREQUENCY_KEY}" \
  --reference-kind="${REFERENCE_KIND}" \
  --reference-dir="${REFERENCE_DIR}" \
  --phi-degrees="${PHI_DEGREES}" \
  --polarization="${POLARIZATION}" \
  --max-final-relative-rms="${MAX_FINAL_RELATIVE_RMS}" \
  --max-sampled-relative-rms="${MAX_SAMPLED_RELATIVE_RMS}" \
  --order="${ORDER}" \
  --esprk-order="${ESPRK_ORDER}" \
  --final-time="${FINAL_TIME}" \
  --cfl="${CFL}" \
  --flux="${FLUX}" \
  --pml-width="${PML_WIDTH}" \
  --sigma-max="${SIGMA_MAX}" \
  --sigma-degree="${SIGMA_DEGREE}" \
  --rcs-start-time="${RCS_START_TIME}" \
  --rcs-every="${RCS_EVERY}" \
  --diagnostics-every="${DIAGNOSTICS_EVERY}" \
  --paraview-every="${PARAVIEW_EVERY}" \
  --output-dir="${OUTPUT_DIR}" \
  "${EXTRA_AUSTIN_ARGS[@]}"

if [[ "${POSTPROCESS}" == "1" ||
      "${POSTPROCESS}" == "true" ||
      "${POSTPROCESS}" == "TRUE" ]]; then
  POSTPROCESS_ARGS=(
    --run-dir="${OUTPUT_DIR}"
    --polarization="${POLARIZATION}"
    --max-final-relative-rms="${MAX_FINAL_RELATIVE_RMS}"
    --max-sampled-relative-rms="${MAX_SAMPLED_RELATIVE_RMS}"
    --title="NASA PEC almond Austin monostatic RCS: ${FREQUENCY_KEY}, p=${ORDER}"
  )
  if [[ "${WRITE_PNG}" == "1" ||
        "${WRITE_PNG}" == "true" ||
        "${WRITE_PNG}" == "TRUE" ]]; then
    POSTPROCESS_ARGS+=(--write-png)
  fi
  if [[ "${SHOW_FAILED}" == "1" ||
        "${SHOW_FAILED}" == "true" ||
        "${SHOW_FAILED}" == "TRUE" ]]; then
    POSTPROCESS_ARGS+=(--show-failed)
  fi
  "${PYTHON}" examples/postprocess_pec_almond_austin_monostatic_rcs.py \
    "${POSTPROCESS_ARGS[@]}"
fi

echo "Austin monostatic NASA almond workflow complete."
echo "Run directory:  ${OUTPUT_DIR}"
echo "Comparison CSV: ${OUTPUT_DIR}/diagnostics/austin_monostatic_rcs_comparison.csv"
echo "Plots:          ${OUTPUT_DIR}/plots"
echo "Case outputs:   ${OUTPUT_DIR}/cases"
