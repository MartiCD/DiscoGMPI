#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --job-name=magq_curl
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

set -euo pipefail

# Parameterized MareNostrum 5 job for one order of the long-time
# magnetic-charge curl-initialization campaign. Submit p=1,2,3,4 with:
#   bash examples/submit_mn5_magnetic_charge_longtime_curl_p_convergence.sh

module purge
module load intel/2023.0
module load impi/2021.10.0
module load julia/1.10.0

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f Project.toml ]] || {
    echo "Submit this job from the DiscoGMPI repository root." >&2
    exit 1
}

ORDER="${ORDER:?Set ORDER to the DG polynomial order}"
[[ "${ORDER}" =~ ^[1-5]$ ]] || {
    echo "Unsupported ORDER=${ORDER}; expected an integer in 1:5." >&2
    exit 1
}

ESPRK_ORDER="${ESPRK_ORDER:-$((ORDER + 1))}"
expected_esprk_order=$((ORDER + 1))
[[ "${ESPRK_ORDER}" == "${expected_esprk_order}" ]] || {
    echo "ESPRK_ORDER must equal ORDER+1=${expected_esprk_order}." >&2
    exit 1
}

FLUX="${FLUX:-alternating}"
case "${FLUX}" in
    alternating|centered) ;;
    *) echo "Unsupported FLUX=${FLUX}" >&2; exit 1 ;;
esac

MAGNETIC_INITIALIZATION="${MAGNETIC_INITIALIZATION:-curl-potential}"
ELECTRIC_INITIALIZATION="${ELECTRIC_INITIALIZATION:-interpolate}"
case "${MAGNETIC_INITIALIZATION}" in
    interpolate|curl-potential) ;;
    *) echo "Unsupported MAGNETIC_INITIALIZATION=${MAGNETIC_INITIALIZATION}" >&2; exit 1 ;;
esac
case "${ELECTRIC_INITIALIZATION}" in
    interpolate|curl-potential) ;;
    *) echo "Unsupported ELECTRIC_INITIALIZATION=${ELECTRIC_INITIALIZATION}" >&2; exit 1 ;;
esac

SLURM_NTASKS="${SLURM_NTASKS:-16}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
PARTITIONS="${PARTITIONS:-${SLURM_NTASKS}}"
[[ "${PARTITIONS}" == "${SLURM_NTASKS}" ]] || {
    echo "PARTITIONS=${PARTITIONS} must equal SLURM_NTASKS=${SLURM_NTASKS}." >&2
    exit 1
}

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

MPIEXEC="${MPIEXEC:-mpiexec}"
MESH_PATH="${MESH_PATH:-examples/meshes/periodic_box_structured_2x1x1.vtk}"
FINAL_TIME="${FINAL_TIME:-1000.0}"
CFL="${CFL:-0.4}"
PRINT_EVERY="${PRINT_EVERY:-1000}"
ENERGY_EVERY="${ENERGY_EVERY:-1}"
FULL_DIAGNOSTICS_EVERY="${FULL_DIAGNOSTICS_EVERY:-10000}"
PARAVIEW_EVERY="${PARAVIEW_EVERY:-1000000000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/magnetic_charge_invariance_test}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${FLUX}/curl_p${ORDER}}"
RESTART_PATH="${RESTART_PATH:-}"

# Each submitted order gets a job-private preparation directory. This avoids
# races while the eight jobs generate their partition and distributed-mesh cache.
PREP_ROOT="${PREP_ROOT:-${OUTPUT_ROOT}/mn5_mesh_cache/job${SLURM_JOB_ID:-local}_p${ORDER}}"
PARTITION_PATH="${PARTITION_PATH:-${PREP_ROOT}/mesh.mesh.epart.${PARTITIONS}}"
DISTRIBUTED_MESH_DIR="${DISTRIBUTED_MESH_DIR:-${PREP_ROOT}/distributed_mesh}"

[[ -f "${MESH_PATH}" ]] || {
    echo "Mesh file not found: ${MESH_PATH}" >&2
    exit 1
}
restart_args=()
if [[ -n "${RESTART_PATH}" ]]; then
    [[ -f "${RESTART_PATH}/checkpoint_manifest.toml" ]] || {
        echo "Checkpoint manifest not found: ${RESTART_PATH}/checkpoint_manifest.toml" >&2
        exit 1
    }
    restart_args+=(--restart="${RESTART_PATH}")
fi
mkdir -p "${OUTPUT_ROOT}" "${PREP_ROOT}"

echo "SLURM_JOB_ID             = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST           = ${SLURM_NODELIST:-local}"
echo "SLURM_NTASKS             = ${SLURM_NTASKS}"
echo "JULIA_NUM_THREADS        = ${JULIA_NUM_THREADS}"
echo "ORDER / ESPRK_ORDER      = ${ORDER} / ${ESPRK_ORDER}"
echo "FLUX                     = ${FLUX}"
echo "MESH_PATH                = ${MESH_PATH}"
echo "FINAL_TIME / CFL         = ${FINAL_TIME} / ${CFL}"
echo "MAGNETIC_INITIALIZATION  = ${MAGNETIC_INITIALIZATION}"
echo "ELECTRIC_INITIALIZATION  = ${ELECTRIC_INITIALIZATION}"
echo "ENERGY_EVERY             = ${ENERGY_EVERY}"
echo "FULL_DIAGNOSTICS_EVERY   = ${FULL_DIAGNOSTICS_EVERY}"
echo "CHECKPOINT_EVERY         = ${CHECKPOINT_EVERY}"
echo "OUTPUT_DIR               = ${OUTPUT_DIR}"
echo "RESTART_PATH             = ${RESTART_PATH:-none}"
echo "PARTITION_PATH           = ${PARTITION_PATH}"
echo "DISTRIBUTED_MESH_DIR     = ${DISTRIBUTED_MESH_DIR}"
echo "Julia                    = $(julia --version)"
echo "MPI launcher             = $(command -v "${MPIEXEC}")"

"${MPIEXEC}" -n "${SLURM_NTASKS}" -genv OPENBLAS_NUM_THREADS 1 \
    julia --project=. examples/distributed_periodic_poisson_bracket_maxwell.jl \
    --mesh="${MESH_PATH}" \
    --partitions="${PARTITIONS}" \
    --partition="${PARTITION_PATH}" \
    --distributed-mesh-dir="${DISTRIBUTED_MESH_DIR}" \
    --order="${ORDER}" \
    --esprk-order="${ESPRK_ORDER}" \
    --final-time="${FINAL_TIME}" \
    --cfl="${CFL}" \
    --flux="${FLUX}" \
    --magnetic-initialization="${MAGNETIC_INITIALIZATION}" \
    --electric-initialization="${ELECTRIC_INITIALIZATION}" \
    --print-every="${PRINT_EVERY}" \
    --energy-every="${ENERGY_EVERY}" \
    --full-diagnostics-every="${FULL_DIAGNOSTICS_EVERY}" \
    --paraview-every="${PARAVIEW_EVERY}" \
    --checkpoint-every="${CHECKPOINT_EVERY}" \
    "${restart_args[@]}" \
    --output-dir="${OUTPUT_DIR}"

echo "Completed magnetic-charge long-time case p=${ORDER}."
echo "Output directory: ${OUTPUT_DIR}"
