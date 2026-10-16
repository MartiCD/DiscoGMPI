#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gpp_data
#SBATCH --job-name=inv_pconv
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1

set -euo pipefail

# MN5 p-convergence campaign for conserved Maxwell quantities.
# The mesh is fixed while p=1,2,3,4 is swept. The audit records energy,
# electric/magnetic charge, weak charge tests, momentum, update defects, and
# high-precision comparisons. Submit both fluxes with:
#   bash examples/submit_mn5_conserved_quantities_p_convergence.sh

module purge
module load intel/2023.0
module load impi/2021.10.0
module load julia/1.10.0

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f Project.toml ]] || {
    echo "Submit this job from the DiscoGMPI repository root." >&2
    exit 1
}

FLUX="${FLUX:?Set FLUX to alternating or centered}"
case "${FLUX}" in
    alternating|centered) ;;
    *) echo "Unsupported FLUX=${FLUX}" >&2; exit 1 ;;
esac

SLURM_NTASKS="${SLURM_NTASKS:-32}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

MPIEXEC="${MPIEXEC:-mpiexec}"
ORDERS="${ORDERS:-1,2,3,4}"
NX_TARGETS="${NX_TARGETS:-16}"
CFLS="${CFLS:-0.1}"
FINAL_TIME="${FINAL_TIME:-1.0}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
STAGE_EVERY="${STAGE_EVERY:-0}"
MAX_TESTS="${MAX_TESTS:-32}"
MAGNETIC_MODES="${MAGNETIC_MODES:-curl-potential}"
ELECTRIC_MODES="${ELECTRIC_MODES:-interpolate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/invariant_audit/mn5}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${FLUX}_p_convergence_job${SLURM_JOB_ID:-local}}"

mkdir -p "${OUTPUT_ROOT}"

echo "SLURM_JOB_ID          = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST        = ${SLURM_NODELIST:-local}"
echo "SLURM_NTASKS          = ${SLURM_NTASKS}"
echo "FLUX                  = ${FLUX}"
echo "NX_TARGETS            = ${NX_TARGETS}"
echo "ORDERS                = ${ORDERS}"
echo "CFLS                  = ${CFLS}"
echo "FINAL_TIME            = ${FINAL_TIME}"
echo "SAMPLE_EVERY          = ${SAMPLE_EVERY}"
echo "MAGNETIC_MODES        = ${MAGNETIC_MODES}"
echo "ELECTRIC_MODES        = ${ELECTRIC_MODES}"
echo "OUTPUT_DIR            = ${OUTPUT_DIR}"
echo "Julia                 = $(julia --version)"
echo "MPI launcher          = $(command -v "${MPIEXEC}")"

"${MPIEXEC}" -n "${SLURM_NTASKS}" -genv OPENBLAS_NUM_THREADS 1 \
    julia --project=. examples/audit_periodic_maxwell_invariants.jl \
    --flux="${FLUX}" \
    --nx-targets="${NX_TARGETS}" \
    --orders="${ORDERS}" \
    --cfls="${CFLS}" \
    --final-time="${FINAL_TIME}" \
    --sample-every="${SAMPLE_EVERY}" \
    --stage-every="${STAGE_EVERY}" \
    --max-tests="${MAX_TESTS}" \
    --magnetic-modes="${MAGNETIC_MODES}" \
    --electric-modes="${ELECTRIC_MODES}" \
    --output-dir="${OUTPUT_DIR}"

echo "Conserved-quantity p-convergence completed."
echo "Audit directory: ${OUTPUT_DIR}"

