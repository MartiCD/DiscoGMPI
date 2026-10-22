#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gpp_data
#SBATCH --job-name=maxwell_conv
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1

set -euo pipefail

# Parameterized MareNostrum 5 job for one Maxwell convergence family.
# Submit all four case/flux combinations with:
#   bash examples/submit_mn5_maxwell_convergence_jobs.sh
#
# Direct submission example:
#   mkdir -p output/slurm
#   sbatch --export=ALL,CONVERGENCE_CASE=pec_cavity,FLUX=alternating \
#     examples/run_mn5_maxwell_convergence.slurm

module purge
module load intel/2023.0
module load impi/2021.10.0
module load julia/1.10.0

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f Project.toml ]] || {
    echo "Submit this job from the DiscoGMPI repository root." >&2
    exit 1
}

CONVERGENCE_CASE="${CONVERGENCE_CASE:?Set CONVERGENCE_CASE to pec_cavity or plane_wave}"
FLUX="${FLUX:?Set FLUX to alternating or centered}"

case "${CONVERGENCE_CASE}" in
    pec_cavity|plane_wave) ;;
    *)
        echo "Unsupported CONVERGENCE_CASE=${CONVERGENCE_CASE}" >&2
        exit 1
        ;;
esac

case "${FLUX}" in
    alternating|centered) ;;
    *)
        echo "Unsupported FLUX=${FLUX}" >&2
        exit 1
        ;;
esac

SLURM_NTASKS="${SLURM_NTASKS:-32}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

MPIEXEC="${MPIEXEC:-mpiexec}"
ORDERS="${ORDERS:-1,2,3,4}"
CFL="${CFL:-0.8}"
CFL_DIVISOR="${CFL_DIVISOR:-4}"
PERIODS="${PERIODS:-1}"
ERROR_MODE="${ERROR_MODE:-final}"
RATE_POLICY="${RATE_POLICY:-report}"
NODAL_SET="${NODAL_SET:-equispaced}"

CAVITY_CELLS="${CAVITY_CELLS:-4,8,16,32}"
CAVITY_JITTER="${CAVITY_JITTER:-0}"
# nx=2 has only 12 tetrahedra, cannot use 32 MPI ranks, and aliases the
# p=1 sine mode. Start at nx=4 for the production sweep.
PLANE_NX_TARGETS="${PLANE_NX_TARGETS:-4,8,16,32}"
PLANE_WAVE_NUMBER="${PLANE_WAVE_NUMBER:-3.141592653589793}"

OUTPUT_DIR="${OUTPUT_DIR:-output/convergence/mn5}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/${CONVERGENCE_CASE}_${FLUX}_structured.csv}"

echo "SLURM_JOB_ID          = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST        = ${SLURM_NODELIST:-local}"
echo "SLURM_NTASKS          = ${SLURM_NTASKS}"
echo "JULIA_NUM_THREADS     = ${JULIA_NUM_THREADS}"
echo "CONVERGENCE_CASE      = ${CONVERGENCE_CASE}"
echo "FLUX                  = ${FLUX}"
echo "ORDERS                = ${ORDERS}"
echo "CFL / divisor         = ${CFL} / ${CFL_DIVISOR}"
echo "PERIODS               = ${PERIODS}"
echo "ERROR_MODE            = ${ERROR_MODE}"
echo "NODAL_SET             = ${NODAL_SET}"
echo "OUTPUT_CSV            = ${OUTPUT_CSV}"
echo "Julia                 = $(julia --version)"
echo "MPI launcher          = $(command -v "${MPIEXEC}")"

common_args=(
    --mesh-family=structured
    --orders="${ORDERS}"
    --periods="${PERIODS}"
    --cfl="${CFL}"
    --cfl-divisor="${CFL_DIVISOR}"
    --flux="${FLUX}"
    --error-mode="${ERROR_MODE}"
    --rate-policy="${RATE_POLICY}"
    --nodal-set="${NODAL_SET}"
    --output="${OUTPUT_CSV}"
)

case "${CONVERGENCE_CASE}" in
    pec_cavity)
        driver="examples/convergence_distributed_poisson_bracket_maxwell.jl"
        case_args=(
            --boundary-condition=pec
            --cells="${CAVITY_CELLS}"
            --jitter="${CAVITY_JITTER}"
        )
        ;;
    plane_wave)
        driver="examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl"
        case_args=(
            --nx-targets="${PLANE_NX_TARGETS}"
            --wave-number="${PLANE_WAVE_NUMBER}"
        )
        ;;
esac

"${MPIEXEC}" -n "${SLURM_NTASKS}" -genv OPENBLAS_NUM_THREADS 1 \
    julia --project=. "${driver}" "${common_args[@]}" "${case_args[@]}"

echo "Completed ${CONVERGENCE_CASE} with ${FLUX} flux."
echo "CSV: ${OUTPUT_CSV}"
