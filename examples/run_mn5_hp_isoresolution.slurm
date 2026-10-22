#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
set -euo pipefail

module purge
module load intel/2023.0 impi/2021.10.0 julia/1.10.0
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
[[ -f Project.toml ]] || { echo 'Project.toml missing' >&2; exit 1; }
case_file="${1:?Pass a prepared case TOML file}"
[[ -f "${case_file}" ]] || { echo "Missing ${case_file}" >&2; exit 1; }
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
echo "CASE_FILE=${case_file}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} NTASKS=${SLURM_NTASKS:-1}"
initial_args=()
if [[ "${INITIAL_ONLY:-0}" == "1" ]]; then
    initial_args+=(--initial-only)
    echo "INITIAL_ONLY=1"
fi
started=${SECONDS}
"${MPIEXEC:-mpiexec}" -n "${SLURM_NTASKS:-1}" -genv OPENBLAS_NUM_THREADS 1 \
    julia --project=. examples/run_hp_isoresolution_case.jl "${case_file}" "${initial_args[@]}"
echo "LAUNCH_WALL_SECONDS=$((SECONDS-started))"
