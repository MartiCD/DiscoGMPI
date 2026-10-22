#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --job-name=mixed_maxwell
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
set -euo pipefail

# Uses the module versions from the repository's existing MN5 launchers.
# Override scheduling resources with sbatch options at submission time.
module purge
module load intel/2023.0 impi/2021.10.0 julia/1.10.0
cd "${SLURM_SUBMIT_DIR:?Submit from the DiscoGMPI repository root}"
[[ -f Project.toml ]] || { echo 'Project.toml missing' >&2; exit 1; }
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PWD}/.julia_depot:}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
printf 'job=%s ranks=%s threads=%s commit=%s\n' "${SLURM_JOB_ID}" \
    "${SLURM_NTASKS}" "${JULIA_NUM_THREADS}" "$(git rev-parse HEAD)"
julia --project=. -e 'using MPI; MPI.versioninfo()'
exec bash examples/run_mixed_order.sh "$@"
