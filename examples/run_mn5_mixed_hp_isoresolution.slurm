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
case_file="${1:?Pass a prepared mixed case TOML file}"
shift
[[ -f "${case_file}" ]] || { echo "Missing ${case_file}" >&2; exit 1; }
# Always put this checkout's writable depot first. A trailing empty entry keeps
# Julia's standard user/system depots available as read-only fallbacks.
export JULIA_DEPOT_PATH="${PWD}/.julia_depot:${JULIA_DEPOT_PATH:-}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
echo "CASE_FILE=${case_file} CODE_ID=${MIXED_CODE_ID:-manual-snapshot}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} NTASKS=${SLURM_NTASKS:-1}"
# Fail once with a readable setup error instead of printing the same import
# failure from every MPI rank. Dependency installation remains an explicit
# preparation step and never occurs inside production jobs.
julia --project=. -e 'using MPI, DiscoGMPI; MPI.versioninfo()'
initial_args=()
if [[ "${INITIAL_ONLY:-0}" == "1" ]]; then initial_args+=(--initial-only); fi
lock_dir="$(dirname "$(dirname "${case_file}")")/locks"
mkdir -p "${lock_dir}" output/slurm
# The kernel releases this lock after a crash/timeout; no stale lock cleanup.
exec {case_lock}>"${lock_dir}/$(basename "${case_file}").${INITIAL_ONLY:-0}.lock"
flock -n "${case_lock}" || { echo 'This case is already running; refusing overlapping writes' >&2; exit 1; }
# Unique, shared path: rank zero and the launcher can be on different nodes.
progress_dir="$(mktemp -d "${SLURM_SUBMIT_DIR}/output/slurm/mixed_hp_progress_${SLURM_JOB_ID:-local}_XXXXXX")"
heartbeat="${progress_dir}/heartbeat.toml"
runner_args=("--checkpoint-every=${MIXED_HP_CHECKPOINT_EVERY:-1000}"
    "--restart=${MIXED_HP_RESTART:-auto}" "--heartbeat=${heartbeat}")
started=${SECONDS}
launch=("${MPIEXEC:-mpiexec}" -n "${SLURM_NTASKS:-1}" -genv OPENBLAS_NUM_THREADS 1
    julia --project=. examples/run_mixed_hp_isoresolution_case.jl
    "${case_file}" "${initial_args[@]}" "${runner_args[@]}" "$@")
set +e
if [[ "${MIXED_HP_WATCHDOG_SECONDS:-2700}" == "0" ]]; then
    "${launch[@]}"
else
    python3 examples/run_with_progress_watchdog.py --heartbeat "${heartbeat}" \
        --stall-seconds "${MIXED_HP_WATCHDOG_SECONDS:-2700}" \
        --startup-seconds "${MIXED_HP_STARTUP_TIMEOUT:-7200}" -- "${launch[@]}"
fi
launch_status=$?
set -e
echo "LAUNCH_WALL_SECONDS=$((SECONDS-started))"
echo "LAUNCH_EXIT_CODE=${launch_status} HEARTBEAT=${heartbeat}"
exit "${launch_status}"
