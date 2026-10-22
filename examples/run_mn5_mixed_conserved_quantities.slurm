#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -euo pipefail
started=${SECONDS}
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
case_file="${1:?Pass the prepared case TOML}"
shift
module purge
module load intel/2023.0 impi/2021.10.0 julia/1.10.0
export JULIA_DEPOT_PATH="${PWD}/.julia_depot:${JULIA_DEPOT_PATH:-}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Respect Slurm's physical-core cpuset and use one MPI rank per core.
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=core I_MPI_PIN_RESPECT_CPUSET=1
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local} NTASKS=${SLURM_NTASKS:-1} TASKS_PER_NODE=${SLURM_TASKS_PER_NODE:-unknown}"
# Lock the actual output, allowing independent benchmarks of the same case.
output=$(python3 - "${case_file}" "$@" <<'PY'
import json, sys
from pathlib import Path
output=None
for line in Path(sys.argv[1]).read_text().splitlines():
    key,sep,value=line.partition('=')
    if sep and key.strip()=='output':
        output=json.loads(value.strip())
for arg in sys.argv[2:]:
    if arg.startswith('--output-dir='):
        output=arg.split('=',1)[1]
if output is None:
    raise SystemExit('Case output not found')
print(Path(output).resolve())
PY
)
mkdir -p "$(dirname "${output}")" output/slurm
exec {case_lock}>"${output}.lock"
flock -n "${case_lock}" || { echo "Output is already active: ${output}" >&2; exit 1; }
julia --project=. -e 'using MPI, DiscoGMPI; MPI.versioninfo()'
progress_dir=$(mktemp -d "${SLURM_SUBMIT_DIR}/output/slurm/mixed_inv_progress_${SLURM_JOB_ID:-local}_XXXXXX")
heartbeat="${progress_dir}/heartbeat.toml"
# Reserve ten minutes for a synchronized checkpoint and final history merge.
budget=$(( ${MIXED_INV_WALL_SECONDS:-129600} - ${SECONDS} + started - ${MIXED_INV_CHECKPOINT_MARGIN:-600} ))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    budget=$(( SLURM_JOB_END_TIME - $(date +%s) - ${MIXED_INV_CHECKPOINT_MARGIN:-600} ))
fi
(( budget>0 )) || { echo 'No integration time left before checkpoint deadline' >&2; exit 1; }
placement=()
if [[ "${SLURM_NTASKS_PER_NODE:-}" =~ ^[0-9]+$ ]]; then
    placement=(-ppn "${SLURM_NTASKS_PER_NODE}")
fi
launch=("${MPIEXEC:-mpiexec}" -n "${SLURM_NTASKS:-1}" "${placement[@]}" -genv OPENBLAS_NUM_THREADS 1
    julia --project=. examples/run_mixed_conserved_quantities.jl "$case_file"
    "--max-wall-seconds=${budget}" "--heartbeat=${heartbeat}" "$@")
set +e
if [[ "${MIXED_INV_WATCHDOG_SECONDS:-2700}" == "0" ]]; then
    "${launch[@]}"
else
    python3 examples/run_with_progress_watchdog.py --heartbeat "${heartbeat}" \
        --stall-seconds "${MIXED_INV_WATCHDOG_SECONDS:-2700}" \
        --startup-seconds "${MIXED_INV_STARTUP_TIMEOUT:-7200}" -- "${launch[@]}"
fi
launch_status=$?
set -e
echo "LAUNCH_WALL_SECONDS=$((SECONDS-started)) LAUNCH_EXIT_CODE=${launch_status} HEARTBEAT=${heartbeat}"
exit "${launch_status}"
