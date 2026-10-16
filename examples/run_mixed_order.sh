#!/usr/bin/env bash
set -euo pipefail

# Arguments: CSV, p_E list, levels, final time, CFL, initialization, flux.
# Prefix --dry-run to print the launch command without executing or writing.
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"
output="${1:-output/mixed_order/${SLURM_JOB_ID:-local}/results.csv}"
orders="${2:-1,2,3,4}"
levels="${3:-iso:1,2,3,4}"
final_time="${4:-1.0}"
cfl="${5:-0.1}"
initialization="${6:-interpolate}"
flux="${7:-both}"
ranks="${SLURM_NTASKS:-${MPI_RANKS:-1}}"
[[ "${ranks}" =~ ^[1-9][0-9]*$ ]] || { echo 'MPI ranks must be positive' >&2; exit 2; }
[[ "${initialization}" == interpolate || "${initialization}" == project ]] || { echo 'Invalid initialization' >&2; exit 2; }
[[ "${flux}" == both || "${flux}" == centered || "${flux}" == alternating ]] || { echo 'Invalid flux' >&2; exit 2; }
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${JULIA_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
command=("${MPIEXEC:-mpiexec}" -n "${ranks}" "${JULIA:-julia}" --project="${repo_root}"
    "${script_dir}/convergence_mixed_order.jl" "${output}" "${orders}" "${levels}"
    "${final_time}" "${cfl}" "${initialization}" "${flux}")
printf 'Launch:'
printf ' %q' "${command[@]}"
printf '\n'
if (( dry_run )); then exit 0; fi
[[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 2; }
mkdir -p -- "$(dirname -- "${output}")"
exec "${command[@]}"
