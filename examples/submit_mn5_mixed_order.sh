#!/usr/bin/env bash
set -euo pipefail

# Prints eight sbatch commands by default. Only --submit submits jobs.
# Remaining arguments are passed as sbatch resource overrides.
# Example: bash examples/submit_mn5_mixed_order.sh --submit --ntasks=8
submit=0
if [[ "${1:-}" == --submit ]]; then submit=1; shift; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
output_root="${MIXED_OUTPUT_ROOT:-output/mixed_order/$(date -u +%Y%m%dT%H%M%SZ)}"
levels="${MIXED_LEVELS:-iso:1,2,3,4}"
final_time="${MIXED_FINAL_TIME:-1.0}"
cfl="${MIXED_CFL:-0.1}"
initialization="${MIXED_INITIALIZATION:-interpolate}"
if (( submit )); then mkdir -p output/slurm; fi
for p in 1 2 3 4; do
    for flux in centered alternating; do
        output="${output_root}/pE${p}_pH$((p+1))_${flux}.csv"
        command=(sbatch --job-name="mixed_p${p}_${flux}" "$@"
            examples/run_mn5_mixed_order.slurm "${output}" "${p}" "${levels}"
            "${final_time}" "${cfl}" "${initialization}" "${flux}")
        printf '%q ' "${command[@]}"
        printf '\n'
        if (( submit )); then "${command[@]}"; fi
    done
done
