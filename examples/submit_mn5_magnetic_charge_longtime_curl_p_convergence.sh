#!/usr/bin/env bash

set -euo pipefail

# Submit one MN5 long-time magnetic-charge job per polynomial order:
#   bash examples/submit_mn5_magnetic_charge_longtime_curl_p_convergence.sh
#
# Select a subset with a comma-separated list, for example:
#   ORDERS=2,3 bash examples/submit_mn5_magnetic_charge_longtime_curl_p_convergence.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

account="${ACCOUNT:-bsc115}"
qos="${QOS:-gp_bsccase}"
nodes="${NODES:-1}"
ntasks="${NTASKS:-16}"
cpus_per_task="${CPUS_PER_TASK:-1}"
walltime="${WALLTIME:-12:00:00}"
orders_csv="${ORDERS:-1,2,3,4}"
fluxes_csv="${FLUXES:-alternating,centered}"
sbatch_command="${SBATCH_COMMAND:-sbatch}"
log_dir="${LOG_DIR:-output/slurm}"
job_script="examples/run_mn5_magnetic_charge_longtime_curl_p_convergence.slurm"

mkdir -p "${log_dir}"

orders_csv="${orders_csv// /}"
IFS=',' read -r -a orders <<< "${orders_csv}"
fluxes_csv="${fluxes_csv// /}"
IFS=',' read -r -a fluxes <<< "${fluxes_csv}"
(( ${#orders[@]} > 0 )) || {
    echo "ORDERS must contain at least one polynomial order." >&2
    exit 1
}
(( ${#fluxes[@]} > 0 )) || {
    echo "FLUXES must contain at least one flux." >&2
    exit 1
}

summary_file="${log_dir}/magnetic_charge_longtime_curl_p_submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\torder\tesprk_order\tflux\n' > "${summary_file}"

for order in "${orders[@]}"; do
    [[ "${order}" =~ ^[1-5]$ ]] || {
        echo "Unsupported polynomial order '${order}'; expected an integer in 1:5." >&2
        exit 1
    }
    esprk_order=$((order + 1))

    for flux in "${fluxes[@]}"; do
        case "${flux}" in
            alternating) job_name="magq_alt_p${order}" ;;
            centered) job_name="magq_ctr_p${order}" ;;
            *)
                echo "Unsupported flux '${flux}'; expected alternating or centered." >&2
                exit 1
                ;;
        esac

        job_id="$("${sbatch_command}" --parsable \
            --account="${account}" \
            --qos="${qos}" \
            --job-name="${job_name}" \
            --nodes="${nodes}" \
            --ntasks="${ntasks}" \
            --cpus-per-task="${cpus_per_task}" \
            --time="${walltime}" \
            --output="${log_dir}/%x_%j.out" \
            --error="${log_dir}/%x_%j.err" \
            --export="ALL,ORDER=${order},ESPRK_ORDER=${esprk_order},FLUX=${flux}" \
            "${job_script}")"

        printf '%s\t%s\t%s\t%s\n' \
            "${job_id}" "${order}" "${esprk_order}" "${flux}" \
            | tee -a "${summary_file}"
    done
done

job_count=$(( ${#orders[@]} * ${#fluxes[@]} ))
echo "Submitted ${job_count} MN5 magnetic-charge long-time job(s)."
echo "Submission summary: ${summary_file}"
echo "Resources per job: ${nodes} node(s), ${ntasks} task(s), ${walltime}."
