#!/usr/bin/env bash

set -euo pipefail

# Submit one fixed-mesh p-convergence audit for each Poisson-bracket flux:
#   bash examples/submit_mn5_conserved_quantities_p_convergence.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

account="${ACCOUNT:-bsc115}"
qos="${QOS:-gpp_data}"
nodes="${NODES:-1}"
ntasks="${NTASKS:-32}"
cpus_per_task="${CPUS_PER_TASK:-1}"
walltime="${WALLTIME:-36:00:00}"
log_dir="${LOG_DIR:-output/slurm}"
job_script="examples/run_mn5_conserved_quantities_p_convergence.slurm"
mkdir -p "${log_dir}"

summary_file="${log_dir}/conserved_quantities_p_submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tflux\n' > "${summary_file}"

for flux in alternating centered; do
    case "${flux}" in
        alternating) job_name="inv_pconv_alt" ;;
        centered)    job_name="inv_pconv_ctr" ;;
    esac

    job_id="$(sbatch --parsable \
        --account="${account}" \
        --qos="${qos}" \
        --job-name="${job_name}" \
        --nodes="${nodes}" \
        --ntasks="${ntasks}" \
        --cpus-per-task="${cpus_per_task}" \
        --time="${walltime}" \
        --output="${log_dir}/%x_%j.out" \
        --error="${log_dir}/%x_%j.err" \
        --export="ALL,FLUX=${flux}" \
        "${job_script}")"

    printf '%s\t%s\n' "${job_id}" "${flux}" | tee -a "${summary_file}"
done

echo "Submitted centered and alternating conserved-quantity p-convergence jobs."
echo "Submission summary: ${summary_file}"

