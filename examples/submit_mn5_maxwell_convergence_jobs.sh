#!/usr/bin/env bash

set -euo pipefail

# Run from the DiscoGMPI repository root on MareNostrum 5:
#   bash examples/submit_mn5_maxwell_convergence_jobs.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

account="${ACCOUNT:-bsc115}"
qos="${QOS:-gpp_data}"
nodes="${NODES:-1}"
ntasks="${NTASKS:-32}"
cpus_per_task="${CPUS_PER_TASK:-1}"
walltime="${WALLTIME:-36:00:00}"
job_script="examples/run_mn5_maxwell_convergence.slurm"
log_dir="${LOG_DIR:-output/slurm}"
mkdir -p "${log_dir}"

cases=(pec_cavity plane_wave)
fluxes=(alternating centered)

summary_file="${log_dir}/maxwell_convergence_submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tcase\tflux\n' > "${summary_file}"

for convergence_case in "${cases[@]}"; do
    for flux in "${fluxes[@]}"; do
        case "${convergence_case}_${flux}" in
            pec_cavity_alternating) job_name="conv_pec_alt" ;;
            pec_cavity_centered)    job_name="conv_pec_ctr" ;;
            plane_wave_alternating) job_name="conv_wave_alt" ;;
            plane_wave_centered)    job_name="conv_wave_ctr" ;;
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
            --export="ALL,CONVERGENCE_CASE=${convergence_case},FLUX=${flux}" \
            "${job_script}")"

        printf '%s\t%s\t%s\n' "${job_id}" "${convergence_case}" "${flux}" \
            | tee -a "${summary_file}"
    done
done

echo "Submitted four MN5 convergence jobs."
echo "Submission summary: ${summary_file}"
echo "Resources per job: ${nodes} node(s), ${ntasks} task(s), ${walltime}."
