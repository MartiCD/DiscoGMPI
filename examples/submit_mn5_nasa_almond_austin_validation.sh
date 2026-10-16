#!/usr/bin/env bash

set -euo pipefail

# Submit a staged Austin IIIA monostatic validation as independent Slurm-array
# cases. The default seven-angle subset is intended to validate the numerical
# setup before launching the full 0:0.5:180 H/V campaign.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

account="${ACCOUNT:-bsc115}"
qos="${QOS:-gp_bsccase}"
nodes="${NODES:-1}"
ntasks="${NTASKS:-16}"
cpus_per_task="${CPUS_PER_TASK:-1}"
walltime="${WALLTIME:-06:00:00}"
phi_values="${PHI_VALUES:-0,30,60,90,120,150,180}"
polarizations="${POLARIZATIONS:-H,V}"
frequency_key="${FREQUENCY_KEY:-fx1}"
order="${ORDER:-2}"
output_dir="${OUTPUT_DIR:-output/MN5/pec_almond_austin_validation_${frequency_key}_p${order}}"
log_dir="${LOG_DIR:-output/slurm}"
job_script="examples/run_nasa_almond_austin_monostatic_rcs.slurm"
sbatch_command="${SBATCH_COMMAND:-sbatch}"

phi_values="${phi_values// /}"
polarizations="${polarizations// /}"
IFS=',' read -r -a phi_array <<< "${phi_values}"
IFS=',' read -r -a polarization_array <<< "${polarizations}"
(( ${#phi_array[@]} > 0 && ${#polarization_array[@]} > 0 )) || {
  echo "PHI_VALUES and POLARIZATIONS must not be empty." >&2
  exit 1
}
for polarization in "${polarization_array[@]}"; do
  [[ "${polarization}" == "H" || "${polarization}" == "V" ]] || {
    echo "Unsupported polarization '${polarization}'; expected H or V." >&2
    exit 1
  }
done

case_count=$(( ${#phi_array[@]} * ${#polarization_array[@]} ))
array_last=$((case_count - 1))
mkdir -p "${log_dir}"

# Slurm uses commas as --export separators, so encode list values with ';'.
job_id="$("${sbatch_command}" --parsable \
  --account="${account}" \
  --qos="${qos}" \
  --job-name="almond_val_${frequency_key}_p${order}" \
  --nodes="${nodes}" \
  --ntasks="${ntasks}" \
  --cpus-per-task="${cpus_per_task}" \
  --time="${walltime}" \
  --array="0-${array_last}" \
  --output="${log_dir}/%x_%A_%a.out" \
  --error="${log_dir}/%x_%A_%a.err" \
  --export="ALL,AUSTIN_ARRAY_PHI_DEGREES=${phi_values//,/;},AUSTIN_ARRAY_POLARIZATIONS=${polarizations//,/;},FREQUENCY_KEY=${frequency_key},ORDER=${order},OUTPUT_DIR=${output_dir},POSTPROCESS=0,CASE_ONLY=1" \
  "${job_script}")"

echo "Submitted Austin monostatic validation array ${job_id}."
echo "Cases:       ${case_count} (${#phi_array[@]} angles x ${#polarization_array[@]} polarizations)"
echo "Angles:      ${phi_values}"
echo "Polarization:${polarizations}"
echo "Output:      ${output_dir}"
echo "Resources:   ${nodes} node(s), ${ntasks} tasks/case, ${walltime}/case"
echo
echo "After the array completes, aggregate with:"
echo "mpiexec -n 1 julia --project=. examples/distributed_pec_almond_austin_monostatic_rcs.jl --collect-only --phi-degrees=${phi_values} --polarization=${polarizations} --frequency-key=${frequency_key} --output-dir=${output_dir}"
echo "python3 examples/postprocess_pec_almond_austin_monostatic_rcs.py --run-dir=${output_dir} --polarization=${polarizations} --show-failed --write-png"
