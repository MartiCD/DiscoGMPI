#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export MESH_PATH="${1:?mesh path}" FLUX="${2:?flux}" OUTPUT_DIR="${3:?output directory}"
export ORDER=2 ESPRK_ORDER=3
export MAGNETIC_INITIALIZATION=interpolate ELECTRIC_INITIALIZATION=interpolate
export FINAL_TIME="${FINAL_TIME:-10000.0}" CFL=0.2
export PRINT_EVERY="${PRINT_EVERY:-1000}" ENERGY_EVERY="${ENERGY_EVERY:-1}"
export FULL_DIAGNOSTICS_EVERY="${FULL_DIAGNOSTICS_EVERY:-10000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
export PARAVIEW_EVERY="${PARAVIEW_EVERY:-1000000000}"
export PREP_ROOT="${PREP_ROOT:-${OUTPUT_DIR}/mesh_cache/job${SLURM_JOB_ID:-local}}"
exec bash examples/run_mn5_magnetic_charge_longtime_curl_p_convergence.slurm
