#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --job-name=mixed_hp_meshes
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export HP_PREPARE_DRIVER=examples/prepare_mixed_hp_isoresolution.py
export OUTPUT_ROOT="${OUTPUT_ROOT:-output/mixed_hp_isoresolution}"
export CFL_EFFECTIVE="${CFL_EFFECTIVE:-0.1}"
# Share the existing MN5 Gmsh environment setup; its SBATCH lines are comments
# when invoked by bash, so the resource request above remains in effect.
exec bash examples/prepare_mn5_hp_isoresolution.slurm "$@"
