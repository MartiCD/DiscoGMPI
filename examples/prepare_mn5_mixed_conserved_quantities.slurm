#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --job-name=mixed_inv_meshes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
# Same mesh-generation environment as the equal-order h campaign.
module purge
module load mkl python/3.10.15-gcc
module load intel/2023.0 impi/2021.8
module load hdf5 gcc opencascade
module load gmsh/4.13.1-occt julia/1.10.0
module swap gcc/12.3.0 gcc/13.2.0
gmsh_executable="$(readlink -f "$(command -v gmsh)")"
gmsh_root="$(dirname "$(dirname "$gmsh_executable")")"
export PYTHONPATH="${gmsh_root}/lib64${PYTHONPATH:+:${PYTHONPATH}}"
export JULIA_DEPOT_PATH="${PWD}/.julia_depot:${JULIA_DEPOT_PATH:-}"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 examples/manage_mixed_conserved_quantities.py prepare \
    --root "${OUTPUT_ROOT:-output/mixed_conserved_quantities}" "$@"
