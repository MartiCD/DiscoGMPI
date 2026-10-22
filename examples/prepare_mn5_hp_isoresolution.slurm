#!/bin/bash
#SBATCH --account=bsc115
#SBATCH --qos=gp_bsccase
#SBATCH --job-name=hp_meshes
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# MN5 assigns memory from requested CPUs. Reserve one full GPP node because
# preparation temporarily holds the finest unstructured mesh in memory.
#SBATCH --cpus-per-task=112
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"

module purge
module load mkl
module load python/3.10.15-gcc
module load intel/2023.0 impi/2021.8
module load hdf5 gcc opencascade
module load gmsh/4.13.1-occt

# Gmsh 4.13.1 requires GLIBCXX_3.4.32.
module swap gcc/12.3.0 gcc/13.2.0

gmsh_executable="$(readlink -f "$(command -v gmsh)")"
gmsh_root="$(dirname "$(dirname "${gmsh_executable}")")"
gmsh_python_dir="${gmsh_root}/lib64"

[[ -f "${gmsh_python_dir}/gmsh.py" ]] || {
    echo "Missing ${gmsh_python_dir}/gmsh.py" >&2
    exit 1
}

export PYTHONPATH="${gmsh_python_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON:-python3}" -c \
  'import gmsh, numpy; print("Gmsh", gmsh.__version__)'
prepare_args=(
    --output-dir "${OUTPUT_ROOT:-output/hp_isoresolution}"
    --levels "${LEVELS:-4}"
    --cfl "${CFL_EFFECTIVE:-0.2}"
    --periods "${PERIODS:-1}"
    --seed "${MESH_SEED:-1234}"
    --quality-min "${QUALITY_MIN:-0.1}"
    --elements-per-rank "${ELEMENTS_PER_RANK:-1000}"
)
# Mixed campaigns are uncapped unless MAX_RANKS or --max-ranks is supplied.
# The equal-order preparer retains its own parser default of 448 ranks.
[[ -z "${MAX_RANKS:-}" ]] || prepare_args+=(--max-ranks "${MAX_RANKS}")
exec "${PYTHON:-python3}" "${HP_PREPARE_DRIVER:-examples/prepare_hp_isoresolution.py}" \
    "${prepare_args[@]}" "$@"
