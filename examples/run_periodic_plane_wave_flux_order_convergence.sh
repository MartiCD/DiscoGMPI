#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

mpi_ranks="${MPI_RANKS:-8}"
output_dir="output/convergence"
flux_types=(alternating centered)
orders=(1 2 3 4)

mkdir -p "${output_dir}"

for flux in "${flux_types[@]}"; do
    for order in "${orders[@]}"; do
        output_file="${output_dir}/plane_wave_${flux}_structured_p${order}.csv"

        echo "Running flux=${flux}, p=${order}"
        mpiexec -n "${mpi_ranks}" julia --project=. \
            examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl \
            --mesh-family=structured \
            --nx-targets=2,4,8,16,32 \
            --orders="${order}" \
            --periods=1 \
            --cfl=0.8 \
            --cfl-divisor=4 \
            --flux="${flux}" \
            --error-mode=final \
            --rate-policy=report \
            --wave-number=3.141592653589793 \
            --output="${output_file}"
    done
done

