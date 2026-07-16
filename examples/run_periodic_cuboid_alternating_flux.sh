#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MPIEXEC="${MPIEXEC:-mpiexec}"
ORDERS=(1 2 3 4)

for order in "${ORDERS[@]}"; do
    esprk_order=$((order + 1))
    output_dir="output/periodic_cuboid_af_p${order}"

    echo "Running periodic cuboid alternating-flux case: order=${order}, esprk-order=${esprk_order}, output=${output_dir}"

    "${MPIEXEC}" -n 1 julia --project=. \
        examples/distributed_periodic_poisson_bracket_maxwell.jl \
        --partitions=1 \
        --order="${order}" \
        --mesh=examples/meshes/periodic_box_unstructured_2x1x1.vtk \
        --esprk-order="${esprk_order}" \
        --final-time=100.0 \
        --cfl=0.8 \
        --flux=alternating \
        --output-dir="${output_dir}"
done
