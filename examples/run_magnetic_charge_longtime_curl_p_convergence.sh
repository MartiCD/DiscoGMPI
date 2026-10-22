#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MPIEXEC="${MPIEXEC:-mpiexec}"
ORDERS=(1 2 3 4)
FLUX_TYPE="alternating"
FINAL_TIME=1000.0
MESH_PATH="examples/meshes/periodic_box_structured_2x1x1.vtk"

for order in "${ORDERS[@]}"; do
    output_dir="output/magnetic_charge_invariance_test/curl_p${order}"
    esprk_order=$((order + 1))

    echo "Running magnetic-charge long-time case: order=${order}, output=${output_dir}"

    "${MPIEXEC}" -n 2 julia --project=. \
        examples/distributed_periodic_poisson_bracket_maxwell.jl \
        --mesh="${MESH_PATH}" \
        --partitions=2 \
        --order="${order}" \
        --esprk-order="${esprk_order}" \
        --final-time="${FINAL_TIME}" \
        --cfl=0.4 \
        --flux="${FLUX_TYPE}" \
        --magnetic-initialization=curl-potential \
        --print-every=1000 \
        --energy-every=1 \
        --full-diagnostics-every=10000 \
        --paraview-every=1000000000 \
        --checkpoint-every=10000 \
        --output-dir="${output_dir}"
done
