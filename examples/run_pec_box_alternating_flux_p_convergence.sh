#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MPIEXEC="${MPIEXEC:-mpiexec}"
ORDERS=(1 2 3 4)

for order in "${ORDERS[@]}"; do
    esprk_order=$((order + 1))
    output_dir="output/pec_box_af_p${order}"

    echo "Running PEC box alternating-flux case: order=${order}, esprk-order=${esprk_order}, output=${output_dir}"

    "${MPIEXEC}" -n 2 julia --project=. \
        examples/distributed_poisson_bracket_maxwell.jl \
        --mesh=examples/meshes/tet_mesh.vtk \
        --order="${order}" \
        --esprk-order="${esprk_order}" \
        --final-time=100.0 \
        --cfl=0.11111111111 \
        --boundary-condition=pec \
        --flux=alternating \
        --output-dir="${output_dir}"
done
