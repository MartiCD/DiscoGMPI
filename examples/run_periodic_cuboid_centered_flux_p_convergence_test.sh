#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MPIEXEC="${MPIEXEC:-mpiexec}"
ORDERS=(1 2 3 4)
FINAL_TIME=10.0

for order in "${ORDERS[@]}"; do
    esprk_order=$((order + 1))
    output_dir="output/test_periodic_cuboid_cf_structured_p${order}"

    echo "Running periodic cuboid centered-flux case: order=${order}, esprk-order=${esprk_order}, output=${output_dir}"

    "${MPIEXEC}" -n 1 julia --project=. \
        examples/distributed_periodic_poisson_bracket_maxwell.jl \
        --partitions=1 \
        --order="${order}" \
    	--mesh=examples/meshes/periodic_box_structured_2x1x1.vtk \
        --esprk-order="${esprk_order}" \
        --final-time="${FINAL_TIME}" \
        --cfl=0.8 \
        --flux=centered \
	--paraview-every=10000 \
        --output-dir="${output_dir}"
done

python3 examples/plot_pec_box_alternating_flux_panel.py \
  --prefix=test_periodic_cuboid_cf_structured_p \
  --summary-output=output/periodic_cuboid_cf_structured_p_convergence_invariants_panel.pdf \
  --components-output=output/periodic_cuboid_cf_structured_p_convergence_momentum_components_panel.pdf \
  --error-output=output/periodic_cuboid_cf_structured_p_convergence_invariants_error_panel.pdf \
  --divergence-output=output/periodic_cuboid_cf_structured_p_convergence_divergence_constraints_panel.pdf
