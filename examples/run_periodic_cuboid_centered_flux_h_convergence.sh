#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MPIEXEC="${MPIEXEC:-mpiexec}"
GMSH="${GMSH:-gmsh}"
RANKS="${RANKS:-4}"

ORDER="${ORDER:-1}"
ESPRK_ORDER="${ESPRK_ORDER:-$((ORDER + 1))}"
FINAL_TIME="${FINAL_TIME:-100.0}"
CFL="${CFL:-0.8}"
ENERGY_EVERY="${ENERGY_EVERY:-1}"
PARAVIEW_EVERY="${PARAVIEW_EVERY:-10}"
REGENERATE_MESHES="${REGENERATE_MESHES:-0}"

GEO_FILE="examples/meshes/periodic_box_structured.geo"

# h-refinement family. The first mesh is intentionally the same one used by
# run_periodic_cuboid_centered_flux_p_convergence.sh.
LEVELS=(0 1 2 3)
NX_TARGETS=(8 16 32 64)
NY_TARGETS=(4 8 16 32)
NZ_TARGETS=(4 8 16 32)
MESHES=(
    "examples/meshes/periodic_box_structured_2x1x1.vtk"
    "examples/meshes/periodic_box_structured_nx16_ny8_nz8.vtk"
    "examples/meshes/periodic_box_structured_nx32_ny16_nz16.vtk"
    "examples/meshes/periodic_box_structured_nx64_ny32_nz32.vtk"
)

if [[ "${ORDER}" -lt 1 ]]; then
    echo "ORDER must be at least 1, got ${ORDER}." >&2
    exit 1
fi

if [[ "${ESPRK_ORDER}" -lt 1 || "${ESPRK_ORDER}" -gt 6 ]]; then
    echo "ESPRK_ORDER must be in 1:6, got ${ESPRK_ORDER}." >&2
    exit 1
fi

if [[ ! -f "${GEO_FILE}" ]]; then
    echo "Missing structured Gmsh geometry: ${GEO_FILE}" >&2
    exit 1
fi

generate_mesh_if_needed() {
    local mesh_path="$1"
    local nx="$2"
    local ny="$3"
    local nz="$4"

    if [[ -f "${mesh_path}" && "${REGENERATE_MESHES}" != "1" ]]; then
        echo "Using existing mesh: ${mesh_path}"
        return
    fi

    echo "Generating structured periodic mesh: ${mesh_path} (nx=${nx}, ny=${ny}, nz=${nz})"
    "${GMSH}" "${GEO_FILE}" \
        -3 \
        -format vtk \
        -o "${mesh_path}" \
        -setnumber NxTarget "${nx}" \
        -setnumber NyTarget "${ny}" \
        -setnumber NzTarget "${nz}"
}

for index in "${!LEVELS[@]}"; do
    level="${LEVELS[index]}"
    nx="${NX_TARGETS[index]}"
    ny="${NY_TARGETS[index]}"
    nz="${NZ_TARGETS[index]}"
    mesh="${MESHES[index]}"
    output_dir="output/periodic_cuboid_cf_structured_p${ORDER}_h${level}"

    generate_mesh_if_needed "${mesh}" "${nx}" "${ny}" "${nz}"

    echo "Running periodic cuboid centered-flux h-convergence case:"
    echo "  level=${level}, nx=${nx}, ny=${ny}, nz=${nz}"
    echo "  order=${ORDER}, esprk-order=${ESPRK_ORDER}, ranks=${RANKS}"
    echo "  mesh=${mesh}"
    echo "  output=${output_dir}"

    "${MPIEXEC}" -n "${RANKS}" julia --project=. \
        examples/distributed_periodic_poisson_bracket_maxwell.jl \
        --partitions="${RANKS}" \
        --mesh="${mesh}" \
        --order="${ORDER}" \
        --esprk-order="${ESPRK_ORDER}" \
        --final-time="${FINAL_TIME}" \
        --cfl="${CFL}" \
        --energy-every="${ENERGY_EVERY}" \
        --paraview-every="${PARAVIEW_EVERY}" \
	--output-dir="${output_dir}" \

done
