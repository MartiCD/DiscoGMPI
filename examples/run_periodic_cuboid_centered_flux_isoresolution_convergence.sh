#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MPIEXEC="${MPIEXEC:-mpiexec}"
GMSH="${GMSH:-gmsh}"
DEFAULT_NPROCS="${DEFAULT_NPROCS:-${NPROCS:-${RANKS:-1}}}"

FINAL_TIME="${FINAL_TIME:-1.0}"
CFL="${CFL:-0.8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/convergence}"
ENERGY_EVERY="${ENERGY_EVERY:-1}"
PARAVIEW_EVERY="${PARAVIEW_EVERY:-10}"
REGENERATE_MESHES="${REGENERATE_MESHES:-0}"
DRY_RUN="${DRY_RUN:-0}"

GEO_FILE="examples/meshes/periodic_box_structured.geo"
BASE_MESH="examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk"
BASE_NX=4
BASE_NY=2
BASE_NZ=2

# Isoresolution matrix:
#   P4M1 P2M1 P1M1
#   P4M2 P2M2 P1M2
#   P4M3 P2M3 P1M3
#   P4M4 P2M4 P1M4
#
# Consecutive M levels halve h. Consecutive polynomial levels in the same M
# level also halve h:
#   P2M1 uses h(P4M1)/2, P1M1 uses h(P4M1)/4.
MESH_LEVELS=(1 2 3 4)
ORDERS=(4)

# Per-case MPI rank overrides use NPROCS_P{order}_M{mesh_level}.
# Example:
#   NPROCS_P4_M1=1 NPROCS_P4_M2=2 NPROCS_P4_M3=4 ./examples/run_periodic_cuboid_centered_flux_isoresolution_convergence.sh
# If a case-specific variable is not set, DEFAULT_NPROCS is used.

validate_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got '${value}'." >&2
        exit 1
    fi
}

case_nprocs() {
    local order="$1"
    local mesh_level="$2"
    local variable_name="NPROCS_P${order}_M${mesh_level}"
    local value="${!variable_name-}"

    if [[ -z "${value}" ]]; then
        value="${DEFAULT_NPROCS}"
    fi

    validate_positive_integer "${variable_name}" "${value}"
    echo "${value}"
}

validate_positive_integer "DEFAULT_NPROCS" "${DEFAULT_NPROCS}"

if [[ ! -f "${GEO_FILE}" ]]; then
    echo "Missing structured Gmsh geometry: ${GEO_FILE}" >&2
    exit 1
fi

order_refinement_offset() {
    local order="$1"
    case "${order}" in
        4) echo 0 ;;
        2) echo 1 ;;
        1) echo 2 ;;
        *)
            echo "Unsupported order '${order}'. Expected one of: 4, 2, 1." >&2
            exit 1
            ;;
    esac
}

mesh_path_for_resolution() {
    local nx="$1"
    local ny="$2"
    local nz="$3"

    if [[ "${nx}" -eq "${BASE_NX}" && "${ny}" -eq "${BASE_NY}" && "${nz}" -eq "${BASE_NZ}" ]]; then
        echo "${BASE_MESH}"
    else
        echo "examples/meshes/periodic_box_structured_nx${nx}_ny${ny}_nz${nz}.vtk"
    fi
}

generate_mesh_if_needed() {
    local mesh_path="$1"
    local nx="$2"
    local ny="$3"
    local nz="$4"

    if [[ -f "${mesh_path}" && "${REGENERATE_MESHES}" != "1" ]]; then
        echo "Using existing mesh: ${mesh_path}"
        return
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] Generate mesh ${mesh_path} with nx=${nx}, ny=${ny}, nz=${nz}"
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

run_case() {
    local order="$1"
    local esprk_order="$2"
    local mesh_level="$3"
    local mesh_path="$4"
    local output_dir="$5"
    local nprocs="$6"

    echo "Running periodic cuboid centered-flux isoresolution case:"
    echo "  P${order}M${mesh_level}"
    echo "  ranks=${nprocs}, esprk-order=${esprk_order}, final-time=${FINAL_TIME}, cfl=${CFL}"
    echo "  mesh=${mesh_path}"
    echo "  output=${output_dir}"

    local command=(
        "${MPIEXEC}" -n "${nprocs}"
        julia --project=.
        examples/distributed_periodic_poisson_bracket_maxwell.jl
        --partitions="${nprocs}"
        --mesh="${mesh_path}"
        --order="${order}"
        --esprk-order="${esprk_order}"
        --final-time="${FINAL_TIME}"
        --cfl="${CFL}"
        --energy-every="${ENERGY_EVERY}"
        --paraview-every="${PARAVIEW_EVERY}"
        --flux=centered
        --output-dir="${output_dir}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run]'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

for mesh_level in "${MESH_LEVELS[@]}"; do
    for order in "${ORDERS[@]}"; do
        offset="$(order_refinement_offset "${order}")"
        effective_refinement=$((mesh_level - 1 + offset))
        nx=$((BASE_NX << effective_refinement))
        ny=$((BASE_NY << effective_refinement))
        nz=$((BASE_NZ << effective_refinement))
        mesh_path="$(mesh_path_for_resolution "${nx}" "${ny}" "${nz}")"
        esprk_order=$((order + 1))
        output_dir="${OUTPUT_ROOT}/periodic_cuboid_cf_p${order}_m${mesh_level}"
        nprocs="$(case_nprocs "${order}" "${mesh_level}")"

        generate_mesh_if_needed "${mesh_path}" "${nx}" "${ny}" "${nz}"
        run_case "${order}" "${esprk_order}" "${mesh_level}" "${mesh_path}" "${output_dir}" "${nprocs}"
    done
done
