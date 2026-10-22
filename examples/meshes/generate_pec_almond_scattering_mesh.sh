#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

geo="${script_dir}/pec_almond_scattering.geo"

surface="${ALMOND_SURFACE:-/home/mcircuns/CODE/AustinCEMBenchmarks/NASA-Almond_trimesh_repaired.stl}"
if [[ ! -f "${surface}" ]]; then
  fallback="${script_dir}/NASA-Almond_trimesh_repaired.stl"
  if [[ -f "${fallback}" ]]; then
    surface="${fallback}"
  else
    cat >&2 <<EOF
Could not find the almond STL surface.

Set ALMOND_SURFACE to a closed NASA almond STL, for example:
  ALMOND_SURFACE=/home/mcircuns/CODE/AustinCEMBenchmarks/NASA-Almond_trimesh_repaired.stl \\
    bash examples/meshes/generate_pec_almond_scattering_mesh.sh
EOF
    exit 1
  fi
fi

frequency_hz="${FREQUENCY_HZ:-3.5e9}"
elements_per_wavelength="${ELEMENTS_PER_WAVELENGTH:-4}"
air_buffer_wavelengths="${AIR_BUFFER_WAVELENGTHS:-1.0}"
pml_width_wavelengths="${PML_WIDTH_WAVELENGTHS:-1.0}"
output="${OUTPUT:-${script_dir}/pec_almond_scattering_fx1_epw${elements_per_wavelength}.vtk}"

gmsh_bin="${GMSH:-gmsh}"
tmp_geo="$(mktemp "${TMPDIR:-/tmp}/pec_almond_import.XXXXXX.geo")"
trap 'rm -f "${tmp_geo}"' EXIT

escaped_surface="${surface//\\/\\\\}"
escaped_surface="${escaped_surface//\"/\\\"}"
escaped_geo="${geo//\\/\\\\}"
escaped_geo="${escaped_geo//\"/\\\"}"

{
  printf 'Merge "%s";\n' "${escaped_surface}"
  printf 'Include "%s";\n' "${escaped_geo}"
} > "${tmp_geo}"

echo "Generating PEC almond scattering mesh"
echo "  geo:                    ${geo}"
echo "  import geo:             ${tmp_geo}"
echo "  surface:                ${surface}"
echo "  output:                 ${output}"
echo "  frequency Hz:           ${frequency_hz}"
echo "  elements/wavelength:    ${elements_per_wavelength}"
echo "  air buffer wavelengths: ${air_buffer_wavelengths}"
echo "  PML width wavelengths:  ${pml_width_wavelengths}"

"${gmsh_bin}" "${tmp_geo}" -3 -format vtk -o "${output}" \
  -setnumber FrequencyHz "${frequency_hz}" \
  -setnumber ElementsPerWavelength "${elements_per_wavelength}" \
  -setnumber AirBufferWavelengths "${air_buffer_wavelengths}" \
  -setnumber PMLWidthWavelengths "${pml_width_wavelengths}" \
  "$@"
