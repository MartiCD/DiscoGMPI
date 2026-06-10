# Run from repo root:
#   julia --project=. examples/write_metis.jl 
#
# Generates:
#   examples/meshes/tet_mesh.mesh

using DiscoGMPI

outdir = joinpath(@__DIR__, "meshes")
mkpath(outdir)

input_path = "examples/meshes/tet_mesh.vtk"
output_path = "examples/meshes/tet_mesh.mesh"

write_metis_mesh_from_vtk(input_path, output_path)