# Run from repo root:
#   julia --project=. examples/write_partition_vtk.jl
#
# Generates:
#   examples/output/partitioned_mesh.vtk
#   examples/output/partition_mesh_rank0.vtk
#   examples/output/partition_mesh_rank1.vtk

using DiscoGMPI

outdir = joinpath(@__DIR__, "visualization")
mkpath(outdir)


global_node_coords, global_elem_vertices = read_mesh_file_tet_vtk("examples/meshes/tet_mesh.vtk")
global_node_coords, global_elem_vertices = global_node_coords', global_elem_vertices'

@show size(global_node_coords, 1)
@show size(global_node_coords, 2)

elem_to_rank = read_metis_epart("examples/meshes/tet_mesh.mesh.epart.2")

# ------------------------------------------------------------
# 1. Full mesh with partition_id as CELL_DATA
# ------------------------------------------------------------

full_vtk = joinpath(outdir, "partitioned_mesh.vtk")

write_partition_vtk(
    full_vtk,
    global_node_coords,
    global_elem_vertices,
    elem_to_rank;
    cell_data_name = "partition_id",
)

# ------------------------------------------------------------
# 2. One VTK file per partition
# ------------------------------------------------------------

piece_basename = joinpath(outdir, "partition_mesh")

piece_files = write_partition_piece_vtks(
    piece_basename,
    global_node_coords,
    global_elem_vertices,
    elem_to_rank,
)

println("Generated VTK files:")
println("  ", full_vtk)
foreach(f -> println("  ", f), piece_files)

println()
println("Open partitioned_mesh.vtk in ParaView and color by `partition_id`.")
println("Use Surface With Edges for a clear partitioned tetra visualization.")