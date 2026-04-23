include("../src/DistributedMesh3D.jl")
using .DistributedMesh3D

# Global tetra mesh data
global_node_coords = [
    0.0 1.0 0.0 0.0 1.0
    0.0 0.0 1.0 0.0 1.0
    0.0 0.0 0.0 1.0 1.0
]  # (3, ntotnode=5)

global_elem_vertices = [
    1  2
    2  3
    3  4
    4  5
]  # (4, ntotelem=2)

elem_to_rank = [0, 1]
material_id_global = [10, 20]

mesh_rank0 = build_distributed_mesh(
    global_node_coords,
    global_elem_vertices,
    elem_to_rank,
    0;
    material_id_global=material_id_global,
)

mesh_rank1 = build_distributed_mesh(
    global_node_coords,
    global_elem_vertices,
    elem_to_rank,
    1;
    material_id_global=material_id_global,
)