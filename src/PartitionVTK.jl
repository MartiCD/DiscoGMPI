module PartitionVTK

export write_partition_vtk, write_partition_piece_vtks

"""
Write one global .vtk file with CELL_DATA partition_id.
Inputs:
  global_node_coords   :: 3 × Np
  global_elem_vertices :: 4 × K
  elem_to_rank         :: K, 0-based rank/partition id
"""
function write_partition_vtk(
    filename::AbstractString,
    global_node_coords::AbstractMatrix{T},
    global_elem_vertices::AbstractMatrix{Int},
    elem_to_rank::AbstractVector{Int};
    cell_data_name::AbstractString = "partition_id",
) where {T<:Real}

    @assert size(global_node_coords, 1) == 3
    @assert size(global_elem_vertices, 1) == 4
    K = size(global_elem_vertices, 2)
    Np = size(global_node_coords, 2)
    @assert length(elem_to_rank) == K

    open(filename, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "DiscoGMPI partitioned tetra mesh")
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        println(io, "POINTS $Np double")
        for n in 1:Np
            println(io, "$(global_node_coords[1,n]) $(global_node_coords[2,n]) $(global_node_coords[3,n])")
        end

        # each tet row: 4 node ids + leading cell size = 5 integers
        println(io, "CELLS $K $(5K)")
        for e in 1:K
            # VTK legacy uses 0-based point indices
            v = global_elem_vertices[:, e] .- 1
            println(io, "4 $(v[1]) $(v[2]) $(v[3]) $(v[4])")
        end

        # VTK_TETRA = 10
        println(io, "CELL_TYPES $K")
        for _ in 1:K
            println(io, "10")
        end

        println(io, "CELL_DATA $K")
        println(io, "SCALARS $cell_data_name int 1")
        println(io, "LOOKUP_TABLE default")
        for e in 1:K
            println(io, elem_to_rank[e])
        end
    end

    return filename
end

"""
Write one .vtk per partition/rank, containing only elements owned by that partition.
"""
function write_partition_piece_vtks(
    basename::AbstractString,
    global_node_coords::AbstractMatrix{T},
    global_elem_vertices::AbstractMatrix{Int},
    elem_to_rank::AbstractVector{Int},
) where {T<:Real}

    nranks = maximum(elem_to_rank) + 1
    files = String[]

    for r in 0:nranks-1
        elems = findall(==(r), elem_to_rank)
        isempty(elems) && continue

        used_nodes = sort!(collect(Set(vec(global_elem_vertices[:, elems]))))
        g2l = Dict(g => i for (i, g) in enumerate(used_nodes))

        coords_r = global_node_coords[:, used_nodes]
        elems_r = Matrix{Int}(undef, 4, length(elems))

        for (j, ge) in enumerate(elems)
            for a in 1:4
                elems_r[a, j] = g2l[global_elem_vertices[a, ge]]
            end
        end

        part_r = fill(r, length(elems))
        file = "$(basename)_rank$(r).vtk"
        write_partition_vtk(file, coords_r, elems_r, part_r)
        push!(files, file)
    end

    return files
end

end