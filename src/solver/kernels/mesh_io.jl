

"""
    read_cell_data(vtk)

Read all cell-data arrays from the VTK file.
"""
function read_cell_data(vtk)
    data = Dict{String, Any}()

    cd = get_cell_data(vtk)

    for name in keys(cd)
        data[String(name)] = collect(get_data(cd[name]))
    end

    return data
end


"""
    split_tets_and_tris(vtk)

Extract tetrahedral cells and triangular cells from the VTK mesh.

Returns
-------
- `tets`
- `tris`
- `tet_cell_ids`
- `tri_cell_ids`
"""
function split_tets_and_tris(vtk)
    cells = to_meshcells(get_cells(vtk))

    tet_cols = Vector{NTuple{4, Int}}()
    tri_cols = Vector{NTuple{3, Int}}()

    tet_cell_ids = Int[]
    tri_cell_ids = Int[]

    for (cid, cell) in enumerate(cells)
        ctype = cell.ctype
        conn = Tuple(Int.(cell.connectivity))

        if ctype == VTKCellTypes.VTK_TETRA
            if length(conn) != 4
                error("VTK_TETRA cell $cid does not have 4 nodes.")
            end

            push!(tet_cols, conn)
            push!(tet_cell_ids, cid)

        elseif ctype == VTKCellTypes.VTK_TRIANGLE
            if length(conn) != 3
                error("VTK_TRIANGLE cell $cid does not have 3 nodes.")
            end

            push!(tri_cols, conn)
            push!(tri_cell_ids, cid)

        else
            # Ignore other cell types for now.
            # Later you may want to support lines, quads, wedges, etc.
        end
    end

    tets = isempty(tet_cols) ? Matrix{Int}(undef, 4, 0) :
           reduce(hcat, collect.(tet_cols))

    tris = isempty(tri_cols) ? Matrix{Int}(undef, 3, 0) :
           reduce(hcat, collect.(tri_cols))

    return tets, tris, tet_cell_ids, tri_cell_ids
end


"""
    read_vtu_mesh(filename)

Read a VTU tetrahedral mesh.

Example
-------
```julia
mesh = read_vtu_mesh("box_with_sphere.vtu")
```
"""
function read_vtu_mesh(filename::AbstractString)
    if !isfile(filename)
        error("File not found: $filename")
    end

    vtk = VTKFile(filename)

    points = Matrix{Float64}(get_points(vtk))

    tets, tris, tet_cell_ids, tri_cell_ids = split_tets_and_tris(vtk)

    cell_data = read_cell_data(vtk)

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        cell_data,
    )

    check_mesh_consistency(mesh)

    return mesh
end


"""
    list_cell_data(mesh)

Print the available cell-data arrays.
"""
function list_cell_data(mesh::RawVTUMesh)
    println("Available cell-data arrays:")

    if isempty(mesh.cell_data)
        println("  none")
        return nothing
    end

    for key in sort(collect(keys(mesh.cell_data)))
        values = mesh.cell_data[key]
        println("  - ", key, " :: length = ", length(values), ", type = ", typeof(values))
    end

    return nothing
end


"""
    print_mesh_summary(mesh)

Print a simple summary of the raw VTU mesh.
"""
function print_mesh_summary(mesh::RawVTUMesh)
    println("Raw VTU mesh")
    println("------------")
    println("Number of points:              ", size(mesh.points, 2))
    println("Number of tetrahedra:          ", size(mesh.tets, 2))
    println("Number of surface triangles:   ", size(mesh.tris, 2))
    println("Number of original tet cells:  ", length(mesh.tet_cell_ids))
    println("Number of original tri cells:  ", length(mesh.tri_cell_ids))
    println()

    list_cell_data(mesh)

    return nothing
end


"""
    tet_data(mesh, name)

Extract a cell-data array restricted to tetrahedral cells.

Example
-------
```julia
region_ids = tet_data(mesh, "region_id")
```
"""
function tet_data(mesh::RawVTUMesh, name::String)
    if !haskey(mesh.cell_data, name)
        error("Cell-data array '$name' not found.")
    end

    data = mesh.cell_data[name]

    return data[mesh.tet_cell_ids]
end


"""
    tri_data(mesh, name)

Extract a cell-data array restricted to triangular surface cells.

Example
-------
```julia
boundary_ids = tri_data(mesh, "boundary_id")
```
"""
function tri_data(mesh::RawVTUMesh, name::String)
    if !haskey(mesh.cell_data, name)
        error("Cell-data array '$name' not found.")
    end

    data = mesh.cell_data[name]

    return data[mesh.tri_cell_ids]
end


"""
    check_mesh_consistency(mesh)

Basic sanity checks.
"""
function check_mesh_consistency(mesh::RawVTUMesh)
    np = size(mesh.points, 2)

    if size(mesh.points, 1) != 3
        error("Expected points to have size 3 × Np.")
    end

    if size(mesh.tets, 1) != 4
        error("Expected tets to have size 4 × Nt.")
    end

    if size(mesh.tris, 1) != 3
        error("Expected tris to have size 3 × Ns.")
    end

    if size(mesh.tets, 2) != length(mesh.tet_cell_ids)
        error("Mismatch between number of tetrahedra and tet_cell_ids.")
    end

    if size(mesh.tris, 2) != length(mesh.tri_cell_ids)
        error("Mismatch between number of triangles and tri_cell_ids.")
    end

    if !isempty(mesh.tets)
        min_tet_id = minimum(mesh.tets)
        max_tet_id = maximum(mesh.tets)

        if min_tet_id < 1 || max_tet_id > np
            error(
                "Tetrahedral connectivity contains invalid node ids. " *
                "Valid range is 1:$np, got $min_tet_id:$max_tet_id."
            )
        end
    end

    if !isempty(mesh.tris)
        min_tri_id = minimum(mesh.tris)
        max_tri_id = maximum(mesh.tris)

        if min_tri_id < 1 || max_tri_id > np
            error(
                "Triangle connectivity contains invalid node ids. " *
                "Valid range is 1:$np, got $min_tri_id:$max_tri_id."
            )
        end
    end

    return nothing
end
