module MetisIO

export write_metis_mesh_from_vtk,
        read_mesh_file_tet_vtk

function write_metis_mesh_from_vtk(vtk_path::AbstractString, metis_path::AbstractString)
    coords, elems = read_mesh_file_tet_vtk(vtk_path)

    if size(elems, 1) != 4 && size(elems, 2) == 4
        elems = permutedims(elems)
    end

    @assert size(elems, 1) == 4 "Only tetrahedral meshes supported"

    nelems = size(elems, 2)

    open(metis_path, "w") do io
        println(io, nelems)
        for e in 1:nelems
            println(io, join(elems[:, e], " "))
        end
    end

    return metis_path
end

function read_mesh_file_tet_vtk(file_path::Union{String,SubString{String}})
    lines = readlines(file_path)

    ndofs = 0
    coords_flat = Float64[]

    total_cells = 0
    all_cells = Vector{Vector{Int}}()
    cell_types = Int[]
    cell_tags  = Int[]

    i = 1
    while i <= length(lines)
        line = strip(lines[i])

        if isempty(line)
            i += 1
            continue
        end

        if startswith(line, "POINTS")
            words = split(line)
            ndofs = parse(Int, words[2])

            needed = 3 * ndofs
            i += 1
            while length(coords_flat) < needed
                s = strip(lines[i])
                if !isempty(s)
                    append!(coords_flat, parse.(Float64, split(s)))
                end
                i += 1
            end
            continue
        end

        if startswith(line, "CELLS")
            words = split(line)
            total_cells = parse(Int, words[2])

            i += 1
            for _ in 1:total_cells
                words = split(strip(lines[i]))
                nv = parse(Int, words[1])
                conn = parse.(Int, words[2:end]) .+ 1
                push!(all_cells, conn)
                i += 1
            end
            continue
        end

        if startswith(line, "CELL_TYPES")
            nct = parse(Int, split(line)[2])
            i += 1
            while length(cell_types) < nct
                s = strip(lines[i])
                if !isempty(s)
                    append!(cell_types, parse.(Int, split(s)))
                end
                i += 1
            end
            continue
        end

        if startswith(line, "CELL_DATA")
            ncd = parse(Int, split(line)[2])
            i += 1

            while i <= length(lines) && !startswith(strip(lines[i]), "SCALARS")
                i += 1
            end
            i += 1  # skip SCALARS line

            if startswith(strip(lines[i]), "LOOKUP_TABLE")
                i += 1
            end

            while length(cell_tags) < ncd
                s = strip(lines[i])
                if !isempty(s)
                    append!(cell_tags, parse.(Int, split(s)))
                end
                i += 1
            end
            continue
        end

        i += 1
    end

    coords = reshape(coords_flat, 3, ndofs)

    length(all_cells) == total_cells ||
        error(
            "VTK CELLS section declares $total_cells cells but contains " *
            "$(length(all_cells)).",
        )
    length(cell_types) == total_cells ||
        error(
            "VTK CELL_TYPES section contains $(length(cell_types)) entries " *
            "for $total_cells cells.",
        )
    isempty(cell_tags) || length(cell_tags) == total_cells ||
        error(
            "VTK CELL_DATA scalar array contains $(length(cell_tags)) entries " *
            "for $total_cells cells.",
        )

    tri_cells = Int[]
    tri_tags  = Int[]
    tet_cells = Int[]
    tet_tags  = Int[]

    for c in 1:total_cells
        ctype = cell_types[c]
        conn  = all_cells[c]
        tag   = isempty(cell_tags) ? 0 : cell_tags[c]

        if ctype == 5 && length(conn) == 3
            append!(tri_cells, conn)
            push!(tri_tags, tag)
        elseif ctype == 10 && length(conn) == 4
            append!(tet_cells, conn)
            push!(tet_tags, tag)
        end
    end

    bfaces = isempty(tri_cells) ? zeros(Int, 3, 0) :
        reshape(tri_cells, 3, :)

    EToN = isempty(tet_cells) ? zeros(Int, 4, 0) :
        reshape(tet_cells, 4, :)

    # return ndofs, coords, size(EToN,1), EToN, tet_tags, bfaces, tri_tags
    return coords, EToN
end


end # module MetisIO
