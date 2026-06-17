#!/usr/bin/env julia

const DEFAULT_NX = 4
const LX = 2.0
const LY = 1.0
const LZ = 1.0

const TET_FACES = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)

function usage(io::IO = stdout)
    println(io, """
Generate a periodic [0,2] x [0,1] x [0,1] tetrahedral VTK mesh.

Usage:
  julia examples/meshes/generate_periodic_box_2x1x1.jl [options]

Options:
  --nx N          Number of structured cells in the x direction.
                  Each cell is split into six tetrahedra. Default: 4
  --output PATH   Output VTK path.
                  Default: examples/meshes/periodic_box_2x1x1_nxN.vtk
  --help          Show this message.

Periodic boundary tags:
  x-min/x-max: 1/2
  y-min/y-max: 3/4
  z-min/z-max: 5/6
""")
end

function option_value(args::Vector{String}, index::Int, option::String)
    argument = args[index]
    prefix = "$option="
    if startswith(argument, prefix)
        value = argument[(length(prefix) + 1):end]
        isempty(value) && error("$option requires a value.")
        return value, index
    end
    index < length(args) || error("$option requires a value.")
    return args[index + 1], index + 1
end

function parse_args(args::Vector{String})
    nx = DEFAULT_NX
    output = nothing
    index = 1

    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            usage()
            exit(0)
        elseif argument == "--nx" || startswith(argument, "--nx=")
            value, index = option_value(args, index, "--nx")
            nx = try
                parse(Int, value)
            catch
                error("--nx must be an integer, got '$value'.")
            end
        elseif argument == "--output" || startswith(argument, "--output=")
            output, index = option_value(args, index, "--output")
        else
            error("Unknown argument '$argument'. Use --help for usage.")
        end
        index += 1
    end

    nx >= 1 || error("--nx must be at least 1, got $nx.")
    if output === nothing
        output = joinpath(
            @__DIR__,
            "periodic_box_2x1x1_nx$nx.vtk",
        )
    end
    return nx, String(output)
end

node_id(i, j, k, nx) = 1 + i + (nx + 1) * (j + 2 * k)

function build_points(nx::Int)
    points = zeros(Float64, 3, (nx + 1) * 2 * 2)
    for k in 0:1, j in 0:1, i in 0:nx
        points[:, node_id(i, j, k, nx)] .=
            (LX * i / nx, LY * j, LZ * k)
    end
    return points
end

function build_tetrahedra(nx::Int)
    tetrahedra = NTuple{4, Int}[]
    for i in 0:(nx - 1)
        v000 = node_id(i, 0, 0, nx)
        v100 = node_id(i + 1, 0, 0, nx)
        v010 = node_id(i, 1, 0, nx)
        v110 = node_id(i + 1, 1, 0, nx)
        v001 = node_id(i, 0, 1, nx)
        v101 = node_id(i + 1, 0, 1, nx)
        v011 = node_id(i, 1, 1, nx)
        v111 = node_id(i + 1, 1, 1, nx)

        append!(
            tetrahedra,
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            ),
        )
    end
    return tetrahedra
end

sorted_face(face) = Tuple(sort(collect(face)))

function boundary_triangles(tetrahedra)
    counts = Dict{NTuple{3, Int}, Int}()
    oriented = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for tet in tetrahedra, local_face in TET_FACES
        face = (
            tet[local_face[1]],
            tet[local_face[2]],
            tet[local_face[3]],
        )
        key = sorted_face(face)
        counts[key] = get(counts, key, 0) + 1
        oriented[key] = face
    end

    triangles = [
        oriented[key] for (key, count) in counts if count == 1
    ]
    sort!(triangles; by = sorted_face)
    return triangles
end

function boundary_tag(points, triangle)
    centroid = (
        sum(points[1, collect(triangle)]) / 3,
        sum(points[2, collect(triangle)]) / 3,
        sum(points[3, collect(triangle)]) / 3,
    )
    tolerance = 1e-12

    abs(centroid[1]) < tolerance && return 1
    abs(centroid[1] - LX) < tolerance && return 2
    abs(centroid[2]) < tolerance && return 3
    abs(centroid[2] - LY) < tolerance && return 4
    abs(centroid[3]) < tolerance && return 5
    abs(centroid[3] - LZ) < tolerance && return 6
    error("Could not classify boundary triangle $triangle at $centroid.")
end

function write_vtk(path, points, tetrahedra, triangles, boundary_tags)
    ncells = length(triangles) + length(tetrahedra)
    connectivity_size = 4 * length(triangles) + 5 * length(tetrahedra)
    dataset_name = splitext(basename(path))[1]

    open(path, "w") do io
        println(io, "# vtk DataFile Version 2.0")
        println(io, dataset_name)
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")
        println(io, "POINTS $(size(points, 2)) double")
        for point in eachcol(points)
            println(io, "$(point[1]) $(point[2]) $(point[3])")
        end

        println(io)
        println(io, "CELLS $ncells $connectivity_size")
        for triangle in triangles
            zero_based = collect(triangle) .- 1
            println(io, "3 ", join(zero_based, " "))
        end
        for tet in tetrahedra
            zero_based = collect(tet) .- 1
            println(io, "4 ", join(zero_based, " "))
        end

        println(io)
        println(io, "CELL_TYPES $ncells")
        foreach(_ -> println(io, 5), triangles)
        foreach(_ -> println(io, 10), tetrahedra)

        println(io)
        println(io, "CELL_DATA $ncells")
        println(io, "SCALARS CellEntityIds int 1")
        println(io, "LOOKUP_TABLE default")
        foreach(tag -> println(io, tag), boundary_tags)
        foreach(_ -> println(io, 7), tetrahedra)

        println(io, "SCALARS boundary_id int 1")
        println(io, "LOOKUP_TABLE default")
        foreach(tag -> println(io, tag), boundary_tags)
        foreach(_ -> println(io, 0), tetrahedra)

        println(io, "SCALARS material_id int 1")
        println(io, "LOOKUP_TABLE default")
        foreach(_ -> println(io, 0), triangles)
        foreach(_ -> println(io, 1), tetrahedra)
    end
end

function main(args::Vector{String})
    nx, output = parse_args(args)
    points = build_points(nx)
    tetrahedra = build_tetrahedra(nx)
    triangles = boundary_triangles(tetrahedra)
    boundary_tags = [
        boundary_tag(points, triangle) for triangle in triangles
    ]

    mkpath(dirname(abspath(output)))
    write_vtk(output, points, tetrahedra, triangles, boundary_tags)

    println("Wrote $output")
    println("x-direction cells:  ", nx)
    println("points:             ", size(points, 2))
    println("tetrahedra:         ", length(tetrahedra))
    println("boundary triangles: ", length(triangles))
    for tag in 1:6
        println("boundary tag $tag:     ", count(==(tag), boundary_tags))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
