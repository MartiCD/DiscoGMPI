#!/usr/bin/env julia

# Generate an unstructured, periodic tetrahedral mesh of
# [0,2] x [0,1] x [0,1] with Gmsh.

const DEFAULT_NX = 8
const DEFAULT_NY = 4
const DEFAULT_NZ = 4
const LX = 2.0
const LY = 1.0
const LZ = 1.0

function usage(io::IO = stdout)
    println(io, """
Generate an unstructured periodic [0,2] x [0,1] x [0,1] tetrahedral VTK mesh.

Usage:
  julia --project=. \\
    examples/meshes/generate_periodic_box_2x1x1_unstructured.jl [options]

Options:
  --nx N          Target number of elements along x. Default: 8
  --ny N          Target number of elements along y. Default: 4
  --nz N          Target number of elements along z. Default: 4
  --output PATH   Output legacy VTK path. Default:
                  examples/meshes/periodic_box_2x1x1_unstructured_nx8_ny4_nz4.vtk
  --gmsh PATH     Gmsh executable. Default: gmsh found in PATH
  --help          Show this message.

The directional counts define target spacings. Gmsh creates an unstructured
3D Delaunay tetrahedralization, so they are not structured-cell counts.

Periodic boundary tags:
  x-min/x-max: 1/2
  y-min/y-max: 3/4
  z-min/z-max: 5/6
""")
end

function option_value(
    args::Vector{String},
    index::Int,
    option::String,
)
    argument = args[index]
    prefix = "$option="
    if startswith(argument, prefix)
        value = argument[(length(prefix) + 1):end]
        isempty(value) && error("$option requires a value.")
        return value, index
    end
    argument == option ||
        error("Unknown argument '$argument'. Use --help for usage.")
    index < length(args) || error("$option requires a value.")
    return args[index + 1], index + 1
end

function parse_positive_integer(value::String, option::String)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        error("$option must be an integer, got '$value'.")
    parsed >= 1 ||
        error("$option must be at least 1, got $parsed.")
    return parsed
end

function parse_args(args::Vector{String})
    nx = DEFAULT_NX
    ny = DEFAULT_NY
    nz = DEFAULT_NZ
    output = nothing
    gmsh = nothing
    index = 1

    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            usage()
            return nothing
        elseif argument == "--nx" || startswith(argument, "--nx=")
            value, index = option_value(args, index, "--nx")
            nx = parse_positive_integer(value, "--nx")
        elseif argument == "--ny" || startswith(argument, "--ny=")
            value, index = option_value(args, index, "--ny")
            ny = parse_positive_integer(value, "--ny")
        elseif argument == "--nz" || startswith(argument, "--nz=")
            value, index = option_value(args, index, "--nz")
            nz = parse_positive_integer(value, "--nz")
        elseif argument == "--output" ||
               startswith(argument, "--output=")
            output, index = option_value(args, index, "--output")
        elseif argument == "--gmsh" || startswith(argument, "--gmsh=")
            gmsh, index = option_value(args, index, "--gmsh")
        else
            error("Unknown argument '$argument'. Use --help for usage.")
        end
        index += 1
    end

    if output === nothing
        output = joinpath(
            @__DIR__,
            "periodic_box_2x1x1_unstructured_nx$(nx)_ny$(ny)_nz$(nz).vtk",
        )
    end

    if gmsh === nothing
        gmsh = Sys.which("gmsh")
        gmsh === nothing &&
            error(
                "Gmsh was not found in PATH. Install Gmsh or pass " *
                "--gmsh=/path/to/gmsh.",
            )
    else
        gmsh = abspath(String(gmsh))
        isfile(gmsh) || error("Gmsh executable not found: $gmsh")
    end

    return (
        nx = nx,
        ny = ny,
        nz = nz,
        output = abspath(String(output)),
        gmsh = String(gmsh),
    )
end

function gmsh_geometry_text(nx::Int, ny::Int, nz::Int)
    hx = LX / nx
    hy = LY / ny
    hz = LZ / nz
    target_h = min(hx, hy, hz)
    tolerance = 1e-8

    return """
SetFactory("OpenCASCADE");

Lx = $LX;
Ly = $LY;
Lz = $LZ;
h = $target_h;
eps = $tolerance;

Box(1) = {0, 0, 0, Lx, Ly, Lz};

xmin[] = Surface In BoundingBox {-eps, -eps, -eps, eps, Ly + eps, Lz + eps};
xmax[] = Surface In BoundingBox {Lx - eps, -eps, -eps, Lx + eps, Ly + eps, Lz + eps};
ymin[] = Surface In BoundingBox {-eps, -eps, -eps, Lx + eps, eps, Lz + eps};
ymax[] = Surface In BoundingBox {-eps, Ly - eps, -eps, Lx + eps, Ly + eps, Lz + eps};
zmin[] = Surface In BoundingBox {-eps, -eps, -eps, Lx + eps, Ly + eps, eps};
zmax[] = Surface In BoundingBox {-eps, -eps, Lz - eps, Lx + eps, Ly + eps, Lz + eps};

// OpenCASCADE Box(1) curve groups by direction. Only the box edges are
// transfinite; all surface interiors and the volume remain unstructured.
Transfinite Curve {9, 10, 11, 12} = $(nx + 1);
Transfinite Curve {2, 4, 6, 8} = $(ny + 1);
Transfinite Curve {1, 3, 5, 7} = $(nz + 1);

Physical Surface(1) = {xmin[]};
Physical Surface(2) = {xmax[]};
Physical Surface(3) = {ymin[]};
Physical Surface(4) = {ymax[]};
Physical Surface(5) = {zmin[]};
Physical Surface(6) = {zmax[]};
Physical Volume(7) = {1};

// OpenCASCADE Box(1) creates the faces in x-, x+, y-, y+, z-, z+ order.
// Explicit IDs are required here: Gmsh does not expand the bounding-box
// arrays reliably inside Periodic Surface statements.
Periodic Surface {2} = {1} Translate {Lx, 0, 0};
Periodic Surface {4} = {3} Translate {0, Ly, 0};
Periodic Surface {6} = {5} Translate {0, 0, Lz};

Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.ElementOrder = 1;
Mesh.Optimize = 1;
Mesh.SaveAll = 1;
Mesh.Binary = 0;
"""
end

function read_ascii_legacy_vtk(path::String)
    lines = readlines(path)
    coordinates = Float64[]
    points = zeros(Float64, 3, 0)
    cells = Vector{Vector{Int}}()
    cell_types = Int[]
    index = 1

    while index <= length(lines)
        line = strip(lines[index])

        if startswith(line, "POINTS")
            words = split(line)
            number_points = parse(Int, words[2])
            needed = 3 * number_points
            index += 1
            while length(coordinates) < needed
                append!(coordinates, parse.(Float64, split(lines[index])))
                index += 1
            end
            points = reshape(coordinates, 3, number_points)
            continue
        elseif startswith(line, "CELLS")
            number_cells = parse(Int, split(line)[2])
            index += 1
            for _ in 1:number_cells
                words = split(strip(lines[index]))
                number_nodes = parse(Int, words[1])
                nodes = parse.(Int, words[2:end]) .+ 1
                length(nodes) == number_nodes ||
                    error("Malformed VTK cell in $path.")
                push!(cells, nodes)
                index += 1
            end
            continue
        elseif startswith(line, "CELL_TYPES")
            number_types = parse(Int, split(line)[2])
            index += 1
            while length(cell_types) < number_types
                append!(cell_types, parse.(Int, split(lines[index])))
                index += 1
            end
            continue
        end

        index += 1
    end

    size(points, 2) > 0 || error("Gmsh VTK output contains no points.")
    length(cells) == length(cell_types) ||
        error("Gmsh VTK cell and cell-type counts differ.")
    return points, cells, cell_types
end

function signed_tet_volume6(
    points::Matrix{Float64},
    tet::NTuple{4, Int},
)
    a = @view points[:, tet[1]]
    b = @view points[:, tet[2]]
    c = @view points[:, tet[3]]
    d = @view points[:, tet[4]]
    bax, bay, baz = b .- a
    cax, cay, caz = c .- a
    dax, day, daz = d .- a
    return (
        bax * (cay * daz - caz * day) -
        bay * (cax * daz - caz * dax) +
        baz * (cax * day - cay * dax)
    )
end

function extract_linear_cells(
    points::Matrix{Float64},
    cells::Vector{Vector{Int}},
    cell_types::Vector{Int},
)
    triangles = NTuple{3, Int}[]
    tetrahedra = NTuple{4, Int}[]

    for (nodes, cell_type) in zip(cells, cell_types)
        if cell_type == 5 && length(nodes) == 3
            push!(triangles, (nodes[1], nodes[2], nodes[3]))
        elseif cell_type == 10 && length(nodes) == 4
            tet = (nodes[1], nodes[2], nodes[3], nodes[4])
            volume6 = signed_tet_volume6(points, tet)
            abs(volume6) > 1e-14 ||
                error("Gmsh generated a degenerate tetrahedron $tet.")
            if volume6 < 0.0
                tet = (tet[1], tet[3], tet[2], tet[4])
            end
            push!(tetrahedra, tet)
        end
    end

    isempty(triangles) &&
        error("Gmsh output contains no boundary triangles.")
    isempty(tetrahedra) &&
        error("Gmsh output contains no tetrahedra.")
    return triangles, tetrahedra
end

function boundary_tag(
    points::Matrix{Float64},
    triangle::NTuple{3, Int};
    tolerance::Float64 = 1e-9,
)
    nodes = collect(triangle)
    centroid = (
        sum(@view points[1, nodes]) / 3.0,
        sum(@view points[2, nodes]) / 3.0,
        sum(@view points[3, nodes]) / 3.0,
    )

    abs(centroid[1]) <= tolerance && return 1
    abs(centroid[1] - LX) <= tolerance && return 2
    abs(centroid[2]) <= tolerance && return 3
    abs(centroid[2] - LY) <= tolerance && return 4
    abs(centroid[3]) <= tolerance && return 5
    abs(centroid[3] - LZ) <= tolerance && return 6
    error("Could not classify boundary triangle $triangle at $centroid.")
end

function periodic_face_signature(
    points::Matrix{Float64},
    triangle::NTuple{3, Int},
    normal_dimension::Int,
)
    tangential_dimensions = filter(!=(normal_dimension), 1:3)
    projected = [
        (
            round(points[tangential_dimensions[1], node]; digits = 10),
            round(points[tangential_dimensions[2], node]; digits = 10),
        )
        for node in triangle
    ]
    sort!(projected)
    return Tuple(projected)
end

function validate_periodic_boundaries(
    points::Matrix{Float64},
    triangles::Vector{NTuple{3, Int}},
    boundary_tags::Vector{Int},
)
    for (minus_tag, plus_tag, dimension) in
        ((1, 2, 1), (3, 4, 2), (5, 6, 3))
        minus_signatures = sort([
            periodic_face_signature(points, triangle, dimension)
            for (triangle, tag) in zip(triangles, boundary_tags)
            if tag == minus_tag
        ])
        plus_signatures = sort([
            periodic_face_signature(points, triangle, dimension)
            for (triangle, tag) in zip(triangles, boundary_tags)
            if tag == plus_tag
        ])
        minus_signatures == plus_signatures ||
            error(
                "Periodic boundary triangulations $minus_tag and $plus_tag " *
                "do not match.",
            )
    end
    return nothing
end

function write_vtk(
    path::String,
    points::Matrix{Float64},
    triangles::Vector{NTuple{3, Int}},
    tetrahedra::Vector{NTuple{4, Int}},
    boundary_tags::Vector{Int},
)
    number_cells = length(triangles) + length(tetrahedra)
    connectivity_size =
        4 * length(triangles) + 5 * length(tetrahedra)
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
        println(io, "CELLS $number_cells $connectivity_size")
        for triangle in triangles
            println(io, "3 ", join(collect(triangle) .- 1, " "))
        end
        for tet in tetrahedra
            println(io, "4 ", join(collect(tet) .- 1, " "))
        end

        println(io)
        println(io, "CELL_TYPES $number_cells")
        foreach(_ -> println(io, 5), triangles)
        foreach(_ -> println(io, 10), tetrahedra)

        println(io)
        println(io, "CELL_DATA $number_cells")
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
    return path
end

function main(args::Vector{String})
    config = parse_args(args)
    config === nothing && return nothing

    hx = LX / config.nx
    hy = LY / config.ny
    hz = LZ / config.nz
    target_h = min(hx, hy, hz)

    mktempdir() do temporary_directory
        geometry_path =
            joinpath(temporary_directory, "periodic_box_unstructured.geo")
        raw_vtk_path =
            joinpath(temporary_directory, "periodic_box_unstructured.vtk")
        open(geometry_path, "w") do io
            write(io, gmsh_geometry_text(config.nx, config.ny, config.nz))
        end

        command = `$(config.gmsh) $geometry_path -3 -format vtk -o $raw_vtk_path -v 1`
        try
            run(command)
        catch caught
            error(
                "Gmsh failed while generating the unstructured mesh: " *
                sprint(showerror, caught),
            )
        end

        points, cells, cell_types =
            read_ascii_legacy_vtk(raw_vtk_path)
        triangles, tetrahedra =
            extract_linear_cells(points, cells, cell_types)
        boundary_tags = [
            boundary_tag(points, triangle) for triangle in triangles
        ]
        validate_periodic_boundaries(points, triangles, boundary_tags)

        mkpath(dirname(config.output))
        write_vtk(
            config.output,
            points,
            triangles,
            tetrahedra,
            boundary_tags,
        )

        println("Wrote ", config.output)
        println("mesher:               Gmsh 3D Delaunay")
        println(
            "target resolution:    ",
            "$(config.nx) x $(config.ny) x $(config.nz)",
        )
        println(
            "target spacings:      ",
            "hx=$(hx), hy=$(hy), hz=$(hz)",
        )
        println("Gmsh target h:        ", target_h)
        println("points:               ", size(points, 2))
        println("tetrahedra:           ", length(tetrahedra))
        println("boundary triangles:   ", length(triangles))
        for tag in 1:6
            println(
                "boundary tag $tag:       ",
                count(==(tag), boundary_tags),
            )
        end
        println("periodic matching:    PASS")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
