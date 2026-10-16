#!/usr/bin/env julia

using DiscoGMPI
using LinearAlgebra
using Printf

struct ReferenceTetNodePlotConfig
    order::Int
    nodal_set::Symbol
    output::String
    azimuth_degrees::Float64
    elevation_degrees::Float64
    width::Int
    height::Int
    point_size::Float64
    label_nodes::Bool
end

function usage()
    println("""
    Plot DiscoGMPI reference tetrahedron nodes.

    Usage:
      julia --project=. examples/plot_reference_tet_nodes.jl [options]

    Options:
      --order=N                  Polynomial degree. Default: 4
      --nodal-set=SET            equispaced or warp_blend. Default: equispaced
      --output=PATH              SVG output path.
                                  Default: output/reference_tet_nodes_pN_SET.svg
      --azimuth-degrees=A        View azimuth angle. Default: 35
      --elevation-degrees=E      View elevation angle. Default: 24
      --width=PX                 SVG width. Default: 1000
      --height=PX                SVG height. Default: 800
      --point-size=R             Base node radius in SVG pixels. Default: 5
      --label-nodes              Label every node with its local index.
      -h, --help                 Show this help message.

    Examples:
      julia --project=. examples/plot_reference_tet_nodes.jl --order=4

      julia --project=. examples/plot_reference_tet_nodes.jl \\
        --order=5 --nodal-set=warp_blend \\
        --output=output/reference_tet_nodes_p5_warp_blend.svg
    """)
end

function parse_nodal_set(value::AbstractString)
    key = lowercase(replace(strip(value), "-" => "_"))

    if key in ("equispaced", "equi", "uniform")
        return :equispaced
    elseif key in ("warp_blend", "warpblend", "hesthaven_warburton")
        return :warp_blend
    else
        throw(
            ArgumentError(
                "Unknown nodal set '$value'. Use equispaced or warp_blend.",
            ),
        )
    end
end

function take_option_value(args::Vector{String}, i::Int)
    arg = args[i]

    if occursin("=", arg)
        key, value = split(arg, "=", limit = 2)
        return key, value, i
    end

    i < length(args) ||
        throw(ArgumentError("Option $arg requires a value."))

    return arg, args[i + 1], i + 1
end

function parse_args(args::Vector{String})
    order = 4
    nodal_set = :equispaced
    output = nothing
    azimuth_degrees = 35.0
    elevation_degrees = 24.0
    width = 1000
    height = 800
    point_size = 5.0
    label_nodes = false

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg in ("-h", "--help")
            usage()
            return nothing
        elseif startswith(arg, "--order")
            _, value, i = take_option_value(args, i)
            order = parse(Int, value)
        elseif startswith(arg, "--nodal-set")
            _, value, i = take_option_value(args, i)
            nodal_set = parse_nodal_set(value)
        elseif startswith(arg, "--output")
            _, value, i = take_option_value(args, i)
            output = value
        elseif startswith(arg, "--azimuth-degrees")
            _, value, i = take_option_value(args, i)
            azimuth_degrees = parse(Float64, value)
        elseif startswith(arg, "--elevation-degrees")
            _, value, i = take_option_value(args, i)
            elevation_degrees = parse(Float64, value)
        elseif startswith(arg, "--width")
            _, value, i = take_option_value(args, i)
            width = parse(Int, value)
        elseif startswith(arg, "--height")
            _, value, i = take_option_value(args, i)
            height = parse(Int, value)
        elseif startswith(arg, "--point-size")
            _, value, i = take_option_value(args, i)
            point_size = parse(Float64, value)
        elseif arg == "--label-nodes"
            label_nodes = true
        else
            throw(ArgumentError("Unknown option: $arg"))
        end

        i += 1
    end

    order >= 0 ||
        throw(ArgumentError("Polynomial order must be nonnegative."))
    width > 200 ||
        throw(ArgumentError("SVG width must be larger than 200 pixels."))
    height > 200 ||
        throw(ArgumentError("SVG height must be larger than 200 pixels."))
    point_size > 0 ||
        throw(ArgumentError("Point size must be positive."))

    if output === nothing
        nodal_label = replace(String(nodal_set), "_" => "-")
        output = joinpath(
            "output",
            "reference_tet_nodes_p$(order)_$(nodal_label).svg",
        )
    end

    return ReferenceTetNodePlotConfig(
        order,
        nodal_set,
        output,
        azimuth_degrees,
        elevation_degrees,
        width,
        height,
        point_size,
        label_nodes,
    )
end

function barycentric_coordinates(p::NTuple{3, Float64})
    r, s, t = p
    lambda2 = 0.5 * (r + 1.0)
    lambda3 = 0.5 * (s + 1.0)
    lambda4 = 0.5 * (t + 1.0)
    lambda1 = 1.0 - lambda2 - lambda3 - lambda4
    return (lambda1, lambda2, lambda3, lambda4)
end

function node_class(p::NTuple{3, Float64}; tol::Float64 = 1e-9)
    lambda = barycentric_coordinates(p)
    nzero = count(abs(value) <= tol for value in lambda)

    if nzero >= 3
        return :vertex
    elseif nzero == 2
        return :edge
    elseif nzero == 1
        return :face
    else
        return :interior
    end
end

function class_style(class::Symbol)
    if class == :vertex
        return ("#7b3294", "#3f1d63", 1.2)
    elseif class == :edge
        return ("#008837", "#005a32", 1.0)
    elseif class == :face
        return ("#2b83ba", "#145985", 0.92)
    else
        return ("#d7191c", "#8f1012", 0.85)
    end
end

function normalize3(v::NTuple{3, Float64})
    nrm = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    nrm > 0.0 ||
        throw(ArgumentError("Cannot normalize a zero vector."))
    return (v[1] / nrm, v[2] / nrm, v[3] / nrm)
end

dot3(a::NTuple{3, Float64}, b::NTuple{3, Float64}) =
    a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

cross3(a::NTuple{3, Float64}, b::NTuple{3, Float64}) = (
    a[2] * b[3] - a[3] * b[2],
    a[3] * b[1] - a[1] * b[3],
    a[1] * b[2] - a[2] * b[1],
)

function projection_basis(config::ReferenceTetNodePlotConfig)
    az = deg2rad(config.azimuth_degrees)
    el = deg2rad(config.elevation_degrees)
    view = normalize3((cos(el) * cos(az), cos(el) * sin(az), sin(el)))

    world_up = (0.0, 0.0, 1.0)
    right = cross3(world_up, view)
    if sqrt(dot3(right, right)) < 1e-12
        right = cross3((0.0, 1.0, 0.0), view)
    end
    right = normalize3(right)
    up = normalize3(cross3(view, right))

    return right, up, view
end

function project_points(
    points::Vector{NTuple{3, Float64}},
    config::ReferenceTetNodePlotConfig,
)
    center = (-0.5, -0.5, -0.5)
    right, up, view = projection_basis(config)

    x = Vector{Float64}(undef, length(points))
    y = Vector{Float64}(undef, length(points))
    depth = Vector{Float64}(undef, length(points))

    for (i, p) in enumerate(points)
        q = (p[1] - center[1], p[2] - center[2], p[3] - center[3])
        x[i] = dot3(q, right)
        y[i] = dot3(q, up)
        depth[i] = dot3(q, view)
    end

    return x, y, depth
end

function svg_escape(value::AbstractString)
    escaped = replace(value, "&" => "&amp;")
    escaped = replace(escaped, "<" => "&lt;")
    escaped = replace(escaped, ">" => "&gt;")
    escaped = replace(escaped, "\"" => "&quot;")
    return escaped
end

fmt(value::Real) = @sprintf("%.6f", value)

function screen_transform(
    x::Vector{Float64},
    y::Vector{Float64},
    config::ReferenceTetNodePlotConfig,
)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    xrange = max(xmax - xmin, eps(Float64))
    yrange = max(ymax - ymin, eps(Float64))

    margin_x = 0.12 * config.width
    margin_y_top = 0.16 * config.height
    margin_y_bottom = 0.16 * config.height

    scale = min(
        (config.width - 2margin_x) / xrange,
        (config.height - margin_y_top - margin_y_bottom) / yrange,
    )

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    cx = 0.5 * config.width
    cy = 0.5 * (margin_y_top + config.height - margin_y_bottom)

    sx = [cx + scale * (value - xmid) for value in x]
    sy = [cy - scale * (value - ymid) for value in y]

    return sx, sy, scale
end

function write_polyline(io, xs, ys, attrs::Pair...)
    attr_text = join(
        [
            "$(first(pair))=\"$(svg_escape(string(last(pair))))\""
            for pair in attrs
        ],
        " ",
    )
    points = join(["$(fmt(xs[i])),$(fmt(ys[i]))" for i in eachindex(xs)], " ")
    println(io, "  <polyline points=\"$points\" $attr_text />")
end

function write_reference_tet_svg(config::ReferenceTetNodePlotConfig)
    ref = build_reference_tet(config.order; nodal_set = config.nodal_set)
    face_nodes = reference_face_nodes(ref; tol = 1e-8)

    vertices = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ]
    nodes = [(ref.r[i], ref.s[i], ref.t[i]) for i in 1:ref.Np]
    all_points = vcat(vertices, nodes)
    x, y, depth = project_points(all_points, config)
    sx, sy, scale = screen_transform(x, y, config)

    vertex_sx = sx[1:4]
    vertex_sy = sy[1:4]
    vertex_depth = depth[1:4]
    node_sx = sx[5:end]
    node_sy = sy[5:end]
    node_depth = depth[5:end]

    faces = [
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
        (2, 3, 4),
    ]
    face_order = sortperm([
        sum(vertex_depth[collect(face)]) / length(face)
        for face in faces
    ])

    edges = [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    node_order = sortperm(node_depth)

    output_dir = dirname(config.output)
    output_dir == "" || mkpath(output_dir)

    nodal_label = replace(String(config.nodal_set), "_" => "-")
    title = "Reference tetrahedron nodes: p=$(config.order), $(nodal_label)"

    open(config.output, "w") do io
        println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        println(
            io,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" " *
            "width=\"$(config.width)\" height=\"$(config.height)\" " *
            "viewBox=\"0 0 $(config.width) $(config.height)\">",
        )
        println(io, "  <defs>")
        println(io, "    <marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"4\" orient=\"auto\" markerUnits=\"strokeWidth\">")
        println(io, "      <path d=\"M0,0 L8,4 L0,8 Z\" fill=\"#303030\" />")
        println(io, "    </marker>")
        println(io, "  </defs>")
        println(io, "  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\" />")

        println(io, "  <text x=\"$(config.width / 2)\" y=\"46\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"28\" fill=\"#111111\">$(svg_escape(title))</text>")
        println(io, "  <text x=\"$(config.width / 2)\" y=\"78\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"16\" fill=\"#555555\">Np=$(ref.Np), Nfp=$(length(face_nodes[1])), view=($(fmt(config.azimuth_degrees))°, $(fmt(config.elevation_degrees))°)</text>")

        println(io, "  <g id=\"faces\">")
        for idx in face_order
            face = faces[idx]
            points = join(
                [
                    "$(fmt(vertex_sx[v])),$(fmt(vertex_sy[v]))"
                    for v in face
                ],
                " ",
            )
            println(io, "    <polygon points=\"$points\" fill=\"#d9e8f5\" fill-opacity=\"0.38\" stroke=\"none\" />")
        end
        println(io, "  </g>")

        println(io, "  <g id=\"edges\" fill=\"none\" stroke=\"#202020\" stroke-width=\"2.2\" stroke-linecap=\"round\">")
        for edge in edges
            xs = [vertex_sx[edge[1]], vertex_sx[edge[2]]]
            ys = [vertex_sy[edge[1]], vertex_sy[edge[2]]]
            write_polyline(
                io,
                xs,
                ys,
                "stroke" => "#202020",
                "stroke-width" => "2.2",
                "fill" => "none",
            )
        end
        println(io, "  </g>")

        println(io, "  <g id=\"nodes\">")
        for node in node_order
            class = node_class(nodes[node])
            fill, stroke, radius_scale = class_style(class)
            radius = config.point_size * radius_scale
            println(
                io,
                "    <circle cx=\"$(fmt(node_sx[node]))\" cy=\"$(fmt(node_sy[node]))\" " *
                "r=\"$(fmt(radius))\" fill=\"$fill\" fill-opacity=\"0.88\" " *
                "stroke=\"$stroke\" stroke-width=\"1.2\" />",
            )

            if config.label_nodes
                println(
                    io,
                    "    <text x=\"$(fmt(node_sx[node] + radius + 2.0))\" " *
                    "y=\"$(fmt(node_sy[node] - radius - 2.0))\" " *
                    "font-family=\"Arial, Helvetica, sans-serif\" font-size=\"10\" " *
                    "fill=\"#222222\">$node</text>",
                )
            end
        end
        println(io, "  </g>")

        legend_x = 42.0
        legend_y = config.height - 122.0
        println(io, "  <g id=\"legend\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"15\" fill=\"#222222\">")
        println(io, "    <rect x=\"22\" y=\"$(fmt(legend_y - 30.0))\" width=\"250\" height=\"118\" rx=\"8\" fill=\"#ffffff\" fill-opacity=\"0.86\" stroke=\"#d0d0d0\" />")
        for (row, (class, label)) in enumerate((
            (:vertex, "vertex nodes"),
            (:edge, "edge nodes"),
            (:face, "face nodes"),
            (:interior, "interior nodes"),
        ))
            fill, stroke, radius_scale = class_style(class)
            yrow = legend_y + 25.0 * (row - 1)
            println(io, "    <circle cx=\"$legend_x\" cy=\"$(fmt(yrow))\" r=\"$(fmt(config.point_size * radius_scale))\" fill=\"$fill\" fill-opacity=\"0.88\" stroke=\"$stroke\" stroke-width=\"1.2\" />")
            println(io, "    <text x=\"$(fmt(legend_x + 18.0))\" y=\"$(fmt(yrow + 5.0))\">$label</text>")
        end
        println(io, "  </g>")

        axis_origin = (config.width - 155.0, config.height - 105.0)
        axis_scale = 0.10 * scale
        basis_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        bx, by, _ = project_points(basis_points, config)
        println(io, "  <g id=\"axis-triad\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"15\" fill=\"#303030\" stroke=\"#303030\" stroke-width=\"1.7\">")
        for (label, idx) in zip(("r", "s", "t"), 2:4)
            dx = axis_scale * (bx[idx] - bx[1])
            dy = -axis_scale * (by[idx] - by[1])
            x1 = axis_origin[1]
            y1 = axis_origin[2]
            x2 = x1 + dx
            y2 = y1 + dy
            println(io, "    <line x1=\"$(fmt(x1))\" y1=\"$(fmt(y1))\" x2=\"$(fmt(x2))\" y2=\"$(fmt(y2))\" marker-end=\"url(#arrow)\" />")
            println(io, "    <text x=\"$(fmt(x2 + 7.0))\" y=\"$(fmt(y2 + 5.0))\">$label</text>")
        end
        println(io, "  </g>")

        println(io, "</svg>")
    end

    println("Wrote reference tetrahedron node plot:")
    println("  ", abspath(config.output))
    println("order:      ", config.order)
    println("nodal set:  ", config.nodal_set)
    println("nodes:      ", ref.Np)
    println("face nodes: ", length(face_nodes[1]))

    return config.output
end

function main(args = ARGS)
    config = parse_args(String.(args))
    config === nothing && return 0
    write_reference_tet_svg(config)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch err
        println(stderr, "ERROR: ", sprint(showerror, err))
        exit(1)
    end
end
