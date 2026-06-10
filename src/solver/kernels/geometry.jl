# -------------------------------------------------------------------------
# DG geometry structs
# -------------------------------------------------------------------------


function point3(points::Matrix{Float64}, node::Int)
    return (
        points[1, node],
        points[2, node],
        points[3, node],
    )
end

function vsub(a, b)
    return (
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
    )
end

function vadd(a, b)
    return (
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
    )
end

function vscale(c, a)
    return (
        c * a[1],
        c * a[2],
        c * a[3],
    )
end

function dot3(a, b)
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

function cross3(a, b)
    return (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

function norm3(a)
    return sqrt(dot3(a, a))
end

function tet_centroid(points::Matrix{Float64}, tet_nodes)
    x1 = point3(points, tet_nodes[1])
    x2 = point3(points, tet_nodes[2])
    x3 = point3(points, tet_nodes[3])
    x4 = point3(points, tet_nodes[4])

    return (
        (x1[1] + x2[1] + x3[1] + x4[1]) / 4,
        (x1[2] + x2[2] + x3[2] + x4[2]) / 4,
        (x1[3] + x2[3] + x3[3] + x4[3]) / 4,
    )
end

function tet_signed_volume(points::Matrix{Float64}, tet_nodes)
    x1 = point3(points, tet_nodes[1])
    x2 = point3(points, tet_nodes[2])
    x3 = point3(points, tet_nodes[3])
    x4 = point3(points, tet_nodes[4])

    a = vsub(x2, x1)
    b = vsub(x3, x1)
    c = vsub(x4, x1)

    return dot3(a, cross3(b, c)) / 6.0
end

function build_cell_geometry(mesh::RawVTUMesh)
    ntets = size(mesh.tets, 2)
    cells = Vector{CellGeometry}(undef, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        c = tet_centroid(mesh.points, tet_nodes)
        v = abs(tet_signed_volume(mesh.points, tet_nodes))

        if v <= 0.0
            error("Detected zero or negative-volume tetrahedron at element $e.")
        end

        cells[e] = CellGeometry(c, v)
    end

    return cells
end

function triangle_centroid(points::Matrix{Float64}, nodes::NTuple{3, Int})
    x1 = point3(points, nodes[1])
    x2 = point3(points, nodes[2])
    x3 = point3(points, nodes[3])

    return (
        (x1[1] + x2[1] + x3[1]) / 3,
        (x1[2] + x2[2] + x3[2]) / 3,
        (x1[3] + x2[3] + x3[3]) / 3,
    )
end

function raw_triangle_normal_area(points::Matrix{Float64}, nodes::NTuple{3, Int})
    x1 = point3(points, nodes[1])
    x2 = point3(points, nodes[2])
    x3 = point3(points, nodes[3])

    a = vsub(x2, x1)
    b = vsub(x3, x1)

    n_raw = cross3(a, b)
    n_norm = norm3(n_raw)

    if n_norm <= 0.0
        error("Detected degenerate triangle face with nodes $nodes.")
    end

    area = 0.5 * n_norm
    unit_normal = vscale(1.0 / n_norm, n_raw)

    return unit_normal, area
end

function oriented_face_geometry(
    points::Matrix{Float64},
    nodes::NTuple{3, Int},
    owner_centroid::NTuple{3, Float64},
)
    fc = triangle_centroid(points, nodes)
    n, area = raw_triangle_normal_area(points, nodes)

    owner_to_face = vsub(fc, owner_centroid)

    if dot3(n, owner_to_face) < 0.0
        n = vscale(-1.0, n)
    end

    return FaceGeometry(fc, n, area)
end

function build_dg_geometry(mesh::RawVTUMesh, topology::DGTopology)
    cell_geom = build_cell_geometry(mesh)

    interior_geom = Vector{FaceGeometry}(undef, length(topology.interior_faces))
    boundary_geom = Vector{FaceGeometry}(undef, length(topology.boundary_faces))

    for i in eachindex(topology.interior_faces)
        f = topology.interior_faces[i]

        owner_centroid = cell_geom[f.left_elem].centroid

        interior_geom[i] = oriented_face_geometry(
            mesh.points,
            f.nodes,
            owner_centroid,
        )

        # Optional consistency check:
        # normal should point roughly from left element to right element.
        cL = cell_geom[f.left_elem].centroid
        cR = cell_geom[f.right_elem].centroid
        L_to_R = vsub(cR, cL)

        if dot3(interior_geom[i].normal, L_to_R) < 0.0
            error("Interior face normal orientation failed at face $i.")
        end
    end

    for i in eachindex(topology.boundary_faces)
        f = topology.boundary_faces[i]

        owner_centroid = cell_geom[f.elem].centroid

        boundary_geom[i] = oriented_face_geometry(
            mesh.points,
            f.nodes,
            owner_centroid,
        )
    end

    return DGGeometry(cell_geom, interior_geom, boundary_geom)
end

function print_geometry_summary(geometry::DGGeometry)
    volumes = [c.volume for c in geometry.cells]
    boundary_areas = [f.area for f in geometry.boundary_faces]
    interior_areas = [f.area for f in geometry.interior_faces]

    println("DG geometry")
    println("-----------")
    println("Number of cell geometries:      ", length(geometry.cells))
    println("Number of interior face geoms:  ", length(geometry.interior_faces))
    println("Number of boundary face geoms:  ", length(geometry.boundary_faces))

    println()
    println("Cell volumes")
    println("------------")
    println("min volume: ", minimum(volumes))
    println("max volume: ", maximum(volumes))
    println("sum volume: ", sum(volumes))

    println()
    println("Face areas")
    println("----------")
    println("min interior area: ", minimum(interior_areas))
    println("max interior area: ", maximum(interior_areas))
    println("min boundary area: ", minimum(boundary_areas))
    println("max boundary area: ", maximum(boundary_areas))

    return nothing
end
