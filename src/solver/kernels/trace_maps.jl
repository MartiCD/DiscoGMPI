# -------------------------------------------------------------------------
# DG trace maps
# -------------------------------------------------------------------------


function physical_point_on_element(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    elem::Int,
    local_node::Int,
)
    tet_nodes = mesh.tets[:, elem]

    return map_to_physical(
        mesh.points,
        tet_nodes,
        ref.r[local_node],
        ref.s[local_node],
        ref.t[local_node],
    )
end


function physical_face_points(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    elem::Int,
    face_nodes::Vector{Int},
)
    pts = Vector{NTuple{3, Float64}}(undef, length(face_nodes))

    for i in eachindex(face_nodes)
        pts[i] = physical_point_on_element(
            mesh,
            ref,
            elem,
            face_nodes[i],
        )
    end

    return pts
end

function squared_distance(a::NTuple{3, Float64}, b::NTuple{3, Float64})
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]

    return dx * dx + dy * dy + dz * dz
end

function match_face_node_permutation(
    minus_points::Vector{NTuple{3, Float64}},
    plus_points::Vector{NTuple{3, Float64}};
    tol::Float64 = 1e-10,
)
    nminus = length(minus_points)
    nplus = length(plus_points)

    if nminus != nplus
        error("Cannot match face nodes: minus has $nminus nodes, plus has $nplus nodes.")
    end

    perm = Vector{Int}(undef, nminus)
    used = falses(nplus)

    tol2 = tol^2

    for i in 1:nminus
        best_j = 0
        best_d2 = Inf

        for j in 1:nplus
            if used[j]
                continue
            end

            d2 = squared_distance(minus_points[i], plus_points[j])

            if d2 < best_d2
                best_d2 = d2
                best_j = j
            end
        end

        if best_j == 0 || best_d2 > tol2
            error(
                "Could not match face node $i. " *
                "Best squared distance = $best_d2, tolerance squared = $tol2."
            )
        end

        perm[i] = best_j
        used[best_j] = true
    end

    return perm
end

function build_dg_trace_maps(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    topology::DGTopology,
    fops::ReferenceTetFaceOperators;
    tol::Float64 = 1e-10,
)
    interior_maps = Vector{InteriorTraceMap}(
        undef,
        length(topology.interior_faces),
    )

    for i in eachindex(topology.interior_faces)
        face = topology.interior_faces[i]

        minus_elem = face.left_elem
        minus_face = face.left_local_face

        plus_elem = face.right_elem
        plus_face = face.right_local_face

        minus_nodes = copy(fops.face_nodes[minus_face])
        plus_nodes = copy(fops.face_nodes[plus_face])

        minus_points = physical_face_points(
            mesh,
            ref,
            minus_elem,
            minus_nodes,
        )

        plus_points = physical_face_points(
            mesh,
            ref,
            plus_elem,
            plus_nodes,
        )

        plus_to_minus_perm = match_face_node_permutation(
            minus_points,
            plus_points;
            tol = tol,
        )

        interior_maps[i] = InteriorTraceMap(
            minus_elem,
            minus_face,
            plus_elem,
            plus_face,
            minus_nodes,
            plus_nodes,
            plus_to_minus_perm,
        )
    end

    boundary_maps = Vector{BoundaryTraceMap}(
        undef,
        length(topology.boundary_faces),
    )

    for i in eachindex(topology.boundary_faces)
        face = topology.boundary_faces[i]

        nodes = copy(fops.face_nodes[face.local_face])

        boundary_maps[i] = BoundaryTraceMap(
            face.elem,
            face.local_face,
            face.boundary_id,
            nodes,
        )
    end

    return DGTraceMaps(interior_maps, boundary_maps)
end

function print_trace_map_summary(trace_maps::DGTraceMaps)
    println("DG trace maps")
    println("-------------")
    println("Number of interior traces: ", length(trace_maps.interior))
    println("Number of boundary traces: ", length(trace_maps.boundary))

    if !isempty(trace_maps.interior)
        nfp_values = unique(length(tr.minus_nodes) for tr in trace_maps.interior)
        println("Interior trace Nfp values: ", sort(collect(nfp_values)))
    end

    if !isempty(trace_maps.boundary)
        nfp_values = unique(length(tr.nodes) for tr in trace_maps.boundary)
        println("Boundary trace Nfp values: ", sort(collect(nfp_values)))
    end

    boundary_ids = sort(unique(tr.boundary_id for tr in trace_maps.boundary))

    println()
    println("Boundary trace counts")
    println("---------------------")

    for bid in boundary_ids
        n = count(tr -> tr.boundary_id == bid, trace_maps.boundary)
        println("  boundary_id = ", bid, " : ", n, " traces")
    end

    println()
    println("Permutation examples")
    println("--------------------")

    nexamples = min(5, length(trace_maps.interior))

    for i in 1:nexamples
        tr = trace_maps.interior[i]

        println(
            "trace ", i,
            ": elem ", tr.minus_elem, " face ", tr.minus_face,
            " ↔ elem ", tr.plus_elem, " face ", tr.plus_face,
            ", perm = ", tr.plus_to_minus_perm,
        )
    end

    return nothing
end

function evaluate_linear_function_at_element_nodes(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    elem::Int,
)
    u = Vector{Float64}(undef, ref.Np)

    tet_nodes = mesh.tets[:, elem]

    for i in 1:ref.Np
        x, y, z = map_to_physical(
            mesh.points,
            tet_nodes,
            ref.r[i],
            ref.s[i],
            ref.t[i],
        )

        u[i] = x + 2.0 * y + 3.0 * z
    end

    return u
end


function test_trace_maps_linear_function(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    trace_maps::DGTraceMaps,
)
    max_jump = 0.0
    worst_trace = 0

    for i in eachindex(trace_maps.interior)
        tr = trace_maps.interior[i]

        u_minus_elem = evaluate_linear_function_at_element_nodes(
            mesh,
            ref,
            tr.minus_elem,
        )

        u_plus_elem = evaluate_linear_function_at_element_nodes(
            mesh,
            ref,
            tr.plus_elem,
        )

        uM = u_minus_elem[tr.minus_nodes]
        uP = u_plus_elem[tr.plus_nodes[tr.plus_to_minus_perm]]

        jump = maximum(abs.(uM .- uP))

        if jump > max_jump
            max_jump = jump
            worst_trace = i
        end
    end

    println("Trace-map continuity test")
    println("-------------------------")
    println("u(x,y,z) = x + 2y + 3z")
    println("max interior trace jump: ", max_jump)
    println("worst trace id:          ", worst_trace)

    if max_jump < 1e-10
        println("✓ interior trace permutations are consistent")
    else
        println("⚠ interior trace permutations may be inconsistent")
    end

    return nothing
end

function test_trace_map_geometry(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    trace_maps::DGTraceMaps,
)
    max_distance = 0.0
    worst_trace = 0

    for i in eachindex(trace_maps.interior)
        tr = trace_maps.interior[i]

        minus_points = physical_face_points(
            mesh,
            ref,
            tr.minus_elem,
            tr.minus_nodes,
        )

        plus_points = physical_face_points(
            mesh,
            ref,
            tr.plus_elem,
            tr.plus_nodes,
        )

        plus_points_aligned = plus_points[tr.plus_to_minus_perm]

        for q in eachindex(minus_points)
            d = sqrt(squared_distance(minus_points[q], plus_points_aligned[q]))

            if d > max_distance
                max_distance = d
                worst_trace = i
            end
        end
    end

    println("Trace-map geometry test")
    println("-----------------------")
    println("max matched-node distance: ", max_distance)
    println("worst trace id:            ", worst_trace)

    if max_distance < 1e-10
        println("✓ matched trace nodes are geometrically consistent")
    else
        println("⚠ matched trace nodes may be geometrically inconsistent")
    end

    return nothing
end
