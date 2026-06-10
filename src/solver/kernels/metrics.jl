# ------------------------------------------------------------------------
# Metrics 
# ------------------------------------------------------------------------


# This will be replacesd by high-order polynomials...
function ref_tet_shape_functions(r::Float64, s::Float64, t::Float64)
    ϕ1 = -(r + s + t + 1.0) / 2.0
    ϕ2 =  (r + 1.0) / 2.0
    ϕ3 =  (s + 1.0) / 2.0
    ϕ4 =  (t + 1.0) / 2.0

    return (ϕ1, ϕ2, ϕ3, ϕ4)
end

function ref_tet_shape_gradients()
    # Each column is ∇_ref ϕᵢ = [∂ϕᵢ/∂r, ∂ϕᵢ/∂s, ∂ϕᵢ/∂t]
    return [
        -0.5   0.5   0.0   0.0
        -0.5   0.0   0.5   0.0
        -0.5   0.0   0.0   0.5
    ]
end

function map_to_physical(
    points::Matrix{Float64},
    tet_nodes,
    r::Float64,
    s::Float64,
    t::Float64,
)
    ϕ = ref_tet_shape_functions(r, s, t)

    x = 0.0
    y = 0.0
    z = 0.0

    for a in 1:4
        node = tet_nodes[a]

        x += ϕ[a] * points[1, node]
        y += ϕ[a] * points[2, node]
        z += ϕ[a] * points[3, node]
    end

    return (x, y, z)
end

function build_tet_mapping(points::Matrix{Float64}, tet_nodes)
    x1 = points[:, tet_nodes[1]]
    x2 = points[:, tet_nodes[2]]
    x3 = points[:, tet_nodes[3]]
    x4 = points[:, tet_nodes[4]]

    J = zeros(Float64, 3, 3)

    J[:, 1] .= 0.5 .* (x2 .- x1)  # ∂x/∂r
    J[:, 2] .= 0.5 .* (x3 .- x1)  # ∂x/∂s
    J[:, 3] .= 0.5 .* (x4 .- x1)  # ∂x/∂t

    detJ = det(J)

    if abs(detJ) <= eps(Float64)
        error("Degenerate tetrahedron detected: detJ = $detJ.")
    end

    invJ = inv(J)

    return TetMapping(
        J,
        invJ,
        detJ,
        abs(detJ),
    )
end

function build_reference_mappings(mesh::RawVTUMesh)
    ntets = size(mesh.tets, 2)

    mappings = Vector{TetMapping}(undef, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]
        mappings[e] = build_tet_mapping(mesh.points, tet_nodes)
    end

    return DGReferenceMapping(mappings)
end

function reference_gradient_to_physical(mapping::TetMapping, grad_ref)
    return mapping.invJ' * grad_ref
end

# For linear basis functions
function physical_shape_gradients(mapping::TetMapping)
    grad_ref = ref_tet_shape_gradients()
    return mapping.invJ' * grad_ref
end

function print_mapping_summary(mesh::RawVTUMesh, geometry::DGGeometry, mappings::DGReferenceMapping)
    ntets = size(mesh.tets, 2)

    detJs = [m.detJ for m in mappings.tet_mappings]
    absdetJs = [m.absdetJ for m in mappings.tet_mappings]

    mapped_volumes = REF_TET_VOLUME .* absdetJs
    geom_volumes = [c.volume for c in geometry.cells]

    volume_errors = abs.(mapped_volumes .- geom_volumes)

    println("Reference-to-physical mappings")
    println("------------------------------")
    println("Number of mappings:       ", length(mappings.tet_mappings))
    println("Number of tetrahedra:     ", ntets)

    println()
    println("Jacobian determinants")
    println("---------------------")
    println("min detJ:              ", minimum(detJs))
    println("max detJ:              ", maximum(detJs))
    println("min |detJ|:            ", minimum(absdetJs))
    println("max |detJ|:            ", maximum(absdetJs))

    println()
    println("Volume consistency")
    println("------------------")
    println("max |Vmap - Vgeom|:   ", maximum(volume_errors))
    println("sum mapped volumes:   ", sum(mapped_volumes))
    println("sum geom volumes:     ", sum(geom_volumes))

    if maximum(volume_errors) < 1e-10
        println("✓ mapped volumes match geometric volumes")
    else
        println("⚠ mapped volumes differ from geometric volumes")
    end

    nnegative = count(<(0.0), detJs)

    println()
    println("Orientation")
    println("-----------")
    println("negative detJ count:   ", nnegative)
    println("positive detJ count:   ", ntets - nnegative)

    return nothing
end
