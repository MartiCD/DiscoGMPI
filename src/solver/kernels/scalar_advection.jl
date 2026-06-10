# ------------------------------------------------------------------------
# Scalar advection
# ------------------------------------------------------------------------
function interpolate_scalar_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    f::Function,
)
    ntets = size(mesh.tets, 2)
    u = zeros(Float64, ref.Np, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        for i in 1:ref.Np
            x, y, z = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[i],
                ref.s[i],
                ref.t[i],
            )

            u[i, e] = f(x, y, z)
        end
    end

    return u
end

function scalar_advection_volume_rhs!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    physops::DGPhysicalOperators,
    β::NTuple{3, Float64},
)
    fill!(rhs, 0.0)

    ne = size(u, 2)

    for e in 1:ne
        op = physops.elements[e]

        rhs[:, e] .-= β[1] .* (op.Dx * u[:, e])
        rhs[:, e] .-= β[2] .* (op.Dy * u[:, e])
        rhs[:, e] .-= β[3] .* (op.Dz * u[:, e])
    end

    return rhs
end

function scalar_advection_surface_rhs!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
    β::NTuple{3, Float64},
)
    for ff in flux_faces.interior
        tr = ff.trace

        n = ff.normal
        an = β[1] * n[1] + β[2] * n[2] + β[3] * n[3]

        uM = u[tr.minus_nodes, tr.minus_elem]
        uP = u[tr.plus_nodes[tr.plus_to_minus_perm], tr.plus_elem]

        uhat = an >= 0.0 ? uM : uP

        fluxM = an .* (uhat .- uM)
        fluxP = an .* (uhat .- uP)

        # Need to embed face flux into volume-node vector.
        faceM = zeros(Float64, ref.Np)
        faceP = zeros(Float64, ref.Np)

        faceM[tr.minus_nodes] .= fluxM
        faceP[tr.plus_nodes[tr.plus_to_minus_perm]] .= -fluxP

        rhs[:, tr.minus_elem] .-= fops.LIFT * faceM
        rhs[:, tr.plus_elem]  .+= fops.LIFT * faceP
    end

    return rhs
end

# Later replace this...
function scalar_boundary_state(
    x::Float64,
    y::Float64,
    z::Float64,
)
    return x + 2.0 * y + 3.0 * z
end

# ... with this 
# if boundary_id == 10
#     # PEC equivalent for Maxwell later
# else
#     # absorbing / inflow / outflow
# end

function test_scalar_advection_volume_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = similar(u)

    scalar_advection_volume_rhs!(rhs, u, physops, β)

    exact = -(β[1] + 2.0 * β[2] + 3.0 * β[3])

    err = maximum(abs.(rhs .- exact))

    println("Scalar advection volume test")
    println("----------------------------")
    println("β:                         ", β)
    println("exact RHS:                 ", exact)
    println("max error:                 ", err)

    if err < 1e-10
        println("✓ scalar volume operator is consistent")
    else
        println("⚠ scalar volume operator has larger-than-expected error")
    end

    return nothing
end

function test_scalar_advection_interior_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = zeros(Float64, size(u))

    scalar_advection_surface_rhs!(
        rhs,
        u,
        ref,
        fops,
        flux_faces,
        β,
    )

    max_rhs = maximum(abs.(rhs))

    println("Scalar advection interior-surface test")
    println("--------------------------------------")
    println("β:                         ", β)
    println("max |surface rhs|:         ", max_rhs)

    if max_rhs < 1e-10
        println("✓ interior surface operator vanishes for continuous linear field")
    else
        println("⚠ interior surface operator is not vanishing as expected")
    end

    return nothing
end

function physical_boundary_trace_points(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    tr::BoundaryTraceMap,
)
    pts = Vector{NTuple{3, Float64}}(undef, length(tr.nodes))

    for i in eachindex(tr.nodes)
        pts[i] = physical_point_on_element(
            mesh,
            ref,
            tr.elem,
            tr.nodes[i],
        )
    end

    return pts
end

function scalar_advection_exact_boundary_state(
    x::Float64,
    y::Float64,
    z::Float64,
)
    return x + 2.0 * y + 3.0 * z
end

function scalar_advection_boundary_surface_rhs!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
    β::NTuple{3, Float64},
    boundary_state::Function,
)
    for ff in flux_faces.boundary
        tr = ff.trace

        n = ff.normal
        an = β[1] * n[1] + β[2] * n[2] + β[3] * n[3]

        uM = u[tr.nodes, tr.elem]

        uB = similar(uM)

        pts = physical_boundary_trace_points(mesh, ref, tr)

        for q in eachindex(pts)
            x, y, z = pts[q]
            uB[q] = boundary_state(x, y, z)
        end

        # Upwind exterior state:
        #
        # If an < 0, information enters the domain, so use uB.
        # If an >= 0, information leaves the domain, so use uM.
        uhat = an < 0.0 ? uB : uM

        # Boundary correction for the interior/minus state.
        flux = an .* (uhat .- uM)

        face_vec = zeros(Float64, ref.Np)
        face_vec[tr.nodes] .= flux

        rhs[:, tr.elem] .-= fops.LIFT * face_vec
    end

    return rhs
end

function scalar_advection_surface_rhs_with_boundaries!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
    β::NTuple{3, Float64},
    boundary_state::Function,
)
    scalar_advection_surface_rhs!(
        rhs,
        u,
        ref,
        fops,
        flux_faces,
        β,
    )

    scalar_advection_boundary_surface_rhs!(
        rhs,
        u,
        mesh,
        ref,
        fops,
        flux_faces,
        β,
        boundary_state,
    )

    return rhs
end

function test_scalar_advection_boundary_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = zeros(Float64, size(u))

    scalar_advection_boundary_surface_rhs!(
        rhs,
        u,
        mesh,
        ref,
        fops,
        flux_faces,
        β,
        scalar_advection_exact_boundary_state,
    )

    max_rhs = maximum(abs.(rhs))

    println("Scalar advection boundary-surface test")
    println("--------------------------------------")
    println("β:                         ", β)
    println("max |boundary rhs|:        ", max_rhs)

    if max_rhs < 1e-10
        println("✓ boundary surface operator vanishes for exact inflow state")
    else
        println("⚠ boundary surface operator is not vanishing as expected")
    end

    return nothing
end

function test_scalar_advection_full_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = zeros(Float64, size(u))

    scalar_advection_surface_rhs_with_boundaries!(
        rhs,
        u,
        mesh,
        ref,
        fops,
        flux_faces,
        β,
        scalar_advection_exact_boundary_state,
    )

    max_rhs = maximum(abs.(rhs))

    println("Scalar advection full-surface test")
    println("----------------------------------")
    println("β:                         ", β)
    println("max |surface rhs|:         ", max_rhs)

    if max_rhs < 1e-10
        println("✓ full surface operator vanishes for continuous field and exact boundary data")
    else
        println("⚠ full surface operator is not vanishing as expected")
    end

    return nothing
end
