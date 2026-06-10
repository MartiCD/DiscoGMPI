# ------------------------------------------------------------------------
# PHYSICAL OPERATORS 
# ------------------------------------------------------------------------


function physical_derivative_matrices(ref::ReferenceTet, invJ::AbstractMatrix{Float64})
    # ∇x = J^{-T} ∇r
    #
    # [∂x]   [invJ[1,1] invJ[2,1] invJ[3,1]] [∂r]
    # [∂y] = [invJ[1,2] invJ[2,2] invJ[3,2]] [∂s]
    # [∂z]   [invJ[1,3] invJ[2,3] invJ[3,3]] [∂t]

    Dx = invJ[1, 1] .* ref.Dr .+
         invJ[2, 1] .* ref.Ds .+
         invJ[3, 1] .* ref.Dt

    Dy = invJ[1, 2] .* ref.Dr .+
         invJ[2, 2] .* ref.Ds .+
         invJ[3, 2] .* ref.Dt

    Dz = invJ[1, 3] .* ref.Dr .+
         invJ[2, 3] .* ref.Ds .+
         invJ[3, 3] .* ref.Dt

    return Dx, Dy, Dz
end

function physical_stiffness_matrices(ref::ReferenceTet, invJ::AbstractMatrix{Float64})
    # Sx[i,j] = ∫_Kref ℓ_i ∂xℓ_j dV, with ∂x mapped to reference
    # derivatives. These are intentionally not multiplied by |detJ|.
    Sx = invJ[1, 1] .* ref.Sr .+
         invJ[2, 1] .* ref.Ss .+
         invJ[3, 1] .* ref.St

    Sy = invJ[1, 2] .* ref.Sr .+
         invJ[2, 2] .* ref.Ss .+
         invJ[3, 2] .* ref.St

    Sz = invJ[1, 3] .* ref.Sr .+
         invJ[2, 3] .* ref.Ss .+
         invJ[3, 3] .* ref.St

    return Sx, Sy, Sz
end

function build_physical_element_operator(ref::ReferenceTet, mapping::TetMapping)
    invJ = mapping.invJ

    Dx, Dy, Dz = physical_derivative_matrices(ref, invJ)
    Sx, Sy, Sz = physical_stiffness_matrices(ref, invJ)
    weak = PhysicalWeakDerivativeOperators(Sx, Sy, Sz)

    return PhysicalElementOperators(
        Dx,
        Dy,
        Dz,
        weak,
        mapping.absdetJ,
    )
end

function physical_weak_derivative_matrices(op::PhysicalElementOperators)
    return op.weak.Sx, op.weak.Sy, op.weak.Sz
end

function physical_weak_derivative_transpose_matrices(op::PhysicalElementOperators)
    return op.weak.SxT, op.weak.SyT, op.weak.SzT
end


function build_physical_operators(ref::ReferenceTet, mappings::DGReferenceMapping)
    ops = Vector{PhysicalElementOperators}(undef, length(mappings.tet_mappings))

    for e in eachindex(mappings.tet_mappings)
        ops[e] = build_physical_element_operator(ref, mappings.tet_mappings[e])
    end

    return DGPhysicalOperators(ops)
end

function print_physical_operator_summary(physops::DGPhysicalOperators)
    println("Physical DG operators")
    println("---------------------")
    println("Number of elements:      ", length(physops.elements))

    dx_norms = [norm(op.Dx) for op in physops.elements]
    dy_norms = [norm(op.Dy) for op in physops.elements]
    dz_norms = [norm(op.Dz) for op in physops.elements]

    sx_norms = [norm(op.weak.Sx) for op in physops.elements]
    sy_norms = [norm(op.weak.Sy) for op in physops.elements]
    sz_norms = [norm(op.weak.Sz) for op in physops.elements]

    println()
    println("Derivative operator norms")
    println("-------------------------")
    println("min ||Dx||:              ", minimum(dx_norms))
    println("max ||Dx||:              ", maximum(dx_norms))
    println("min ||Dy||:              ", minimum(dy_norms))
    println("max ||Dy||:              ", maximum(dy_norms))
    println("min ||Dz||:              ", minimum(dz_norms))
    println("max ||Dz||:              ", maximum(dz_norms))

    println()
    println("Weak derivative matrix norms")
    println("----------------------------")
    println("min ||Sx||:              ", minimum(sx_norms))
    println("max ||Sx||:              ", maximum(sx_norms))
    println("min ||Sy||:              ", minimum(sy_norms))
    println("max ||Sy||:              ", maximum(sy_norms))
    println("min ||Sz||:              ", minimum(sz_norms))
    println("max ||Sz||:              ", maximum(sz_norms))

    scales = [op.mass_scale for op in physops.elements]

    println()
    println("Mass scaling")
    println("------------")
    println("min |detJ|:              ", minimum(scales))
    println("max |detJ|:              ", maximum(scales))

    return nothing
end

function test_physical_stiffness_consistency(
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
)
    max_err_x = 0.0
    max_err_y = 0.0
    max_err_z = 0.0

    for op in physops.elements
        max_err_x = max(max_err_x, norm(op.weak.Sx - ref.M * op.Dx, Inf))
        max_err_y = max(max_err_y, norm(op.weak.Sy - ref.M * op.Dy, Inf))
        max_err_z = max(max_err_z, norm(op.weak.Sz - ref.M * op.Dz, Inf))
    end

    println("Physical stiffness consistency test")
    println("-----------------------------------")
    println("max ||Sx - M Dx||∞:      ", max_err_x)
    println("max ||Sy - M Dy||∞:      ", max_err_y)
    println("max ||Sz - M Dz||∞:      ", max_err_z)

    return maximum((max_err_x, max_err_y, max_err_z))
end

function test_physical_derivatives_linear(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
)
    max_err_x = 0.0
    max_err_y = 0.0
    max_err_z = 0.0

    ntets = size(mesh.tets, 2)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        # For N = 1, these are exactly the vertices.
        # For N > 1, this assumes you later map all reference interpolation nodes.
        # For now, evaluate the physical coordinates of all reference nodes.
        x = zeros(Float64, ref.Np)
        y = zeros(Float64, ref.Np)
        z = zeros(Float64, ref.Np)

        for i in 1:ref.Np
            p = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[i],
                ref.s[i],
                ref.t[i],
            )

            x[i] = p[1]
            y[i] = p[2]
            z[i] = p[3]
        end

        u = x .+ 2.0 .* y .+ 3.0 .* z

        ux = physops.elements[e].Dx * u
        uy = physops.elements[e].Dy * u
        uz = physops.elements[e].Dz * u

        max_err_x = max(max_err_x, maximum(abs.(ux .- 1.0)))
        max_err_y = max(max_err_y, maximum(abs.(uy .- 2.0)))
        max_err_z = max(max_err_z, maximum(abs.(uz .- 3.0)))
    end

    println("Physical derivative test")
    println("------------------------")
    println("u(x,y,z) = x + 2y + 3z")
    println("max error ux:            ", max_err_x)
    println("max error uy:            ", max_err_y)
    println("max error uz:            ", max_err_z)

    return nothing
end
