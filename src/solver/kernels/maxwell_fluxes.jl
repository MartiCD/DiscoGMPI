# -------------------------------------------------------------------------
# Select different types of NUMERICAL FLUXES
# -------------------------------------------------------------------------

function maxwell_impedance(; ε::Float64 = 1.0, μ::Float64 = 1.0)
    if ε <= 0.0 || μ <= 0.0
        error("ε and μ must be positive.")
    end

    return sqrt(μ / ε)
end


function maxwell_admittance(; ε::Float64 = 1.0, μ::Float64 = 1.0)
    return 1.0 / maxwell_impedance(; ε = ε, μ = μ)
end

function cross_n_cross_n_vec(
    n::NTuple{3, Float64},
    vx::AbstractVector{Float64},
    vy::AbstractVector{Float64},
    vz::AbstractVector{Float64},
)
    cx, cy, cz = cross_n_vec(n, vx, vy, vz)

    return cross_n_vec(n, cx, cy, cz)
end

function maxwell_surface_flux_values(
    minus,
    plus,
    n::NTuple{3, Float64};
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    dEx = plus.Ex .- minus.Ex
    dEy = plus.Ey .- minus.Ey
    dEz = plus.Ez .- minus.Ez

    dHx = plus.Hx .- minus.Hx
    dHy = plus.Hy .- minus.Hy
    dHz = plus.Hz .- minus.Hz

    # Central part:
    #
    # E correction =  1/2 n × (H⁺ - H⁻)
    # H correction = -1/2 n × (E⁺ - E⁻)
    fluxEx, fluxEy, fluxEz = cross_n_vec(n, dHx, dHy, dHz)
    fluxEx .*= 0.5
    fluxEy .*= 0.5
    fluxEz .*= 0.5

    fluxHx, fluxHy, fluxHz = cross_n_vec(n, dEx, dEy, dEz)
    fluxHx .*= -0.5
    fluxHy .*= -0.5
    fluxHz .*= -0.5

    if flux_kind == MaxwellFlux_Central
        return (
            fluxEx = fluxEx,
            fluxEy = fluxEy,
            fluxEz = fluxEz,
            fluxHx = fluxHx,
            fluxHy = fluxHy,
            fluxHz = fluxHz,
        )

    elseif flux_kind == MaxwellFlux_Upwind
        Z = maxwell_impedance(; ε = ε, μ = μ)
        Y = 1.0 / Z

        # Upwind penalty:
        #
        # E correction += -1/2 Y n × (n × (E⁺ - E⁻))
        # H correction += -1/2 Z n × (n × (H⁺ - H⁻))
        nnEx, nnEy, nnEz = cross_n_cross_n_vec(n, dEx, dEy, dEz)
        nnHx, nnHy, nnHz = cross_n_cross_n_vec(n, dHx, dHy, dHz)

        fluxEx .-= 0.5 * Y .* nnEx
        fluxEy .-= 0.5 * Y .* nnEy
        fluxEz .-= 0.5 * Y .* nnEz

        fluxHx .-= 0.5 * Z .* nnHx
        fluxHy .-= 0.5 * Z .* nnHy
        fluxHz .-= 0.5 * Z .* nnHz

        return (
            fluxEx = fluxEx,
            fluxEy = fluxEy,
            fluxEz = fluxEz,
            fluxHx = fluxHx,
            fluxHy = fluxHy,
            fluxHz = fluxHz,
        )

    else
        error("Unsupported Maxwell flux kind: $flux_kind")
    end
end

function maxwell_poisson_bracket_surface_flux_values(
    minus,
    plus,
    n::NTuple{3, Float64};
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    if flux_kind == MaxwellFlux_Central
        dHx = minus.Hx .- plus.Hx
        dHy = minus.Hy .- plus.Hy
        dHz = minus.Hz .- plus.Hz

        sumEx = minus.Ex .+ plus.Ex
        sumEy = minus.Ey .+ plus.Ey
        sumEz = minus.Ez .+ plus.Ez

        fluxEx, fluxEy, fluxEz = cross_n_vec(n, dHx, dHy, dHz)
        fluxEx .*= -0.5 / ε
        fluxEy .*= -0.5 / ε
        fluxEz .*= -0.5 / ε

        fluxHx, fluxHy, fluxHz = cross_n_vec(n, sumEx, sumEy, sumEz)
        fluxHx .*= -0.5 / μ
        fluxHy .*= -0.5 / μ
        fluxHz .*= -0.5 / μ

    elseif flux_kind == MaxwellFlux_Alternating
        θ = poisson_bracket_alternating_theta(n)
        α = 1.0 - θ

        dHx = minus.Hx .- plus.Hx
        dHy = minus.Hy .- plus.Hy
        dHz = minus.Hz .- plus.Hz

        Ehatx = θ .* plus.Ex .+ α .* minus.Ex
        Ehaty = θ .* plus.Ey .+ α .* minus.Ey
        Ehatz = θ .* plus.Ez .+ α .* minus.Ez

        fluxEx, fluxEy, fluxEz = cross_n_vec(n, dHx, dHy, dHz)
        fluxEx .*= -α / ε
        fluxEy .*= -α / ε
        fluxEz .*= -α / ε

        fluxHx, fluxHy, fluxHz = cross_n_vec(n, Ehatx, Ehaty, Ehatz)
        fluxHx .*= -1.0 / μ
        fluxHy .*= -1.0 / μ
        fluxHz .*= -1.0 / μ

    else
        error("Unsupported Poisson-bracket Maxwell flux kind: $flux_kind")
    end

    return (
        fluxEx = fluxEx,
        fluxEy = fluxEy,
        fluxEz = fluxEz,
        fluxHx = fluxHx,
        fluxHy = fluxHy,
        fluxHz = fluxHz,
    )
end

@inline function poisson_bracket_alternating_theta(
    n::NTuple{3, Float64},
)
    s = n[1] + n[2] + n[3]

    if s > 0.0
        return 1.0
    elseif s < 0.0
        return 0.0
    end

    return 0.5
end

@inline function cross_n_components(
    n::NTuple{3, Float64},
    vx::Float64,
    vy::Float64,
    vz::Float64,
)
    return (
        n[2] * vz - n[3] * vy,
        n[3] * vx - n[1] * vz,
        n[1] * vy - n[2] * vx,
    )
end

function maxwell_surface_flux_values!(
    flux::MaxwellFaceWorkspace,
    minus::MaxwellFaceWorkspace,
    plus::MaxwellFaceWorkspace,
    n::NTuple{3, Float64};
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
)
    if flux_kind == MaxwellFlux_Central
        @inbounds for q in eachindex(minus.Ex)
            dEx = plus.Ex[q] - minus.Ex[q]
            dEy = plus.Ey[q] - minus.Ey[q]
            dEz = plus.Ez[q] - minus.Ez[q]
            dHx = plus.Hx[q] - minus.Hx[q]
            dHy = plus.Hy[q] - minus.Hy[q]
            dHz = plus.Hz[q] - minus.Hz[q]

            nx_dH = cross_n_components(n, dHx, dHy, dHz)
            nx_dE = cross_n_components(n, dEx, dEy, dEz)

            flux.Ex[q] = 0.5 * nx_dH[1]
            flux.Ey[q] = 0.5 * nx_dH[2]
            flux.Ez[q] = 0.5 * nx_dH[3]
            flux.Hx[q] = -0.5 * nx_dE[1]
            flux.Hy[q] = -0.5 * nx_dE[2]
            flux.Hz[q] = -0.5 * nx_dE[3]
        end
    elseif flux_kind == MaxwellFlux_Upwind
        Z = maxwell_impedance(; ε = ε, μ = μ)
        Y = 1.0 / Z

        @inbounds for q in eachindex(minus.Ex)
            dEx = plus.Ex[q] - minus.Ex[q]
            dEy = plus.Ey[q] - minus.Ey[q]
            dEz = plus.Ez[q] - minus.Ez[q]
            dHx = plus.Hx[q] - minus.Hx[q]
            dHy = plus.Hy[q] - minus.Hy[q]
            dHz = plus.Hz[q] - minus.Hz[q]

            nx_dH = cross_n_components(n, dHx, dHy, dHz)
            nx_dE = cross_n_components(n, dEx, dEy, dEz)
            nn_dE = cross_n_components(n, nx_dE[1], nx_dE[2], nx_dE[3])
            nn_dH = cross_n_components(n, nx_dH[1], nx_dH[2], nx_dH[3])

            flux.Ex[q] = 0.5 * nx_dH[1] - 0.5 * Y * nn_dE[1]
            flux.Ey[q] = 0.5 * nx_dH[2] - 0.5 * Y * nn_dE[2]
            flux.Ez[q] = 0.5 * nx_dH[3] - 0.5 * Y * nn_dE[3]
            flux.Hx[q] = -0.5 * nx_dE[1] - 0.5 * Z * nn_dH[1]
            flux.Hy[q] = -0.5 * nx_dE[2] - 0.5 * Z * nn_dH[2]
            flux.Hz[q] = -0.5 * nx_dE[3] - 0.5 * Z * nn_dH[3]
        end
    else
        error("Unsupported Maxwell flux kind: $flux_kind")
    end

    return flux
end

function maxwell_poisson_bracket_surface_flux_values!(
    flux::MaxwellFaceWorkspace,
    minus::MaxwellFaceWorkspace,
    plus::MaxwellFaceWorkspace,
    n::NTuple{3, Float64};
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    ε::Float64,
    μ::Float64,
)
    if flux_kind == MaxwellFlux_Central
        @inbounds for q in eachindex(minus.Ex)
            dHx = minus.Hx[q] - plus.Hx[q]
            dHy = minus.Hy[q] - plus.Hy[q]
            dHz = minus.Hz[q] - plus.Hz[q]
            sumEx = minus.Ex[q] + plus.Ex[q]
            sumEy = minus.Ey[q] + plus.Ey[q]
            sumEz = minus.Ez[q] + plus.Ez[q]

            nx_dH = cross_n_components(n, dHx, dHy, dHz)
            nx_sumE = cross_n_components(n, sumEx, sumEy, sumEz)

            flux.Ex[q] = -0.5 * nx_dH[1] / ε
            flux.Ey[q] = -0.5 * nx_dH[2] / ε
            flux.Ez[q] = -0.5 * nx_dH[3] / ε
            flux.Hx[q] = -0.5 * nx_sumE[1] / μ
            flux.Hy[q] = -0.5 * nx_sumE[2] / μ
            flux.Hz[q] = -0.5 * nx_sumE[3] / μ
        end
    elseif flux_kind == MaxwellFlux_Alternating
        θ = poisson_bracket_alternating_theta(n)
        α = 1.0 - θ

        @inbounds for q in eachindex(minus.Ex)
            dHx = minus.Hx[q] - plus.Hx[q]
            dHy = minus.Hy[q] - plus.Hy[q]
            dHz = minus.Hz[q] - plus.Hz[q]
            Ehatx = θ * plus.Ex[q] + α * minus.Ex[q]
            Ehaty = θ * plus.Ey[q] + α * minus.Ey[q]
            Ehatz = θ * plus.Ez[q] + α * minus.Ez[q]

            nx_dH = cross_n_components(n, dHx, dHy, dHz)
            nx_Ehat = cross_n_components(n, Ehatx, Ehaty, Ehatz)

            flux.Ex[q] = -α * nx_dH[1] / ε
            flux.Ey[q] = -α * nx_dH[2] / ε
            flux.Ez[q] = -α * nx_dH[3] / ε
            flux.Hx[q] = -nx_Ehat[1] / μ
            flux.Hy[q] = -nx_Ehat[2] / μ
            flux.Hz[q] = -nx_Ehat[3] / μ
        end
    else
        error("Unsupported Poisson-bracket Maxwell flux kind: $flux_kind")
    end

    return flux
end

function require_poisson_bracket_surface_flux(formulation::PoissonBracketFormulation)
    if formulation.flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating)
        return nothing
    end

    throw(
        ArgumentError(
            "PoissonBracketFormulation supports MaxwellFlux_Central and " *
            "MaxwellFlux_Alternating. Upwind Maxwell fluxes are dissipative " *
            "and do not define the partitioned Poisson-bracket operator."
        ),
    )
end

function require_poisson_bracket_central_flux(formulation::PoissonBracketFormulation)
    if formulation.flux_kind != MaxwellFlux_Central
        throw(
            ArgumentError(
                "PoissonBracketFormulation only supports MaxwellFlux_Central. " *
                "Upwind Maxwell fluxes are dissipative and do not define the " *
                "partitioned Poisson-bracket operator."
            ),
        )
    end

    return nothing
end

function add_lifted_maxwell_surface_flux!(
    rhs::MaxwellRHS,
    elem::Int,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    face::Int,
    nodes::Vector{Int},
    flux,
    area::Float64,
)
    add_lifted_face_contribution!(
        rhs.rhsEx, elem, ref, fops, mappings,
        face, nodes, flux.fluxEx, area,
    )

    add_lifted_face_contribution!(
        rhs.rhsEy, elem, ref, fops, mappings,
        face, nodes, flux.fluxEy, area,
    )

    add_lifted_face_contribution!(
        rhs.rhsEz, elem, ref, fops, mappings,
        face, nodes, flux.fluxEz, area,
    )

    add_lifted_face_contribution!(
        rhs.rhsHx, elem, ref, fops, mappings,
        face, nodes, flux.fluxHx, area,
    )

    add_lifted_face_contribution!(
        rhs.rhsHy, elem, ref, fops, mappings,
        face, nodes, flux.fluxHy, area,
    )

    add_lifted_face_contribution!(
        rhs.rhsHz, elem, ref, fops, mappings,
        face, nodes, flux.fluxHz, area,
    )

    return rhs
end

function add_lifted_maxwell_surface_flux!(
    rhs::MaxwellRHS,
    elem::Int,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    face::Int,
    flux::MaxwellFaceWorkspace,
    area::Float64,
    workspace::MaxwellSurfaceWorkspace,
)
    add_lifted_face_contribution!(
        rhs.rhsEx, elem, fops, mappings,
        face, flux.Ex, area, workspace.lifted,
    )

    add_lifted_face_contribution!(
        rhs.rhsEy, elem, fops, mappings,
        face, flux.Ey, area, workspace.lifted,
    )

    add_lifted_face_contribution!(
        rhs.rhsEz, elem, fops, mappings,
        face, flux.Ez, area, workspace.lifted,
    )

    add_lifted_face_contribution!(
        rhs.rhsHx, elem, fops, mappings,
        face, flux.Hx, area, workspace.lifted,
    )

    add_lifted_face_contribution!(
        rhs.rhsHy, elem, fops, mappings,
        face, flux.Hy, area, workspace.lifted,
    )

    add_lifted_face_contribution!(
        rhs.rhsHz, elem, fops, mappings,
        face, flux.Hz, area, workspace.lifted,
    )

    return rhs
end

function add_lifted_permuted_maxwell_surface_flux!(
    rhs::MaxwellRHS,
    elem::Int,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    face::Int,
    flux::MaxwellFaceWorkspace,
    plus_to_minus_perm::AbstractVector{Int},
    area::Float64,
    workspace::MaxwellSurfaceWorkspace,
)
    unpermute_plus_face_values!(
        workspace.face_values,
        flux.Ex,
        plus_to_minus_perm,
    )
    add_lifted_face_contribution!(
        rhs.rhsEx, elem, fops, mappings,
        face, workspace.face_values, area, workspace.lifted,
    )

    unpermute_plus_face_values!(
        workspace.face_values,
        flux.Ey,
        plus_to_minus_perm,
    )
    add_lifted_face_contribution!(
        rhs.rhsEy, elem, fops, mappings,
        face, workspace.face_values, area, workspace.lifted,
    )

    unpermute_plus_face_values!(
        workspace.face_values,
        flux.Ez,
        plus_to_minus_perm,
    )
    add_lifted_face_contribution!(
        rhs.rhsEz, elem, fops, mappings,
        face, workspace.face_values, area, workspace.lifted,
    )

    unpermute_plus_face_values!(
        workspace.face_values,
        flux.Hx,
        plus_to_minus_perm,
    )
    add_lifted_face_contribution!(
        rhs.rhsHx, elem, fops, mappings,
        face, workspace.face_values, area, workspace.lifted,
    )

    unpermute_plus_face_values!(
        workspace.face_values,
        flux.Hy,
        plus_to_minus_perm,
    )
    add_lifted_face_contribution!(
        rhs.rhsHy, elem, fops, mappings,
        face, workspace.face_values, area, workspace.lifted,
    )

    unpermute_plus_face_values!(
        workspace.face_values,
        flux.Hz,
        plus_to_minus_perm,
    )
    add_lifted_face_contribution!(
        rhs.rhsHz, elem, fops, mappings,
        face, workspace.face_values, area, workspace.lifted,
    )

    return rhs
end

function maxwell_surface_workspace(
    workspace::Union{Nothing, MaxwellSurfaceWorkspace},
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
)
    if workspace === nothing
        return MaxwellSurfaceWorkspace(ref, fops)
    end

    matches_workspace(workspace, ref, fops) ||
        throw(ArgumentError("Maxwell surface workspace has incompatible dimensions."))
    return workspace
end

function maxwell_interior_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in flux_faces.interior
        maxwell_interior_surface_face_rhs!(
            rhs,
            U,
            ref,
            fops,
            mappings,
            ff;
            flux_kind = flux_kind,
            ε = ε,
            μ = μ,
            workspace = surface_workspace,
        )
    end

    return rhs
end

function maxwell_interior_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    return maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_interior_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in flux_faces.interior
        maxwell_interior_surface_face_rhs!(
            rhs,
            U,
            ref,
            fops,
            mappings,
            ff,
            formulation;
            ε = ε,
            μ = μ,
            workspace = surface_workspace,
        )
    end

    return rhs
end

function maxwell_periodic_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic::DGPeriodicFluxFaces;
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in periodic.faces
        maxwell_periodic_surface_face_rhs!(
            rhs,
            U,
            ref,
            fops,
            mappings,
            ff;
            flux_kind = flux_kind,
            ε = ε,
            μ = μ,
            workspace = surface_workspace,
        )
    end

    return rhs
end

function maxwell_periodic_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic::DGPeriodicFluxFaces,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    return maxwell_periodic_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        periodic;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_periodic_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic::DGPeriodicFluxFaces,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in periodic.faces
        maxwell_periodic_surface_face_rhs!(
            rhs,
            U,
            ref,
            fops,
            mappings,
            ff,
            formulation;
            ε = ε,
            μ = μ,
            workspace = surface_workspace,
        )
    end

    return rhs
end

function maxwell_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in flux_faces.boundary
        maxwell_boundary_surface_face_rhs!(
            rhs,
            U,
            ref,
            fops,
            mappings,
            ff,
            registry;
            flux_kind = flux_kind,
            ε = ε,
            μ = μ,
            workspace = surface_workspace,
        )
    end

    return rhs
end

function maxwell_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    return maxwell_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in flux_faces.boundary
        maxwell_boundary_surface_face_rhs!(
            rhs,
            U,
            ref,
            fops,
            mappings,
            ff,
            registry,
            formulation;
            ε = ε,
            μ = μ,
            workspace = surface_workspace,
        )
    end

    return rhs
end

function maxwell_interior_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::InteriorFluxFace;
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)
    tr = ff.trace
    n = ff.normal

    minus, plus = interior_face_traces!(surface_workspace, U, tr)

    maxwell_surface_flux_values!(
        surface_workspace.flux,
        minus,
        plus,
        n;
        flux_kind = flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_maxwell_surface_flux!(
        rhs,
        tr.minus_elem,
        fops,
        mappings,
        tr.minus_face,
        surface_workspace.flux,
        ff.area,
        surface_workspace,
    )

    nplus = (-n[1], -n[2], -n[3])

    maxwell_surface_flux_values!(
        surface_workspace.flux,
        plus,
        minus,
        nplus;
        flux_kind = flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_permuted_maxwell_surface_flux!(
        rhs,
        tr.plus_elem,
        fops,
        mappings,
        tr.plus_face,
        surface_workspace.flux,
        tr.plus_to_minus_perm,
        ff.area,
        surface_workspace,
    )

    return rhs
end

function maxwell_interior_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::InteriorFluxFace,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    return maxwell_interior_surface_face_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        ff;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_interior_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::InteriorFluxFace,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    tr = ff.trace
    n = ff.normal

    minus, plus = interior_face_traces!(surface_workspace, U, tr)

    maxwell_poisson_bracket_surface_flux_values!(
        surface_workspace.flux,
        minus,
        plus,
        n;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_maxwell_surface_flux!(
        rhs,
        tr.minus_elem,
        fops,
        mappings,
        tr.minus_face,
        surface_workspace.flux,
        ff.area,
        surface_workspace,
    )

    nplus = (-n[1], -n[2], -n[3])

    maxwell_poisson_bracket_surface_flux_values!(
        surface_workspace.flux,
        plus,
        minus,
        nplus;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_permuted_maxwell_surface_flux!(
        rhs,
        tr.plus_elem,
        fops,
        mappings,
        tr.plus_face,
        surface_workspace.flux,
        tr.plus_to_minus_perm,
        ff.area,
        surface_workspace,
    )

    return rhs
end

function maxwell_periodic_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::PeriodicFluxFace;
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)
    tr = ff.trace
    n = ff.normal

    minus, plus = periodic_face_traces!(surface_workspace, U, tr)

    maxwell_surface_flux_values!(
        surface_workspace.flux,
        minus,
        plus,
        n;
        flux_kind = flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_maxwell_surface_flux!(
        rhs,
        tr.minus_elem,
        fops,
        mappings,
        tr.minus_face,
        surface_workspace.flux,
        ff.area,
        surface_workspace,
    )

    nplus = (-n[1], -n[2], -n[3])

    maxwell_surface_flux_values!(
        surface_workspace.flux,
        plus,
        minus,
        nplus;
        flux_kind = flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_permuted_maxwell_surface_flux!(
        rhs,
        tr.plus_elem,
        fops,
        mappings,
        tr.plus_face,
        surface_workspace.flux,
        tr.plus_to_minus_perm,
        ff.area,
        surface_workspace,
    )

    return rhs
end

function maxwell_periodic_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::PeriodicFluxFace,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    return maxwell_periodic_surface_face_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        ff;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_periodic_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::PeriodicFluxFace,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    tr = ff.trace
    n = ff.normal

    minus, plus = periodic_face_traces!(surface_workspace, U, tr)

    maxwell_poisson_bracket_surface_flux_values!(
        surface_workspace.flux,
        minus,
        plus,
        n;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_maxwell_surface_flux!(
        rhs,
        tr.minus_elem,
        fops,
        mappings,
        tr.minus_face,
        surface_workspace.flux,
        ff.area,
        surface_workspace,
    )

    nplus = (-n[1], -n[2], -n[3])

    maxwell_poisson_bracket_surface_flux_values!(
        surface_workspace.flux,
        plus,
        minus,
        nplus;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
    )

    add_lifted_permuted_maxwell_surface_flux!(
        rhs,
        tr.plus_elem,
        fops,
        mappings,
        tr.plus_face,
        surface_workspace.flux,
        tr.plus_to_minus_perm,
        ff.area,
        surface_workspace,
    )

    return rhs
end

function maxwell_boundary_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::BoundaryFluxFace,
    registry::MaxwellBoundaryRegistry;
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)
    kind = boundary_kind(registry, ff.boundary_id)

    if kind == MaxwellBC_None
        return rhs

    elseif kind in (MaxwellBC_PEC, MaxwellBC_PMC, MaxwellBC_Absorbing)
        tr = ff.trace
        n = ff.normal

        minus = boundary_face_minus_trace!(surface_workspace, U, tr)
        plus = maxwell_boundary_plus_trace!(
            surface_workspace.plus,
            minus,
            n,
            kind;
            ε = ε,
            μ = μ,
        )

        maxwell_surface_flux_values!(
            surface_workspace.flux,
            minus,
            plus,
            n;
            flux_kind = flux_kind,
            ε = ε,
            μ = μ,
        )

        add_lifted_maxwell_surface_flux!(
            rhs,
            tr.elem,
            fops,
            mappings,
            tr.face,
            surface_workspace.flux,
            ff.area,
            surface_workspace,
        )

        return rhs

    else
        error("Unsupported Maxwell boundary kind $kind for boundary_id = $(ff.boundary_id).")
    end
end

function maxwell_boundary_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::BoundaryFluxFace,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    return maxwell_boundary_surface_face_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        ff,
        registry;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_boundary_surface_face_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    ff::BoundaryFluxFace,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    kind = boundary_kind(registry, ff.boundary_id)

    if kind == MaxwellBC_None
        return rhs

    elseif kind in (MaxwellBC_PEC, MaxwellBC_PMC, MaxwellBC_Absorbing)
        tr = ff.trace
        n = ff.normal

        if formulation.flux_kind == MaxwellFlux_Alternating
            if kind == MaxwellBC_PEC
                return rhs
            end

            throw(
                ArgumentError(
                    "PoissonBracketFormulation with MaxwellFlux_Alternating " *
                    "currently supports interior/periodic faces and PEC " *
                    "boundaries. Boundary kind $kind is not implemented.",
                ),
            )
        end

        minus = boundary_face_minus_trace!(surface_workspace, U, tr)
        plus = maxwell_boundary_plus_trace!(
            surface_workspace.plus,
            minus,
            n,
            kind;
            ε = ε,
            μ = μ,
        )

        maxwell_poisson_bracket_surface_flux_values!(
            surface_workspace.flux,
            minus,
            plus,
            n;
            flux_kind = formulation.flux_kind,
            ε = ε,
            μ = μ,
        )

        add_lifted_maxwell_surface_flux!(
            rhs,
            tr.elem,
            fops,
            mappings,
            tr.face,
            surface_workspace.flux,
            ff.area,
            surface_workspace,
        )

        return rhs

    else
        error("Unsupported Maxwell boundary kind $kind for boundary_id = $(ff.boundary_id).")
    end
end

struct MaxwellVolumeThreadWorkspace
    work::Vector{MaxwellElementScratch}
    nrows::Int
    nthreads::Int
end

function MaxwellVolumeThreadWorkspace(nrows::Int)
    return MaxwellVolumeThreadWorkspace(
        [MaxwellElementScratch(nrows) for _ in 1:Base.Threads.nthreads()],
        nrows,
        Base.Threads.nthreads(),
    )
end

function matches_workspace(
    workspace::MaxwellVolumeThreadWorkspace,
    nrows::Int,
)
    return workspace.nrows == nrows &&
           workspace.nthreads == Base.Threads.nthreads()
end

function maxwell_volume_thread_workspace!(
    backend::ThreadedBackend,
    U::MaxwellField,
)
    nrows = size(U.Ex, 1)
    workspace = backend.maxwell_volume_workspace[]

    if !(workspace isa MaxwellVolumeThreadWorkspace) ||
       !matches_workspace(workspace, nrows)
        workspace = MaxwellVolumeThreadWorkspace(nrows)
        backend.maxwell_volume_workspace[] = workspace
    end

    return workspace::MaxwellVolumeThreadWorkspace
end

function maxwell_volume_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    physops::DGPhysicalOperators;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing,MaxwellVolumeThreadWorkspace} = nothing,
)
    ne = size(U.Ex, 2)

    if ne == 0
        return rhs
    end

    work = workspace === nothing ?
           MaxwellVolumeThreadWorkspace(size(U.Ex, 1)).work :
           workspace.work

    Base.Threads.@threads :static for e in 1:ne
        scratch = work[Base.Threads.threadid()]

        @views begin
            op = physops.elements[e]

            curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Hx[:, e],
                U.Hy[:, e],
                U.Hz[:, e],
                op,
            )

            rhs.rhsEx[:, e] .+=  (1.0 / ε) .* scratch.curl_x
            rhs.rhsEy[:, e] .+=  (1.0 / ε) .* scratch.curl_y
            rhs.rhsEz[:, e] .+=  (1.0 / ε) .* scratch.curl_z

            curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Ex[:, e],
                U.Ey[:, e],
                U.Ez[:, e],
                op,
            )

            rhs.rhsHx[:, e] .+= -(1.0 / μ) .* scratch.curl_x
            rhs.rhsHy[:, e] .+= -(1.0 / μ) .* scratch.curl_y
            rhs.rhsHz[:, e] .+= -(1.0 / μ) .* scratch.curl_z
        end
    end

    return rhs
end

function maxwell_volume_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing,MaxwellVolumeThreadWorkspace} = nothing,
)
    return maxwell_volume_rhs_threaded!(
        rhs,
        U,
        physops;
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_volume_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing,MaxwellVolumeThreadWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)

    ne = size(U.Ex, 2)

    if ne == 0
        return rhs
    end

    work = workspace === nothing ?
           MaxwellVolumeThreadWorkspace(size(U.Ex, 1)).work :
           workspace.work
    mass_factor = cholesky(ref.M)

    Base.Threads.@threads :static for e in 1:ne
        scratch = work[Base.Threads.threadid()]

        @views begin
            op = physops.elements[e]
            Sx, Sy, Sz = physical_weak_derivative_matrices(op)
            SxT, SyT, SzT = physical_weak_derivative_transpose_matrices(op)

            weak_curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Hx[:, e],
                U.Hy[:, e],
                U.Hz[:, e],
                Sx,
                Sy,
                Sz,
            )

            ldiv!(mass_factor, scratch.curl_x)
            ldiv!(mass_factor, scratch.curl_y)
            ldiv!(mass_factor, scratch.curl_z)

            rhs.rhsEx[:, e] .+= (1.0 / ε) .* scratch.curl_x
            rhs.rhsEy[:, e] .+= (1.0 / ε) .* scratch.curl_y
            rhs.rhsEz[:, e] .+= (1.0 / ε) .* scratch.curl_z

            weak_curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Ex[:, e],
                U.Ey[:, e],
                U.Ez[:, e],
                SxT,
                SyT,
                SzT,
            )

            ldiv!(mass_factor, scratch.curl_x)
            ldiv!(mass_factor, scratch.curl_y)
            ldiv!(mass_factor, scratch.curl_z)

            rhs.rhsHx[:, e] .+= (1.0 / μ) .* scratch.curl_x
            rhs.rhsHy[:, e] .+= (1.0 / μ) .* scratch.curl_y
            rhs.rhsHz[:, e] .+= (1.0 / μ) .* scratch.curl_z
        end
    end

    return rhs
end

struct MaxwellSurfaceThreadWorkspace
    work::Vector{MaxwellSurfaceWorkspace}
    nfp::Int
    nrows::Int
    nthreads::Int
end

function MaxwellSurfaceThreadWorkspace(
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
)
    return MaxwellSurfaceThreadWorkspace(
        [
            MaxwellSurfaceWorkspace(ref, fops)
            for _ in 1:Base.Threads.nthreads()
        ],
        length(fops.face_nodes[1]),
        ref.Np,
        Base.Threads.nthreads(),
    )
end

function matches_workspace(
    workspace::MaxwellSurfaceThreadWorkspace,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
)
    return workspace.nfp == length(fops.face_nodes[1]) &&
           workspace.nrows == ref.Np &&
           workspace.nthreads == Base.Threads.nthreads()
end

function maxwell_surface_thread_workspace!(
    backend::ThreadedBackend,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
)
    workspace = backend.maxwell_surface_workspace[]

    if !(workspace isa MaxwellSurfaceThreadWorkspace) ||
       !matches_workspace(workspace, ref, fops)
        workspace = MaxwellSurfaceThreadWorkspace(ref, fops)
        backend.maxwell_surface_workspace[] = workspace
    end

    return workspace::MaxwellSurfaceThreadWorkspace
end

function maxwell_surface_workspace!(
    backend::SerialBackend,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
)
    workspace = backend.maxwell_surface_workspace[]

    if !(workspace isa MaxwellSurfaceWorkspace) ||
       !matches_workspace(workspace, ref, fops)
        workspace = MaxwellSurfaceWorkspace(ref, fops)
        backend.maxwell_surface_workspace[] = workspace
    end

    return workspace::MaxwellSurfaceWorkspace
end

function maxwell_surface_workspace!(
    backend::ThreadedBackend,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
)
    return maxwell_surface_thread_workspace!(backend, ref, fops).work[1]
end

function maxwell_interior_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    work = workspace === nothing ?
           MaxwellSurfaceThreadWorkspace(ref, fops).work :
           workspace.work

    for color in flux_faces.interior_colors
        Base.Threads.@threads :static for j in eachindex(color)
            surface_workspace = work[Base.Threads.threadid()]
            maxwell_interior_surface_face_rhs!(
                rhs,
                U,
                ref,
                fops,
                mappings,
                flux_faces.interior[color[j]];
                flux_kind = flux_kind,
                ε = ε,
                μ = μ,
                workspace = surface_workspace,
            )
        end
    end

    return rhs
end

function maxwell_interior_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    return maxwell_interior_surface_rhs_threaded!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_interior_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    work = workspace === nothing ?
           MaxwellSurfaceThreadWorkspace(ref, fops).work :
           workspace.work

    for color in flux_faces.interior_colors
        Base.Threads.@threads :static for j in eachindex(color)
            surface_workspace = work[Base.Threads.threadid()]
            maxwell_interior_surface_face_rhs!(
                rhs,
                U,
                ref,
                fops,
                mappings,
                flux_faces.interior[color[j]],
                formulation;
                ε = ε,
                μ = μ,
                workspace = surface_workspace,
            )
        end
    end

    return rhs
end

function maxwell_periodic_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic::DGPeriodicFluxFaces;
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    work = workspace === nothing ?
           MaxwellSurfaceThreadWorkspace(ref, fops).work :
           workspace.work

    for color in periodic.colors
        Base.Threads.@threads :static for j in eachindex(color)
            surface_workspace = work[Base.Threads.threadid()]
            maxwell_periodic_surface_face_rhs!(
                rhs,
                U,
                ref,
                fops,
                mappings,
                periodic.faces[color[j]];
                flux_kind = flux_kind,
                ε = ε,
                μ = μ,
                workspace = surface_workspace,
            )
        end
    end

    return rhs
end

function maxwell_periodic_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic::DGPeriodicFluxFaces,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    return maxwell_periodic_surface_rhs_threaded!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        periodic;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_periodic_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic::DGPeriodicFluxFaces,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    work = workspace === nothing ?
           MaxwellSurfaceThreadWorkspace(ref, fops).work :
           workspace.work

    for color in periodic.colors
        Base.Threads.@threads :static for j in eachindex(color)
            surface_workspace = work[Base.Threads.threadid()]
            maxwell_periodic_surface_face_rhs!(
                rhs,
                U,
                ref,
                fops,
                mappings,
                periodic.faces[color[j]],
                formulation;
                ε = ε,
                μ = μ,
                workspace = surface_workspace,
            )
        end
    end

    return rhs
end

function maxwell_boundary_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    flux_kind::MaxwellFluxKind,
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    work = workspace === nothing ?
           MaxwellSurfaceThreadWorkspace(ref, fops).work :
           workspace.work

    for color in flux_faces.boundary_colors
        Base.Threads.@threads :static for j in eachindex(color)
            surface_workspace = work[Base.Threads.threadid()]
            maxwell_boundary_surface_face_rhs!(
                rhs,
                U,
                ref,
                fops,
                mappings,
                flux_faces.boundary[color[j]],
                registry;
                flux_kind = flux_kind,
                ε = ε,
                μ = μ,
                workspace = surface_workspace,
            )
        end
    end

    return rhs
end

function maxwell_boundary_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    return maxwell_boundary_surface_rhs_threaded!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry;
        flux_kind = formulation.flux_kind,
        ε = ε,
        μ = μ,
        workspace = workspace,
    )
end

function maxwell_boundary_surface_rhs_threaded!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64,
    μ::Float64,
    workspace::Union{Nothing, MaxwellSurfaceThreadWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)
    work = workspace === nothing ?
           MaxwellSurfaceThreadWorkspace(ref, fops).work :
           workspace.work

    for color in flux_faces.boundary_colors
        Base.Threads.@threads :static for j in eachindex(color)
            surface_workspace = work[Base.Threads.threadid()]
            maxwell_boundary_surface_face_rhs!(
                rhs,
                U,
                ref,
                fops,
                mappings,
                flux_faces.boundary[color[j]],
                registry,
                formulation;
                ε = ε,
                μ = μ,
                workspace = surface_workspace,
            )
        end
    end

    return rhs
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    formulation = HesthavenWarburtonFormulation(flux_kind)

    return maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    fill_maxwell_rhs!(rhs, 0.0)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    maxwell_volume_rhs!(
        rhs,
        U,
        ref,
        physops,
        formulation;
        ε = ε,
        μ = μ,
        reset = false,
    )

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)

    fill_maxwell_rhs!(rhs, 0.0)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    maxwell_volume_rhs!(
        rhs,
        U,
        ref,
        physops,
        formulation;
        ε = ε,
        μ = μ,
        reset = false,
    )

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return maxwell_rhs!(
        rhs,
        U,
        dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return maxwell_rhs!(
        rhs,
        U,
        dg,
        registry,
        formulation,
        dg.backend;
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation,
    backend::ThreadedBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    fill_maxwell_rhs!(rhs, 0.0)
    volume_workspace = maxwell_volume_thread_workspace!(backend, U)
    surface_workspace = maxwell_surface_thread_workspace!(
        backend,
        dg.ref,
        dg.fops,
    )

    maxwell_volume_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation;
        ε = ε,
        μ = μ,
        workspace = volume_workspace,
    )

    maxwell_interior_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation,
    backend::SerialBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    surface_workspace = maxwell_surface_workspace!(backend, dg.ref, dg.fops)
    return maxwell_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.physops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return maxwell_rhs!(
        rhs,
        U,
        dg,
        registry,
        formulation,
        dg.backend;
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    backend::SerialBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    surface_workspace = maxwell_surface_workspace!(backend, dg.ref, dg.fops)
    return maxwell_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.physops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    backend::ThreadedBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    require_poisson_bracket_surface_flux(formulation)

    fill_maxwell_rhs!(rhs, 0.0)
    volume_workspace = maxwell_volume_thread_workspace!(backend, U)
    surface_workspace = maxwell_surface_thread_workspace!(
        backend,
        dg.ref,
        dg.fops,
    )

    maxwell_volume_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation;
        ε = ε,
        μ = μ,
        workspace = volume_workspace,
    )

    maxwell_interior_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    formulation = HesthavenWarburtonFormulation(flux_kind)

    return maxwell_rhs_periodic!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    fill_maxwell_rhs!(rhs, 0.0)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    maxwell_volume_rhs!(
        rhs,
        U,
        ref,
        physops,
        formulation;
        ε = ε,
        μ = μ,
        reset = false,
    )

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_periodic_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        periodic,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    require_poisson_bracket_surface_flux(formulation)

    fill_maxwell_rhs!(rhs, 0.0)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    maxwell_volume_rhs!(
        rhs,
        U,
        ref,
        physops,
        formulation;
        ε = ε,
        μ = μ,
        reset = false,
    )

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_periodic_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        periodic,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return maxwell_rhs_periodic!(
        rhs,
        U,
        dg,
        periodic,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation,
    backend::ThreadedBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    fill_maxwell_rhs!(rhs, 0.0)
    volume_workspace = maxwell_volume_thread_workspace!(backend, U)
    surface_workspace = maxwell_surface_thread_workspace!(
        backend,
        dg.ref,
        dg.fops,
    )

    maxwell_volume_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation;
        ε = ε,
        μ = μ,
        workspace = volume_workspace,
    )

    maxwell_interior_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_periodic_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        periodic,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return maxwell_rhs_periodic!(
        rhs,
        U,
        dg,
        periodic,
        registry,
        formulation,
        dg.backend;
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation,
    backend::SerialBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    surface_workspace = maxwell_surface_workspace!(backend, dg.ref, dg.fops)
    return maxwell_rhs_periodic!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.physops,
        dg.mappings,
        dg.flux_faces,
        periodic,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return maxwell_rhs_periodic!(
        rhs,
        U,
        dg,
        periodic,
        registry,
        formulation,
        dg.backend;
        ε = ε,
        μ = μ,
    )
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    backend::SerialBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    surface_workspace = maxwell_surface_workspace!(backend, dg.ref, dg.fops)
    return maxwell_rhs_periodic!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.physops,
        dg.mappings,
        dg.flux_faces,
        periodic,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    backend::ThreadedBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    require_poisson_bracket_surface_flux(formulation)

    fill_maxwell_rhs!(rhs, 0.0)
    volume_workspace = maxwell_volume_thread_workspace!(backend, U)
    surface_workspace = maxwell_surface_thread_workspace!(
        backend,
        dg.ref,
        dg.fops,
    )

    maxwell_volume_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation;
        ε = ε,
        μ = μ,
        workspace = volume_workspace,
    )

    maxwell_interior_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_periodic_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        periodic,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    maxwell_boundary_surface_rhs_threaded!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
        workspace = surface_workspace,
    )

    return rhs
end

function make_maxwell_periodic_rhs_function(
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return make_maxwell_periodic_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function make_maxwell_periodic_rhs_function(
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        maxwell_rhs_periodic!(
            rhs,
            U,
            ref,
            fops,
            physops,
            mappings,
            flux_faces,
            periodic_faces,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )

        return rhs
    end
end

function make_maxwell_periodic_rhs_function(
    dg::DGDiscretization,
    periodic_faces::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return make_maxwell_periodic_rhs_function(
        dg,
        periodic_faces,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function make_maxwell_periodic_rhs_function(
    dg::DGDiscretization,
    periodic_faces::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        maxwell_rhs_periodic!(
            rhs,
            U,
            dg,
            periodic_faces,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )

        return rhs
    end
end

function test_maxwell_upwind_interior_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    Efun = (x, y, z) -> (
        2.0 * y + 3.0 * z,
        4.0 * z + 5.0 * x,
        6.0 * x + 7.0 * y,
    )

    Hfun = (x, y, z) -> (
        3.0 * y - 2.0 * z,
        5.0 * z - 4.0 * x,
        7.0 * x - 6.0 * y,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        flux_kind = MaxwellFlux_Upwind,
        ε = 1.0,
        μ = 1.0,
    )

    maxerr = maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))

    println("Maxwell upwind interior surface consistency test")
    println("------------------------------------------------")
    println("max surface RHS: ", maxerr)

    if maxerr < 1e-10
        println("✓ upwind interior surface operator vanishes for continuous field")
    else
        println("⚠ upwind interior surface operator failed continuous-field test")
    end

    return nothing
end

function test_maxwell_upwind_periodic_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    periodic_faces::DGPeriodicFluxFaces,
)
    Efun = (x, y, z) -> (
        sin(2.0 * pi * x) * cos(2.0 * pi * y),
        sin(2.0 * pi * y) * cos(2.0 * pi * z),
        sin(2.0 * pi * z) * cos(2.0 * pi * x),
    )

    Hfun = (x, y, z) -> (
        cos(2.0 * pi * x) * sin(2.0 * pi * z),
        cos(2.0 * pi * y) * sin(2.0 * pi * x),
        cos(2.0 * pi * z) * sin(2.0 * pi * y),
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_periodic_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        periodic_faces;
        flux_kind = MaxwellFlux_Upwind,
        ε = 1.0,
        μ = 1.0,
    )

    maxerr = maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))

    println("Maxwell upwind periodic surface consistency test")
    println("-----------------------------------------------")
    println("max periodic surface RHS: ", maxerr)

    if maxerr < 1e-10
        println("✓ upwind periodic surface operator vanishes for periodic field")
    else
        println("⚠ upwind periodic surface operator failed periodic-field test")
    end

    return nothing
end

function test_maxwell_upwind_periodic_rhs_zero_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    registry = periodic_box_with_pec_sphere_registry()

    maxwell_rhs_periodic!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
        flux_kind = MaxwellFlux_Upwind,
    )

    maxerr = maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))

    println("Maxwell upwind periodic RHS zero-field test")
    println("-------------------------------------------")
    println("max error:   ", maxerr)

    if maxerr < 1e-12
        println("✓ upwind periodic Maxwell RHS vanishes for zero field")
    else
        println("⚠ upwind periodic Maxwell RHS zero-field test failed")
    end

    return nothing
end
