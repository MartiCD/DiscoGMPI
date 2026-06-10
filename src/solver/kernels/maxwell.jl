# -------------------------------------------------------------------------
# MAXWELL
#--------------------------------------------------------------------------

function maxwell_minus_trace(U::MaxwellField, tr::InteriorTraceMap)
    nodes = tr.minus_nodes
    e = tr.minus_elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],
        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end

function maxwell_plus_trace(U::MaxwellField, tr::InteriorTraceMap)
    nodes = tr.plus_nodes[tr.plus_to_minus_perm]
    e = tr.plus_elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],
        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end

function maxwell_boundary_minus_trace(U::MaxwellField, tr::BoundaryTraceMap)
    nodes = tr.nodes
    e = tr.elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],
        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end

function reflect_pec_E(
    Ex::Float64,
    Ey::Float64,
    Ez::Float64,
    n::NTuple{3, Float64},
)
    ndotE = n[1] * Ex + n[2] * Ey + n[3] * Ez

    Epx = -Ex + 2.0 * ndotE * n[1]
    Epy = -Ey + 2.0 * ndotE * n[2]
    Epz = -Ez + 2.0 * ndotE * n[3]

    return Epx, Epy, Epz
end

function pec_boundary_plus_trace(
    minus_trace,
    n::NTuple{3, Float64},
)
    Nfp = length(minus_trace.Ex)

    ExP = similar(minus_trace.Ex)
    EyP = similar(minus_trace.Ey)
    EzP = similar(minus_trace.Ez)

    HxP = copy(minus_trace.Hx)
    HyP = copy(minus_trace.Hy)
    HzP = copy(minus_trace.Hz)

    for q in 1:Nfp
        ExP[q], EyP[q], EzP[q] = reflect_pec_E(
            minus_trace.Ex[q],
            minus_trace.Ey[q],
            minus_trace.Ez[q],
            n,
        )
    end

    return (
        Ex = ExP,
        Ey = EyP,
        Ez = EzP,
        Hx = HxP,
        Hy = HyP,
        Hz = HzP,
    )
end

function interpolate_maxwell_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    Efun::Function,
    Hfun::Function,
)
    ntets = size(mesh.tets, 2)

    Ex = zeros(Float64, ref.Np, ntets)
    Ey = zeros(Float64, ref.Np, ntets)
    Ez = zeros(Float64, ref.Np, ntets)

    Hx = zeros(Float64, ref.Np, ntets)
    Hy = zeros(Float64, ref.Np, ntets)
    Hz = zeros(Float64, ref.Np, ntets)

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

            Exi, Eyi, Ezi = Efun(x, y, z)
            Hxi, Hyi, Hzi = Hfun(x, y, z)

            Ex[i, e] = Exi
            Ey[i, e] = Eyi
            Ez[i, e] = Ezi

            Hx[i, e] = Hxi
            Hy[i, e] = Hyi
            Hz[i, e] = Hzi
        end
    end

    return MaxwellField(Ex, Ey, Ez, Hx, Hy, Hz)
end

function cross_norm_E(
    n::NTuple{3, Float64},
    Ex::Float64,
    Ey::Float64,
    Ez::Float64,
)
    cx = n[2] * Ez - n[3] * Ey
    cy = n[3] * Ex - n[1] * Ez
    cz = n[1] * Ey - n[2] * Ex

    return sqrt(cx * cx + cy * cy + cz * cz)
end

function test_maxwell_pec_boundary_reflection(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    Efun = (x, y, z) -> (
        x + 0.25 * y,
        2.0 * y - 0.5 * z,
        3.0 * z + 0.1 * x,
    )

    Hfun = (x, y, z) -> (
        z + 0.2 * x,
        x - 0.3 * y,
        y + 0.4 * z,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    max_cross = 0.0
    max_normal_error = 0.0
    max_H_error = 0.0

    checked_faces = 0
    checked_nodes = 0
    worst_face = 0

    for i in eachindex(flux_faces.boundary)
        ff = flux_faces.boundary[i]

        if ff.boundary_id != pec_boundary_id
            continue
        end

        checked_faces += 1

        tr = ff.trace
        n = ff.normal

        minus = maxwell_boundary_minus_trace(U, tr)
        plus = pec_boundary_plus_trace(minus, n)

        for q in eachindex(minus.Ex)
            Eavg_x = 0.5 * (minus.Ex[q] + plus.Ex[q])
            Eavg_y = 0.5 * (minus.Ey[q] + plus.Ey[q])
            Eavg_z = 0.5 * (minus.Ez[q] + plus.Ez[q])

            # PEC condition on the averaged/interface field.
            cross_val = cross_norm_E(n, Eavg_x, Eavg_y, Eavg_z)

            if cross_val > max_cross
                max_cross = cross_val
                worst_face = i
            end

            # Optional: verify Eavg is exactly the normal projection of Eminus.
            ndotE = n[1] * minus.Ex[q] +
                    n[2] * minus.Ey[q] +
                    n[3] * minus.Ez[q]

            En_x = ndotE * n[1]
            En_y = ndotE * n[2]
            En_z = ndotE * n[3]

            normal_error = sqrt(
                (Eavg_x - En_x)^2 +
                (Eavg_y - En_y)^2 +
                (Eavg_z - En_z)^2
            )

            max_normal_error = max(max_normal_error, normal_error)

            # PEC reflection keeps H unchanged in this simple exterior-state construction.
            H_error = sqrt(
                (plus.Hx[q] - minus.Hx[q])^2 +
                (plus.Hy[q] - minus.Hy[q])^2 +
                (plus.Hz[q] - minus.Hz[q])^2
            )

            max_H_error = max(max_H_error, H_error)

            checked_nodes += 1
        end
    end

    println("Maxwell PEC boundary reflection test")
    println("------------------------------------")
    println("PEC boundary_id:                  ", pec_boundary_id)
    println("checked PEC faces:                ", checked_faces)
    println("checked PEC face nodes:           ", checked_nodes)
    println("max ||n × Eavg||:                 ", max_cross)
    println("max ||Eavg - (n·E)n||:            ", max_normal_error)
    println("max ||Hplus - Hminus||:           ", max_H_error)
    println("worst boundary face id:           ", worst_face)

    if checked_faces == 0
        println("⚠ no PEC boundary faces found")
    elseif max_cross < 1e-12 &&
           max_normal_error < 1e-12 &&
           max_H_error < 1e-12
        println("✓ PEC reflection state satisfies n × Eavg = 0")
    else
        println("⚠ PEC reflection test has larger-than-expected error")
    end

    return nothing
end

function curl_element(
    Fx::AbstractVector{Float64},
    Fy::AbstractVector{Float64},
    Fz::AbstractVector{Float64},
    op::PhysicalElementOperators,
)
    scratch = MaxwellElementScratch(length(Fx))

    curl_element!(
        scratch.curl_x,
        scratch.curl_y,
        scratch.curl_z,
        scratch.tmp,
        Fx,
        Fy,
        Fz,
        op,
    )

    return copy(scratch.curl_x), copy(scratch.curl_y), copy(scratch.curl_z)
end

struct MaxwellElementScratch
    tmp::Vector{Float64}
    curl_x::Vector{Float64}
    curl_y::Vector{Float64}
    curl_z::Vector{Float64}
end

function MaxwellElementScratch(n::Int)
    return MaxwellElementScratch(
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
    )
end

function curl_element!(
    curl_x::AbstractVector{Float64},
    curl_y::AbstractVector{Float64},
    curl_z::AbstractVector{Float64},
    tmp::AbstractVector{Float64},
    Fx::AbstractVector{Float64},
    Fy::AbstractVector{Float64},
    Fz::AbstractVector{Float64},
    op::PhysicalElementOperators,
)
    weak_curl_element!(
        curl_x,
        curl_y,
        curl_z,
        tmp,
        Fx,
        Fy,
        Fz,
        op.Dx,
        op.Dy,
        op.Dz,
    )

    return curl_x, curl_y, curl_z
end

function weak_curl_element!(
    curl_x::AbstractVector{Float64},
    curl_y::AbstractVector{Float64},
    curl_z::AbstractVector{Float64},
    tmp::AbstractVector{Float64},
    Fx::AbstractVector{Float64},
    Fy::AbstractVector{Float64},
    Fz::AbstractVector{Float64},
    Dx::AbstractMatrix{Float64},
    Dy::AbstractMatrix{Float64},
    Dz::AbstractMatrix{Float64},
)
    mul!(curl_x, Dy, Fz)
    mul!(tmp, Dz, Fy)
    curl_x .-= tmp

    mul!(curl_y, Dz, Fx)
    mul!(tmp, Dx, Fz)
    curl_y .-= tmp

    mul!(curl_z, Dx, Fy)
    mul!(tmp, Dy, Fx)
    curl_z .-= tmp

    return curl_x, curl_y, curl_z
end

function similar_maxwell_rhs(U::MaxwellField)
    return MaxwellRHS(
        similar(U.Ex),
        similar(U.Ey),
        similar(U.Ez),
        similar(U.Hx),
        similar(U.Hy),
        similar(U.Hz),
    )
end


function fill_maxwell_rhs!(rhs::MaxwellRHS, value::Float64)
    fill!(rhs.rhsEx, value)
    fill!(rhs.rhsEy, value)
    fill!(rhs.rhsEz, value)

    fill!(rhs.rhsHx, value)
    fill!(rhs.rhsHy, value)
    fill!(rhs.rhsHz, value)

    return rhs
end

function add_maxwell_rhs!(dest::MaxwellRHS, src::MaxwellRHS)
    dest.rhsEx .+= src.rhsEx
    dest.rhsEy .+= src.rhsEy
    dest.rhsEz .+= src.rhsEz

    dest.rhsHx .+= src.rhsHx
    dest.rhsHy .+= src.rhsHy
    dest.rhsHz .+= src.rhsHz

    return dest
end


function default_maxwell_boundary_registry()
    return MaxwellBoundaryRegistry(
        Dict(
            1  => MaxwellBC_None,
            2  => MaxwellBC_None,
            3  => MaxwellBC_None,
            4  => MaxwellBC_None,
            5  => MaxwellBC_None,
            6  => MaxwellBC_None,
            10 => MaxwellBC_PEC,
        ),
    )
end

function boundary_kind(registry::MaxwellBoundaryRegistry, boundary_id::Int)
    return get(registry.kinds, boundary_id, MaxwellBC_None)
end

function maxwell_volume_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    physops::DGPhysicalOperators;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    reset::Bool = true,
)
    if reset
        fill_maxwell_rhs!(rhs, 0.0)
    end

    ne = size(U.Ex, 2)
    scratch = MaxwellElementScratch(size(U.Ex, 1))

    for e in 1:ne
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

function maxwell_volume_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
    formulation::HesthavenWarburtonFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    reset::Bool = true,
)
    return maxwell_volume_rhs!(
        rhs,
        U,
        physops;
        ε = ε,
        μ = μ,
        reset = reset,
    )
end

function maxwell_volume_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    reset::Bool = true,
)
    if reset
        fill_maxwell_rhs!(rhs, 0.0)
    end

    ne = size(U.Ex, 2)
    scratch = MaxwellElementScratch(size(U.Ex, 1))
    mass_factor = cholesky(ref.M)

    for e in 1:ne
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


function test_maxwell_volume_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
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

    maxwell_volume_rhs!(
        rhs,
        U,
        physops;
        ε = 1.0,
        μ = 1.0,
    )

    exact_rhsE = (-11.0, -9.0, -7.0)
    exact_rhsH = (-3.0, 3.0, -3.0)

    errEx = maximum(abs.(rhs.rhsEx .- exact_rhsE[1]))
    errEy = maximum(abs.(rhs.rhsEy .- exact_rhsE[2]))
    errEz = maximum(abs.(rhs.rhsEz .- exact_rhsE[3]))

    errHx = maximum(abs.(rhs.rhsHx .- exact_rhsH[1]))
    errHy = maximum(abs.(rhs.rhsHy .- exact_rhsH[2]))
    errHz = maximum(abs.(rhs.rhsHz .- exact_rhsH[3]))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell volume operator test")
    println("----------------------------")
    println("Expected rhsE:             ", exact_rhsE)
    println("Expected rhsH:             ", exact_rhsH)
    println("max error rhsEx:           ", errEx)
    println("max error rhsEy:           ", errEy)
    println("max error rhsEz:           ", errEz)
    println("max error rhsHx:           ", errHx)
    println("max error rhsHy:           ", errHy)
    println("max error rhsHz:           ", errHz)
    println("max error:                 ", maxerr)

    if maxerr < 1e-10
        println("✓ Maxwell volume curl operator is consistent")
    else
        println("⚠ Maxwell volume curl operator has larger-than-expected error")
    end

    return nothing
end


function cross_n_vec(
    n::NTuple{3, Float64},
    vx::AbstractVector{Float64},
    vy::AbstractVector{Float64},
    vz::AbstractVector{Float64},
)
    cx = n[2] .* vz .- n[3] .* vy
    cy = n[3] .* vx .- n[1] .* vz
    cz = n[1] .* vy .- n[2] .* vx

    return cx, cy, cz
end


function unpermute_plus_face_values(
    values_minus_order::AbstractVector{Float64},
    plus_to_minus_perm::Vector{Int},
)
    values_plus_order = similar(values_minus_order)

    for i in eachindex(plus_to_minus_perm)
        values_plus_order[plus_to_minus_perm[i]] = values_minus_order[i]
    end

    return values_plus_order
end

function add_lifted_face_contribution!(
    rhs_component::Matrix{Float64},
    elem::Int,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    local_face::Int,
    face_nodes::Vector{Int},
    face_values::AbstractVector{Float64},
    physical_face_area::Float64,
)
    reference_area = reference_face_area(local_face)

    surface_scale = physical_face_area / reference_area
    volume_scale = mappings.tet_mappings[elem].absdetJ

    face_rhs = fops.face_mass[local_face] * face_values

    embedded = zeros(Float64, ref.Np)
    embedded[face_nodes] .= face_rhs

    rhs_component[:, elem] .+= (surface_scale / volume_scale) .* (ref.M \ embedded)

    return nothing
end


function test_maxwell_interior_surface_operator(
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
        flux_faces,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell interior surface operator test")
    println("--------------------------------------")
    println("max |rhsEx|:               ", errEx)
    println("max |rhsEy|:               ", errEy)
    println("max |rhsEz|:               ", errEz)
    println("max |rhsHx|:               ", errHx)
    println("max |rhsHy|:               ", errHy)
    println("max |rhsHz|:               ", errHz)
    println("max error:                 ", maxerr)

    if maxerr < 1e-10
        println("✓ Maxwell interior surface operator vanishes for continuous field")
    else
        println("⚠ Maxwell interior surface operator is not vanishing as expected")
    end

    return nothing
end

function maxwell_boundary_plus_trace(
    minus_trace,
    normal::NTuple{3, Float64},
    boundary_id::Int;
    pec_boundary_id::Int = 10,
)
    if boundary_id == pec_boundary_id
        return pec_boundary_plus_trace(minus_trace, normal)
    else
        error(
            "No Maxwell boundary state implemented for boundary_id = $boundary_id. " *
            "Currently only PEC boundary_id = $pec_boundary_id is supported."
        )
    end
end

function maxwell_boundary_surface_flux_values(
    minus,
    plus,
    n::NTuple{3, Float64},
)
    # Central boundary state.
    Ehat_x = 0.5 .* (minus.Ex .+ plus.Ex)
    Ehat_y = 0.5 .* (minus.Ey .+ plus.Ey)
    Ehat_z = 0.5 .* (minus.Ez .+ plus.Ez)

    Hhat_x = 0.5 .* (minus.Hx .+ plus.Hx)
    Hhat_y = 0.5 .* (minus.Hy .+ plus.Hy)
    Hhat_z = 0.5 .* (minus.Hz .+ plus.Hz)

    # E correction: n × (Hhat - Hminus)
    dHx = Hhat_x .- minus.Hx
    dHy = Hhat_y .- minus.Hy
    dHz = Hhat_z .- minus.Hz

    fluxEx, fluxEy, fluxEz = cross_n_vec(n, dHx, dHy, dHz)

    # H correction: -n × (Ehat - Eminus)
    dEx = Ehat_x .- minus.Ex
    dEy = Ehat_y .- minus.Ey
    dEz = Ehat_z .- minus.Ez

    fluxHx, fluxHy, fluxHz = cross_n_vec(n, dEx, dEy, dEz)

    fluxHx .*= -1.0
    fluxHy .*= -1.0
    fluxHz .*= -1.0

    return (
        fluxEx = fluxEx,
        fluxEy = fluxEy,
        fluxEz = fluxEz,
        fluxHx = fluxHx,
        fluxHy = fluxHy,
        fluxHz = fluxHz,
    )
end

function maxwell_pec_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    for ff in flux_faces.boundary
        if ff.boundary_id != pec_boundary_id
            continue
        end

        tr = ff.trace
        n = ff.normal

        minus = maxwell_boundary_minus_trace(U, tr)

        plus = maxwell_boundary_plus_trace(
            minus,
            n,
            ff.boundary_id;
            pec_boundary_id = pec_boundary_id,
        )

        flux = maxwell_boundary_surface_flux_values(minus, plus, n)

        add_lifted_face_contribution!(
            rhs.rhsEx,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxEx,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEy,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxEy,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEz,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxEz,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHx,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxHx,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHy,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxHy,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHz,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxHz,
            ff.area,
        )
    end

    return rhs
end

function test_maxwell_pec_local_flux_zero(
    flux_faces::DGFluxFaces,
    ref::ReferenceTet;
    pec_boundary_id::Int = 10,
)
    max_flux = 0.0
    checked_faces = 0
    checked_nodes = 0
    worst_face = 0

    for i in eachindex(flux_faces.boundary)
        ff = flux_faces.boundary[i]

        if ff.boundary_id != pec_boundary_id
            continue
        end

        checked_faces += 1

        n = ff.normal
        Nfp = length(ff.trace.nodes)

        # PEC-compatible trace: E is purely normal.
        α = collect(range(1.0, 2.0; length = Nfp))

        Ex = α .* n[1]
        Ey = α .* n[2]
        Ez = α .* n[3]

        # Arbitrary H. Since Hplus = Hminus for PEC in this construction,
        # the central H jump contribution is zero.
        Hx = collect(range(0.2, 0.7; length = Nfp))
        Hy = collect(range(-0.5, 0.3; length = Nfp))
        Hz = collect(range(1.1, 1.6; length = Nfp))

        minus = (
            Ex = Ex,
            Ey = Ey,
            Ez = Ez,
            Hx = Hx,
            Hy = Hy,
            Hz = Hz,
        )

        plus = pec_boundary_plus_trace(minus, n)

        flux = maxwell_boundary_surface_flux_values(minus, plus, n)

        local_max = maximum((
            maximum(abs.(flux.fluxEx)),
            maximum(abs.(flux.fluxEy)),
            maximum(abs.(flux.fluxEz)),
            maximum(abs.(flux.fluxHx)),
            maximum(abs.(flux.fluxHy)),
            maximum(abs.(flux.fluxHz)),
        ))

        if local_max > max_flux
            max_flux = local_max
            worst_face = i
        end

        checked_nodes += Nfp
    end

    println("Maxwell PEC local flux zero test")
    println("--------------------------------")
    println("PEC boundary_id:          ", pec_boundary_id)
    println("checked PEC faces:        ", checked_faces)
    println("checked PEC face nodes:   ", checked_nodes)
    println("max local flux magnitude: ", max_flux)
    println("worst boundary face id:   ", worst_face)

    if checked_faces == 0
        println("⚠ no PEC faces found")
    elseif max_flux < 1e-12
        println("✓ PEC-compatible traces produce zero boundary correction")
    else
        println("⚠ PEC-compatible trace produced nonzero boundary correction")
    end

    return nothing
end

function test_maxwell_pec_boundary_surface_zero_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_pec_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = pec_boundary_id,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell PEC boundary surface zero-field test")
    println("--------------------------------------------")
    println("max |rhsEx|:        ", errEx)
    println("max |rhsEy|:        ", errEy)
    println("max |rhsEz|:        ", errEz)
    println("max |rhsHx|:        ", errHx)
    println("max |rhsHy|:        ", errHy)
    println("max |rhsHz|:        ", errHz)
    println("max error:          ", maxerr)

    if maxerr < 1e-12
        println("✓ PEC boundary surface RHS vanishes for zero field")
    else
        println("⚠ PEC zero-field boundary RHS is not zero")
    end

    return nothing
end

function test_maxwell_pec_boundary_surface_arbitrary_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    Efun = (x, y, z) -> (
        x + 0.25 * y,
        2.0 * y - 0.5 * z,
        3.0 * z + 0.1 * x,
    )

    Hfun = (x, y, z) -> (
        z + 0.2 * x,
        x - 0.3 * y,
        y + 0.4 * z,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_pec_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = pec_boundary_id,
    )

    maxE = maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
    ))

    maxH = maximum((
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))

    println("Maxwell PEC boundary surface arbitrary-field test")
    println("-------------------------------------------------")
    println("max electric correction:   ", maxE)
    println("max magnetic correction:   ", maxH)

    if maxE < 1e-12 && maxH > 0.0
        println("✓ central PEC flux gives zero E correction and nonzero H correction")
    else
        println("⚠ PEC arbitrary-field correction differs from expected central-flux behavior")
    end

    return nothing
end


function empty_maxwell_boundary_registry()
    return MaxwellBoundaryRegistry(Dict{Int, MaxwellBoundaryKind}())
end

function test_maxwell_rhs_matches_volume_without_boundaries(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
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

    rhs_full = similar_maxwell_rhs(U)
    rhs_vol = similar_maxwell_rhs(U)

    registry = empty_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs_full,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    maxwell_volume_rhs!(
        rhs_vol,
        U,
        physops;
        ε = 1.0,
        μ = 1.0,
        reset = true,
    )

    errEx = maximum(abs.(rhs_full.rhsEx .- rhs_vol.rhsEx))
    errEy = maximum(abs.(rhs_full.rhsEy .- rhs_vol.rhsEy))
    errEz = maximum(abs.(rhs_full.rhsEz .- rhs_vol.rhsEz))

    errHx = maximum(abs.(rhs_full.rhsHx .- rhs_vol.rhsHx))
    errHy = maximum(abs.(rhs_full.rhsHy .- rhs_vol.rhsHy))
    errHz = maximum(abs.(rhs_full.rhsHz .- rhs_vol.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell full RHS no-boundary consistency test")
    println("---------------------------------------------")
    println("max |rhs_full - rhs_vol| Ex: ", errEx)
    println("max |rhs_full - rhs_vol| Ey: ", errEy)
    println("max |rhs_full - rhs_vol| Ez: ", errEz)
    println("max |rhs_full - rhs_vol| Hx: ", errHx)
    println("max |rhs_full - rhs_vol| Hy: ", errHy)
    println("max |rhs_full - rhs_vol| Hz: ", errHz)
    println("max error:                  ", maxerr)

    if maxerr < 1e-10
        println("✓ full RHS matches volume RHS when boundary fluxes are disabled")
    else
        println("⚠ full RHS differs from volume RHS unexpectedly")
    end

    return nothing
end

function test_maxwell_rhs_zero_field_with_pec(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    registry = default_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell full RHS zero-field PEC test")
    println("------------------------------------")
    println("max |rhsEx|: ", errEx)
    println("max |rhsEy|: ", errEy)
    println("max |rhsEz|: ", errEz)
    println("max |rhsHx|: ", errHx)
    println("max |rhsHy|: ", errHy)
    println("max |rhsHz|: ", errHz)
    println("max error:   ", maxerr)

    if maxerr < 1e-12
        println("✓ full Maxwell RHS vanishes for zero field with PEC enabled")
    else
        println("⚠ full Maxwell RHS zero-field test failed")
    end

    return nothing
end

function test_maxwell_rhs_linear_field_no_boundaries(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
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

    registry = empty_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    exact_rhsE = (-11.0, -9.0, -7.0)
    exact_rhsH = (-3.0, 3.0, -3.0)

    errEx = maximum(abs.(rhs.rhsEx .- exact_rhsE[1]))
    errEy = maximum(abs.(rhs.rhsEy .- exact_rhsE[2]))
    errEz = maximum(abs.(rhs.rhsEz .- exact_rhsE[3]))

    errHx = maximum(abs.(rhs.rhsHx .- exact_rhsH[1]))
    errHy = maximum(abs.(rhs.rhsHy .- exact_rhsH[2]))
    errHz = maximum(abs.(rhs.rhsHz .- exact_rhsH[3]))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell full RHS linear-field no-boundary test")
    println("----------------------------------------------")
    println("Expected rhsE: ", exact_rhsE)
    println("Expected rhsH: ", exact_rhsH)
    println("max error Ex:  ", errEx)
    println("max error Ey:  ", errEy)
    println("max error Ez:  ", errEz)
    println("max error Hx:  ", errHx)
    println("max error Hy:  ", errHy)
    println("max error Hz:  ", errHz)
    println("max error:     ", maxerr)

    if maxerr < 1e-10
        println("✓ full Maxwell RHS reproduces exact linear-field volume curl when boundaries are disabled")
    else
        println("⚠ full Maxwell RHS linear-field test failed")
    end

    return nothing
end

# -------------------------------------------------------------------------
# Maxwell energy diagnostics
# -------------------------------------------------------------------------


function mass_quadratic_form(M::AbstractMatrix{Float64}, u::AbstractVector{Float64})
    return dot(u, M * u)
end

function maxwell_energy(
    U::MaxwellField,
    ref::ReferenceTet,
    mappings::DGReferenceMapping;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    ne = size(U.Ex, 2)

    if length(mappings.tet_mappings) != ne
        error(
            "Mismatch between number of field elements and mappings: " *
            "$ne vs $(length(mappings.tet_mappings))."
        )
    end

    Ex_energy = 0.0
    Ey_energy = 0.0
    Ez_energy = 0.0

    Hx_energy = 0.0
    Hy_energy = 0.0
    Hz_energy = 0.0

    M = ref.M

    for e in 1:ne
        J = mappings.tet_mappings[e].absdetJ

        Ex_energy += 0.5 * ε * J * mass_quadratic_form(M, U.Ex[:, e])
        Ey_energy += 0.5 * ε * J * mass_quadratic_form(M, U.Ey[:, e])
        Ez_energy += 0.5 * ε * J * mass_quadratic_form(M, U.Ez[:, e])

        Hx_energy += 0.5 * μ * J * mass_quadratic_form(M, U.Hx[:, e])
        Hy_energy += 0.5 * μ * J * mass_quadratic_form(M, U.Hy[:, e])
        Hz_energy += 0.5 * μ * J * mass_quadratic_form(M, U.Hz[:, e])
    end

    electric = Ex_energy + Ey_energy + Ez_energy
    magnetic = Hx_energy + Hy_energy + Hz_energy
    total = electric + magnetic

    return MaxwellEnergy(
        electric,
        magnetic,
        total,
        Ex_energy,
        Ey_energy,
        Ez_energy,
        Hx_energy,
        Hy_energy,
        Hz_energy,
    )
end

function print_maxwell_energy_summary(energy::MaxwellEnergy)
    println("Maxwell energy diagnostics")
    println("--------------------------")
    println("Electric energy:        ", energy.electric)
    println("Magnetic energy:        ", energy.magnetic)
    println("Total energy:           ", energy.total)

    println()
    println("Electric components")
    println("-------------------")
    println("Ex energy:              ", energy.Ex)
    println("Ey energy:              ", energy.Ey)
    println("Ez energy:              ", energy.Ez)

    println()
    println("Magnetic components")
    println("-------------------")
    println("Hx energy:              ", energy.Hx)
    println("Hy energy:              ", energy.Hy)
    println("Hz energy:              ", energy.Hz)

    return nothing
end

function mesh_volume_from_mappings(mappings::DGReferenceMapping)
    volume = 0.0

    for mapping in mappings.tet_mappings
        volume += REF_TET_VOLUME * mapping.absdetJ
    end

    return volume
end

function test_maxwell_energy_constant_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    mappings::DGReferenceMapping;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    Efun = (x, y, z) -> (1.0, 2.0, 3.0)
    Hfun = (x, y, z) -> (4.0, 5.0, 6.0)

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    energy = maxwell_energy(
        U,
        ref,
        mappings;
        ε = ε,
        μ = μ,
    )

    volume = mesh_volume_from_mappings(mappings)

    expected_electric = 0.5 * ε * (1.0^2 + 2.0^2 + 3.0^2) * volume
    expected_magnetic = 0.5 * μ * (4.0^2 + 5.0^2 + 6.0^2) * volume
    expected_total = expected_electric + expected_magnetic

    err_electric = abs(energy.electric - expected_electric)
    err_magnetic = abs(energy.magnetic - expected_magnetic)
    err_total = abs(energy.total - expected_total)

    println("Maxwell constant-field energy test")
    println("----------------------------------")
    println("mesh volume:               ", volume)
    println("computed electric energy:  ", energy.electric)
    println("expected electric energy:  ", expected_electric)
    println("electric energy error:     ", err_electric)
    println()
    println("computed magnetic energy:  ", energy.magnetic)
    println("expected magnetic energy:  ", expected_magnetic)
    println("magnetic energy error:     ", err_magnetic)
    println()
    println("computed total energy:     ", energy.total)
    println("expected total energy:     ", expected_total)
    println("total energy error:        ", err_total)

    if err_total < 1e-10
        println("✓ constant-field Maxwell energy is consistent")
    else
        println("⚠ constant-field Maxwell energy has larger-than-expected error")
    end

    return nothing
end

function maxwell_energy_rate(
    U::MaxwellField,
    rhs::MaxwellRHS,
    ref::ReferenceTet,
    mappings::DGReferenceMapping;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    ne = size(U.Ex, 2)

    rate = 0.0
    M = ref.M

    for e in 1:ne
        J = mappings.tet_mappings[e].absdetJ

        rate += ε * J * dot(U.Ex[:, e], M * rhs.rhsEx[:, e])
        rate += ε * J * dot(U.Ey[:, e], M * rhs.rhsEy[:, e])
        rate += ε * J * dot(U.Ez[:, e], M * rhs.rhsEz[:, e])

        rate += μ * J * dot(U.Hx[:, e], M * rhs.rhsHx[:, e])
        rate += μ * J * dot(U.Hy[:, e], M * rhs.rhsHy[:, e])
        rate += μ * J * dot(U.Hz[:, e], M * rhs.rhsHz[:, e])
    end

    return rate
end

function test_maxwell_energy_rate_zero_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    registry = default_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rate = maxwell_energy_rate(
        U,
        rhs,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    println("Maxwell zero-field energy-rate test")
    println("-----------------------------------")
    println("dEnergy/dt: ", rate)

    if abs(rate) < 1e-12
        println("✓ zero-field energy rate is zero")
    else
        println("⚠ zero-field energy rate is not zero")
    end

    return nothing
end

function test_maxwell_energy_rate_no_boundaries(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
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

    registry = empty_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rate = maxwell_energy_rate(
        U,
        rhs,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    energy = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    println("Maxwell no-boundary energy-rate diagnostic")
    println("------------------------------------------")
    println("Total energy:      ", energy.total)
    println("dEnergy/dt:        ", rate)
    println("relative rate:     ", abs(rate) / max(energy.total, eps(Float64)))

    return nothing
end
