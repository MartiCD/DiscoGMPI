function homogeneous_maxwell_materials(
    nelements::Integer;
    epsilon::Real = 1.0,
    permeability::Real = 1.0,
)
    material = MaxwellMaterial(epsilon, permeability)
    return MaxwellElementMaterials(
        fill(material.epsilon, nelements),
        fill(material.permeability, nelements),
    )
end

function maxwell_element_materials(
    material_ids::AbstractVector{<:Integer},
    material_table::AbstractDict{<:Integer, MaxwellMaterial},
)
    epsilon = Vector{Float64}(undef, length(material_ids))
    permeability = similar(epsilon)

    for elem in eachindex(material_ids)
        material_id = material_ids[elem]
        haskey(material_table, material_id) ||
            throw(
                ArgumentError(
                    "No Maxwell material is registered for material_id=$material_id.",
                ),
            )
        material = material_table[material_id]
        epsilon[elem] = material.epsilon
        permeability[elem] = material.permeability
    end

    return MaxwellElementMaterials(epsilon, permeability)
end

function maxwell_element_materials(
    mesh::RawVTUMesh,
    material_table::AbstractDict{<:Integer, MaxwellMaterial};
    material_tag_name::String = "material_id",
)
    return maxwell_element_materials(
        Int.(tet_data(mesh, material_tag_name)),
        material_table,
    )
end

function maxwell_element_materials(
    distributed_dg::DistributedDGDiscretization,
    material_table::AbstractDict{<:Integer, MaxwellMaterial},
)
    return maxwell_element_materials(
        distributed_dg.distributed_mesh.elements.material_id,
        material_table,
    )
end

function validate_maxwell_materials(
    materials::MaxwellElementMaterials,
    nelements::Integer,
)
    length(materials.epsilon) == nelements ||
        throw(
            ArgumentError(
                "MaxwellElementMaterials has $(length(materials.epsilon)) entries, " *
                "but the field has $nelements elements.",
            ),
        )
    return materials
end

function element_material(
    materials::MaxwellElementMaterials,
    elem::Integer,
)
    return MaxwellMaterial(
        materials.epsilon[elem],
        materials.permeability[elem],
    )
end

function maxwell_material_surface_flux_values(
    minus,
    plus,
    n::NTuple{3, Float64},
    minus_material::MaxwellMaterial,
    plus_material::MaxwellMaterial;
    flux_kind::MaxwellFluxKind,
)
    dEx = plus.Ex .- minus.Ex
    dEy = plus.Ey .- minus.Ey
    dEz = plus.Ez .- minus.Ez
    dHx = plus.Hx .- minus.Hx
    dHy = plus.Hy .- minus.Hy
    dHz = plus.Hz .- minus.Hz

    fluxEx, fluxEy, fluxEz = cross_n_vec(n, dHx, dHy, dHz)
    fluxHx, fluxHy, fluxHz = cross_n_vec(n, dEx, dEy, dEz)

    if flux_kind == MaxwellFlux_Central
        fluxEx .*= 0.5
        fluxEy .*= 0.5
        fluxEz .*= 0.5
        fluxHx .*= -0.5
        fluxHy .*= -0.5
        fluxHz .*= -0.5
    elseif flux_kind == MaxwellFlux_Upwind
        Zminus = maxwell_impedance(
            ε = minus_material.epsilon,
            μ = minus_material.permeability,
        )
        Zplus = maxwell_impedance(
            ε = plus_material.epsilon,
            μ = plus_material.permeability,
        )
        Yminus = 1.0 / Zminus
        Yplus = 1.0 / Zplus
        nnEx, nnEy, nnEz = cross_n_cross_n_vec(n, dEx, dEy, dEz)
        nnHx, nnHy, nnHz = cross_n_cross_n_vec(n, dHx, dHy, dHz)

        fluxEx .= (Zplus .* fluxEx .- nnEx) ./ (Zminus + Zplus)
        fluxEy .= (Zplus .* fluxEy .- nnEy) ./ (Zminus + Zplus)
        fluxEz .= (Zplus .* fluxEz .- nnEz) ./ (Zminus + Zplus)

        fluxHx .= -(Yplus .* fluxHx .+ nnHx) ./ (Yminus + Yplus)
        fluxHy .= -(Yplus .* fluxHy .+ nnHy) ./ (Yminus + Yplus)
        fluxHz .= -(Yplus .* fluxHz .+ nnHz) ./ (Yminus + Yplus)
    else
        error("Unsupported Maxwell flux kind: $flux_kind")
    end

    fluxEx ./= minus_material.epsilon
    fluxEy ./= minus_material.epsilon
    fluxEz ./= minus_material.epsilon
    fluxHx ./= minus_material.permeability
    fluxHy ./= minus_material.permeability
    fluxHz ./= minus_material.permeability

    return (
        fluxEx = fluxEx,
        fluxEy = fluxEy,
        fluxEz = fluxEz,
        fluxHx = fluxHx,
        fluxHy = fluxHy,
        fluxHz = fluxHz,
    )
end

function maxwell_material_surface_flux_values!(
    flux::MaxwellFaceWorkspace,
    minus::MaxwellFaceWorkspace,
    plus::MaxwellFaceWorkspace,
    n::NTuple{3, Float64};
    flux_kind::MaxwellFluxKind,
    minus_epsilon::Float64,
    minus_permeability::Float64,
    plus_epsilon::Float64,
    plus_permeability::Float64,
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

            flux.Ex[q] = 0.5 * nx_dH[1] / minus_epsilon
            flux.Ey[q] = 0.5 * nx_dH[2] / minus_epsilon
            flux.Ez[q] = 0.5 * nx_dH[3] / minus_epsilon
            flux.Hx[q] = -0.5 * nx_dE[1] / minus_permeability
            flux.Hy[q] = -0.5 * nx_dE[2] / minus_permeability
            flux.Hz[q] = -0.5 * nx_dE[3] / minus_permeability
        end
    elseif flux_kind == MaxwellFlux_Upwind
        Zminus = maxwell_impedance(
            ε = minus_epsilon,
            μ = minus_permeability,
        )
        Zplus = maxwell_impedance(
            ε = plus_epsilon,
            μ = plus_permeability,
        )
        Yminus = 1.0 / Zminus
        Yplus = 1.0 / Zplus

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

            flux.Ex[q] =
                (Zplus * nx_dH[1] - nn_dE[1]) /
                (Zminus + Zplus) / minus_epsilon
            flux.Ey[q] =
                (Zplus * nx_dH[2] - nn_dE[2]) /
                (Zminus + Zplus) / minus_epsilon
            flux.Ez[q] =
                (Zplus * nx_dH[3] - nn_dE[3]) /
                (Zminus + Zplus) / minus_epsilon

            flux.Hx[q] =
                -(Yplus * nx_dE[1] + nn_dH[1]) /
                (Yminus + Yplus) / minus_permeability
            flux.Hy[q] =
                -(Yplus * nx_dE[2] + nn_dH[2]) /
                (Yminus + Yplus) / minus_permeability
            flux.Hz[q] =
                -(Yplus * nx_dE[3] + nn_dH[3]) /
                (Yminus + Yplus) / minus_permeability
        end
    else
        error("Unsupported Maxwell flux kind: $flux_kind")
    end

    return flux
end

function maxwell_material_face_flux!(
    flux::MaxwellFaceWorkspace,
    minus::MaxwellFaceWorkspace,
    plus::MaxwellFaceWorkspace,
    normal::NTuple{3, Float64},
    formulation::HesthavenWarburtonFormulation;
    minus_epsilon::Float64,
    minus_permeability::Float64,
    plus_epsilon::Float64,
    plus_permeability::Float64,
)
    return maxwell_material_surface_flux_values!(
        flux,
        minus,
        plus,
        normal;
        flux_kind = formulation.flux_kind,
        minus_epsilon = minus_epsilon,
        minus_permeability = minus_permeability,
        plus_epsilon = plus_epsilon,
        plus_permeability = plus_permeability,
    )
end

function maxwell_material_face_flux!(
    flux::MaxwellFaceWorkspace,
    minus::MaxwellFaceWorkspace,
    plus::MaxwellFaceWorkspace,
    normal::NTuple{3, Float64},
    formulation::PoissonBracketFormulation;
    minus_epsilon::Float64,
    minus_permeability::Float64,
    plus_epsilon::Float64,
    plus_permeability::Float64,
)
    require_poisson_bracket_surface_flux(formulation)
    return maxwell_poisson_bracket_surface_flux_values!(
        flux,
        minus,
        plus,
        normal;
        ε = minus_epsilon,
        μ = minus_permeability,
    )
end

function maxwell_volume_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
    formulation::HesthavenWarburtonFormulation,
    materials::MaxwellElementMaterials;
    reset::Bool = true,
)
    reset && fill_maxwell_rhs!(rhs, 0.0)
    nelements = size(U.Ex, 2)
    validate_maxwell_materials(materials, nelements)
    scratch = MaxwellElementScratch(size(U.Ex, 1))

    for elem in 1:nelements
        @views begin
            op = physops.elements[elem]
            curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Hx[:, elem],
                U.Hy[:, elem],
                U.Hz[:, elem],
                op,
            )
            rhs.rhsEx[:, elem] .+=
                scratch.curl_x ./ materials.epsilon[elem]
            rhs.rhsEy[:, elem] .+=
                scratch.curl_y ./ materials.epsilon[elem]
            rhs.rhsEz[:, elem] .+=
                scratch.curl_z ./ materials.epsilon[elem]

            curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Ex[:, elem],
                U.Ey[:, elem],
                U.Ez[:, elem],
                op,
            )
            rhs.rhsHx[:, elem] .-=
                scratch.curl_x ./ materials.permeability[elem]
            rhs.rhsHy[:, elem] .-=
                scratch.curl_y ./ materials.permeability[elem]
            rhs.rhsHz[:, elem] .-=
                scratch.curl_z ./ materials.permeability[elem]
        end
    end

    return rhs
end

function maxwell_volume_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
    formulation::PoissonBracketFormulation,
    materials::MaxwellElementMaterials;
    reset::Bool = true,
)
    require_poisson_bracket_surface_flux(formulation)
    reset && fill_maxwell_rhs!(rhs, 0.0)
    nelements = size(U.Ex, 2)
    validate_maxwell_materials(materials, nelements)
    scratch = MaxwellElementScratch(size(U.Ex, 1))
    mass_factor = cholesky(ref.M)

    for elem in 1:nelements
        @views begin
            op = physops.elements[elem]
            Sx, Sy, Sz = physical_weak_derivative_matrices(op)
            SxT, SyT, SzT = physical_weak_derivative_transpose_matrices(op)

            weak_curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Hx[:, elem],
                U.Hy[:, elem],
                U.Hz[:, elem],
                Sx,
                Sy,
                Sz,
            )
            ldiv!(mass_factor, scratch.curl_x)
            ldiv!(mass_factor, scratch.curl_y)
            ldiv!(mass_factor, scratch.curl_z)
            rhs.rhsEx[:, elem] .+=
                scratch.curl_x ./ materials.epsilon[elem]
            rhs.rhsEy[:, elem] .+=
                scratch.curl_y ./ materials.epsilon[elem]
            rhs.rhsEz[:, elem] .+=
                scratch.curl_z ./ materials.epsilon[elem]

            weak_curl_element!(
                scratch.curl_x,
                scratch.curl_y,
                scratch.curl_z,
                scratch.tmp,
                U.Ex[:, elem],
                U.Ey[:, elem],
                U.Ez[:, elem],
                SxT,
                SyT,
                SzT,
            )
            ldiv!(mass_factor, scratch.curl_x)
            ldiv!(mass_factor, scratch.curl_y)
            ldiv!(mass_factor, scratch.curl_z)
            rhs.rhsHx[:, elem] .+=
                scratch.curl_x ./ materials.permeability[elem]
            rhs.rhsHy[:, elem] .+=
                scratch.curl_y ./ materials.permeability[elem]
            rhs.rhsHz[:, elem] .+=
                scratch.curl_z ./ materials.permeability[elem]
        end
    end

    return rhs
end

function maxwell_material_face_flux(
    minus,
    plus,
    normal::NTuple{3, Float64},
    formulation::HesthavenWarburtonFormulation,
    minus_material::MaxwellMaterial,
    plus_material::MaxwellMaterial,
)
    return maxwell_material_surface_flux_values(
        minus,
        plus,
        normal,
        minus_material,
        plus_material;
        flux_kind = formulation.flux_kind,
    )
end

function maxwell_material_face_flux(
    minus,
    plus,
    normal::NTuple{3, Float64},
    formulation::PoissonBracketFormulation,
    minus_material::MaxwellMaterial,
    plus_material::MaxwellMaterial,
)
    require_poisson_bracket_surface_flux(formulation)
    return maxwell_poisson_bracket_surface_flux_values(
        minus,
        plus,
        normal;
        ε = minus_material.epsilon,
        μ = minus_material.permeability,
    )
end

function add_material_interior_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in flux_faces.interior
        tr = ff.trace
        minus, plus = interior_face_traces!(surface_workspace, U, tr)
        minus_epsilon = materials.epsilon[tr.minus_elem]
        minus_permeability = materials.permeability[tr.minus_elem]
        plus_epsilon = materials.epsilon[tr.plus_elem]
        plus_permeability = materials.permeability[tr.plus_elem]

        maxwell_material_face_flux!(
            surface_workspace.flux,
            minus,
            plus,
            ff.normal,
            formulation;
            minus_epsilon = minus_epsilon,
            minus_permeability = minus_permeability,
            plus_epsilon = plus_epsilon,
            plus_permeability = plus_permeability,
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

        plus_normal = (-ff.normal[1], -ff.normal[2], -ff.normal[3])
        maxwell_material_face_flux!(
            surface_workspace.flux,
            plus,
            minus,
            plus_normal,
            formulation;
            minus_epsilon = plus_epsilon,
            minus_permeability = plus_permeability,
            plus_epsilon = minus_epsilon,
            plus_permeability = minus_permeability,
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
    end

    return rhs
end

function add_material_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
    workspace::Union{Nothing, MaxwellSurfaceWorkspace} = nothing;
    mesh::Union{Nothing, RawVTUMesh} = nothing,
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    surface_workspace = maxwell_surface_workspace(workspace, ref, fops)

    for ff in flux_faces.boundary
        kind = boundary_kind(registry, ff.boundary_id)
        kind == MaxwellBC_None && continue

        tr = ff.trace
        epsilon = materials.epsilon[tr.elem]
        permeability = materials.permeability[tr.elem]
        minus = boundary_face_minus_trace!(surface_workspace, U, tr)
        plus = maxwell_boundary_plus_trace!(
            surface_workspace.plus,
            minus,
            mesh,
            ref,
            tr,
            ff.normal,
            kind,
            boundary_data;
            ε = epsilon,
            μ = permeability,
        )
        maxwell_material_face_flux!(
            surface_workspace.flux,
            minus,
            plus,
            ff.normal,
            formulation;
            minus_epsilon = epsilon,
            minus_permeability = permeability,
            plus_epsilon = epsilon,
            plus_permeability = permeability,
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
    end

    return rhs
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials;
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    fill_maxwell_rhs!(rhs, 0.0)
    surface_workspace = maxwell_surface_workspace!(dg.backend, dg.ref, dg.fops)
    maxwell_volume_rhs!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation,
        materials;
        reset = false,
    )
    add_material_interior_surface_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation,
        materials,
        surface_workspace,
    )
    add_material_boundary_surface_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation,
        materials,
        surface_workspace,
        mesh = dg.mesh,
        boundary_data = boundary_data,
    )
    return rhs
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials;
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    exchange_maxwell_ghost_traces!(U, distributed_dg)
    maxwell_rhs!(
        rhs,
        U,
        distributed_dg.dg,
        registry,
        formulation,
        materials,
        boundary_data = boundary_data,
    )
    return zero_ghost_maxwell_rhs!(rhs, distributed_dg)
end

function make_distributed_maxwell_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        return maxwell_rhs!(
            rhs,
            U,
            distributed_dg,
            registry,
            formulation,
            materials,
            boundary_data = boundary_data,
        )
    end
end

function distributed_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
    ;
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    rhs_function! = make_distributed_maxwell_rhs_function(
        distributed_dg,
        registry,
        formulation,
        materials,
        boundary_data,
    )
    return rk_step!(U, work, scheme, dt, rhs_function!)
end

function distributed_partitioned_symplectic_rk_step!(
    U::MaxwellField,
    work::MaxwellPartitionedRKWorkspace,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    materials::MaxwellElementMaterials,
    ;
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    rhs_function! = make_distributed_maxwell_rhs_function(
        distributed_dg,
        registry,
        formulation,
        materials,
        boundary_data,
    )
    return partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )
end

function maxwell_energy(
    U::MaxwellField,
    ref::ReferenceTet,
    mappings::DGReferenceMapping,
    materials::MaxwellElementMaterials,
)
    validate_maxwell_materials(materials, size(U.Ex, 2))
    components = zeros(Float64, 6)

    for elem in axes(U.Ex, 2)
        J = mappings.tet_mappings[elem].absdetJ
        epsilon = materials.epsilon[elem]
        permeability = materials.permeability[elem]
        components[1] +=
            0.5 * epsilon * J *
            mass_quadratic_form(ref.M, @view U.Ex[:, elem])
        components[2] +=
            0.5 * epsilon * J *
            mass_quadratic_form(ref.M, @view U.Ey[:, elem])
        components[3] +=
            0.5 * epsilon * J *
            mass_quadratic_form(ref.M, @view U.Ez[:, elem])
        components[4] +=
            0.5 * permeability * J *
            mass_quadratic_form(ref.M, @view U.Hx[:, elem])
        components[5] +=
            0.5 * permeability * J *
            mass_quadratic_form(ref.M, @view U.Hy[:, elem])
        components[6] +=
            0.5 * permeability * J *
            mass_quadratic_form(ref.M, @view U.Hz[:, elem])
    end

    electric = sum(@view components[1:3])
    magnetic = sum(@view components[4:6])
    return MaxwellEnergy(
        electric,
        magnetic,
        electric + magnetic,
        components...,
    )
end

function distributed_maxwell_energy(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials,
)
    validate_maxwell_materials(materials, size(U.Ex, 2))
    ref = distributed_dg.dg.ref
    mappings = distributed_dg.dg.mappings.tet_mappings
    local_components = zeros(Float64, 6)

    for elem in distributed_dg.distributed_mesh.partition.owned
        J = mappings[elem].absdetJ
        epsilon = materials.epsilon[elem]
        permeability = materials.permeability[elem]
        local_components[1] +=
            0.5 * epsilon * J *
            mass_quadratic_form(ref.M, @view U.Ex[:, elem])
        local_components[2] +=
            0.5 * epsilon * J *
            mass_quadratic_form(ref.M, @view U.Ey[:, elem])
        local_components[3] +=
            0.5 * epsilon * J *
            mass_quadratic_form(ref.M, @view U.Ez[:, elem])
        local_components[4] +=
            0.5 * permeability * J *
            mass_quadratic_form(ref.M, @view U.Hx[:, elem])
        local_components[5] +=
            0.5 * permeability * J *
            mass_quadratic_form(ref.M, @view U.Hy[:, elem])
        local_components[6] +=
            0.5 * permeability * J *
            mass_quadratic_form(ref.M, @view U.Hz[:, elem])
    end

    components =
        MPI.Allreduce(local_components, +, distributed_dg.comm)
    electric = sum(@view components[1:3])
    magnetic = sum(@view components[4:6])
    return MaxwellEnergy(
        electric,
        magnetic,
        electric + magnetic,
        components...,
    )
end

function integrate_nodal_values(
    values::AbstractVector{Float64},
    mass_ones::AbstractVector{Float64},
    absdetJ::Float64,
)
    return absdetJ * dot(mass_ones, values)
end

function local_maxwell_invariant_values(
    U::MaxwellField,
    dg::DGDiscretization,
    materials::MaxwellElementMaterials,
    elements,
)
    mass_ones = dg.ref.M * ones(Float64, dg.ref.Np)
    values = zeros(Float64, 8)
    tmp1 = zeros(Float64, dg.ref.Np)
    tmp2 = similar(tmp1)
    tmp3 = similar(tmp1)

    for elem in elements
        op = dg.physops.elements[elem]
        J = dg.mappings.tet_mappings[elem].absdetJ
        epsilon = materials.epsilon[elem]
        permeability = materials.permeability[elem]

        mul!(tmp1, op.Dx, @view U.Ex[:, elem])
        mul!(tmp2, op.Dy, @view U.Ey[:, elem])
        mul!(tmp3, op.Dz, @view U.Ez[:, elem])
        tmp1 .+= tmp2 .+ tmp3
        values[1] +=
            epsilon * integrate_nodal_values(tmp1, mass_ones, J)

        mul!(tmp1, op.Dx, @view U.Hx[:, elem])
        mul!(tmp2, op.Dy, @view U.Hy[:, elem])
        mul!(tmp3, op.Dz, @view U.Hz[:, elem])
        tmp1 .+= tmp2 .+ tmp3
        values[2] +=
            permeability * integrate_nodal_values(tmp1, mass_ones, J)

        px =
            epsilon * permeability .*
            (U.Ey[:, elem] .* U.Hz[:, elem] .-
             U.Ez[:, elem] .* U.Hy[:, elem])
        py =
            epsilon * permeability .*
            (U.Ez[:, elem] .* U.Hx[:, elem] .-
             U.Ex[:, elem] .* U.Hz[:, elem])
        pz =
            epsilon * permeability .*
            (U.Ex[:, elem] .* U.Hy[:, elem] .-
             U.Ey[:, elem] .* U.Hx[:, elem])

        values[3] += integrate_nodal_values(px, mass_ones, J)
        values[4] += integrate_nodal_values(py, mass_ones, J)
        values[5] += integrate_nodal_values(pz, mass_ones, J)

        x = Vector{Float64}(undef, dg.ref.Np)
        y = similar(x)
        z = similar(x)
        tet_nodes = dg.mesh.tets[:, elem]
        for node in 1:dg.ref.Np
            x[node], y[node], z[node] = map_to_physical(
                dg.mesh.points,
                tet_nodes,
                dg.ref.r[node],
                dg.ref.s[node],
                dg.ref.t[node],
            )
        end

        values[6] +=
            integrate_nodal_values(y .* pz .- z .* py, mass_ones, J)
        values[7] +=
            integrate_nodal_values(z .* px .- x .* pz, mass_ones, J)
        values[8] +=
            integrate_nodal_values(x .* py .- y .* px, mass_ones, J)
    end

    return values
end

function maxwell_invariants(
    U::MaxwellField,
    dg::DGDiscretization,
    materials::MaxwellElementMaterials,
)
    validate_maxwell_materials(materials, size(U.Ex, 2))
    values = local_maxwell_invariant_values(
        U,
        dg,
        materials,
        axes(U.Ex, 2),
    )
    return MaxwellInvariantDiagnostics(
        maxwell_energy(U, dg.ref, dg.mappings, materials),
        values[1],
        values[2],
        (values[3], values[4], values[5]),
        (values[6], values[7], values[8]),
    )
end

function distributed_maxwell_invariants(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials,
)
    validate_maxwell_materials(materials, size(U.Ex, 2))
    local_values = local_maxwell_invariant_values(
        U,
        distributed_dg.dg,
        materials,
        distributed_dg.distributed_mesh.partition.owned,
    )
    values = MPI.Allreduce(local_values, +, distributed_dg.comm)
    return MaxwellInvariantDiagnostics(
        distributed_maxwell_energy(U, distributed_dg, materials),
        values[1],
        values[2],
        (values[3], values[4], values[5]),
        (values[6], values[7], values[8]),
    )
end

mutable struct DistributedPeriodicFace
    pair_id::Int
    name::Symbol
    local_elem::Int
    local_face::Int
    local_nodes::Vector{Int}
    boundary_id::Int
    partner_rank::Int
    partner_global_elem::Int
    partner_face::Int
    partner_local_elem::Int
    partner_local_nodes::Vector{Int}
    partner_to_local_perm::Vector{Int}
    partner_epsilon::Float64
    partner_permeability::Float64
    normal::NTuple{3, Float64}
    area::Float64
    remote_offset::Int
end

struct DistributedPeriodicMaxwellExchange
    faces::Vector{DistributedPeriodicFace}
    neighbors::Vector{Int}
    remote_faces::Dict{Int, Vector{DistributedPeriodicFace}}
    send_buffers::Dict{Int, Vector{Float64}}
    recv_buffers::Dict{Int, Vector{Float64}}
end

function owned_periodic_boundary_descriptors(
    distributed_dg::DistributedDGDiscretization,
    boundary_ids::Set{Int},
    materials::Union{Nothing, MaxwellElementMaterials},
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    owned = Set(distributed_dg.distributed_mesh.partition.owned)
    descriptors = NamedTuple[]

    for ff in distributed_dg.dg.flux_faces.boundary
        ff.boundary_id in boundary_ids || continue
        ff.trace.elem in owned || continue
        elem = ff.trace.elem
        push!(
            descriptors,
            (
                rank = rank,
                global_elem =
                    distributed_dg.distributed_mesh.elements.global_ids[elem],
                face = ff.trace.face,
                boundary_id = ff.boundary_id,
                centroid = ff.centroid,
                normal = ff.normal,
                area = ff.area,
                points = physical_face_points(
                    distributed_dg.dg.mesh,
                    distributed_dg.dg.ref,
                    elem,
                    ff.trace.nodes,
                ),
                epsilon =
                    materials === nothing ? NaN : materials.epsilon[elem],
                permeability =
                    materials === nothing ? NaN : materials.permeability[elem],
            ),
        )
    end

    return descriptors
end

function pair_distributed_periodic_descriptors(
    descriptors,
    specs;
    centroid_tol::Float64,
    node_tol::Float64,
    area_rtol::Float64,
)
    pairs = NamedTuple[]

    for spec in specs
        minus_faces = filter(
            face -> face.boundary_id == spec.minus_boundary_id,
            descriptors,
        )
        plus_faces = filter(
            face -> face.boundary_id == spec.plus_boundary_id,
            descriptors,
        )
        length(minus_faces) == length(plus_faces) ||
            error(
                "Distributed periodic pair $(spec.name) has incompatible face " *
                "counts: $(length(minus_faces)) and $(length(plus_faces)).",
            )
        used_plus = falses(length(plus_faces))

        for minus in minus_faces
            best = 0
            best_distance = Inf
            for index in eachindex(plus_faces)
                used_plus[index] && continue
                shifted = shift_point(
                    plus_faces[index].centroid,
                    spec.plus_to_minus_shift,
                )
                distance = squared_distance(minus.centroid, shifted)
                if distance < best_distance
                    best = index
                    best_distance = distance
                end
            end
            best > 0 && best_distance <= centroid_tol^2 ||
                error(
                    "Could not pair distributed periodic face at $(minus.centroid) " *
                    "for $(spec.name).",
                )
            used_plus[best] = true
            plus = plus_faces[best]
            relative_area_error =
                abs(minus.area - plus.area) /
                max(abs(minus.area), abs(plus.area), eps(Float64))
            relative_area_error <= area_rtol ||
                error(
                    "Distributed periodic face area mismatch for $(spec.name): " *
                    "$relative_area_error.",
                )
            permutation = match_periodic_face_node_permutation(
                minus.points,
                plus.points,
                spec.plus_to_minus_shift;
                tol = node_tol,
            )
            push!(
                pairs,
                (
                    pair_id = length(pairs) + 1,
                    name = spec.name,
                    minus = minus,
                    plus = plus,
                    plus_to_minus_perm = permutation,
                ),
            )
        end
    end

    return pairs
end

function local_periodic_boundary_lookup(
    distributed_dg::DistributedDGDiscretization,
)
    lookup = Dict{Tuple{Int, Int}, BoundaryFluxFace}()
    global_ids = distributed_dg.distributed_mesh.elements.global_ids
    for ff in distributed_dg.dg.flux_faces.boundary
        lookup[(global_ids[ff.trace.elem], ff.trace.face)] = ff
    end
    return lookup
end

function build_distributed_periodic_maxwell_exchange(
    distributed_dg::DistributedDGDiscretization,
    specs = default_unit_box_periodic_specs();
    root::Int = 0,
    centroid_tol::Float64 = 1e-9,
    node_tol::Float64 = 1e-9,
    area_rtol::Float64 = 1e-10,
    materials::Union{Nothing, MaxwellElementMaterials} = nothing,
)
    comm = distributed_dg.comm
    rank = MPI.Comm_rank(comm)
    boundary_ids = Set{Int}()
    for spec in specs
        push!(boundary_ids, spec.minus_boundary_id)
        push!(boundary_ids, spec.plus_boundary_id)
    end
    materials === nothing ||
        validate_maxwell_materials(
            materials,
            size(distributed_dg.dg.mesh.tets, 2),
        )
    local_descriptors = owned_periodic_boundary_descriptors(
        distributed_dg,
        boundary_ids,
        materials,
    )
    gathered = MPI.gather(local_descriptors, comm; root = root)
    pairs = nothing
    pairing_error = nothing

    if rank == root
        try
            descriptors = reduce(vcat, gathered; init = NamedTuple[])
            pairs = pair_distributed_periodic_descriptors(
                descriptors,
                specs;
                centroid_tol = centroid_tol,
                node_tol = node_tol,
                area_rtol = area_rtol,
            )
        catch error
            pairing_error = sprint(showerror, error)
        end
    end

    pairing_error = MPI.bcast(pairing_error, comm; root = root)
    pairing_error === nothing ||
        error("Distributed periodic face pairing failed: $pairing_error")
    pairs = MPI.bcast(pairs, comm; root = root)

    lookup = local_periodic_boundary_lookup(distributed_dg)
    faces = DistributedPeriodicFace[]

    for pair in pairs
        if pair.minus.rank == rank
            local_ff = lookup[(pair.minus.global_elem, pair.minus.face)]
            partner_local_elem =
                pair.plus.rank == rank ?
                lookup[(pair.plus.global_elem, pair.plus.face)].trace.elem : 0
            partner_nodes =
                pair.plus.rank == rank ?
                copy(lookup[(pair.plus.global_elem, pair.plus.face)].trace.nodes) :
                Int[]
            push!(
                faces,
                DistributedPeriodicFace(
                    pair.pair_id,
                    pair.name,
                    local_ff.trace.elem,
                    local_ff.trace.face,
                    copy(local_ff.trace.nodes),
                    local_ff.boundary_id,
                    pair.plus.rank,
                    pair.plus.global_elem,
                    pair.plus.face,
                    partner_local_elem,
                    partner_nodes,
                    copy(pair.plus_to_minus_perm),
                    pair.plus.epsilon,
                    pair.plus.permeability,
                    local_ff.normal,
                    local_ff.area,
                    0,
                ),
            )
        end

        if pair.plus.rank == rank
            local_ff = lookup[(pair.plus.global_elem, pair.plus.face)]
            partner_local_elem =
                pair.minus.rank == rank ?
                lookup[(pair.minus.global_elem, pair.minus.face)].trace.elem : 0
            partner_nodes =
                pair.minus.rank == rank ?
                copy(lookup[(pair.minus.global_elem, pair.minus.face)].trace.nodes) :
                Int[]
            push!(
                faces,
                DistributedPeriodicFace(
                    pair.pair_id,
                    pair.name,
                    local_ff.trace.elem,
                    local_ff.trace.face,
                    copy(local_ff.trace.nodes),
                    local_ff.boundary_id,
                    pair.minus.rank,
                    pair.minus.global_elem,
                    pair.minus.face,
                    partner_local_elem,
                    partner_nodes,
                    invperm(pair.plus_to_minus_perm),
                    pair.minus.epsilon,
                    pair.minus.permeability,
                    local_ff.normal,
                    local_ff.area,
                    0,
                ),
            )
        end
    end

    sort!(faces; by = face -> (face.partner_rank, face.pair_id))
    remote_faces = Dict{Int, Vector{DistributedPeriodicFace}}()
    nfp = length(distributed_dg.dg.fops.face_nodes[1])
    for face in faces
        face.partner_rank == rank && continue
        push!(
            get!(remote_faces, face.partner_rank, DistributedPeriodicFace[]),
            face,
        )
    end
    neighbors = sort!(collect(keys(remote_faces)))
    send_buffers = Dict{Int, Vector{Float64}}()
    recv_buffers = Dict{Int, Vector{Float64}}()

    for neighbor in neighbors
        sort!(remote_faces[neighbor]; by = face -> face.pair_id)
        for (index, face) in enumerate(remote_faces[neighbor])
            face.remote_offset = (index - 1) * 6 * nfp + 1
        end
        nvalues = length(remote_faces[neighbor]) * 6 * nfp
        send_buffers[neighbor] = Vector{Float64}(undef, nvalues)
        recv_buffers[neighbor] = similar(send_buffers[neighbor])
    end

    return DistributedPeriodicMaxwellExchange(
        faces,
        neighbors,
        remote_faces,
        send_buffers,
        recv_buffers,
    )
end

function pack_distributed_periodic_traces!(
    buffer::Vector{Float64},
    U::MaxwellField,
    faces::Vector{DistributedPeriodicFace},
)
    components = (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz)
    offset = 1
    for face in faces, component in components, node in face.local_nodes
        buffer[offset] = component[node, face.local_elem]
        offset += 1
    end
    return buffer
end

function exchange_distributed_periodic_traces!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange;
    tag::Int = 24018,
)
    requests = MPI.Request[]
    for neighbor in periodic.neighbors
        pack_distributed_periodic_traces!(
            periodic.send_buffers[neighbor],
            U,
            periodic.remote_faces[neighbor],
        )
        push!(
            requests,
            MPI.Irecv!(
                periodic.recv_buffers[neighbor],
                distributed_dg.comm;
                source = neighbor,
                tag = tag,
            ),
        )
    end
    for neighbor in periodic.neighbors
        push!(
            requests,
            MPI.Isend(
                periodic.send_buffers[neighbor],
                distributed_dg.comm;
                dest = neighbor,
                tag = tag,
            ),
        )
    end
    isempty(requests) || MPI.Waitall(requests)
    return periodic
end

function distributed_periodic_plus_trace(
    U::MaxwellField,
    periodic::DistributedPeriodicMaxwellExchange,
    face::DistributedPeriodicFace,
    rank::Int,
)
    components = (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz)
    permutation = face.partner_to_local_perm

    if face.partner_rank == rank
        values = map(
            component ->
                component[face.partner_local_nodes, face.partner_local_elem][permutation],
            components,
        )
    else
        buffer = periodic.recv_buffers[face.partner_rank]
        nfp = length(face.local_nodes)
        values = ntuple(6) do component
            first = face.remote_offset + (component - 1) * nfp
            buffer[first:(first + nfp - 1)][permutation]
        end
    end

    return (
        Ex = values[1],
        Ey = values[2],
        Ez = values[3],
        Hx = values[4],
        Hy = values[5],
        Hz = values[6],
    )
end

function distributed_periodic_plus_trace!(
    dest::MaxwellFaceWorkspace,
    U::MaxwellField,
    periodic::DistributedPeriodicMaxwellExchange,
    face::DistributedPeriodicFace,
    rank::Int,
)
    permutation = face.partner_to_local_perm

    if face.partner_rank == rank
        @inbounds for q in eachindex(permutation)
            node = face.partner_local_nodes[permutation[q]]
            elem = face.partner_local_elem
            dest.Ex[q] = U.Ex[node, elem]
            dest.Ey[q] = U.Ey[node, elem]
            dest.Ez[q] = U.Ez[node, elem]
            dest.Hx[q] = U.Hx[node, elem]
            dest.Hy[q] = U.Hy[node, elem]
            dest.Hz[q] = U.Hz[node, elem]
        end
    else
        buffer = periodic.recv_buffers[face.partner_rank]
        nfp = length(face.local_nodes)

        @inbounds for q in eachindex(permutation)
            node = permutation[q]
            dest.Ex[q] = buffer[face.remote_offset + node - 1]
            dest.Ey[q] = buffer[face.remote_offset + nfp + node - 1]
            dest.Ez[q] = buffer[face.remote_offset + 2 * nfp + node - 1]
            dest.Hx[q] = buffer[face.remote_offset + 3 * nfp + node - 1]
            dest.Hy[q] = buffer[face.remote_offset + 4 * nfp + node - 1]
            dest.Hz[q] = buffer[face.remote_offset + 5 * nfp + node - 1]
        end
    end

    return dest
end

function add_distributed_periodic_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    materials::Union{Nothing, MaxwellElementMaterials} = nothing,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    dg = distributed_dg.dg
    surface_workspace = maxwell_surface_workspace!(dg.backend, dg.ref, dg.fops)

    for face in periodic.faces
        minus = gather_face_values!(
            surface_workspace.minus,
            U,
            face.local_nodes,
            face.local_elem,
        )
        plus = distributed_periodic_plus_trace!(
            surface_workspace.plus,
            U,
            periodic,
            face,
            rank,
        )

        if materials === nothing
            if formulation isa PoissonBracketFormulation
                maxwell_poisson_bracket_surface_flux_values!(
                    surface_workspace.flux,
                    minus,
                    plus,
                    face.normal;
                    flux_kind = formulation.flux_kind,
                    ε = ε,
                    μ = μ,
                )
            else
                maxwell_surface_flux_values!(
                    surface_workspace.flux,
                    minus,
                    plus,
                    face.normal;
                    flux_kind = formulation.flux_kind,
                    ε = ε,
                    μ = μ,
                )
            end
        else
            isfinite(face.partner_epsilon) &&
                isfinite(face.partner_permeability) ||
                throw(
                    ArgumentError(
                        "The distributed periodic exchange was built without " *
                        "material metadata. Rebuild it with materials=materials.",
                    ),
                )
            maxwell_material_face_flux!(
                surface_workspace.flux,
                minus,
                plus,
                face.normal,
                formulation;
                minus_epsilon = materials.epsilon[face.local_elem],
                minus_permeability = materials.permeability[face.local_elem],
                plus_epsilon = face.partner_epsilon,
                plus_permeability = face.partner_permeability,
            )
        end

        add_lifted_maxwell_surface_flux!(
            rhs,
            face.local_elem,
            dg.fops,
            dg.mappings,
            face.local_face,
            surface_workspace.flux,
            face.area,
            surface_workspace,
        )
    end

    return rhs
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    exchange_maxwell_ghost_traces!(U, distributed_dg)
    exchange_distributed_periodic_traces!(U, distributed_dg, periodic)
    dg = distributed_dg.dg
    fill_maxwell_rhs!(rhs, 0.0)
    maxwell_volume_rhs!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation;
        ε = ε,
        μ = μ,
        reset = false,
    )
    maxwell_interior_surface_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation;
        ε = ε,
        μ = μ,
    )
    add_distributed_periodic_surface_rhs!(
        rhs,
        U,
        distributed_dg,
        periodic,
        formulation;
        ε = ε,
        μ = μ,
    )
    maxwell_boundary_surface_rhs!(
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
    )
    return zero_ghost_maxwell_rhs!(rhs, distributed_dg)
end

function maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
)
    exchange_maxwell_ghost_traces!(U, distributed_dg)
    exchange_distributed_periodic_traces!(U, distributed_dg, periodic)
    dg = distributed_dg.dg
    fill_maxwell_rhs!(rhs, 0.0)
    maxwell_volume_rhs!(
        rhs,
        U,
        dg.ref,
        dg.physops,
        formulation,
        materials;
        reset = false,
    )
    add_material_interior_surface_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation,
        materials,
    )
    add_distributed_periodic_surface_rhs!(
        rhs,
        U,
        distributed_dg,
        periodic,
        formulation;
        materials = materials,
    )
    add_material_boundary_surface_rhs!(
        rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation,
        materials,
    )
    return zero_ghost_maxwell_rhs!(rhs, distributed_dg)
end

function make_distributed_periodic_maxwell_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        return maxwell_rhs_periodic!(
            rhs,
            U,
            distributed_dg,
            periodic,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )
    end
end

function distributed_periodic_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    rhs_function! = make_distributed_periodic_maxwell_rhs_function(
        distributed_dg,
        periodic,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )
    return rk_step!(U, work, scheme, dt, rhs_function!)
end

function distributed_periodic_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
)
    rhs_function! = make_distributed_periodic_maxwell_rhs_function(
        distributed_dg,
        periodic,
        registry,
        formulation,
        materials,
    )
    return rk_step!(U, work, scheme, dt, rhs_function!)
end

function distributed_periodic_partitioned_symplectic_rk_step!(
    U::MaxwellField,
    work::MaxwellPartitionedRKWorkspace,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    rhs_function! = make_distributed_periodic_maxwell_rhs_function(
        distributed_dg,
        periodic,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )
    return partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )
end

function distributed_periodic_partitioned_symplectic_rk_step!(
    U::MaxwellField,
    work::MaxwellPartitionedRKWorkspace,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    materials::MaxwellElementMaterials,
)
    rhs_function! = make_distributed_periodic_maxwell_rhs_function(
        distributed_dg,
        periodic,
        registry,
        formulation,
        materials,
    )
    return partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )
end

function make_distributed_periodic_maxwell_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        return maxwell_rhs_periodic!(
            rhs,
            U,
            distributed_dg,
            periodic,
            registry,
            formulation,
            materials,
        )
    end
end
