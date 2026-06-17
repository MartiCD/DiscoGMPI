"""
    MaxwellNonlinearPML

Nodal damping data for the six-field nonlinear PML of Abarbanel, Gottlieb,
and Hesthaven (2006), equations (2.9)-(2.13) and Appendix A.

The implementation follows the component equations and the energy identity
in the paper. In particular, the magnetic source is
`(sigma .* P_E) x E`, where `P_E = -(E x H) / D_E`.
"""
struct MaxwellNonlinearPML
    sigma_x::Matrix{Float64}
    sigma_y::Matrix{Float64}
    sigma_z::Matrix{Float64}
    a::Float64
    regularization::Float64

    function MaxwellNonlinearPML(
        sigma_x::AbstractMatrix{<:Real},
        sigma_y::AbstractMatrix{<:Real},
        sigma_z::AbstractMatrix{<:Real};
        a::Real = 0.5,
        regularization::Real = 1e-12,
    )
        size(sigma_x) == size(sigma_y) == size(sigma_z) ||
            throw(
                ArgumentError(
                    "Nonlinear PML sigma arrays must have identical sizes.",
                ),
            )
        0.0 < a < 1.0 ||
            throw(ArgumentError("Nonlinear PML parameter a must lie in (0,1)."))
        regularization > 0.0 ||
            throw(
                ArgumentError(
                    "Nonlinear PML regularization must be positive.",
                ),
            )

        sigma_arrays = (sigma_x, sigma_y, sigma_z)
        all(sigma -> all(isfinite, sigma), sigma_arrays) ||
            throw(ArgumentError("Nonlinear PML sigma values must be finite."))
        all(sigma -> all(>=(0), sigma), sigma_arrays) ||
            throw(
                ArgumentError(
                    "Nonlinear PML sigma values must be non-negative.",
                ),
            )

        return new(
            Matrix{Float64}(sigma_x),
            Matrix{Float64}(sigma_y),
            Matrix{Float64}(sigma_z),
            Float64(a),
            Float64(regularization),
        )
    end
end

"""
    polynomial_pml_sigma(
        coordinate,
        interface,
        outer_boundary;
        sigma_max,
        degree=2,
    )

Evaluate a one-sided polynomial PML profile. The value is zero on the
interior side of `interface`, rises toward `sigma_max` through the layer,
and is clamped to `sigma_max` beyond `outer_boundary`.
"""
function polynomial_pml_sigma(
    coordinate::Real,
    interface::Real,
    outer_boundary::Real;
    sigma_max::Real,
    degree::Integer = 2,
)
    isfinite(coordinate) ||
        throw(ArgumentError("PML coordinates must be finite."))
    isfinite(interface) && isfinite(outer_boundary) ||
        throw(ArgumentError("PML layer bounds must be finite."))
    outer_boundary != interface ||
        throw(ArgumentError("PML interface and outer boundary must differ."))
    sigma_max >= 0.0 ||
        throw(ArgumentError("PML sigma_max must be non-negative."))
    degree >= 1 ||
        throw(ArgumentError("PML polynomial degree must be at least one."))

    depth = if outer_boundary > interface
        (coordinate - interface) / (outer_boundary - interface)
    else
        (interface - coordinate) / (interface - outer_boundary)
    end
    depth = clamp(depth, 0.0, 1.0)
    return Float64(sigma_max) * depth^degree
end

function build_maxwell_nonlinear_pml(
    mesh::RawVTUMesh,
    ref::ReferenceTet;
    sigma_x::Function = (x, y, z) -> 0.0,
    sigma_y::Function = (x, y, z) -> 0.0,
    sigma_z::Function = (x, y, z) -> 0.0,
    a::Real = 0.5,
    regularization::Real = 1e-12,
)
    nelements = size(mesh.tets, 2)
    sigma_x_values = zeros(Float64, ref.Np, nelements)
    sigma_y_values = similar(sigma_x_values)
    sigma_z_values = similar(sigma_x_values)

    for elem in 1:nelements
        tet_nodes = @view mesh.tets[:, elem]
        for node in 1:ref.Np
            x, y, z = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[node],
                ref.s[node],
                ref.t[node],
            )
            sigma_x_values[node, elem] = sigma_x(x, y, z)
            sigma_y_values[node, elem] = sigma_y(x, y, z)
            sigma_z_values[node, elem] = sigma_z(x, y, z)
        end
    end

    return MaxwellNonlinearPML(
        sigma_x_values,
        sigma_y_values,
        sigma_z_values;
        a = a,
        regularization = regularization,
    )
end

function build_maxwell_nonlinear_pml(
    dg::DGDiscretization;
    kwargs...,
)
    return build_maxwell_nonlinear_pml(dg.mesh, dg.ref; kwargs...)
end

function build_maxwell_nonlinear_pml(
    distributed_dg::DistributedDGDiscretization;
    kwargs...,
)
    return build_maxwell_nonlinear_pml(distributed_dg.dg; kwargs...)
end

"""
    nonlinear_pml_source(E, H, sigma; a=0.5, regularization=1e-12)

Return the electric and magnetic nonlinear PML source vectors at one point.
This is the primary energy-decaying system in Appendix A of the paper.
"""
@inline function _nonlinear_pml_source(
    electric::NTuple{3, <:Real},
    magnetic::NTuple{3, <:Real},
    sigma::NTuple{3, <:Real};
    a::Real = 0.5,
    regularization::Real = 1e-12,
)
    Ex, Ey, Ez = electric
    Hx, Hy, Hz = magnetic
    sigma_x, sigma_y, sigma_z = sigma

    cross_x = Ey * Hz - Ez * Hy
    cross_y = Ez * Hx - Ex * Hz
    cross_z = Ex * Hy - Ey * Hx
    electric_squared = Ex^2 + Ey^2 + Ez^2
    magnetic_squared = Hx^2 + Hy^2 + Hz^2
    denominator_h =
        a * magnetic_squared + (1.0 - a) * electric_squared +
        regularization
    denominator_e =
        a * electric_squared + (1.0 - a) * magnetic_squared +
        regularization

    p_h_x = cross_x / denominator_h
    p_h_y = cross_y / denominator_h
    p_h_z = cross_z / denominator_h
    p_e_x = -cross_x / denominator_e
    p_e_y = -cross_y / denominator_e
    p_e_z = -cross_z / denominator_e

    sigma_p_h_x = sigma_x * p_h_x
    sigma_p_h_y = sigma_y * p_h_y
    sigma_p_h_z = sigma_z * p_h_z
    sigma_p_e_x = sigma_x * p_e_x
    sigma_p_e_y = sigma_y * p_e_y
    sigma_p_e_z = sigma_z * p_e_z

    electric_source = (
        sigma_p_h_y * Hz - sigma_p_h_z * Hy,
        sigma_p_h_z * Hx - sigma_p_h_x * Hz,
        sigma_p_h_x * Hy - sigma_p_h_y * Hx,
    )
    magnetic_source = (
        sigma_p_e_y * Ez - sigma_p_e_z * Ey,
        sigma_p_e_z * Ex - sigma_p_e_x * Ez,
        sigma_p_e_x * Ey - sigma_p_e_y * Ex,
    )

    return (
        electric = electric_source,
        magnetic = magnetic_source,
    )
end

function nonlinear_pml_source(
    electric::NTuple{3, <:Real},
    magnetic::NTuple{3, <:Real},
    sigma::NTuple{3, <:Real};
    a::Real = 0.5,
    regularization::Real = 1e-12,
)
    0.0 < a < 1.0 ||
        throw(ArgumentError("Nonlinear PML parameter a must lie in (0,1)."))
    regularization > 0.0 ||
        throw(ArgumentError("Nonlinear PML regularization must be positive."))
    all(isfinite, electric) && all(isfinite, magnetic) ||
        throw(ArgumentError("Maxwell field values must be finite."))
    all(isfinite, sigma) && all(>=(0), sigma) ||
        throw(
            ArgumentError(
                "Nonlinear PML sigma values must be finite and non-negative.",
            ),
        )
    return _nonlinear_pml_source(
        electric,
        magnetic,
        sigma;
        a = a,
        regularization = regularization,
    )
end

function validate_maxwell_nonlinear_pml(
    pml::MaxwellNonlinearPML,
    U::MaxwellField,
)
    field_size = size(U.Ex)
    size(pml.sigma_x) == field_size ||
        throw(
            ArgumentError(
                "Nonlinear PML data has size $(size(pml.sigma_x)), " *
                "but the Maxwell field has size $field_size.",
            ),
        )
    return pml
end

"""
    add_maxwell_nonlinear_pml_source!(rhs, U, pml; elements=axes(U.Ex, 2))

Add the local nonlinear PML source to an existing Maxwell RHS. Distributed
callers should pass only their owned element indices.
"""
function add_maxwell_nonlinear_pml_source!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    pml::MaxwellNonlinearPML;
    elements = axes(U.Ex, 2),
)
    validate_maxwell_nonlinear_pml(pml, U)
    size(rhs.rhsEx) == size(U.Ex) ||
        throw(ArgumentError("Maxwell RHS and field sizes must match."))

    @inbounds for elem in elements, node in axes(U.Ex, 1)
        source = _nonlinear_pml_source(
            (
                U.Ex[node, elem],
                U.Ey[node, elem],
                U.Ez[node, elem],
            ),
            (
                U.Hx[node, elem],
                U.Hy[node, elem],
                U.Hz[node, elem],
            ),
            (
                pml.sigma_x[node, elem],
                pml.sigma_y[node, elem],
                pml.sigma_z[node, elem],
            );
            a = pml.a,
            regularization = pml.regularization,
        )
        rhs.rhsEx[node, elem] += source.electric[1]
        rhs.rhsEy[node, elem] += source.electric[2]
        rhs.rhsEz[node, elem] += source.electric[3]
        rhs.rhsHx[node, elem] += source.magnetic[1]
        rhs.rhsHy[node, elem] += source.magnetic[2]
        rhs.rhsHz[node, elem] += source.magnetic[3]
    end

    return rhs
end

function maxwell_nonlinear_pml_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    maxwell_rhs!(rhs, U, dg, registry, formulation; ε = 1.0, μ = 1.0)
    return add_maxwell_nonlinear_pml_source!(rhs, U, pml)
end

function maxwell_nonlinear_pml_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization,
    periodic::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    maxwell_rhs_periodic!(
        rhs,
        U,
        dg,
        periodic,
        registry,
        formulation;
        ε = 1.0,
        μ = 1.0,
    )
    return add_maxwell_nonlinear_pml_source!(rhs, U, pml)
end

function maxwell_nonlinear_pml_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    maxwell_rhs!(rhs, U, distributed_dg, registry, formulation; ε = 1.0, μ = 1.0)
    add_maxwell_nonlinear_pml_source!(
        rhs,
        U,
        pml;
        elements = distributed_dg.distributed_mesh.partition.owned,
    )
    return zero_ghost_maxwell_rhs!(rhs, distributed_dg)
end

function maxwell_nonlinear_pml_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    maxwell_rhs_periodic!(
        rhs,
        U,
        distributed_dg,
        periodic,
        registry,
        formulation;
        ε = 1.0,
        μ = 1.0,
    )
    add_maxwell_nonlinear_pml_source!(
        rhs,
        U,
        pml;
        elements = distributed_dg.distributed_mesh.partition.owned,
    )
    return zero_ghost_maxwell_rhs!(rhs, distributed_dg)
end

function make_maxwell_nonlinear_pml_rhs_function(
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    return (rhs, U) ->
        maxwell_nonlinear_pml_rhs!(
            rhs,
            U,
            dg,
            registry,
            formulation,
            pml,
        )
end

function make_distributed_maxwell_nonlinear_pml_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    return (rhs, U) ->
        maxwell_nonlinear_pml_rhs!(
            rhs,
            U,
            distributed_dg,
            registry,
            formulation,
            pml,
        )
end

function make_distributed_periodic_maxwell_nonlinear_pml_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    return (rhs, U) ->
        maxwell_nonlinear_pml_rhs_periodic!(
            rhs,
            U,
            distributed_dg,
            periodic,
            registry,
            formulation,
            pml,
        )
end

function maxwell_nonlinear_pml_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    rhs_function! = make_maxwell_nonlinear_pml_rhs_function(
        dg,
        registry,
        formulation,
        pml,
    )
    return rk_step!(U, work, scheme, dt, rhs_function!)
end

function distributed_maxwell_nonlinear_pml_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    rhs_function! = make_distributed_maxwell_nonlinear_pml_rhs_function(
        distributed_dg,
        registry,
        formulation,
        pml,
    )
    return rk_step!(U, work, scheme, dt, rhs_function!)
end

function distributed_periodic_maxwell_nonlinear_pml_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    pml::MaxwellNonlinearPML,
)
    rhs_function! =
        make_distributed_periodic_maxwell_nonlinear_pml_rhs_function(
            distributed_dg,
            periodic,
            registry,
            formulation,
            pml,
        )
    return rk_step!(U, work, scheme, dt, rhs_function!)
end

"""
    run_maxwell_nonlinear_pml_time_steps!(
        U,
        dg,
        registry,
        formulation::PoissonBracketFormulation,
        pml;
        rk_order=4,
        dt,
        nsteps,
        energy_every=1,
    )

Advance the Poisson-bracket spatial discretization with the nonlinear PML
source using a standard explicit Runge--Kutta method. The PML source couples
the electric and magnetic partitions and is dissipative, so the conservative
partitioned symplectic integrators are not applicable.
"""
function run_maxwell_nonlinear_pml_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    pml::MaxwellNonlinearPML;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    energy_every::Int = 1,
)
    require_poisson_bracket_central_flux(formulation)
    dt > 0.0 || throw(ArgumentError("PML time step must be positive."))
    nsteps >= 0 || throw(ArgumentError("PML step count must be non-negative."))
    energy_every >= 1 ||
        throw(ArgumentError("PML energy interval must be at least one."))

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)
    energy0 = maxwell_energy(U, dg.ref, dg.mappings)

    println("Poisson-bracket Maxwell nonlinear PML time marching")
    println("----------------------------------------------------")
    println("RK scheme:        ", scheme.name)
    println("dt:               ", dt)
    println("nsteps:           ", nsteps)
    println("initial energy:   ", energy0.total)

    for step in 1:nsteps
        maxwell_nonlinear_pml_rk_step!(
            U,
            work,
            scheme,
            dt,
            dg,
            registry,
            formulation,
            pml,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, dg.ref, dg.mappings)
            ratio = energy.total / max(energy0.total, eps(Float64))
            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", E/E0 = ", ratio,
            )
        end
    end

    return U
end

"""
    run_distributed_maxwell_nonlinear_pml_time_steps!(
        U,
        distributed_dg,
        registry,
        formulation::PoissonBracketFormulation,
        pml;
        rk_order=4,
        dt,
        nsteps,
        energy_every=1,
    )

Distributed counterpart of [`run_maxwell_nonlinear_pml_time_steps!`](@ref).
Only owned elements receive PML sources and contribute to the reported global
energy.
"""
function run_distributed_maxwell_nonlinear_pml_time_steps!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    pml::MaxwellNonlinearPML;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    energy_every::Int = 1,
)
    require_poisson_bracket_central_flux(formulation)
    dt > 0.0 || throw(ArgumentError("PML time step must be positive."))
    nsteps >= 0 || throw(ArgumentError("PML step count must be non-negative."))
    energy_every >= 1 ||
        throw(ArgumentError("PML energy interval must be at least one."))

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)
    rank = MPI.Comm_rank(distributed_dg.comm)
    energy0 = distributed_maxwell_energy(U, distributed_dg)

    if rank == 0
        println("Distributed Poisson-bracket nonlinear PML time marching")
        println("-------------------------------------------------------")
        println("MPI ranks:        ", MPI.Comm_size(distributed_dg.comm))
        println("RK scheme:        ", scheme.name)
        println("dt:               ", dt)
        println("nsteps:           ", nsteps)
        println("initial energy:   ", energy0.total)
    end

    for step in 1:nsteps
        distributed_maxwell_nonlinear_pml_rk_step!(
            U,
            work,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation,
            pml,
        )

        if step % energy_every == 0 || step == nsteps
            energy = distributed_maxwell_energy(U, distributed_dg)
            if rank == 0
                ratio = energy.total / max(energy0.total, eps(Float64))
                println(
                    "step = ", step,
                    ", time = ", step * dt,
                    ", energy = ", energy.total,
                    ", E/E0 = ", ratio,
                )
            end
        end
    end

    return U
end
