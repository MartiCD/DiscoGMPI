# -------------------------------------------------------------------------
# Explicit Runge-Kutta time integration
# -------------------------------------------------------------------------


function num_stages(scheme::ExplicitRKScheme)
    return length(scheme.b)
end

function num_stages(scheme::ExplicitPartitionedSymplecticRKScheme)
    return length(scheme.first_weights)
end

function explicit_rk_scheme(order::Int)
    if order == 1
        A = zeros(Float64, 1, 1)
        b = [1.0]
        c = [0.0]

        return ExplicitRKScheme(
            1,
            "RK1 / Forward Euler",
            A,
            b,
            c,
        )

    elseif order == 2
        # Explicit midpoint method.
        A = zeros(Float64, 2, 2)
        A[2, 1] = 0.5

        b = [0.0, 1.0]
        c = [0.0, 0.5]

        return ExplicitRKScheme(
            2,
            "RK2 / Explicit midpoint",
            A,
            b,
            c,
        )

    elseif order == 3
        # Classical third-order RK.
        A = zeros(Float64, 3, 3)
        A[2, 1] = 0.5
        A[3, 1] = -1.0
        A[3, 2] = 2.0

        b = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
        c = [0.0, 0.5, 1.0]

        return ExplicitRKScheme(
            3,
            "RK3 / Classical third-order",
            A,
            b,
            c,
        )

    elseif order == 4
        # Classical RK4.
        A = zeros(Float64, 4, 4)
        A[2, 1] = 0.5
        A[3, 2] = 0.5
        A[4, 3] = 1.0

        b = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
        c = [0.0, 0.5, 0.5, 1.0]

        return ExplicitRKScheme(
            4,
            "RK4 / Classical fourth-order",
            A,
            b,
            c,
        )

    elseif order == 5
        # Dormand-Prince fifth-order update.
        #
        # This is the 5th-order solution of the common DOPRI5(4) pair.
        # We only use the 5th-order weights here, not adaptive stepping.
        A = zeros(Float64, 7, 7)

        A[2, 1] = 1.0 / 5.0

        A[3, 1] = 3.0 / 40.0
        A[3, 2] = 9.0 / 40.0

        A[4, 1] = 44.0 / 45.0
        A[4, 2] = -56.0 / 15.0
        A[4, 3] = 32.0 / 9.0

        A[5, 1] = 19372.0 / 6561.0
        A[5, 2] = -25360.0 / 2187.0
        A[5, 3] = 64448.0 / 6561.0
        A[5, 4] = -212.0 / 729.0

        A[6, 1] = 9017.0 / 3168.0
        A[6, 2] = -355.0 / 33.0
        A[6, 3] = 46732.0 / 5247.0
        A[6, 4] = 49.0 / 176.0
        A[6, 5] = -5103.0 / 18656.0

        A[7, 1] = 35.0 / 384.0
        A[7, 2] = 0.0
        A[7, 3] = 500.0 / 1113.0
        A[7, 4] = 125.0 / 192.0
        A[7, 5] = -2187.0 / 6784.0
        A[7, 6] = 11.0 / 84.0

        b = [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ]

        c = [
            0.0,
            1.0 / 5.0,
            3.0 / 10.0,
            4.0 / 5.0,
            8.0 / 9.0,
            1.0,
            1.0,
        ]

        return ExplicitRKScheme(
            5,
            "RK5 / Dormand-Prince fifth-order update",
            A,
            b,
            c,
        )

    else
        error("Unsupported RK order $order. Supported orders are 1, 2, 3, 4, 5.")
    end
end

function normalize_maxwell_partition(partition::Symbol)
    if partition in (:E, :electric, :Electric)
        return :E
    elseif partition in (:H, :magnetic, :Magnetic)
        return :H
    else
        error(
            "Unsupported Maxwell partition $partition. " *
            "Use :E or :H."
        )
    end
end

function complementary_maxwell_partition(partition::Symbol)
    p = normalize_maxwell_partition(partition)

    if p == :E
        return :H
    else
        return :E
    end
end

function validate_partitioned_symplectic_rk_scheme(
    scheme::ExplicitPartitionedSymplecticRKScheme,
)
    normalize_maxwell_partition(scheme.first_partition)

    if length(scheme.first_weights) != length(scheme.second_weights)
        error(
            "Partitioned RK scheme has $(length(scheme.first_weights)) " *
            "first-partition weights but $(length(scheme.second_weights)) " *
            "second-partition weights."
        )
    end

    return scheme
end

function esprk_coefficients(order::Int)
    if order == 1
        return (
            first = [1.0],
            second = [1.0],
        )

    elseif order == 2
        return (
            first = [0.0, 1.0],
            second = [0.5, 0.5],
        )

    elseif order == 3
        return (
            first = [
                0.2916666666666667,
                0.75,
                -0.041666666666666664,
            ],
            second = [
                0.6666666666666666,
                -0.6666666666666666,
                1.0,
            ],
        )

    elseif order == 4
        return (
            first = [
                7.0 / 48.0,
                3.0 / 8.0,
                -1.0 / 48.0,
                -1.0 / 48.0,
                3.0 / 8.0,
                7.0 / 48.0,
            ],
            second = [
                1.0 / 3.0,
                -1.0 / 3.0,
                1.0,
                -1.0 / 3.0,
                1.0 / 3.0,
                0.0,
            ],
        )

    elseif order == 5
        return (
            first = [
                0.11939002928756727,
                0.6989273703824752,
                -0.17131235827160077,
                0.40126950225135344,
                0.010705081848235983,
                -0.058979625498031166,
            ],
            second = [
                0.33983962583911,
                -0.08860133690302732,
                0.5858564768259621,
                -0.6030393565364912,
                0.3235807965546976,
                0.4423637942197495,
            ],
        )

    elseif order == 6
        return (
            first = [
                0.0502627644003922,
                0.413514300428344,
                0.0450798897943977,
                -0.188054853819569,
                0.541960678450780,
                -0.725525558508690,
                0.541960678450780,
                -0.188054853819569,
                0.0450798897943977,
                0.413514300428344,
                0.0502627644003922,
            ],
            second = [
                0.148816447901042,
                -0.132385865767784,
                0.067307604692185,
                0.432666402578175,
                -0.016404589403618,
                -0.016404589403618,
                0.432666402578175,
                0.067307604692185,
                -0.132385865767784,
                0.148816447901042,
                0.0,
            ],
        )

    else
        error(
            "Unsupported ESPRK order $order. " *
            "Supported orders are 1, 2, 3, 4, 5, 6."
        )
    end
end

function explicit_partitioned_symplectic_rk_scheme(
    order::Int;
    first_partition::Symbol = :E,
)
    first_partition = normalize_maxwell_partition(first_partition)
    second_partition = complementary_maxwell_partition(first_partition)
    label = string(first_partition, "-", second_partition)
    coeffs = esprk_coefficients(order)

    return ExplicitPartitionedSymplecticRKScheme(
        order,
        "ESPRK$order / Explicit symplectic partitioned RK ($label)",
        first_partition,
        coeffs.first,
        coeffs.second,
    )
end

function print_rk_scheme_summary(scheme::ExplicitRKScheme)
    println("Explicit Runge-Kutta scheme")
    println("---------------------------")
    println("Name:          ", scheme.name)
    println("Order:         ", scheme.order)
    println("Stages:        ", num_stages(scheme))
    println("b weights:     ", scheme.b)
    println("c nodes:       ", scheme.c)

    return nothing
end

function print_partitioned_symplectic_rk_scheme_summary(
    scheme::ExplicitPartitionedSymplecticRKScheme,
)
    validate_partitioned_symplectic_rk_scheme(scheme)

    first_partition = normalize_maxwell_partition(scheme.first_partition)
    second_partition = complementary_maxwell_partition(first_partition)

    println("Explicit partitioned symplectic Runge-Kutta scheme")
    println("---------------------------------------------------")
    println("Name:              ", scheme.name)
    println("Order:             ", scheme.order)
    println("Stages:            ", num_stages(scheme))
    println("First partition:   ", first_partition)
    println("Second partition:  ", second_partition)
    println("First weights:     ", scheme.first_weights)
    println("Second weights:    ", scheme.second_weights)

    return nothing
end

function similar_maxwell_field(U::MaxwellField)
    return MaxwellField(
        similar(U.Ex),
        similar(U.Ey),
        similar(U.Ez),
        similar(U.Hx),
        similar(U.Hy),
        similar(U.Hz),
    )
end


function copy_maxwell_field!(dest::MaxwellField, src::MaxwellField)
    copyto!(dest.Ex, src.Ex)
    copyto!(dest.Ey, src.Ey)
    copyto!(dest.Ez, src.Ez)

    copyto!(dest.Hx, src.Hx)
    copyto!(dest.Hy, src.Hy)
    copyto!(dest.Hz, src.Hz)

    return dest
end


function fill_maxwell_field!(U::MaxwellField, value::Float64)
    fill!(U.Ex, value)
    fill!(U.Ey, value)
    fill!(U.Ez, value)

    fill!(U.Hx, value)
    fill!(U.Hy, value)
    fill!(U.Hz, value)

    return U
end


function add_scaled_rhs_to_field!(
    U::MaxwellField,
    rhs::MaxwellRHS,
    α::Float64,
)
    U.Ex .+= α .* rhs.rhsEx
    U.Ey .+= α .* rhs.rhsEy
    U.Ez .+= α .* rhs.rhsEz

    U.Hx .+= α .* rhs.rhsHx
    U.Hy .+= α .* rhs.rhsHy
    U.Hz .+= α .* rhs.rhsHz

    return U
end

function add_scaled_electric_rhs_to_field!(
    U::MaxwellField,
    rhs::MaxwellRHS,
    α::Float64,
)
    U.Ex .+= α .* rhs.rhsEx
    U.Ey .+= α .* rhs.rhsEy
    U.Ez .+= α .* rhs.rhsEz

    return U
end

function add_scaled_magnetic_rhs_to_field!(
    U::MaxwellField,
    rhs::MaxwellRHS,
    α::Float64,
)
    U.Hx .+= α .* rhs.rhsHx
    U.Hy .+= α .* rhs.rhsHy
    U.Hz .+= α .* rhs.rhsHz

    return U
end

function add_scaled_partition_rhs_to_field!(
    U::MaxwellField,
    rhs::MaxwellRHS,
    partition::Symbol,
    α::Float64,
)
    p = normalize_maxwell_partition(partition)

    if p == :E
        return add_scaled_electric_rhs_to_field!(U, rhs, α)
    else
        return add_scaled_magnetic_rhs_to_field!(U, rhs, α)
    end
end


function MaxwellRKWorkspace(U::MaxwellField, scheme::ExplicitRKScheme)
    s = num_stages(scheme)

    U0 = similar_maxwell_field(U)
    Ustage = similar_maxwell_field(U)

    K = [similar_maxwell_rhs(U) for _ in 1:s]

    return MaxwellRKWorkspace(U0, Ustage, K)
end

function MaxwellPartitionedRKWorkspace(U::MaxwellField)
    return MaxwellPartitionedRKWorkspace(similar_maxwell_rhs(U))
end

function MaxwellPartitionedRKWorkspace(
    U::MaxwellField,
    scheme::ExplicitPartitionedSymplecticRKScheme,
)
    validate_partitioned_symplectic_rk_scheme(scheme)

    return MaxwellPartitionedRKWorkspace(U)
end

function build_rk_stage_field!(
    Ustage::MaxwellField,
    U0::MaxwellField,
    K::Vector{MaxwellRHS},
    scheme::ExplicitRKScheme,
    stage::Int,
    dt::Float64,
)
    copy_maxwell_field!(Ustage, U0)

    for j in 1:(stage - 1)
        aij = scheme.A[stage, j]

        if aij != 0.0
            add_scaled_rhs_to_field!(
                Ustage,
                K[j],
                dt * aij,
            )
        end
    end

    return Ustage
end

function rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    rhs_function!::Function,
)
    if length(work.K) != num_stages(scheme)
        error(
            "RK workspace has $(length(work.K)) stages, " *
            "but scheme requires $(num_stages(scheme))."
        )
    end

    copy_maxwell_field!(work.U0, U)

    s = num_stages(scheme)

    for i in 1:s
        build_rk_stage_field!(
            work.Ustage,
            work.U0,
            work.K,
            scheme,
            i,
            dt,
        )

        rhs_function!(work.K[i], work.Ustage)
    end

    # Final update:
    # U^{n+1} = U0 + dt * Σ bᵢ Kᵢ
    copy_maxwell_field!(U, work.U0)

    for i in 1:s
        bi = scheme.b[i]

        if bi != 0.0
            add_scaled_rhs_to_field!(
                U,
                work.K[i],
                dt * bi,
            )
        end
    end

    return U
end

function partitioned_symplectic_rk_step!(
    U::MaxwellField,
    work::MaxwellPartitionedRKWorkspace,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    rhs_function!::Function,
)
    validate_partitioned_symplectic_rk_scheme(scheme)

    first_partition = normalize_maxwell_partition(scheme.first_partition)
    second_partition = complementary_maxwell_partition(first_partition)

    for i in 1:num_stages(scheme)
        first_weight = scheme.first_weights[i]

        if first_weight != 0.0
            rhs_function!(work.rhs, U)
            add_scaled_partition_rhs_to_field!(
                U,
                work.rhs,
                first_partition,
                dt * first_weight,
            )
        end

        second_weight = scheme.second_weights[i]

        if second_weight != 0.0
            rhs_function!(work.rhs, U)
            add_scaled_partition_rhs_to_field!(
                U,
                work.rhs,
                second_partition,
                dt * second_weight,
            )
        end
    end

    return U
end

function make_maxwell_rhs_function(
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
    return make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function make_maxwell_rhs_function(
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        maxwell_rhs!(
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

        return rhs
    end
end

function make_maxwell_rhs_function(
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return make_maxwell_rhs_function(
        dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function make_maxwell_rhs_function(
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        maxwell_rhs!(
            rhs,
            U,
            dg,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )

        return rhs
    end
end

function max_abs_maxwell_field(U::MaxwellField)
    return maximum((
        maximum(abs.(U.Ex)),
        maximum(abs.(U.Ey)),
        maximum(abs.(U.Ez)),
        maximum(abs.(U.Hx)),
        maximum(abs.(U.Hy)),
        maximum(abs.(U.Hz)),
    ))
end

function test_rk_zero_field_step(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    rk_order::Int = 4,
    dt::Float64 = 1e-4,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    registry = default_maxwell_boundary_registry()

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    max_field = max_abs_maxwell_field(U)

    println("RK zero-field step test")
    println("-----------------------")
    println("RK scheme:       ", scheme.name)
    println("dt:              ", dt)
    println("max |U| after step: ", max_field)

    if max_field < 1e-14
        println("✓ RK step preserves the zero Maxwell field")
    else
        println("⚠ RK zero-field test failed")
    end

    return nothing
end

function test_rk_one_step_energy_diagnostic(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    rk_order::Int = 4,
    dt::Float64 = 1e-5,
)
    Efun = (x, y, z) -> (
        sinpi(x) * sinpi(y) * sinpi(z),
        cospi(x) * sinpi(y) * sinpi(z),
        sinpi(x) * cospi(y) * sinpi(z),
    )

    Hfun = (x, y, z) -> (
        sinpi(x) * sinpi(y) * cospi(z),
        cospi(x) * sinpi(y) * cospi(z),
        sinpi(x) * cospi(y) * cospi(z),
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    energy0 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    registry = default_maxwell_boundary_registry()

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    energy1 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    ΔE = energy1.total - energy0.total
    relΔE = ΔE / max(energy0.total, eps(Float64))

    println("RK one-step energy diagnostic")
    println("-----------------------------")
    println("RK scheme:        ", scheme.name)
    println("dt:               ", dt)
    println("energy before:    ", energy0.total)
    println("energy after:     ", energy1.total)
    println("ΔE:               ", ΔE)
    println("relative ΔE:      ", relΔE)

    return nothing
end

function test_partitioned_symplectic_rk_zero_field_step(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    psrk_order::Int = 2,
    first_partition::Symbol = :E,
    dt::Float64 = 1e-4,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)

    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = first_partition,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    registry = default_maxwell_boundary_registry()

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    max_field = max_abs_maxwell_field(U)

    println("Partitioned symplectic RK zero-field step test")
    println("----------------------------------------------")
    println("PSRK scheme:      ", scheme.name)
    println("dt:               ", dt)
    println("max |U| after step: ", max_field)

    if max_field < 1e-14
        println("✓ PSRK step preserves the zero Maxwell field")
    else
        println("⚠ PSRK zero-field test failed")
    end

    return nothing
end

function test_partitioned_symplectic_rk_one_step_energy_diagnostic(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    psrk_order::Int = 2,
    first_partition::Symbol = :E,
    dt::Float64 = 1e-5,
)
    Efun = (x, y, z) -> (
        sinpi(x) * sinpi(y) * sinpi(z),
        cospi(x) * sinpi(y) * sinpi(z),
        sinpi(x) * cospi(y) * sinpi(z),
    )

    Hfun = (x, y, z) -> (
        sinpi(x) * sinpi(y) * cospi(z),
        cospi(x) * sinpi(y) * cospi(z),
        sinpi(x) * cospi(y) * cospi(z),
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    energy0 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = first_partition,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    registry = default_maxwell_boundary_registry()

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    energy1 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    ΔE = energy1.total - energy0.total
    relΔE = ΔE / max(energy0.total, eps(Float64))

    println("Partitioned symplectic RK one-step energy diagnostic")
    println("----------------------------------------------------")
    println("PSRK scheme:       ", scheme.name)
    println("dt:                ", dt)
    println("energy before:     ", energy0.total)
    println("energy after:      ", energy1.total)
    println("ΔE:                ", ΔE)
    println("relative ΔE:       ", relΔE)

    return nothing
end

function run_maxwell_time_steps!(
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = ε,
        μ = μ,
    )

    println("Maxwell time marching")
    println("---------------------")
    println("RK scheme:       ", scheme.name)
    println("dt:              ", dt)
    println("nsteps:          ", nsteps)

    energy0 = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
    println("initial energy:  ", energy0.total)

    for step in 1:nsteps
        rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end

function run_maxwell_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return run_maxwell_time_steps!(
        U,
        dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        rk_order = rk_order,
        dt = dt,
        nsteps = nsteps,
        ε = ε,
        μ = μ,
        energy_every = energy_every,
    )
end

function run_maxwell_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_rhs_function(
        dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    println("Maxwell time marching")
    println("---------------------")
    println("RK scheme:       ", scheme.name)
    println("dt:              ", dt)
    println("nsteps:          ", nsteps)

    energy0 = maxwell_energy(U, dg.ref, dg.mappings; ε = ε, μ = μ)
    println("initial energy:  ", energy0.total)

    for step in 1:nsteps
        rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, dg.ref, dg.mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end

function run_maxwell_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    throw(
        ArgumentError(
            "PoissonBracketFormulation is partitioned and must be marched " *
            "with run_maxwell_partitioned_symplectic_time_steps!(...; " *
            "psrk_order = 1:6, first_partition = :H)."
        ),
    )
end

function run_maxwell_partitioned_symplectic_time_steps!(
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    psrk_order::Int = 2,
    first_partition::Symbol = :E,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = first_partition,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = ε,
        μ = μ,
    )

    println("Maxwell partitioned symplectic time marching")
    println("--------------------------------------------")
    println("PSRK scheme:      ", scheme.name)
    println("dt:               ", dt)
    println("nsteps:           ", nsteps)

    energy0 = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
    println("initial energy:   ", energy0.total)

    for step in 1:nsteps
        partitioned_symplectic_rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end

function run_maxwell_partitioned_symplectic_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry;
    psrk_order::Int = 2,
    first_partition::Symbol = :E,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return run_maxwell_partitioned_symplectic_time_steps!(
        U,
        dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        psrk_order = psrk_order,
        first_partition = first_partition,
        dt = dt,
        nsteps = nsteps,
        ε = ε,
        μ = μ,
        energy_every = energy_every,
    )
end

function run_maxwell_partitioned_symplectic_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    psrk_order::Int = 2,
    first_partition::Symbol = :E,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = first_partition,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_rhs_function(
        dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    println("Maxwell partitioned symplectic time marching")
    println("--------------------------------------------")
    println("PSRK scheme:      ", scheme.name)
    println("dt:               ", dt)
    println("nsteps:           ", nsteps)

    energy0 = maxwell_energy(U, dg.ref, dg.mappings; ε = ε, μ = μ)
    println("initial energy:   ", energy0.total)

    for step in 1:nsteps
        partitioned_symplectic_rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, dg.ref, dg.mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end

function run_maxwell_partitioned_symplectic_time_steps!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    psrk_order::Int = 1,
    first_partition::Symbol = :H,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    require_poisson_bracket_central_flux(formulation)

    if normalize_maxwell_partition(first_partition) != :H
        throw(
            ArgumentError(
                "PoissonBracketFormulation must update the magnetic " *
                "partition first. Use first_partition = :H."
            ),
        )
    end

    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = :H,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_rhs_function(
        dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    println("Poisson-bracket Maxwell partitioned symplectic time marching")
    println("------------------------------------------------------------")
    println("PSRK scheme:      ", scheme.name)
    println("dt:               ", dt)
    println("nsteps:           ", nsteps)
    println("Stage order:      H partition, then E partition")

    energy0 = maxwell_energy(U, dg.ref, dg.mappings; ε = ε, μ = μ)
    println("initial energy:   ", energy0.total)

    for step in 1:nsteps
        partitioned_symplectic_rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, dg.ref, dg.mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end
