#!/usr/bin/env julia

# MPI convergence study for the periodic Poisson-bracket Maxwell plane wave.

# From the DiscoGMPI repository root:
#   mpiexec -n 2 julia --project=. \
#     examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl

include(joinpath(@__DIR__, "distributed_periodic_poisson_bracket_maxwell.jl"))

using Printf

const PERIODIC_CONVERGENCE_RATE_TOLERANCE = 0.5
const PERIODIC_COMPONENT_RATE_ERROR_FLOOR = 1e-12
const PERIODIC_REFERENCE_TET_VOLUME = 4.0 / 3.0



struct DistributedPeriodicConvergenceResult
    mpi_ranks::Int
    boundary_condition::Symbol
    flux::String
    mesh_family::Symbol
    order::Int
    esprk_order::Int
    cubature_order::Int
    mesh_level::Int
    nx_target::Int
    target_h::Float64
    nelements::Int
    min_owned_elements::Int
    max_owned_elements::Int
    characteristic_h::Float64
    h_min::Float64
    h_max::Float64
    h_ratio::Float64
    volume_min::Float64
    volume_max::Float64
    volume_total::Float64
    volume_ratio::Float64
    edge_min::Float64
    edge_max::Float64
    edge_ratio::Float64
    mean_ratio_min::Float64
    mean_ratio_avg::Float64
    dt::Float64
    nsteps::Int
    elapsed_seconds::Float64
    l2_electric_error::Float64
    l2_magnetic_error::Float64
    l2_total_error::Float64
    max_l2_electric_error::Float64
    max_l2_magnetic_error::Float64
    max_l2_total_error::Float64
    max_l2_component_errors::NTuple{6, Float64}
    l2_ex_error::Float64
    l2_ey_error::Float64
    l2_ez_error::Float64
    l2_hx_error::Float64
    l2_hy_error::Float64
    l2_hz_error::Float64
    relative_total_error::Float64
    linf_electric_error::Float64
    linf_magnetic_error::Float64
    linf_total_error::Float64
    energy_error::Float64
    relative_energy_error::Float64
    electric_charge::Float64
    magnetic_charge::Float64

    initial_energy::Float64
    final_energy::Float64
end

function parse_periodic_integer_list(value::AbstractString)
    values = [
        parse(Int, strip(entry))
        for entry in split(value, ",")
        if !isempty(strip(entry))
    ]
    isempty(values) &&
        throw(ArgumentError("Expected a comma-separated integer list."))
    return values
end

function periodic_wave_period(config::DistributedPeriodicConvergenceConfig)
    angular_frequency =
        config.wave_number / sqrt(config.epsilon * config.mu)
    return 2.0 * pi / angular_frequency
end

function print_periodic_convergence_usage(io::IO = stdout)
    println(io, """
Distributed periodic Poisson-bracket Maxwell convergence study

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl [options]

Options:
  --mesh-family=NAME  Mesh family: structured or unstructured.
                      Structured uses in-memory [0,2]x[0,1]x[0,1]
                      meshes with ny=nz=nx/2. Default: unstructured
  --nx-target=N       Base NxTarget / x-direction cell count. Four levels use
                      N, 2N, 4N, and 8N. A comma-separated list is accepted as
                      an alias for --nx-targets. Default: 2
  --levels=L          Number of uniform h-refinement levels. Default: 4
  --nx-targets=a,b    Explicit NxTarget family. Values must be strictly
                      increasing. Structured meshes additionally require
                      even values that double consecutively. This overrides
                      --nx-target and --levels.
  --geo=PATH          Parameterized periodic Gmsh geometry for unstructured
                      meshes. Default: examples/meshes/periodic_box_unstructured.geo
  --mesh-dir=PATH     Generated unstructured VTK mesh directory. Default:
                      <output-directory>/periodic_convergence_meshes
  --orders=a,b,c      DG polynomial orders in 1:5. Default: 2,3,4
  --nodal-set=NAME    equispaced (default) or warp-blend.
  --magnetic-initialization=MODE  interpolate (default) or curl-potential.
  --electric-initialization=MODE  interpolate (default) or curl-potential.
  --rate-policy=NAME  fail (default) or report. Results are written before failure.
  --rate-tolerance=X  Allowed shortfall from the expected rate. Default: 0.5
  --error-mode=MODE   Error sampling: final or max-time. Default: max-time.
  --error-every=N     In max-time mode, sample every N steps and at the final
                      step. Default: 1.
  --final-time=T      Final physical time. Default: 0.25
  --time=T            Alias for --final-time.
  --periods=P         Final time as P plane-wave periods. Mutually exclusive
                      with --final-time and --time.
  --cfl=C             Maxwell CFL factor. Default: 0.05
  --cfl-divisor=D     Divide the CFL factor by D for temporal-error isolation.
                      Use D=4 or D=8 to rerun with smaller time steps.
                      Default: 1.0
  --epsilon=X         Electric permittivity. The analytical wave requires 1.
                      Default: 1.0
  --mu=X              Magnetic permeability. The analytical wave requires 1.
                      Default: 1.0
  --wave-number=K     Positive x-directed wave number. Default: 2*pi
  --flux=NAME         Poisson-bracket surface flux: centered or alternating.
                      Default: centered
  --output=PATH       Output CSV. Default:
                      output/convergence_distributed_periodic_poisson_bracket.csv
  --help              Show this message.

Method:
  - axis-aligned periodic box from either structured or Gmsh meshes
  - periodic boundary pairs 1<->2, 3<->4, and 5<->6
  - x-directed analytical plane wave
  - structured or unstructured mesh family with target h, h/2, h/4, and h/8
  - PoissonBracketFormulation with selectable centered/alternating flux
  - H-first ESPRK with time order = DG order + 1
  - Jaskowiec-Sukumar cubature order max(2, 2N + 4)
  - continuous aggregate and component-wise L2 errors
  - convergence rates based on the measured characteristic mesh size
  - strict finest-pair aggregate-rate checks: E = N+1 and H = N,
    with a 0.5 rate tolerance

Interpretation:
  On a general tetrahedral family, nominally zero transverse components can
  be generated at order N and may not have meaningful rates when the coarse
  error is at roundoff level. Inspect the component table and transverse
  leakage printed below the aggregate table.
""")
end

function parse_periodic_convergence_arguments(args::Vector{String})
    mesh_family = :unstructured
    base_nx_target = 2
    number_levels = 4
    explicit_nx_targets = nothing
    orders = [2, 3, 4]
    final_time = 0.25
    final_time_explicit = false
    periods = nothing
    cfl = 0.05
    cfl_divisor = 1.0
    epsilon = 1.0
    mu = 1.0
    wave_number = DEFAULT_WAVE_NUMBER
    flux_kind = MaxwellFlux_Central
    nodal_set = :equispaced
    magnetic_initialization = :interpolate
    electric_initialization = :interpolate
    rate_policy = :fail
    rate_tolerance = PERIODIC_CONVERGENCE_RATE_TOLERANCE
    error_every = 1
    error_mode = :max_time
    geo_path = joinpath(
        @__DIR__,
        "meshes",
        "periodic_box_unstructured.geo",
    )
    mesh_dir = ""
    output = joinpath(
        "output",
        "convergence_distributed_periodic_poisson_bracket.csv",
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--mesh-family=") ||
               startswith(arg, "--mesh-type=")
            mesh_family = Symbol(lowercase(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--nx-target=")
            value = split(arg, "=", limit = 2)[2]
            if occursin(',', value)
                explicit_nx_targets = parse_periodic_integer_list(value)
            else
                base_nx_target = parse(Int, value)
            end
        elseif startswith(arg, "--levels=")
            number_levels = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nx-targets=") ||
               startswith(arg, "--cells=")
            explicit_nx_targets =
                parse_periodic_integer_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--geo=")
            geo_path = abspath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mesh-dir=")
            mesh_dir = abspath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--orders=")
            orders =
                parse_periodic_integer_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--magnetic-initialization=")
            magnetic_initialization = parse_magnetic_initialization(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--electric-initialization=")
            electric_initialization = parse_electric_initialization(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nodal-set=")
            nodal_set = parse_convergence_nodal_set(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--rate-policy=")
            rate_policy = Symbol(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--rate-tolerance=")
            rate_tolerance = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--error-every=")
            error_every = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--error-mode=")
            error_mode = Symbol(replace(lowercase(split(arg, "=", limit = 2)[2]), '-' => '_'))
        elseif startswith(arg, "--final-time=") || startswith(arg, "--time=")
            periods === nothing ||
                throw(
                    ArgumentError(
                        "--periods is mutually exclusive with --final-time and --time.",
                    ),
                )
            final_time = parse(Float64, split(arg, "=", limit = 2)[2])
            final_time_explicit = true
        elseif startswith(arg, "--periods=")
            final_time_explicit &&
                throw(
                    ArgumentError(
                        "--periods is mutually exclusive with --final-time and --time.",
                    ),
                )
            periods = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl-divisor=") ||
               startswith(arg, "--cfl-division=")
            cfl_divisor = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--epsilon=") || startswith(arg, "--eps=")
            epsilon = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--wave-number=")
            wave_number = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--flux=")
            flux_kind = parse_maxwell_flux_kind(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        else
            throw(
                ArgumentError(
                    "Unknown argument '$arg'. Run with --help for usage.",
                ),
            )
        end
    end

    mesh_family in (:structured, :unstructured) ||
        throw(ArgumentError("--mesh-family must be structured or unstructured."))
    base_nx_target >= 1 ||
        throw(ArgumentError("--nx-target must be positive."))
    number_levels >= 2 ||
        throw(ArgumentError("--levels must be at least two."))
    nx_targets = explicit_nx_targets === nothing ?
                 [base_nx_target * 2^level for level in 0:(number_levels - 1)] :
                 explicit_nx_targets
    length(nx_targets) >= 1 ||
        throw(ArgumentError("At least one mesh level is required."))
    all(>=(1), nx_targets) ||
        throw(ArgumentError("NxTarget values must be positive."))
    all(nx_targets[level + 1] > nx_targets[level]
        for level in 1:(length(nx_targets) - 1)) ||
        throw(ArgumentError("NxTarget values must be strictly increasing."))
    if mesh_family == :structured
        all(nx_targets[level + 1] == 2 * nx_targets[level]
            for level in 1:(length(nx_targets) - 1)) ||
            throw(
                ArgumentError(
                    "Structured NxTarget values must define uniform halving: " *
                    "N,2N,4N,...",
                ),
            )
        all(iseven, nx_targets) ||
            throw(
                ArgumentError(
                    "Structured periodic NxTarget values must be even so ny=nz=nx/2 is integral.",
                ),
            )
    end
    all(order -> 1 <= order <= 5, orders) ||
        throw(ArgumentError("--orders must be in 1:5."))
    if periods === nothing
        final_time > 0.0 ||
            throw(ArgumentError("--final-time must be positive."))
    else
        periods > 0.0 ||
            throw(ArgumentError("--periods must be positive."))
    end
    cfl > 0.0 ||
        throw(ArgumentError("--cfl must be positive."))
    cfl_divisor > 0.0 ||
        throw(ArgumentError("--cfl-divisor must be positive."))
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))
    wave_number > 0.0 ||
        throw(ArgumentError("--wave-number must be positive."))
    flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating) ||
        throw(ArgumentError("--flux must be centered or alternating."))
    isapprox(epsilon, 1.0; rtol = 0.0, atol = 1e-14) ||
        throw(
            ArgumentError(
                "The periodic analytical plane wave requires epsilon=1.",
            ),
        )
    isapprox(mu, 1.0; rtol = 0.0, atol = 1e-14) ||
        throw(
            ArgumentError(
                "The periodic analytical plane wave requires mu=1.",
            ),
        )

    if periods !== nothing
        angular_frequency = wave_number / sqrt(epsilon * mu)
        final_time = periods * 2.0 * pi / angular_frequency
    end

    if mesh_family == :unstructured
        isfile(geo_path) ||
            throw(ArgumentError("Gmsh geometry not found: $geo_path"))
        isempty(mesh_dir) &&
            (mesh_dir = joinpath(dirname(abspath(output)), "periodic_convergence_meshes"))
    end

    validate_convergence_policy(rate_policy, rate_tolerance)
    validate_convergence_error_sampling(error_mode, error_every)
    length(nx_targets) * length(orders) == 1 && (rate_policy = :report)
    return DistributedPeriodicConvergenceConfig(
        mesh_family,
        nx_targets,
        orders,
        final_time,
        periods,
        cfl,
        cfl_divisor,
        epsilon,
        mu,
        wave_number,
        flux_kind,
        geo_path,
        mesh_dir,
        abspath(output),
        nodal_set,
        rate_tolerance,
        rate_policy,
        magnetic_initialization,
        electric_initialization,
        error_every,
        error_mode,
    )
end

function effective_periodic_convergence_cfl(
    config::DistributedPeriodicConvergenceConfig,
)
    return config.cfl / config.cfl_divisor
end

function generate_periodic_convergence_mesh(
    config::DistributedPeriodicConvergenceConfig,
    nx_target::Int,
)
    gmsh = Sys.which("gmsh")
    gmsh === nothing &&
        error("Generating the convergence mesh family requires Gmsh.")
    mkpath(config.mesh_dir)
    mesh_path = joinpath(
        config.mesh_dir,
        "periodic_box_unstructured_nx$(nx_target).vtk",
    )
    if isfile(mesh_path)
        return mesh_path, load_periodic_mesh(mesh_path)
    end
    command = `$(gmsh) $(config.geo_path) -3 -format vtk -bin 0 -nt 1 -v 1 -setnumber NxTarget $(nx_target) -o $(mesh_path)`
    run(command)
    isfile(mesh_path) ||
        error("Gmsh did not create the expected mesh $mesh_path.")
    return mesh_path, load_periodic_mesh(mesh_path)
end

function build_structured_periodic_convergence_mesh(nx::Int)
    nx >= 2 && iseven(nx) ||
        throw(ArgumentError("Structured periodic nx must be even and at least 2."))
    return structured_box_mesh(
        nx,
        nx ÷ 2,
        nx ÷ 2;
        lower = (0.0, 0.0, 0.0),
        upper = (2.0, 1.0, 1.0),
        boundary_id = periodic_box_boundary_id,
    )
end

function root_periodic_convergence_mesh(
    config::DistributedPeriodicConvergenceConfig,
    nx_target::Int,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    mesh_path = ""
    mesh = nothing
    generation_error = nothing
    if rank == 0
        try
            if config.mesh_family == :structured
                mesh = build_structured_periodic_convergence_mesh(nx_target)
            else
                mesh_path, mesh =
                    generate_periodic_convergence_mesh(config, nx_target)
            end
        catch error
            generation_error = sprint(showerror, error)
        end
    end
    generation_error = MPI.bcast(generation_error, comm; root = 0)
    generation_error === nothing ||
        error("Periodic convergence mesh generation failed: $generation_error")
    mesh_path = MPI.bcast(mesh_path, comm; root = 0)
    return mesh_path, mesh
end

function periodic_distributed_characteristic_h(
    distributed_dg::DistributedDGDiscretization,
)
    return distributed_mesh_characteristic_h(distributed_dg)
end

function tetrahedron_diameter(
    points::AbstractMatrix{<:Real},
    tet_nodes,
)
    h = 0.0

    @inbounds for a in 1:3
        ia = tet_nodes[a]
        xa = points[1, ia]
        ya = points[2, ia]
        za = points[3, ia]

        for b in (a + 1):4
            ib = tet_nodes[b]
            dx = xa - points[1, ib]
            dy = ya - points[2, ib]
            dz = za - points[3, ib]
            h = max(h, sqrt(dx^2 + dy^2 + dz^2))
        end
    end

    return h
end

function periodic_distributed_h_min_h_max(
    distributed_dg::DistributedDGDiscretization,
)
    mesh = distributed_dg.dg.mesh

    local_h_min = Inf
    local_h_max = 0.0

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view mesh.tets[:, elem]
        hK = tetrahedron_diameter(mesh.points, tet_nodes)

        local_h_min = min(local_h_min, hK)
        local_h_max = max(local_h_max, hK)
    end

    global_h_min = MPI.Allreduce(local_h_min, min, distributed_dg.comm)
    global_h_max = MPI.Allreduce(local_h_max, max, distributed_dg.comm)

    return global_h_min, global_h_max
end

function periodic_space_l2_error_vector(
    workspace::DistributedMaxwellComponentL2Workspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    workspace.cubature_order == cubature_order ||
        throw(ArgumentError("Component-error workspace cubature order mismatch."))
    components = distributed_periodic_component_l2_errors!(
        workspace,
        U,
        distributed_dg,
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )
    electric = sqrt(sum(components[index]^2 for index in 1:3))
    magnetic = sqrt(sum(components[index]^2 for index in 4:6))
    total = sqrt(electric^2 + magnetic^2)
    return (electric, magnetic, total, components...)
end

function periodic_time_error_update!(
    l2_squared::Vector{Float64},
    linf::Vector{Float64},
    previous,
    current,
    dt::Float64,
)
    length(l2_squared) == length(linf) == length(previous) == length(current) ||
        throw(ArgumentError("Time-error vectors must have matching lengths."))
    dt > 0.0 || throw(ArgumentError("Time-error integration requires dt > 0."))
    for index in eachindex(l2_squared)
        l2_squared[index] +=
            0.5 * dt * (previous[index]^2 + current[index]^2)
        linf[index] = max(linf[index], current[index])
    end
    return nothing
end

function advance_distributed_periodic_convergence_case!(
    U::MaxwellField, distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange, registry::MaxwellBoundaryRegistry,
    scheme::ExplicitPartitionedSymplecticRKScheme, dt::Float64, nsteps::Int;
    epsilon::Float64, mu::Float64, wave::PlaneWaveParameters,
    flux_kind::MaxwellFluxKind, cubature_order::Int,
    track_time_l2_maxima::Bool = true, compute_time_error_norms::Bool = false,
    error_every::Int = 1,
)
    workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    error_workspace = track_time_l2_maxima ?
        DistributedMaxwellComponentL2Workspace(distributed_dg, cubature_order) : nothing
    if error_workspace !== nothing
        electric_scale = -wave.wave_number * wave.magnetic_amplitude /
                         (wave.angular_frequency * epsilon)
        phase_x(x) = wave.wave_number * (x - wave.x_origin)
        precompute_maxwell_exact_spatial_modes!(
            error_workspace,
            (x, y, z) -> (0.0, 0.0, electric_scale * sin(phase_x(x)),
                          0.0, wave.magnetic_amplitude * sin(phase_x(x)), 0.0),
            (x, y, z) -> (0.0, 0.0, electric_scale * cos(phase_x(x)),
                          0.0, wave.magnetic_amplitude * cos(phase_x(x)), 0.0),
        )
    end
    formulation = PoissonBracketFormulation(flux_kind)
    rhs_partition_function! = make_distributed_periodic_maxwell_partition_rhs_function(
        distributed_dg, periodic, registry, formulation; ε = epsilon, μ = mu)
    function evaluate(t)
        c = cos(wave.angular_frequency * t)
        s = -sin(wave.angular_frequency * t)
        components = distributed_maxwell_component_l2_errors_separable!(
            error_workspace, U, distributed_dg, ntuple(_ -> c, 6), ntuple(_ -> s, 6))
        electric = sqrt(sum(components[index]^2 for index in 1:3))
        magnetic = sqrt(sum(components[index]^2 for index in 4:6))
        return (electric, magnetic, hypot(electric, magnetic), components...)
    end
    previous = track_time_l2_maxima ? evaluate(0.0) : nothing
    maxima = previous === nothing ? ntuple(_ -> 0.0, 9) :
             maxwell_error_maxima(ntuple(_ -> 0.0, 9), previous)
    time_l2_squared = zeros(Float64, 9)
    previous_sample_step = 0
    for step in 1:nsteps
        partitioned_symplectic_rk_step_partitioned_rhs!(
            U, workspace, scheme, dt, rhs_partition_function!)
        if track_time_l2_maxima && (step == nsteps || step % error_every == 0)
            current = evaluate(step * dt)
            maxima = maxwell_error_maxima(maxima, current)
            if compute_time_error_norms && previous !== nothing
                sample_dt = (step - previous_sample_step) * dt
                for i in 1:9
                    time_l2_squared[i] +=
                        0.5 * sample_dt * (previous[i]^2 + current[i]^2)
                end
            end
            previous = current
            previous_sample_step = step
        end
    end
    return (
        electric = maxima[1], magnetic = maxima[2], total = maxima[3],
        time_linf = maxima,
        time_l2 = compute_time_error_norms ? Tuple(sqrt.(time_l2_squared)) : nothing,
    )
end

function run_distributed_periodic_convergence_case(
    nx_target::Int,
    polynomial_order::Int,
    mesh_level::Int,
    config::DistributedPeriodicConvergenceConfig,
    comm::MPI.Comm,
    root_mesh::Union{Nothing, RawVTUMesh},
    ;
    root_partition = nothing,
    nsteps_override::Union{Nothing, Int} = nothing,
    dt_limit_out = nothing,
    track_time_l2_maxima::Bool = true,
    time_error_norms_out = nothing,
    enforce_cfl_limit::Bool = true,
    validity_callback = nothing,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_partition = rank == 0 && root_partition === nothing ?
                     balanced_spatial_partition(root_mesh, nranks) : root_partition
    nelements = MPI.bcast(
        rank == 0 ? size(root_mesh.tets, 2) : 0,
        comm;
        root = 0,
    )

    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        polynomial_order;
        comm = comm,
        nodal_set = config.nodal_set,
    )
    box = distributed_periodic_box(distributed_dg)
    periods = config.wave_number * periodic_box_lengths(box)[1] / (2.0 * pi)
    isapprox(periods, round(Int, periods); rtol = 1e-10, atol = 1e-10) ||
        throw(
            ArgumentError(
                "Wave number $(config.wave_number) is not periodic over " *
                "the generated mesh x extent $(periodic_box_lengths(box)[1]).",
            ),
        )
    wave = PlaneWaveParameters(
        config.wave_number,
        config.wave_number / sqrt(config.epsilon * config.mu),
        MAGNETIC_AMPLITUDE,
        box.lower[1],
    )
    owned_count =
        length(distributed_dg.distributed_mesh.partition.owned)
    min_owned = MPI.Allreduce(owned_count, min, comm)
    max_owned = MPI.Allreduce(owned_count, max, comm)

    electric, magnetic = exact_periodic_wave_functions(
        0.0;
        epsilon = config.epsilon,
        mu = config.mu,
        wave = wave,
    )
    U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    registry = MaxwellBoundaryRegistry(
        Dict(boundary_id => MaxwellBC_None for boundary_id in 1:6),
    )
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        periodic_boundary_specs(box),
    )

    if config.magnetic_initialization == :curl_potential
        materials = homogeneous_maxwell_materials(size(U.Hx, 2);
            epsilon = config.epsilon, permeability = config.mu)
        potential = (x, y, z, t) -> periodic_magnetic_vector_potential(x, y, z, t;
            mu = config.mu, wave)
        initialize_magnetic_from_potential!(U, distributed_dg, materials, potential;
            nodal_set = config.nodal_set, periodic)
        initial_divergence = magnetic_initialization_diagnostics(U, distributed_dg, materials)
        rank == 0 && println("Initial magnetic divergence: ", initial_divergence)
    end

    if config.electric_initialization == :curl_potential
        materials = homogeneous_maxwell_materials(size(U.Ex, 2);
            epsilon = config.epsilon, permeability = config.mu)
        potential = (x, y, z, t) -> periodic_electric_vector_potential(x, y, z, t;
            epsilon = config.epsilon, wave)
        initialize_electric_from_potential!(U, distributed_dg, materials, potential;
            nodal_set = config.nodal_set, periodic)
        initial_divergence = electric_initialization_diagnostics(U, distributed_dg, materials)
        rank == 0 && println("Initial electric divergence: ", initial_divergence)
    end

    validity_callback === nothing || validity_callback("initial", U, distributed_dg)
    local_dt, _ = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = effective_periodic_convergence_cfl(config),
        ε = config.epsilon,
        μ = config.mu,
    )
    estimated_dt = MPI.Allreduce(local_dt, min, comm)
    dt_limit_out !== nothing && (dt_limit_out[] = estimated_dt)
    nsteps = if nsteps_override === nothing
        max(1, ceil(Int, config.final_time / estimated_dt))
    else
        nsteps_override > 0 ||
            throw(ArgumentError("The overridden time-step count must be positive."))
        nsteps_override
    end
    dt = config.final_time / nsteps
    !enforce_cfl_limit || dt <= estimated_dt * (1.0 + 64.0 * eps(Float64)) ||
        throw(
            ArgumentError(
                "Planned dt=$dt exceeds the global CFL limit $estimated_dt " *
                "for N=$polynomial_order, mesh level $mesh_level.",
            ),
        )
    esprk_order = polynomial_order + 1
    scheme = explicit_partitioned_symplectic_rk_scheme(
        esprk_order;
        first_partition = :H,
    )
    cubature_order = max(2, 2 * polynomial_order + 4)
    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )

    MPI.Barrier(comm)
    time_l2_maxima = Ref{Any}(nothing)
    local_elapsed = @elapsed begin
        time_l2_maxima[] = advance_distributed_periodic_convergence_case!(
            U,
            distributed_dg,
            periodic,
            registry,
            scheme,
            dt,
            nsteps;
            epsilon = config.epsilon,
            mu = config.mu,
            wave = wave,
            flux_kind = config.flux_kind,
            cubature_order = cubature_order,
            track_time_l2_maxima = track_time_l2_maxima,
            compute_time_error_norms = time_error_norms_out !== nothing,
            error_every = config.error_every,
        )
    end
    if time_error_norms_out !== nothing
        time_error_norms_out[] = (
            l2 = time_l2_maxima[].time_l2,
            linf = time_l2_maxima[].time_linf,
        )
    end
    elapsed = MPI.Allreduce(local_elapsed, max, comm)

    diagnostics = distributed_periodic_quadrature_diagnostics(
        U,
        distributed_dg,
        config.final_time,
        cubature_order;
        epsilon = config.epsilon,
        mu = config.mu,
        wave = wave,
    )
    component_errors = distributed_periodic_component_l2_errors(
        U,
        distributed_dg,
        config.final_time,
        cubature_order;
        epsilon = config.epsilon,
        mu = config.mu,
        wave = wave,
    )
    linf_electric, linf_magnetic, linf_total =
        distributed_periodic_linf_errors(
            U,
            distributed_dg,
            config.final_time,
            cubature_order;
            epsilon = config.epsilon,
            mu = config.mu,
            wave = wave,
        )
    final_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    validity_callback === nothing || validity_callback("final", U, distributed_dg)
    relative_energy_error =
        (final_energy.total - initial_energy.total) /
        max(initial_energy.total, eps(Float64))
    l2_maxima = time_l2_maxima[]
    if config.error_mode == :final
        final_electric = sqrt(sum(component_errors[index]^2 for index in 1:3))
        final_magnetic = sqrt(sum(component_errors[index]^2 for index in 4:6))
        final_total = hypot(final_electric, final_magnetic)
        final_vector = (final_electric, final_magnetic, final_total, component_errors...)
        l2_maxima = (electric = final_electric, magnetic = final_magnetic,
                     total = final_total, time_linf = final_vector, time_l2 = nothing)
    end

    characteristic_h = periodic_distributed_characteristic_h(distributed_dg)
    quality = distributed_mesh_quality_metrics(distributed_dg)

    result = DistributedPeriodicConvergenceResult(
        nranks,
        :periodic,
        maxwell_flux_kind_label(config.flux_kind),
        config.mesh_family,
        polynomial_order,
        esprk_order,
        cubature_order,
        mesh_level,
        nx_target,
        periodic_box_lengths(box)[1] / nx_target,
        nelements,
        min_owned,
        max_owned,
        characteristic_h,
        quality.h_min,
        quality.h_max,
        quality.h_ratio,
        quality.volume_min,
        quality.volume_max,
        quality.volume_total,
        quality.volume_ratio,
        quality.edge_min,
        quality.edge_max,
        quality.edge_ratio,
        quality.mean_ratio_min,
        quality.mean_ratio_avg,
        dt,
        nsteps,
        elapsed,
        diagnostics.electric_error_l2,
        diagnostics.magnetic_error_l2,
        diagnostics.field_error_l2,
        l2_maxima.electric,
        l2_maxima.magnetic,
        l2_maxima.total,
        time_l2_maxima[].time_linf[4:9],
        component_errors...,
        diagnostics.field_relative_error,
        linf_electric,
        linf_magnetic,
        linf_total,
        diagnostics.total_energy - diagnostics.exact_total_energy,
        relative_energy_error,
        diagnostics.electric_charge,
        diagnostics.magnetic_charge,
        initial_energy.total,
        final_energy.total,
    )

    MPI.Barrier(comm)
    return result
end

function periodic_convergence_row(r::DistributedPeriodicConvergenceResult, config)
    maxima = (r.max_l2_electric_error, r.max_l2_magnetic_error, r.max_l2_total_error,
              r.max_l2_component_errors...)
    final_errors = (r.l2_electric_error, r.l2_magnetic_error, r.l2_total_error,
                    r.l2_ex_error, r.l2_ey_error, r.l2_ez_error,
                    r.l2_hx_error, r.l2_hy_error, r.l2_hz_error)
    metadata = merge(maxwell_convergence_metadata(r), (
        case = "periodic", nodal_set = config.nodal_set, mesh_parameter = r.nx_target,
        magnetic_initialization = string(config.magnetic_initialization),
        electric_initialization = string(config.electric_initialization),
        mesh_source = config.mesh_family == :structured ? "structured_periodic_box" :
                      joinpath(config.mesh_dir, "periodic_box_unstructured_nx$(r.nx_target).vtk"),
        final_time = config.final_time, epsilon = config.epsilon, mu = config.mu,
        wave_number = config.wave_number,
        effective_cfl = effective_periodic_convergence_cfl(config),
        nodes_per_element = num_tet_nodes(r.order),
        error_samples = config.error_mode == :final ? 1 :
                        2 + fld(r.nsteps - 1, config.error_every),
        error_mode = replace(string(config.error_mode), '_' => '-'),
        error_every = config.error_every,
        relative_energy_drift = r.relative_energy_error,
        rate_tolerance = config.rate_tolerance, rate_policy = config.rate_policy))
    return maxwell_convergence_row(metadata, maxima, final_errors)
end

function run_periodic_convergence_study(
    config::DistributedPeriodicConvergenceConfig,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    if rank == 0
        println(
            "Distributed periodic Poisson-bracket Maxwell convergence study",
        )
        println(
            "--------------------------------------------------------------",
        )
        println("MPI ranks:          ", nranks)
        println("DG orders:          ", config.orders)
        println("mesh family:        ", config.mesh_family)
        if config.mesh_family == :unstructured
            println("Gmsh geometry:      ", config.geo_path)
            println("generated meshes:   ", config.mesh_dir)
        end
        println("NxTarget levels:    ", config.nx_targets)
        target_h_ratios = config.nx_targets[1] ./ config.nx_targets
        println("target h ratios:    ", join(target_h_ratios, ", "))
        if config.periods !== nothing
            println("wave periods:       ", config.periods)
            println("wave period:        ", periodic_wave_period(config))
        end
        println("final time:         ", config.final_time)
        println("CFL:                ", config.cfl)
        println("CFL divisor:        ", config.cfl_divisor)
        println("effective CFL:      ", effective_periodic_convergence_cfl(config))
        println("wave number:        ", config.wave_number)
        println("boundary condition: periodic")
        println("flux:               ", maxwell_flux_kind_label(config.flux_kind))
        println("ESPRK rule:         order N+1, H-first")
        println("cubature rule:      Jaskowiec-Sukumar max(2,2N+4)")
        println("analytical mode:    traveling plane wave along +x")
        println("output:             ", config.output)
        println("error mode/every:   ", replace(string(config.error_mode), '_' => '-'), " / ", config.error_every)
        println()
    end

    results = DistributedPeriodicConvergenceResult[]
    root_meshes = Dict{Int, Union{Nothing, RawVTUMesh}}()
    root_partitions = Dict{Int, Any}()
    for nx_target in config.nx_targets
        _, root_mesh =
            root_periodic_convergence_mesh(config, nx_target, comm)
        root_meshes[nx_target] = root_mesh
        root_partitions[nx_target] = rank == 0 ?
            balanced_spatial_partition(root_mesh, nranks) : nothing
    end

    for order in config.orders
        for (level_index, nx_target) in enumerate(config.nx_targets)
            mesh_level = level_index - 1
            if rank == 0
                @printf(
                    "Running N=%d, ESPRK=%d, level=%d, %s=%d\n",
                    order,
                    order + 1,
                    mesh_level,
                    config.mesh_family == :structured ? "x-cells" : "NxTarget",
                    nx_target,
                )
            end
            push!(
                results,
                run_distributed_periodic_convergence_case(
                    nx_target,
                    order,
                    mesh_level,
                    config,
                    comm,
                    root_meshes[nx_target];
                    root_partition = root_partitions[nx_target],
                    track_time_l2_maxima = config.error_mode == :max_time,
                ),
            )
        end
    end

    rows = rank == 0 ? [periodic_convergence_row(r, config) for r in results] : NamedTuple[]
    finish_maxwell_convergence(rows, config.output, comm; policy = config.rate_policy)
    return results
end

function main_periodic_convergence(args::Vector{String})
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    try
        config = parse_periodic_convergence_arguments(args)
        if config === nothing
            rank == 0 && print_periodic_convergence_usage()
            return nothing
        end
        run_periodic_convergence_study(config, comm)
    catch error
        rank == 0 && println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    finally
        if initialized_here && !MPI.Finalized()
            MPI.Finalize()
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_periodic_convergence(ARGS)
end
