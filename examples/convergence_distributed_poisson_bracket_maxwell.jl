#!/usr/bin/env julia

# MPI convergence study for the distributed Poisson-bracket Maxwell cavity.
#
# Structured example:
#   mpiexec -n 2 julia --project=. \
#     examples/convergence_distributed_poisson_bracket_maxwell.jl \
#     --mesh-family=structured --cells=2,3,4,5
#
# Unstructured example:
#   mpiexec -n 2 julia --project=. \
#     examples/convergence_distributed_poisson_bracket_maxwell.jl \
#     --mesh-family=unstructured --mesh-dir=examples/meshes/pec_box

include(joinpath(@__DIR__, "distributed_poisson_bracket_maxwell.jl"))

using Printf

const REFERENCE_TET_VOLUME = 4.0 / 3.0
const CONVERGENCE_RATE_TOLERANCE = 0.5
const COMPONENT_RATE_ERROR_FLOOR = 1e-12
const UNSTRUCTURED_MESH_LEVELS = collect(0:3)

Base.@kwdef mutable struct DistributedConvergenceResult
    mpi_ranks::Int
    boundary_condition::Symbol
    flux::String
    mesh_family::Symbol
    order::Int
    esprk_order::Int
    cubature_order::Int
    mesh_level::Int
    mesh_parameter::Int
    nelements::Int
    min_owned_elements::Int
    max_owned_elements::Int
    characteristic_h::Float64
    h_min::Float64
    h_max::Float64
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
    final_errors::NTuple{9, Float64}
    linf_l2_electric_error::Float64
    linf_l2_magnetic_error::Float64
    linf_l2_u_error::Float64
    linf_l2_ex_error::Float64
    linf_l2_ey_error::Float64
    linf_l2_ez_error::Float64
    linf_l2_hx_error::Float64
    linf_l2_hy_error::Float64
    linf_l2_hz_error::Float64
    initial_energy::Float64
    final_energy::Float64
    relative_energy_drift::Float64

end

function parse_integer_list(value::AbstractString)
    values = [
        parse(Int, strip(entry))
        for entry in split(value, ",")
        if !isempty(strip(entry))
    ]
    isempty(values) &&
        throw(ArgumentError("Expected a comma-separated integer list."))
    return values
end

function print_convergence_usage(io::IO = stdout)
    println(io, """
Distributed Poisson-bracket Maxwell cavity convergence study

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/convergence_distributed_poisson_bracket_maxwell.jl [options]

Structured mesh study
---------------------
The driver generates each tetrahedral unit-cube mesh in memory. Every Cartesian
cell is split into six tetrahedra. Interior vertices may be jittered.

  --mesh-family=structured
  --cells=a,b,c,d     Cells per coordinate direction, coarse to fine.
                      Default: 2,3,4,5
  --jitter=J          Interior-node jitter as a fraction of cell spacing.
                      Default: 0.08
  --seed=N            Deterministic jitter seed. Default: 1234

Example:
  mpiexec -n 4 julia --project=. \\
    examples/convergence_distributed_poisson_bracket_maxwell.jl \\
    --mesh-family=structured --cells=2,3,4,5 --orders=1,2,3

Unstructured mesh study
-----------------------
The driver does not generate meshes. It loads these four pre-generated files
from --mesh-dir and uses the same family for every polynomial degree:

  pec_box_m0.vtk
  pec_box_m1.vtk
  pec_box_m2.vtk
  pec_box_m3.vtk

  --mesh-family=unstructured
  --mesh-dir=PATH     Directory containing the four files above.
                      Default: examples/meshes
  --mesh-levels=a,b   Select unstructured mesh indices from 0,1,2,3.
                      Default: 0,1,2,3.

The --cells, --jitter, and --seed options are structured-only controls.

Example:
  mpiexec -n 4 julia --project=. \\
    examples/convergence_distributed_poisson_bracket_maxwell.jl \\
    --mesh-family=unstructured --mesh-dir=examples/meshes/pec_box \\
    --orders=1,2,3

Common options
--------------
  --orders=a,b,c      DG polynomial orders. Default: 1,2,3
  --nodal-set=NAME    equispaced (default) or warp-blend.
  --rate-policy=NAME  fail (default) or report. Results are written before failure.
  --rate-tolerance=X  Allowed shortfall from the expected rate. Default: 0.5
  --error-mode=MODE   Error sampling: final or max-time. Default: max-time.
  --error-every=N     In max-time mode, sample every N steps and at the final
                      step. Default: 1.
  --final-time=T      Final physical time. Default: 0.25
  --time=T            Alias for --final-time.
  --periods=P         Final time as P cavity-wave periods. Mutually exclusive
                      with --final-time and --time.
  --cfl=C             Maxwell CFL factor. Default: 0.05
  --cfl-divisor=D     Divide CFL by D for temporal-error isolation.
                      Default: 1.0
  --epsilon=X         Electric permittivity. Default: 1.0
  --mu=X              Magnetic permeability. Default: 1.0
  --boundary-condition=NAME
                      Exterior boundary condition: pec or pmc.
                      Default: pec
  --flux=NAME         Poisson-bracket surface flux: centered or alternating.
                      Default: centered
  --output=PATH       Output CSV. Default:
                      output/convergence_distributed_poisson_bracket.csv
  --help              Show this message.

Reported diagnostics
--------------------
  - L-infinity-in-time, L2-in-space errors for Ex,Ey,Ez,Hx,Hy,Hz,E,H,U
  - h_min, h_max, and characteristic h
  - actual timestep dt used by each simulation
  - component, E, H, and U rates computed with h_max
  - initial/final energy and relative energy drift
  - mesh quality and element ownership metadata

Method
------
  - unit-cube PEC eigenmode or its electromagnetic-dual PMC mode
  - PoissonBracketFormulation with selectable centered/alternating flux
  - H-first ESPRK with time order N+1
  - Jaskowiec-Sukumar cubature order max(2,2N+4)
  - finest-pair checks: E components N+1, H components N
""")
end

function cavity_wave_period(epsilon::Float64, mu::Float64)
    omega = sqrt(3.0) * pi / sqrt(epsilon * mu)
    return 2.0 * pi / omega
end

function parse_convergence_arguments(args::Vector{String})
    mesh_parameters = [2, 3, 4, 5]
    explicit_cells = false
    mesh_family = :structured
    mesh_dir = joinpath(@__DIR__, "meshes")
    explicit_mesh_dir = false
    explicit_mesh_levels = false
    orders = [1, 2, 3]
    final_time = 0.25
    final_time_explicit = false
    periods = nothing
    cfl = 0.05
    cfl_divisor = 1.0
    jitter = 0.08
    explicit_jitter = false
    seed = 1234
    explicit_seed = false
    epsilon = 1.0
    mu = 1.0
    boundary_condition = :pec
    flux_kind = MaxwellFlux_Central
    nodal_set = :equispaced
    rate_policy = :fail
    rate_tolerance = CONVERGENCE_RATE_TOLERANCE
    error_every = 1
    error_mode = :max_time
    output =
        joinpath("output", "convergence_distributed_poisson_bracket.csv")

    for arg in args
        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--mesh-family=") ||
               startswith(arg, "--mesh-type=")
            mesh_family =
                Symbol(lowercase(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--cells=")
            mesh_parameters =
                parse_integer_list(split(arg, "=", limit = 2)[2])
            explicit_cells = true
        elseif startswith(arg, "--mesh-dir=")
            mesh_dir = abspath(split(arg, "=", limit = 2)[2])
            explicit_mesh_dir = true
        elseif startswith(arg, "--mesh-levels=")
            mesh_parameters = parse_integer_list(split(arg, "=", limit = 2)[2])
            explicit_mesh_levels = true
        elseif startswith(arg, "--orders=")
            orders = parse_integer_list(split(arg, "=", limit = 2)[2])
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
        elseif startswith(arg, "--jitter=")
            jitter = parse(Float64, split(arg, "=", limit = 2)[2])
            explicit_jitter = true
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit = 2)[2])
            explicit_seed = true
        elseif startswith(arg, "--epsilon=") || startswith(arg, "--eps=")
            epsilon = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--boundary-condition=")
            boundary_condition =
                Symbol(lowercase(split(arg, "=", limit = 2)[2]))
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
    if mesh_family == :structured
        explicit_mesh_levels &&
            throw(ArgumentError("--mesh-levels is only valid with --mesh-family=unstructured."))
        explicit_mesh_dir &&
            throw(
                ArgumentError(
                    "--mesh-dir is only valid with --mesh-family=unstructured.",
                ),
            )
        length(mesh_parameters) >= 1 ||
            throw(ArgumentError("At least one structured mesh level is required."))
        issorted(mesh_parameters) ||
            throw(ArgumentError("--cells must be ordered from coarse to fine."))
        length(unique(mesh_parameters)) == length(mesh_parameters) ||
            throw(ArgumentError("--cells entries must be distinct."))
        all(>=(1), mesh_parameters) ||
            throw(ArgumentError("--cells entries must be positive."))
    else
        explicit_cells &&
            throw(
                ArgumentError(
                    "--cells is only valid with --mesh-family=structured. " *
                    "Unstructured mode always uses pec_box_m0.vtk through pec_box_m3.vtk.",
                ),
            )
        explicit_jitter &&
            throw(
                ArgumentError(
                    "--jitter is only valid with --mesh-family=structured.",
                ),
            )
        explicit_seed &&
            throw(
                ArgumentError(
                    "--seed is only valid with --mesh-family=structured.",
                ),
            )
        mesh_parameters = explicit_mesh_levels ? mesh_parameters : copy(UNSTRUCTURED_MESH_LEVELS)
        all(level -> level in UNSTRUCTURED_MESH_LEVELS, mesh_parameters) ||
            throw(ArgumentError("--mesh-levels entries must be among 0,1,2,3."))
        length(unique(mesh_parameters)) == length(mesh_parameters) ||
            throw(ArgumentError("--mesh-levels entries must be distinct."))
    end

    all(order -> 1 <= order <= 5, orders) ||
        throw(
            ArgumentError(
                "--orders must be in 1:5 because ESPRK order is N+1.",
            ),
        )
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
    0.0 <= jitter < 0.25 ||
        throw(ArgumentError("--jitter must lie in [0, 0.25)."))
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))
    boundary_condition in (:pec, :pmc) ||
        throw(ArgumentError("--boundary-condition must be pec or pmc."))
    flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating) ||
        throw(ArgumentError("--flux must be centered or alternating."))
    if flux_kind == MaxwellFlux_Alternating && boundary_condition != :pec
        throw(
            ArgumentError(
                "--flux=alternating currently supports --boundary-condition=pec.",
            ),
        )
    end

    if periods !== nothing
        final_time = periods * cavity_wave_period(epsilon, mu)
    end

    validate_convergence_policy(rate_policy, rate_tolerance)
    validate_convergence_error_sampling(error_mode, error_every)
    length(mesh_parameters) * length(orders) == 1 && (rate_policy = :report)
    return DistributedConvergenceConfig(
        mesh_parameters,
        mesh_family,
        abspath(mesh_dir),
        orders,
        final_time,
        periods,
        cfl,
        cfl_divisor,
        jitter,
        seed,
        epsilon,
        mu,
        boundary_condition,
        flux_kind,
        abspath(output),
        nodal_set,
        rate_tolerance,
        rate_policy,
        error_every,
        error_mode,
    )
end

effective_convergence_cfl(config::DistributedConvergenceConfig) =
    config.cfl / config.cfl_divisor

function build_convergence_mesh(
    cells_per_axis::Int;
    jitter::Float64,
    seed::Int,
)
    return structured_box_mesh(
        cells_per_axis,
        cells_per_axis,
        cells_per_axis;
        boundary_id = PEC_BOUNDARY_ID,
        jitter = jitter,
        seed = seed,
    )
end

unstructured_convergence_mesh_path(
    config::DistributedConvergenceConfig,
    mesh_level::Int,
) = joinpath(config.mesh_dir, "pec_box_m$(mesh_level).vtk")

function validate_unstructured_mesh_family(
    config::DistributedConvergenceConfig,
    comm::MPI.Comm,
)
    config.mesh_family == :unstructured || return nothing
    rank = MPI.Comm_rank(comm)
    validation_error = nothing
    if rank == 0
        missing_paths = filter(
            path -> !isfile(path),
            [
                unstructured_convergence_mesh_path(config, level)
                for level in config.mesh_parameters
            ],
        )
        if !isempty(missing_paths)
            validation_error =
                "Missing unstructured convergence meshes: " *
                join(missing_paths, ", ")
        end
    end
    validation_error = MPI.bcast(validation_error, comm; root = 0)
    validation_error === nothing || error(validation_error)
    return nothing
end

function root_convergence_mesh(
    config::DistributedConvergenceConfig,
    mesh_parameter::Int,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    root_mesh = nothing
    mesh_path = ""
    load_error = nothing

    if rank == 0
        try
            if config.mesh_family == :structured
                root_mesh = build_convergence_mesh(
                    mesh_parameter;
                    jitter = config.jitter,
                    seed = config.seed,
                )
            else
                mesh_path =
                    unstructured_convergence_mesh_path(config, mesh_parameter)
                isfile(mesh_path) ||
                    error(
                        "Unstructured convergence mesh not found: $mesh_path. " *
                        "Expected pec_box_m0.vtk through pec_box_m3.vtk in " *
                        "$(config.mesh_dir).",
                    )
                root_mesh = load_pec_mesh(mesh_path)
            end
        catch error
            load_error = sprint(showerror, error)
        end
    end

    load_error = MPI.bcast(load_error, comm; root = 0)
    load_error === nothing ||
        error("Convergence mesh loading failed: $load_error")
    mesh_path = MPI.bcast(mesh_path, comm; root = 0)
    return root_mesh, mesh_path
end

function distributed_characteristic_h(
    distributed_dg::DistributedDGDiscretization,
)
    return distributed_mesh_characteristic_h(distributed_dg)
end

function cavity_space_l2_error_vector!(
    workspace::DistributedMaxwellComponentL2Workspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    components = distributed_maxwell_component_l2_errors!(
        workspace,
        U,
        distributed_dg;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
    electric = sqrt(sum(components[index]^2 for index in 1:3))
    magnetic = sqrt(sum(components[index]^2 for index in 4:6))
    total = hypot(electric, magnetic)
    return (electric, magnetic, total, components...)
end

function update_linf_l2_errors(
    current::NTuple{9, Float64},
    candidate::NTuple{9, Float64},
)
    return maxwell_error_maxima(current, candidate)
end

function advance_distributed_convergence_case!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    nsteps::Int;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol,
    flux_kind::MaxwellFluxKind,
    cubature_order::Int,
    error_every::Int = 1,
    error_mode::Symbol = :max_time,
)
    rk_workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    error_workspace = error_mode == :max_time ?
        DistributedMaxwellComponentL2Workspace(distributed_dg, cubature_order) : nothing
    omega = sqrt(3.0) * pi / sqrt(epsilon * mu)
    electric0, _ = exact_cavity_mode_functions(0.0;
        epsilon = epsilon, mu = mu, boundary_condition = boundary_condition)
    _, magnetic_quarter = exact_cavity_mode_functions(pi / (2.0 * omega);
        epsilon = epsilon, mu = mu, boundary_condition = boundary_condition)
    error_workspace === nothing || precompute_maxwell_exact_spatial_modes!(
        error_workspace,
        (x, y, z) -> (electric0(x, y, z)..., magnetic_quarter(x, y, z)...))
    formulation = PoissonBracketFormulation(flux_kind)
    rhs_partition_function! = make_distributed_maxwell_partition_rhs_function(
        distributed_dg, registry, formulation; ε = epsilon, μ = mu)
    function evaluate(t)
        c = cos(omega * t)
        s = sin(omega * t)
        factors = (c, c, c, s, s, s)
        components = distributed_maxwell_component_l2_errors_separable!(
            error_workspace, U, distributed_dg, factors, ntuple(_ -> 0.0, 6))
        electric = sqrt(sum(components[index]^2 for index in 1:3))
        magnetic = sqrt(sum(components[index]^2 for index in 4:6))
        return (electric, magnetic, hypot(electric, magnetic), components...)
    end
    maxima = error_mode == :max_time ? evaluate(0.0) : ntuple(_ -> 0.0, 9)

    for step in 1:nsteps
        partitioned_symplectic_rk_step_partitioned_rhs!(
            U, rk_workspace, scheme, dt, rhs_partition_function!)
        if error_mode == :max_time && (step == nsteps || step % error_every == 0)
            candidate = evaluate(step * dt)
            maxima = update_linf_l2_errors(maxima, candidate)
        end
    end
    return maxima
end

function run_distributed_convergence_case(
    mesh_parameter::Int,
    polynomial_order::Int,
    mesh_level::Int,
    config::DistributedConvergenceConfig,
    comm::MPI.Comm,
    root_mesh_override = nothing,
    root_partition_override = nothing,
    root_mesh_preloaded::Bool = false,
    ; nsteps_override::Union{Nothing, Int} = nothing,
    validity_callback = nothing,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_mesh = root_mesh_override
    if !root_mesh_preloaded
        root_mesh, _ = root_convergence_mesh(config, mesh_parameter, comm)
    end
    root_partition = rank == 0 && root_partition_override === nothing ?
                     balanced_spatial_partition(root_mesh, nranks) :
                     root_partition_override
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
    owned_count =
        length(distributed_dg.distributed_mesh.partition.owned)
    min_owned = MPI.Allreduce(owned_count, min, comm)
    max_owned = MPI.Allreduce(owned_count, max, comm)

    electric, magnetic = exact_cavity_mode_functions(
        0.0;
        epsilon = config.epsilon,
        mu = config.mu,
        boundary_condition = config.boundary_condition,
    )
    U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    validity_callback === nothing || validity_callback("initial", U, distributed_dg)
    boundary_kind = config.boundary_condition == :pec ?
                    MaxwellBC_PEC :
                    MaxwellBC_PMC
    registry = MaxwellBoundaryRegistry(
        Dict(PEC_BOUNDARY_ID => boundary_kind),
    )

    local_dt, _ = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = effective_convergence_cfl(config),
        ε = config.epsilon,
        μ = config.mu,
    )
    estimated_dt = MPI.Allreduce(local_dt, min, comm)
    nsteps = nsteps_override === nothing ?
             max(1, ceil(Int, config.final_time / estimated_dt)) : nsteps_override
    nsteps > 0 || throw(ArgumentError("Time-step count must be positive."))
    dt = config.final_time / nsteps
    dt <= estimated_dt * (1 + 64eps(Float64)) ||
        throw(ArgumentError("Planned dt=$dt exceeds CFL limit $estimated_dt."))
    esprk_order = polynomial_order + 1
    scheme = explicit_partitioned_symplectic_rk_scheme(
        esprk_order;
        first_partition = :H,
    )
    cubature_order = max(2, 2 * polynomial_order + 4)
    quality = distributed_mesh_quality_metrics(distributed_dg)
    characteristic_h = distributed_characteristic_h(distributed_dg)
    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    ).total

    MPI.Barrier(comm)
    started = time_ns()
    linf_l2_errors = advance_distributed_convergence_case!(
        U,
        distributed_dg,
        registry,
        scheme,
        dt,
        nsteps;
        epsilon = config.epsilon,
        mu = config.mu,
        boundary_condition = config.boundary_condition,
        flux_kind = config.flux_kind,
        cubature_order = cubature_order,
        error_every = config.error_every,
        error_mode = config.error_mode,
    )
    elapsed = MPI.Allreduce((time_ns() - started) * 1e-9, max, comm)
    final_errors = cavity_space_l2_error_vector!(
        DistributedMaxwellComponentL2Workspace(distributed_dg, cubature_order),
        U, distributed_dg, config.final_time;
        epsilon = config.epsilon, mu = config.mu,
        boundary_condition = config.boundary_condition)
    config.error_mode == :final && (linf_l2_errors = final_errors)
    validity_callback === nothing || validity_callback("final", U, distributed_dg)
    final_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    ).total
    relative_energy_drift =
        (final_energy - initial_energy) /
        max(abs(initial_energy), eps(Float64))

    result = DistributedConvergenceResult(
        mpi_ranks = nranks,
        boundary_condition = config.boundary_condition,
        flux = maxwell_flux_kind_label(config.flux_kind),
        mesh_family = config.mesh_family,
        order = polynomial_order,
        esprk_order = esprk_order,
        cubature_order = cubature_order,
        mesh_level = mesh_level,
        mesh_parameter = mesh_parameter,
        nelements = nelements,
        min_owned_elements = min_owned,
        max_owned_elements = max_owned,
        characteristic_h = characteristic_h,
        h_min = quality.h_min,
        h_max = quality.h_max,
        volume_min = quality.volume_min,
        volume_max = quality.volume_max,
        volume_total = quality.volume_total,
        volume_ratio = quality.volume_ratio,
        edge_min = quality.edge_min,
        edge_max = quality.edge_max,
        edge_ratio = quality.edge_ratio,
        mean_ratio_min = quality.mean_ratio_min,
        mean_ratio_avg = quality.mean_ratio_avg,
        dt = dt,
        nsteps = nsteps,
        elapsed_seconds = elapsed,
        final_errors = final_errors,
        linf_l2_electric_error = linf_l2_errors[1],
        linf_l2_magnetic_error = linf_l2_errors[2],
        linf_l2_u_error = linf_l2_errors[3],
        linf_l2_ex_error = linf_l2_errors[4],
        linf_l2_ey_error = linf_l2_errors[5],
        linf_l2_ez_error = linf_l2_errors[6],
        linf_l2_hx_error = linf_l2_errors[7],
        linf_l2_hy_error = linf_l2_errors[8],
        linf_l2_hz_error = linf_l2_errors[9],
        initial_energy = initial_energy,
        final_energy = final_energy,
        relative_energy_drift = relative_energy_drift,
    )

    MPI.Barrier(comm)
    return result
end

function cavity_convergence_row(r::DistributedConvergenceResult, config)
    maxima = (r.linf_l2_electric_error, r.linf_l2_magnetic_error, r.linf_l2_u_error,
              r.linf_l2_ex_error, r.linf_l2_ey_error, r.linf_l2_ez_error,
              r.linf_l2_hx_error, r.linf_l2_hy_error, r.linf_l2_hz_error)
    metadata = merge(maxwell_convergence_metadata(r), (
        case = "cavity",
        nodal_set = config.nodal_set,
        mesh_source = config.mesh_family == :structured ? "structured_unit_cube" :
                      unstructured_convergence_mesh_path(config, r.mesh_parameter),
        final_time = config.final_time, epsilon = config.epsilon, mu = config.mu,
        wave_number = sqrt(3.0) * pi, jitter = config.jitter, seed = config.seed,
        effective_cfl = effective_convergence_cfl(config),
        error_mode = replace(string(config.error_mode), '_' => '-'),
        error_every = config.error_every,
        error_samples = config.error_mode == :final ? 1 :
                        2 + fld(r.nsteps - 1, config.error_every),
        nodes_per_element = num_tet_nodes(r.order),
        h_ratio = r.h_max / r.h_min,
        rate_tolerance = config.rate_tolerance, rate_policy = config.rate_policy))
    return maxwell_convergence_row(metadata, maxima, r.final_errors)
end

function run_convergence_study(
    config::DistributedConvergenceConfig,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    validate_unstructured_mesh_family(config, comm)

    if rank == 0
        println("Distributed Poisson-bracket Maxwell convergence study")
        println("-----------------------------------------------------")
        println("MPI ranks:          ", nranks)
        println("DG orders:          ", config.orders)
        println("mesh family:        ", config.mesh_family)
        if config.mesh_family == :structured
            println("cells per axis:     ", config.mesh_parameters)
            println("interior jitter:    ", config.jitter)
        else
            println("mesh directory:     ", config.mesh_dir)
            println(
                "mesh files:         ",
                join(
                    [
                        "pec_box_m$(level).vtk"
                        for level in UNSTRUCTURED_MESH_LEVELS
                    ],
                    ", ",
                ),
            )
        end
        if config.periods !== nothing
            println("wave periods:       ", config.periods)
            println(
                "wave period:        ",
                cavity_wave_period(config.epsilon, config.mu),
            )
        end
        println("final time:         ", config.final_time)
        println("CFL:                ", config.cfl)
        println("CFL divisor:        ", config.cfl_divisor)
        println("effective CFL:      ", effective_convergence_cfl(config))
        println(
            "boundary condition: ",
            uppercase(string(config.boundary_condition)),
        )
        println("flux:               ", maxwell_flux_kind_label(config.flux_kind))
        println("ESPRK rule:         order N+1, H-first")
        println("rate length scale:  h_max")
        println("error norm:         L-infinity time, L2 space")
        println("output:             ", config.output)
        println("error mode/every:   ", replace(string(config.error_mode), '_' => '-'), " / ", config.error_every)
        println()
    end

    results = DistributedConvergenceResult[]
    root_meshes = Dict{Int, Any}()
    root_partitions = Dict{Int, Any}()
    for mesh_parameter in config.mesh_parameters
        root_mesh, _ = root_convergence_mesh(config, mesh_parameter, comm)
        root_meshes[mesh_parameter] = root_mesh
        root_partitions[mesh_parameter] = rank == 0 ?
            balanced_spatial_partition(root_mesh, nranks) : nothing
    end
    for order in config.orders
        for (index, mesh_parameter) in enumerate(config.mesh_parameters)
            mesh_level = index - 1
            if rank == 0
                mesh_description =
                    config.mesh_family == :structured ?
                    "cells=$mesh_parameter" :
                    "pec_box_m$(mesh_parameter).vtk"
                @printf(
                    "Running N=%d, ESPRK=%d, level=%d, %s\n",
                    order,
                    order + 1,
                    mesh_level,
                    mesh_description,
                )
            end
            push!(
                results,
                run_distributed_convergence_case(
                    mesh_parameter,
                    order,
                    mesh_level,
                    config,
                    comm,
                    root_meshes[mesh_parameter],
                    root_partitions[mesh_parameter],
                    true,
                ),
            )
        end
    end

    rows = rank == 0 ? [cavity_convergence_row(r, config) for r in results] : NamedTuple[]
    finish_maxwell_convergence(rows, config.output, comm; policy = config.rate_policy)
    return results
end

function main_convergence(args::Vector{String})
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    try
        config = parse_convergence_arguments(args)
        if config === nothing
            rank == 0 && print_convergence_usage()
            return nothing
        end
        run_convergence_study(config, comm)
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
    main_convergence(ARGS)
end
