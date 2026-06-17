#!/usr/bin/env julia

# Standalone distributed periodic Poisson-bracket Maxwell isoresolution study.
#
# This script intentionally does not include the existing convergence drivers.
# It reuses the imperative periodic driver for mesh loading, periodic-boundary
# conventions, exact-wave setup, and the distributed time-stepper API.

include(joinpath(@__DIR__, "distributed_periodic_poisson_bracket_maxwell.jl"))

using Printf

const ISO_ORDERS = (1, 2, 4)
const ISO_LEVELS = 0:3
const ISO_BASE_INTERVALS = 4
const ISO_WAVE_NUMBER = 2.0 * pi
const ISO_MAGNETIC_AMPLITUDE = 1.0
const ISO_DEFAULT_CFL = 0.05
const ISO_DEFAULT_FINAL_TIME = 0.25
const ISO_DEFAULT_RATE_TOLERANCE = 0.25
const ISO_DEFAULT_TARGET_WORK_PER_RANK = 2.0e6
const ISO_DEFAULT_MIN_ELEMENTS_PER_RANK = 16
const ISO_REFERENCE_TET_VOLUME = 4.0 / 3.0
const ISO_ERROR_NAMES = (
    "electric",
    "magnetic",
    "total",
    "ex",
    "ey",
    "ez",
    "hx",
    "hy",
    "hz",
)
const ISO_EMPTY_RATES = convert(
    NTuple{9, Union{Missing, Float64}},
    ntuple(_ -> missing, 9),
)

struct PeriodicIsoresolutionCase
    order::Int
    level::Int
    order_label::String
    mesh_label::String
    exponent::Int
    intervals_per_wavelength::Int
    nx_target::Int
    h_target::Float64
end

struct PeriodicIsoresolutionConfig
    orders::Vector{Int}
    levels::Vector{Int}
    base_intervals::Int
    final_time::Float64
    cfl::Float64
    cfl_divisor::Float64
    epsilon::Float64
    mu::Float64
    wave_number::Float64
    geo_path::String
    mesh_dir::String
    gmsh::String
    output::String
    latex_output::String
    force_mesh::Bool
    smoke::Bool
    dry_run::Bool
    target_work_per_rank::Float64
    min_elements_per_rank::Int
    max_active_ranks::Int
    rate_tolerance::Float64
end

mutable struct PeriodicIsoresolutionResult
    case::PeriodicIsoresolutionCase
    active_ranks::Int
    nelements::Int
    min_owned_elements::Int
    max_owned_elements::Int
    nodes_per_element::Int
    work_units::Float64
    cubature_order::Int
    h_target::Float64
    h_characteristic::Float64
    h_min::Float64
    h_max::Float64
    h_ratio::Float64
    volume_total::Float64
    volume_min::Float64
    volume_max::Float64
    volume_ratio::Float64
    mean_ratio_min::Float64
    mean_ratio_avg::Float64
    dt::Float64
    nsteps::Int
    elapsed_seconds::Float64
    final_space_l2_errors::NTuple{9, Float64}
    time_linf_space_l2_errors::NTuple{9, Float64}
    rates_time_linf_space_l2::NTuple{9, Union{Missing, Float64}}
    initial_energy::Float64
    final_energy::Float64
    relative_energy_error::Float64
end

function parse_integer_list(value::AbstractString)
    values = [
        parse(Int, strip(entry))
        for entry in split(value, ',')
        if !isempty(strip(entry))
    ]
    isempty(values) &&
        throw(ArgumentError("Expected a comma-separated integer list."))
    return values
end

function default_latex_output(csv_path::AbstractString)
    if endswith(lowercase(csv_path), ".csv")
        return csv_path[1:(lastindex(csv_path) - 4)] * ".tex"
    end
    return string(csv_path, ".tex")
end

function periodic_wave_period(epsilon::Float64, mu::Float64, wave_number::Float64)
    angular_frequency = wave_number / sqrt(epsilon * mu)
    return 2.0 * pi / angular_frequency
end

function print_periodic_isoresolution_usage(io::IO = stdout)
    println(io, """
Distributed periodic Poisson-bracket Maxwell isoresolution convergence study

Usage:
  mpiexec -n <max-ranks> julia --project=. \\
    examples/convergence_distributed_periodic_isoresolution.jl [options]

Options:
  --orders=a,b,c             DG orders. Default: 1,2,4
  --levels=L                 Number of mesh levels, interpreted as 0:L-1.
                             Default: 4
  --base-intervals=N         P4/M0 intervals per wavelength. Default: 4
  --final-time=T, --time=T   Final physical time. Default: 0.25
  --periods=P                Final time as P wave periods.
  --cfl=C                    CFL factor before cfl-divisor. Default: 0.05
  --cfl-divisor=D            Divide CFL by D for temporal-error isolation.
                             Default: 1
  --epsilon=X                Electric permittivity. Default: 1
  --mu=X                     Magnetic permeability. Default: 1
  --wave-number=K            Positive x-directed wave number. Default: 2*pi
  --geo=PATH                 Periodic Gmsh .geo file. Default:
                             examples/meshes/periodic_box_unstructured.geo
  --mesh-dir=PATH            Generated mesh directory. Default:
                             output/periodic_isoresolution_meshes
  --gmsh=PATH                Gmsh executable. Default: gmsh from PATH
  --force-mesh               Regenerate meshes even when cached VTK files exist.
  --target-work-per-rank=X   Rank heuristic target for Ne*Np^2. Default: 2e6
  --min-elements-per-rank=N  Do not split below this owned-element target.
                             Default: 16
  --max-active-ranks=N       Cap per-case active MPI ranks. Default: all ranks
  --rate-tolerance=T         Strict rate tolerance. Default: 0.25
  --output=PATH              CSV output. Default:
                             output/convergence_distributed_periodic_isoresolution.csv
  --latex-output=PATH        LaTeX table output. Default: output path with .tex
  --smoke                    Run a capped two-case P4 smoke solve.
  --dry-run                  Print the planned cases and rank counts only.
  --help                     Show this message.

Method:
  - periodic plane wave on the Gmsh geometry domain
  - Gmsh-generated unstructured periodic tetrahedral meshes
  - isoresolution cases P1/P2/P4 with q = base*2^(level+shift(P))
  - PoissonBracketFormulation with centered flux
  - H-first ESPRK with time order P+1
  - primary errors are L2 in space and Linf in time
  - strict expected-rate checks use electric/Ez -> P+1 and magnetic/Hy/total -> P
""")
end

function parse_periodic_isoresolution_arguments(args::Vector{String})
    orders = collect(ISO_ORDERS)
    levels = collect(ISO_LEVELS)
    base_intervals = ISO_BASE_INTERVALS
    final_time = ISO_DEFAULT_FINAL_TIME
    final_time_explicit = false
    periods = nothing
    cfl = ISO_DEFAULT_CFL
    cfl_divisor = 1.0
    epsilon = 1.0
    mu = 1.0
    wave_number = ISO_WAVE_NUMBER
    geo_path = joinpath(@__DIR__, "meshes", "periodic_box_unstructured.geo")
    mesh_dir = joinpath("output", "periodic_isoresolution_meshes")
    gmsh = ""
    output = joinpath(
        "output",
        "convergence_distributed_periodic_isoresolution.csv",
    )
    latex_output = ""
    force_mesh = false
    smoke = false
    dry_run = false
    target_work_per_rank = ISO_DEFAULT_TARGET_WORK_PER_RANK
    min_elements_per_rank = ISO_DEFAULT_MIN_ELEMENTS_PER_RANK
    max_active_ranks = 0
    rate_tolerance = ISO_DEFAULT_RATE_TOLERANCE

    for arg in args
        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--orders=")
            orders = parse_integer_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--levels=")
            count = parse(Int, split(arg, "=", limit = 2)[2])
            count >= 1 || throw(ArgumentError("--levels must be positive."))
            levels = collect(0:(count - 1))
        elseif startswith(arg, "--base-intervals=")
            base_intervals = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--final-time=") || startswith(arg, "--time=")
            periods === nothing ||
                throw(ArgumentError("--periods is mutually exclusive with --final-time."))
            final_time = parse(Float64, split(arg, "=", limit = 2)[2])
            final_time_explicit = true
        elseif startswith(arg, "--periods=")
            final_time_explicit &&
                throw(ArgumentError("--periods is mutually exclusive with --final-time."))
            periods = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl-divisor=")
            cfl_divisor = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--epsilon=") || startswith(arg, "--eps=")
            epsilon = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--wave-number=")
            wave_number = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--geo=")
            geo_path = abspath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mesh-dir=")
            mesh_dir = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--gmsh=")
            gmsh = split(arg, "=", limit = 2)[2]
        elseif arg == "--force-mesh"
            force_mesh = true
        elseif startswith(arg, "--target-work-per-rank=")
            target_work_per_rank =
                parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--min-elements-per-rank=")
            min_elements_per_rank =
                parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--max-active-ranks=") ||
               startswith(arg, "--max-ranks=")
            max_active_ranks = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--rate-tolerance=")
            rate_tolerance = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--latex-output=")
            latex_output = split(arg, "=", limit = 2)[2]
        elseif arg == "--smoke"
            smoke = true
        elseif arg == "--dry-run"
            dry_run = true
        else
            throw(ArgumentError("Unknown argument '$arg'. Run with --help."))
        end
    end

    all(order -> order in ISO_ORDERS, orders) ||
        throw(ArgumentError("This isoresolution study supports only orders 1, 2, and 4."))
    !isempty(levels) || throw(ArgumentError("At least one mesh level is required."))
    minimum(levels) == 0 ||
        throw(ArgumentError("Mesh levels must start at 0."))
    levels == collect(0:maximum(levels)) ||
        throw(ArgumentError("Mesh levels must be contiguous from 0."))
    base_intervals >= 1 ||
        throw(ArgumentError("--base-intervals must be positive."))
    final_time > 0.0 || throw(ArgumentError("--final-time must be positive."))
    cfl > 0.0 || throw(ArgumentError("--cfl must be positive."))
    cfl_divisor > 0.0 ||
        throw(ArgumentError("--cfl-divisor must be positive."))
    epsilon > 0.0 || throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 || throw(ArgumentError("--mu must be positive."))
    wave_number > 0.0 ||
        throw(ArgumentError("--wave-number must be positive."))
    target_work_per_rank > 0.0 ||
        throw(ArgumentError("--target-work-per-rank must be positive."))
    min_elements_per_rank >= 1 ||
        throw(ArgumentError("--min-elements-per-rank must be positive."))
    max_active_ranks >= 0 ||
        throw(ArgumentError("--max-active-ranks must be non-negative."))
    rate_tolerance >= 0.0 ||
        throw(ArgumentError("--rate-tolerance must be non-negative."))
    isfile(geo_path) || throw(ArgumentError("Gmsh geometry not found: $geo_path"))

    if periods !== nothing
        periods > 0.0 || throw(ArgumentError("--periods must be positive."))
        final_time = periods * periodic_wave_period(epsilon, mu, wave_number)
    end

    isempty(latex_output) && (latex_output = default_latex_output(output))
    if smoke
        orders = [4]
        levels = [0, 1]
        base_intervals = min(base_intervals, 2)
        final_time = min(final_time, 2.0e-4)
    end

    return PeriodicIsoresolutionConfig(
        sort(unique(orders)),
        levels,
        base_intervals,
        final_time,
        cfl,
        cfl_divisor,
        epsilon,
        mu,
        wave_number,
        geo_path,
        mesh_dir,
        gmsh,
        output,
        latex_output,
        force_mesh,
        smoke,
        dry_run,
        target_work_per_rank,
        min_elements_per_rank,
        max_active_ranks,
        rate_tolerance,
    )
end

periodic_isoresolution_shift(order::Int) =
    order == 4 ? 0 :
    order == 2 ? 1 :
    order == 1 ? 2 :
    throw(ArgumentError("Isoresolution order must be one of 1, 2, or 4."))

function periodic_isoresolution_case(
    order::Int,
    level::Int;
    base_intervals::Int = ISO_BASE_INTERVALS,
)
    exponent = level + periodic_isoresolution_shift(order)
    q = base_intervals * 2^exponent
    return PeriodicIsoresolutionCase(
        order,
        level,
        "P$order",
        "M$level",
        exponent,
        q,
        q,
        1.0 / q,
    )
end

function periodic_isoresolution_cases(config::PeriodicIsoresolutionConfig)
    return [
        periodic_isoresolution_case(
            order,
            level;
            base_intervals = config.base_intervals,
        )
        for order in config.orders for level in config.levels
    ]
end

function periodic_isoresolution_cases(; smoke::Bool = false)
    config = parse_periodic_isoresolution_arguments(
        smoke ? ["--smoke", "--dry-run"] : ["--dry-run"],
    )
    return periodic_isoresolution_cases(config)
end

periodic_isoresolution_case_key(case::PeriodicIsoresolutionCase) =
    case.order_label * case.mesh_label

function periodic_isoresolution_unique_resolutions(cases)
    return sort(unique(case.intervals_per_wavelength for case in cases))
end

function periodic_isoresolution_mesh_path(
    config::PeriodicIsoresolutionConfig,
    q::Int,
)
    return joinpath(config.mesh_dir, @sprintf("periodic_box_iso_q%04d.vtk", q))
end

function resolved_gmsh(config::PeriodicIsoresolutionConfig)
    if !isempty(config.gmsh)
        path = abspath(config.gmsh)
        isfile(path) || throw(ArgumentError("Gmsh executable not found: $path"))
        return path
    end
    gmsh = Sys.which("gmsh")
    gmsh === nothing &&
        throw(ArgumentError("Gmsh was not found in PATH. Use --gmsh=/path/to/gmsh."))
    return gmsh
end

function generate_periodic_isoresolution_mesh!(
    config::PeriodicIsoresolutionConfig,
    q::Int,
)
    mesh_path = periodic_isoresolution_mesh_path(config, q)
    if isfile(mesh_path) && !config.force_mesh
        return mesh_path
    end

    mkpath(dirname(mesh_path))
    gmsh = resolved_gmsh(config)
    command = `$(gmsh) $(config.geo_path) -3 -format vtk -bin 0 -nt 1 -v 1 -setnumber NxTarget $(q) -o $(mesh_path)`
    run(command)
    isfile(mesh_path) ||
        error("Gmsh did not create the expected mesh: $mesh_path")
    return mesh_path
end

function estimate_periodic_isoresolution_elements(q::Int)
    transverse = max(1, round(Int, q / 4))
    return 6 * q * transverse^2
end

function tetrahedron_diameter(points::AbstractMatrix{<:Real}, tet_nodes)
    diameter = 0.0
    for a in 1:3
        ia = tet_nodes[a]
        xa = points[1, ia]
        ya = points[2, ia]
        za = points[3, ia]
        for b in (a + 1):4
            ib = tet_nodes[b]
            dx = xa - points[1, ib]
            dy = ya - points[2, ib]
            dz = za - points[3, ib]
            diameter = max(diameter, sqrt(dx^2 + dy^2 + dz^2))
        end
    end
    return diameter
end

function periodic_balanced_spatial_partition(mesh::RawVTUMesh, nranks::Int)
    nranks >= 1 || throw(ArgumentError("nranks must be positive."))
    nelements = size(mesh.tets, 2)
    nelements >= nranks ||
        throw(ArgumentError("Cannot split $nelements elements over $nranks ranks."))

    centroids = Vector{NTuple{3, Float64}}(undef, nelements)
    for elem in 1:nelements
        nodes = @view mesh.tets[:, elem]
        centroids[elem] = (
            sum(mesh.points[1, node] for node in nodes) / 4.0,
            sum(mesh.points[2, node] for node in nodes) / 4.0,
            sum(mesh.points[3, node] for node in nodes) / 4.0,
        )
    end

    order = sortperm(
        collect(1:nelements);
        by = elem -> begin
            c = centroids[elem]
            (c[1], c[2], c[3])
        end,
    )
    partition = Vector{Int}(undef, nelements)
    for (position, elem) in enumerate(order)
        partition[elem] = min(nranks - 1, fld((position - 1) * nranks, nelements))
    end
    return partition
end

function isoresolution_work_units(nelements::Int, order::Int)
    nodes = num_tet_nodes(order)
    return Float64(nelements) * Float64(nodes)^2
end

function periodic_isoresolution_rank_count(
    config::PeriodicIsoresolutionConfig,
    nelements::Int,
    order::Int,
    launcher_ranks::Int,
)
    launcher_ranks >= 1 ||
        throw(ArgumentError("The launcher must provide at least one MPI rank."))
    nelements >= 1 || throw(ArgumentError("A case must contain at least one element."))
    work = isoresolution_work_units(nelements, order)
    requested = max(1, ceil(Int, work / config.target_work_per_rank))
    rank_cap = config.max_active_ranks == 0 ?
               launcher_ranks :
               min(launcher_ranks, config.max_active_ranks)
    element_cap = max(1, fld(nelements, config.min_elements_per_rank))
    return clamp(requested, 1, min(rank_cap, element_cap))
end

function periodic_isoresolution_case_comm(comm::MPI.Comm, active_ranks::Int)
    rank = MPI.Comm_rank(comm)
    return MPI.Comm_split(comm, rank < active_ranks ? 0 : nothing, rank)
end

function periodic_space_l2_error_vector!(
    workspace::DistributedMaxwellComponentL2Workspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
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
    total = hypot(electric, magnetic)
    return (electric, magnetic, total, components...)
end

function update_linf_errors(
    current::NTuple{9, Float64},
    candidate::NTuple{9, Float64},
)
    return ntuple(index -> max(current[index], candidate[index]), 9)
end

function run_periodic_isoresolution_case(
    case::PeriodicIsoresolutionCase,
    config::PeriodicIsoresolutionConfig,
    comm::MPI.Comm,
    root_mesh::Union{Nothing, RawVTUMesh},
    active_ranks::Int,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    nranks == active_ranks ||
        throw(ArgumentError("Case communicator size does not match active ranks."))

    root_partition =
        rank == 0 ? periodic_balanced_spatial_partition(root_mesh, nranks) : nothing
    nelements = MPI.bcast(
        rank == 0 ? size(root_mesh.tets, 2) : 0,
        comm;
        root = 0,
    )

    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        case.order;
        comm = comm,
    )
    box = distributed_periodic_box(distributed_dg)
    Lx = periodic_box_lengths(box)[1]
    periods = config.wave_number * Lx / (2.0 * pi)
    isapprox(periods, round(Int, periods); rtol = 1e-10, atol = 1e-10) ||
        throw(
            ArgumentError(
                "Wave number $(config.wave_number) is not periodic over x extent $Lx.",
            ),
        )

    wave = PlaneWaveParameters(
        config.wave_number,
        config.wave_number / sqrt(config.epsilon * config.mu),
        ISO_MAGNETIC_AMPLITUDE,
        box.lower[1],
    )
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        0.0;
        epsilon = config.epsilon,
        mu = config.mu,
        wave = wave,
    )
    U = interpolate_maxwell_field(distributed_dg, exact_electric, exact_magnetic)

    registry = MaxwellBoundaryRegistry(
        Dict(boundary_id => MaxwellBC_None for boundary_id in 1:6),
    )
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        periodic_boundary_specs(box),
    )
    formulation = PoissonBracketFormulation()
    esprk_order = case.order + 1
    scheme = explicit_partitioned_symplectic_rk_scheme(
        esprk_order;
        first_partition = :H,
    )
    rk_workspace = MaxwellPartitionedRKWorkspace(U, scheme)

    local_dt, _ = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = config.cfl / config.cfl_divisor,
        ε = config.epsilon,
        μ = config.mu,
    )
    dt_limit = MPI.Allreduce(local_dt, min, comm)
    nsteps = max(1, ceil(Int, config.final_time / dt_limit))
    dt = config.final_time / nsteps
    cubature_order = max(2, 2 * case.order + 4)
    error_workspace = DistributedMaxwellComponentL2Workspace(
        distributed_dg,
        cubature_order,
    )

    owned_count = length(distributed_dg.distributed_mesh.partition.owned)
    min_owned = MPI.Allreduce(owned_count, min, comm)
    max_owned = MPI.Allreduce(owned_count, max, comm)
    quality = distributed_mesh_quality_metrics(distributed_dg)
    h_characteristic = (quality.volume_total / nelements)^(1.0 / 3.0)
    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )

    initial_errors = periodic_space_l2_error_vector!(
        error_workspace,
        U,
        distributed_dg,
        0.0;
        epsilon = config.epsilon,
        mu = config.mu,
        wave = wave,
    )
    time_linf_errors = initial_errors
    final_errors = initial_errors

    MPI.Barrier(comm)
    elapsed_local = @elapsed begin
        for step in 1:nsteps
            distributed_periodic_partitioned_symplectic_rk_step!(
                U,
                rk_workspace,
                scheme,
                dt,
                distributed_dg,
                periodic,
                registry,
                formulation;
                ε = config.epsilon,
                μ = config.mu,
            )
            final_errors = periodic_space_l2_error_vector!(
                error_workspace,
                U,
                distributed_dg,
                step * dt;
                epsilon = config.epsilon,
                mu = config.mu,
                wave = wave,
            )
            time_linf_errors = update_linf_errors(time_linf_errors, final_errors)
        end
    end
    elapsed = MPI.Allreduce(elapsed_local, max, comm)
    final_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    relative_energy_error =
        (final_energy.total - initial_energy.total) /
        max(abs(initial_energy.total), eps(Float64))

    return PeriodicIsoresolutionResult(
        case,
        active_ranks,
        nelements,
        min_owned,
        max_owned,
        num_tet_nodes(case.order),
        isoresolution_work_units(nelements, case.order),
        cubature_order,
        case.h_target,
        h_characteristic,
        quality.h_min,
        quality.h_max,
        quality.h_ratio,
        quality.volume_total,
        quality.volume_min,
        quality.volume_max,
        quality.volume_ratio,
        quality.mean_ratio_min,
        quality.mean_ratio_avg,
        dt,
        nsteps,
        elapsed,
        final_errors,
        time_linf_errors,
        ISO_EMPTY_RATES,
        initial_energy.total,
        final_energy.total,
        relative_energy_error,
    )
end

function isoresolution_convergence_rate(
    fine_error::Float64,
    coarse_error::Float64,
    fine_h::Float64,
    coarse_h::Float64,
)
    fine_error > 0.0 && coarse_error > 0.0 || return missing
    fine_h > 0.0 && coarse_h > 0.0 || return missing
    fine_h != coarse_h || return missing
    return log(coarse_error / fine_error) / log(coarse_h / fine_h)
end

function add_periodic_isoresolution_rates!(results)
    for order in sort(unique(result.case.order for result in results))
        subset = sort(
            filter(result -> result.case.order == order, results);
            by = result -> result.case.level,
        )
        previous = nothing
        for result in subset
            if previous === nothing
                result.rates_time_linf_space_l2 = ISO_EMPTY_RATES
            else
                result.rates_time_linf_space_l2 = ntuple(
                    index -> isoresolution_convergence_rate(
                        result.time_linf_space_l2_errors[index],
                        previous.time_linf_space_l2_errors[index],
                        result.h_target,
                        previous.h_target,
                    ),
                    9,
                )
            end
            previous = result
        end
    end
    return results
end

formatted_rate(rate::Union{Missing, Float64}) =
    ismissing(rate) ? "-" : @sprintf("%.3f", rate)

function rate_pass(
    rate::Union{Missing, Float64},
    expected::Float64,
    tolerance::Float64,
)
    ismissing(rate) && return false
    return isfinite(rate) && rate >= expected - tolerance
end

function result_physical_verdict(
    result::PeriodicIsoresolutionResult,
    tolerance::Float64,
)
    order = result.case.order
    specs = (
        ("E", result.rates_time_linf_space_l2[1], order + 1.0),
        ("H", result.rates_time_linf_space_l2[2], Float64(order)),
        ("total", result.rates_time_linf_space_l2[3], Float64(order)),
        ("Ez", result.rates_time_linf_space_l2[6], order + 1.0),
        ("Hy", result.rates_time_linf_space_l2[8], Float64(order)),
    )
    if result.case.level == 0
        return missing
    end
    return all(rate_pass(rate, expected, tolerance) for (_, rate, expected) in specs)
end

function periodic_isoresolution_verdict(
    results,
    tolerance::Float64,
)
    messages = String[]
    pass = true
    for result in sort(results; by = r -> (r.case.order, r.case.level))
        result.case.level == 0 && continue
        verdict = result_physical_verdict(result, tolerance)
        pass &= verdict === true
        push!(
            messages,
            @sprintf(
                "%s: Ez=%s/%s, Hy=%s/%s, E=%s/%s, H=%s/%s, total=%s/%s => %s",
                periodic_isoresolution_case_key(result.case),
                formatted_rate(result.rates_time_linf_space_l2[6]),
                formatted_rate(result.case.order + 1.0),
                formatted_rate(result.rates_time_linf_space_l2[8]),
                formatted_rate(Float64(result.case.order)),
                formatted_rate(result.rates_time_linf_space_l2[1]),
                formatted_rate(result.case.order + 1.0),
                formatted_rate(result.rates_time_linf_space_l2[2]),
                formatted_rate(Float64(result.case.order)),
                formatted_rate(result.rates_time_linf_space_l2[3]),
                formatted_rate(Float64(result.case.order)),
                verdict === true ? "PASS" : "FAIL",
            ),
        )
    end
    return pass, messages
end

function csv_rate(rate::Union{Missing, Float64})
    return ismissing(rate) ? "" : string(rate)
end

function periodic_isoresolution_csv_header()
    final_columns =
        Tuple("final_l2_space_$(name)_error" for name in ISO_ERROR_NAMES)
    time_columns =
        Tuple("linf_time_l2_space_$(name)_error" for name in ISO_ERROR_NAMES)
    rate_columns =
        Tuple("rate_linf_time_l2_space_$(name)" for name in ISO_ERROR_NAMES)
    return join(
        (
            "case",
            "order",
            "level",
            "intervals_per_wavelength",
            "nx_target",
            "active_ranks",
            "nelements",
            "min_owned_elements",
            "max_owned_elements",
            "nodes_per_element",
            "work_units",
            "cubature_order",
            "h_target",
            "h_characteristic",
            "h_min",
            "h_max",
            "h_ratio",
            "volume_total",
            "volume_min",
            "volume_max",
            "volume_ratio",
            "mean_ratio_min",
            "mean_ratio_avg",
            "dt",
            "nsteps",
            "elapsed_seconds",
            final_columns...,
            time_columns...,
            rate_columns...,
            "initial_energy",
            "final_energy",
            "relative_energy_error",
            "physical_rate_verdict",
        ),
        ',',
    )
end

function write_periodic_isoresolution_csv(
    path::String,
    results,
    config::PeriodicIsoresolutionConfig,
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, periodic_isoresolution_csv_header())
        for result in sort(results; by = r -> (r.case.order, r.case.level))
            verdict = config.smoke ?
                      missing :
                      result_physical_verdict(result, config.rate_tolerance)
            values = (
                periodic_isoresolution_case_key(result.case),
                result.case.order,
                result.case.level,
                result.case.intervals_per_wavelength,
                result.case.nx_target,
                result.active_ranks,
                result.nelements,
                result.min_owned_elements,
                result.max_owned_elements,
                result.nodes_per_element,
                result.work_units,
                result.cubature_order,
                result.h_target,
                result.h_characteristic,
                result.h_min,
                result.h_max,
                result.h_ratio,
                result.volume_total,
                result.volume_min,
                result.volume_max,
                result.volume_ratio,
                result.mean_ratio_min,
                result.mean_ratio_avg,
                result.dt,
                result.nsteps,
                result.elapsed_seconds,
                result.final_space_l2_errors...,
                result.time_linf_space_l2_errors...,
                (csv_rate(rate) for rate in result.rates_time_linf_space_l2)...,
                result.initial_energy,
                result.final_energy,
                result.relative_energy_error,
                ismissing(verdict) ? "" : (verdict ? "PASS" : "FAIL"),
            )
            println(io, join(values, ','))
        end
    end
    return path
end

function latex_number(value::Float64)
    if value == 0.0
        return "0"
    end
    return @sprintf("%.3e", value)
end

function latex_rate(rate::Union{Missing, Float64})
    return ismissing(rate) ? "--" : @sprintf("%.2f", rate)
end

function write_periodic_isoresolution_latex(
    path::String,
    results,
    config::PeriodicIsoresolutionConfig,
)
    mkpath(dirname(path))
    sorted_results = sort(results; by = r -> (r.case.order, r.case.level))
    open(path, "w") do io
        println(io, "% Generated by examples/convergence_distributed_periodic_isoresolution.jl")
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Periodic Poisson-bracket isoresolution convergence. Errors are \$L_\\infty(0,T;L^2(\\Omega))\$.}")
        println(io, "\\begin{tabular}{rrrrrrrrrrrr}")
        println(io, "\\hline")
        println(io, "\$P\$ & level & \$q\$ & ranks & \$N_e\$ & \$h\$ & \$E_z\$ & rate & \$H_y\$ & rate & \$E\$ rate & \$H\$ rate \\\\")
        println(io, "\\hline")
        for result in sorted_results
            println(
                io,
                @sprintf(
                    "%d & %d & %d & %d & %d & %.3e & %s & %s & %s & %s & %s & %s \\\\",
                    result.case.order,
                    result.case.level,
                    result.case.intervals_per_wavelength,
                    result.active_ranks,
                    result.nelements,
                    result.h_target,
                    latex_number(result.time_linf_space_l2_errors[6]),
                    latex_rate(result.rates_time_linf_space_l2[6]),
                    latex_number(result.time_linf_space_l2_errors[8]),
                    latex_rate(result.rates_time_linf_space_l2[8]),
                    latex_rate(result.rates_time_linf_space_l2[1]),
                    latex_rate(result.rates_time_linf_space_l2[2]),
                ),
            )
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
        println(io)

        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Transverse component leakage for the periodic plane wave.}")
        println(io, "\\begin{tabular}{rrrrrrrr}")
        println(io, "\\hline")
        println(io, "\$P\$ & level & \$q\$ & \$E_x\$ & \$E_y\$ & \$H_x\$ & \$H_z\$ & rel. energy \\\\")
        println(io, "\\hline")
        for result in sorted_results
            println(
                io,
                @sprintf(
                    "%d & %d & %d & %s & %s & %s & %s & %.3e \\\\",
                    result.case.order,
                    result.case.level,
                    result.case.intervals_per_wavelength,
                    latex_number(result.time_linf_space_l2_errors[4]),
                    latex_number(result.time_linf_space_l2_errors[5]),
                    latex_number(result.time_linf_space_l2_errors[7]),
                    latex_number(result.time_linf_space_l2_errors[9]),
                    result.relative_energy_error,
                ),
            )
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
    return path
end

function print_periodic_isoresolution_plan(
    cases,
    nelements_by_q,
    config::PeriodicIsoresolutionConfig,
    launcher_ranks::Int,
)
    println("Planned periodic isoresolution cases")
    println("------------------------------------")
    println(
        rpad("case", 8),
        rpad("P", 4),
        rpad("level", 8),
        rpad("q", 8),
        rpad("Ne", 10),
        rpad("Np", 8),
        rpad("work", 14),
        "ranks",
    )
    for case in cases
        nelements = nelements_by_q[case.intervals_per_wavelength]
        ranks = periodic_isoresolution_rank_count(
            config,
            nelements,
            case.order,
            launcher_ranks,
        )
        println(
            rpad(periodic_isoresolution_case_key(case), 8),
            rpad(string(case.order), 4),
            rpad(string(case.level), 8),
            rpad(string(case.intervals_per_wavelength), 8),
            rpad(string(nelements), 10),
            rpad(string(num_tet_nodes(case.order)), 8),
            rpad(@sprintf("%.3e", isoresolution_work_units(nelements, case.order)), 14),
            ranks,
        )
    end
end

function print_periodic_isoresolution_results(
    results,
    config::PeriodicIsoresolutionConfig,
)
    println()
    println("Periodic Poisson-bracket isoresolution convergence")
    println("--------------------------------------------------")
    println(
        rpad("case", 8),
        rpad("P", 4),
        rpad("q", 8),
        rpad("ranks", 8),
        rpad("Ne", 10),
        rpad("h", 12),
        rpad("dt", 12),
        rpad("Ez", 13),
        rpad("r Ez", 8),
        rpad("Hy", 13),
        rpad("r Hy", 8),
        rpad("E", 13),
        rpad("r E", 8),
        rpad("H", 13),
        "r H",
    )
    for result in sort(results; by = r -> (r.case.order, r.case.level))
        println(
            rpad(periodic_isoresolution_case_key(result.case), 8),
            rpad(string(result.case.order), 4),
            rpad(string(result.case.intervals_per_wavelength), 8),
            rpad(string(result.active_ranks), 8),
            rpad(string(result.nelements), 10),
            rpad(@sprintf("%.3e", result.h_target), 12),
            rpad(@sprintf("%.3e", result.dt), 12),
            rpad(@sprintf("%.3e", result.time_linf_space_l2_errors[6]), 13),
            rpad(formatted_rate(result.rates_time_linf_space_l2[6]), 8),
            rpad(@sprintf("%.3e", result.time_linf_space_l2_errors[8]), 13),
            rpad(formatted_rate(result.rates_time_linf_space_l2[8]), 8),
            rpad(@sprintf("%.3e", result.time_linf_space_l2_errors[1]), 13),
            rpad(formatted_rate(result.rates_time_linf_space_l2[1]), 8),
            rpad(@sprintf("%.3e", result.time_linf_space_l2_errors[2]), 13),
            formatted_rate(result.rates_time_linf_space_l2[2]),
        )
    end
end

function root_prepare_mesh_counts(
    config::PeriodicIsoresolutionConfig,
    cases,
    rank::Int,
    comm::MPI.Comm,
)
    resolutions = periodic_isoresolution_unique_resolutions(cases)
    counts = zeros(Int, length(resolutions))
    paths = Dict{Int, String}()

    if rank == 0
        for (index, q) in enumerate(resolutions)
            if config.dry_run
                counts[index] = estimate_periodic_isoresolution_elements(q)
                paths[q] = periodic_isoresolution_mesh_path(config, q)
            else
                mesh_path = generate_periodic_isoresolution_mesh!(config, q)
                mesh = load_periodic_mesh(mesh_path)
                counts[index] = size(mesh.tets, 2)
                paths[q] = mesh_path
            end
        end
    end

    counts = MPI.bcast(counts, comm; root = 0)
    if rank != 0
        for q in resolutions
            paths[q] = periodic_isoresolution_mesh_path(config, q)
        end
    end
    nelements_by_q = Dict(q => counts[index] for (index, q) in enumerate(resolutions))
    return nelements_by_q, paths
end

function run_periodic_isoresolution_study(
    config::PeriodicIsoresolutionConfig,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    launcher_ranks = MPI.Comm_size(comm)
    cases = periodic_isoresolution_cases(config)
    nelements_by_q, mesh_paths =
        root_prepare_mesh_counts(config, cases, rank, comm)

    if rank == 0
        println("Distributed periodic Poisson-bracket isoresolution study")
        println("---------------------------------------------------------")
        println("launcher MPI ranks:      ", launcher_ranks)
        println("orders:                  ", config.orders)
        println("levels:                  ", config.levels)
        println("final time:              ", config.final_time)
        println("CFL:                     ", config.cfl)
        println("CFL divisor:             ", config.cfl_divisor)
        println("effective CFL:           ", config.cfl / config.cfl_divisor)
        println("Gmsh geometry:           ", config.geo_path)
        println("mesh directory:          ", config.mesh_dir)
        println("target work per rank:    ", config.target_work_per_rank)
        println("min elements per rank:   ", config.min_elements_per_rank)
        println("max active ranks:        ", config.max_active_ranks == 0 ? launcher_ranks : config.max_active_ranks)
        println("primary norm:            Linf(time) L2(space)")
        println("expected rates:          E/Ez=P+1, H/Hy/total=P")
        println("CSV output:              ", config.output)
        println("LaTeX output:            ", config.latex_output)
        println()
        print_periodic_isoresolution_plan(
            cases,
            nelements_by_q,
            config,
            launcher_ranks,
        )
    end

    config.dry_run && return PeriodicIsoresolutionResult[]

    results = PeriodicIsoresolutionResult[]
    for case in cases
        q = case.intervals_per_wavelength
        nelements = nelements_by_q[q]
        active_ranks = periodic_isoresolution_rank_count(
            config,
            nelements,
            case.order,
            launcher_ranks,
        )
        rank == 0 && @printf(
            "\nRunning %s: P=%d, q=%d, Ne=%d, active ranks=%d\n",
            periodic_isoresolution_case_key(case),
            case.order,
            q,
            nelements,
            active_ranks,
        )

        case_comm = periodic_isoresolution_case_comm(comm, active_ranks)
        try
            if case_comm != MPI.COMM_NULL
                case_rank = MPI.Comm_rank(case_comm)
                root_mesh =
                    case_rank == 0 ? load_periodic_mesh(mesh_paths[q]) : nothing
                result = run_periodic_isoresolution_case(
                    case,
                    config,
                    case_comm,
                    root_mesh,
                    active_ranks,
                )
                if rank == 0
                    push!(results, result)
                    @printf(
                        "Completed %s: dt=%.3e, steps=%d, Ez=%.3e, Hy=%.3e, elapsed=%.3f s\n",
                        periodic_isoresolution_case_key(case),
                        result.dt,
                        result.nsteps,
                        result.time_linf_space_l2_errors[6],
                        result.time_linf_space_l2_errors[8],
                        result.elapsed_seconds,
                    )
                end
            end
        finally
            case_comm != MPI.COMM_NULL && MPI.free(case_comm)
        end
        MPI.Barrier(comm)
    end

    if rank == 0
        add_periodic_isoresolution_rates!(results)
        print_periodic_isoresolution_results(results, config)
        write_periodic_isoresolution_csv(config.output, results, config)
        write_periodic_isoresolution_latex(config.latex_output, results, config)
        println()
        if config.smoke
            println("Smoke mode completed; strict convergence verdict skipped.")
        else
            pass, messages =
                periodic_isoresolution_verdict(results, config.rate_tolerance)
            println("Strict physical-component convergence checks")
            println("--------------------------------------------")
            foreach(println, messages)
            println("Overall verdict: ", pass ? "PASS" : "FAIL")
        end
        println("Wrote CSV:   ", config.output)
        println("Wrote LaTeX: ", config.latex_output)
    end

    return results
end

function main_periodic_isoresolution(args::Vector{String})
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    try
        config = parse_periodic_isoresolution_arguments(args)
        if config === nothing
            rank == 0 && print_periodic_isoresolution_usage()
            return nothing
        end
        run_periodic_isoresolution_study(config, comm)
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
    main_periodic_isoresolution(ARGS)
end
