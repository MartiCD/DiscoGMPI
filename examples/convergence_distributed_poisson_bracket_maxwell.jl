#!/usr/bin/env julia

# MPI convergence study for the distributed Poisson-bracket Maxwell cavity.
#
# From the DiscoGMPI repository root:
#   mpiexec -n 2 julia --project=. \
#     examples/convergence_distributed_poisson_bracket_maxwell.jl

include(joinpath(@__DIR__, "distributed_poisson_bracket_maxwell.jl"))

using Printf
using Random

const REFERENCE_TET_VOLUME = 4.0 / 3.0

struct DistributedConvergenceConfig
    cells_per_axis::Vector{Int}
    orders::Vector{Int}
    final_time::Float64
    cfl::Float64
    jitter::Float64
    seed::Int
    epsilon::Float64
    mu::Float64
    output::String
end

struct DistributedConvergenceResult
    mpi_ranks::Int
    order::Int
    esprk_order::Int
    cubature_order::Int
    mesh_level::Int
    cells_per_axis::Int
    nelements::Int
    min_owned_elements::Int
    max_owned_elements::Int
    characteristic_h::Float64
    dt::Float64
    nsteps::Int
    elapsed_seconds::Float64
    l2_electric_error::Float64
    l2_magnetic_error::Float64
    l2_total_error::Float64
    relative_total_error::Float64
    linf_electric_error::Float64
    linf_magnetic_error::Float64
    linf_total_error::Float64
    energy_error::Float64
    relative_energy_error::Float64
    electric_charge::Float64
    magnetic_charge::Float64
    rate_electric::Union{Missing, Float64}
    rate_magnetic::Union{Missing, Float64}
    rate_total::Union{Missing, Float64}
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
Distributed Poisson-bracket Maxwell convergence study

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/convergence_distributed_poisson_bracket_maxwell.jl [options]

Options:
  --cells=a,b,c,d     Unit-cube cells per axis. Default: 2,3,4,5
  --orders=a,b,c      DG polynomial orders. Default: 1,2,3
  --final-time=T      Final physical time. Default: 0.25
  --time=T            Alias for --final-time.
  --cfl=C             Maxwell CFL factor. Default: 0.05
  --jitter=J          Interior-node jitter fraction. Default: 0.08
  --seed=N            Deterministic jitter seed. Default: 1234
  --epsilon=X         Electric permittivity. Default: 1.0
  --mu=X              Magnetic permeability. Default: 1.0
  --output=PATH       Output CSV.
                      Default: output/convergence_distributed_poisson_bracket.csv
  --help              Show this message.

Method:
  - unit-cube PEC eigenmode used by distributed_poisson_bracket_maxwell.jl
  - PoissonBracketFormulation with centered flux
  - H-first ESPRK with time order = DG order + 1
  - Jaskowiec-Sukumar cubature order max(2, 2N + 4)
  - continuous L2 and quadrature-point Linf errors
""")
end

function parse_convergence_arguments(args::Vector{String})
    cells_per_axis = [2, 3, 4, 5]
    orders = [1, 2, 3]
    final_time = 0.25
    cfl = 0.05
    jitter = 0.08
    seed = 1234
    epsilon = 1.0
    mu = 1.0
    output =
        joinpath("output", "convergence_distributed_poisson_bracket.csv")

    for arg in args
        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--cells=")
            cells_per_axis =
                parse_integer_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--orders=")
            orders = parse_integer_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--final-time=") || startswith(arg, "--time=")
            final_time = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--jitter=")
            jitter = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--epsilon=") || startswith(arg, "--eps=")
            epsilon = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
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

    length(cells_per_axis) >= 2 ||
        throw(ArgumentError("At least two mesh levels are required."))
    issorted(cells_per_axis) ||
        throw(ArgumentError("--cells must be ordered from coarse to fine."))
    length(unique(cells_per_axis)) == length(cells_per_axis) ||
        throw(ArgumentError("--cells entries must be distinct."))
    all(>=(1), cells_per_axis) ||
        throw(ArgumentError("--cells entries must be positive."))
    all(order -> 1 <= order <= 5, orders) ||
        throw(
            ArgumentError(
                "--orders must be in 1:5 because ESPRK order is N+1.",
            ),
        )
    final_time > 0.0 ||
        throw(ArgumentError("--final-time must be positive."))
    cfl > 0.0 ||
        throw(ArgumentError("--cfl must be positive."))
    0.0 <= jitter < 0.25 ||
        throw(ArgumentError("--jitter must lie in [0, 0.25)."))
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))

    return DistributedConvergenceConfig(
        cells_per_axis,
        orders,
        final_time,
        cfl,
        jitter,
        seed,
        epsilon,
        mu,
        abspath(output),
    )
end

function convergence_node_id(
    i::Int,
    j::Int,
    k::Int,
    cells_per_axis::Int,
)
    nodes_per_axis = cells_per_axis + 1
    return 1 + i + nodes_per_axis * (j + nodes_per_axis * k)
end

function build_convergence_points(
    cells_per_axis::Int;
    jitter::Float64,
    seed::Int,
)
    nodes_per_axis = cells_per_axis + 1
    coordinates = collect(range(0.0, 1.0; length = nodes_per_axis))
    mesh_spacing = 1.0 / cells_per_axis
    random = MersenneTwister(seed + cells_per_axis)
    points = zeros(Float64, 3, nodes_per_axis^3)

    for k in 0:cells_per_axis
        for j in 0:cells_per_axis
            for i in 0:cells_per_axis
                node = convergence_node_id(i, j, k, cells_per_axis)
                x = coordinates[i + 1]
                y = coordinates[j + 1]
                z = coordinates[k + 1]

                if 0 < i < cells_per_axis &&
                   0 < j < cells_per_axis &&
                   0 < k < cells_per_axis
                    scale = jitter * mesh_spacing
                    x += scale * (2.0 * rand(random) - 1.0)
                    y += scale * (2.0 * rand(random) - 1.0)
                    z += scale * (2.0 * rand(random) - 1.0)
                end

                points[:, node] .= (x, y, z)
            end
        end
    end

    return points
end

function build_convergence_tets(cells_per_axis::Int)
    tetrahedra = NTuple{4, Int}[]

    for k in 0:(cells_per_axis - 1)
        for j in 0:(cells_per_axis - 1)
            for i in 0:(cells_per_axis - 1)
                v000 = convergence_node_id(i, j, k, cells_per_axis)
                v100 = convergence_node_id(i + 1, j, k, cells_per_axis)
                v010 = convergence_node_id(i, j + 1, k, cells_per_axis)
                v110 =
                    convergence_node_id(i + 1, j + 1, k, cells_per_axis)
                v001 = convergence_node_id(i, j, k + 1, cells_per_axis)
                v101 =
                    convergence_node_id(i + 1, j, k + 1, cells_per_axis)
                v011 =
                    convergence_node_id(i, j + 1, k + 1, cells_per_axis)
                v111 =
                    convergence_node_id(
                        i + 1,
                        j + 1,
                        k + 1,
                        cells_per_axis,
                    )

                push!(tetrahedra, (v000, v100, v110, v111))
                push!(tetrahedra, (v000, v110, v010, v111))
                push!(tetrahedra, (v000, v010, v011, v111))
                push!(tetrahedra, (v000, v011, v001, v111))
                push!(tetrahedra, (v000, v001, v101, v111))
                push!(tetrahedra, (v000, v101, v100, v111))
            end
        end
    end

    return reduce(hcat, collect.(tetrahedra))
end

function build_convergence_mesh(
    cells_per_axis::Int;
    jitter::Float64,
    seed::Int,
)
    points = build_convergence_points(
        cells_per_axis;
        jitter = jitter,
        seed = seed,
    )
    tets = build_convergence_tets(cells_per_axis)
    tris = build_boundary_tris(tets)
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))
    boundary_ids = zeros(Int, ntets + ntris)
    boundary_ids[tri_cell_ids] .= PEC_BOUNDARY_ID

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_ids),
    )
    check_mesh_consistency(mesh)
    return mesh
end

function balanced_spatial_partition(mesh::RawVTUMesh, nranks::Int)
    nelements = size(mesh.tets, 2)
    nelements >= nranks ||
        throw(
            ArgumentError(
                "Mesh has $nelements tetrahedra for $nranks MPI ranks.",
            ),
        )

    centroids = Vector{NTuple{3, Float64}}(undef, nelements)
    for elem in 1:nelements
        nodes = @view mesh.tets[:, elem]
        centroids[elem] = (
            sum(@view mesh.points[1, nodes]) / 4.0,
            sum(@view mesh.points[2, nodes]) / 4.0,
            sum(@view mesh.points[3, nodes]) / 4.0,
        )
    end

    order = sortperm(1:nelements; by = elem -> centroids[elem])
    partition = zeros(Int, nelements)
    for (position, elem) in enumerate(order)
        partition[elem] =
            min(div((position - 1) * nranks, nelements), nranks - 1)
    end
    return partition
end

function distributed_characteristic_h(
    distributed_dg::DistributedDGDiscretization,
)
    local_volume = 0.0
    for elem in distributed_dg.distributed_mesh.partition.owned
        local_volume +=
            REFERENCE_TET_VOLUME *
            distributed_dg.dg.mappings.tet_mappings[elem].absdetJ
    end
    global_volume = MPI.Allreduce(local_volume, +, distributed_dg.comm)
    nelements = MPI.Allreduce(
        length(distributed_dg.distributed_mesh.partition.owned),
        +,
        distributed_dg.comm,
    )
    return (global_volume / nelements)^(1.0 / 3.0)
end

function distributed_linf_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
)
    cubature_points, _, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
    )
    mesh = distributed_dg.dg.mesh
    local_maxima = zeros(Float64, 3)

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view mesh.tets[:, elem]

        @views begin
            Ex = U.Ex[:, elem]
            Ey = U.Ey[:, elem]
            Ez = U.Ez[:, elem]
            Hx = U.Hx[:, elem]
            Hy = U.Hy[:, elem]
            Hz = U.Hz[:, elem]

            for q in 1:number_cubature_points
                r = cubature_points[q, 1]
                s = cubature_points[q, 2]
                t = cubature_points[q, 3]
                x, y, z = DiscoGMPI.map_to_physical(
                    mesh.points,
                    tet_nodes,
                    r,
                    s,
                    t,
                )
                exact_Ex, exact_Ey, exact_Ez = exact_electric(x, y, z)
                exact_Hx, exact_Hy, exact_Hz = exact_magnetic(x, y, z)
                row = view(interpolation, q, :)
                electric_error_squared =
                    (dot(row, Ex) - exact_Ex)^2 +
                    (dot(row, Ey) - exact_Ey)^2 +
                    (dot(row, Ez) - exact_Ez)^2
                magnetic_error_squared =
                    (dot(row, Hx) - exact_Hx)^2 +
                    (dot(row, Hy) - exact_Hy)^2 +
                    (dot(row, Hz) - exact_Hz)^2
                local_maxima[1] =
                    max(local_maxima[1], sqrt(electric_error_squared))
                local_maxima[2] =
                    max(local_maxima[2], sqrt(magnetic_error_squared))
                local_maxima[3] =
                    max(
                        local_maxima[3],
                        sqrt(electric_error_squared + magnetic_error_squared),
                    )
            end
        end
    end

    maxima = MPI.Allreduce(local_maxima, max, distributed_dg.comm)
    return maxima[1], maxima[2], maxima[3]
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
)
    workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    formulation = PoissonBracketFormulation()

    for _ in 1:nsteps
        distributed_partitioned_symplectic_rk_step!(
            U,
            workspace,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation;
            ε = epsilon,
            μ = mu,
        )
    end
    return U
end

function run_distributed_convergence_case(
    cells_per_axis::Int,
    polynomial_order::Int,
    mesh_level::Int,
    config::DistributedConvergenceConfig,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_mesh = rank == 0 ? build_convergence_mesh(
        cells_per_axis;
        jitter = config.jitter,
        seed = config.seed,
    ) : nothing
    root_partition =
        rank == 0 ? balanced_spatial_partition(root_mesh, nranks) : nothing
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
    )
    owned_count =
        length(distributed_dg.distributed_mesh.partition.owned)
    min_owned = MPI.Allreduce(owned_count, min, comm)
    max_owned = MPI.Allreduce(owned_count, max, comm)

    electric, magnetic = exact_cavity_mode_functions(
        0.0;
        epsilon = config.epsilon,
        mu = config.mu,
    )
    U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    registry = MaxwellBoundaryRegistry(
        Dict(PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )

    local_dt, _ = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = config.cfl,
        ε = config.epsilon,
        μ = config.mu,
    )
    estimated_dt = MPI.Allreduce(local_dt, min, comm)
    nsteps = max(1, ceil(Int, config.final_time / estimated_dt))
    dt = config.final_time / nsteps
    esprk_order = polynomial_order + 1
    scheme = explicit_partitioned_symplectic_rk_scheme(
        esprk_order;
        first_partition = :H,
    )
    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )

    MPI.Barrier(comm)
    local_elapsed = @elapsed advance_distributed_convergence_case!(
        U,
        distributed_dg,
        registry,
        scheme,
        dt,
        nsteps;
        epsilon = config.epsilon,
        mu = config.mu,
    )
    elapsed = MPI.Allreduce(local_elapsed, max, comm)

    cubature_order = max(2, 2 * polynomial_order + 4)
    diagnostics = distributed_quadrature_diagnostics(
        U,
        distributed_dg,
        config.final_time,
        cubature_order;
        epsilon = config.epsilon,
        mu = config.mu,
    )
    linf_electric, linf_magnetic, linf_total =
        distributed_linf_errors(
            U,
            distributed_dg,
            config.final_time,
            cubature_order;
            epsilon = config.epsilon,
            mu = config.mu,
        )
    final_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    relative_energy_error =
        (final_energy.total - initial_energy.total) /
        max(initial_energy.total, eps(Float64))

    result = DistributedConvergenceResult(
        nranks,
        polynomial_order,
        esprk_order,
        cubature_order,
        mesh_level,
        cells_per_axis,
        nelements,
        min_owned,
        max_owned,
        distributed_characteristic_h(distributed_dg),
        dt,
        nsteps,
        elapsed,
        diagnostics.electric_error_l2,
        diagnostics.magnetic_error_l2,
        diagnostics.field_error_l2,
        diagnostics.field_relative_error,
        linf_electric,
        linf_magnetic,
        linf_total,
        diagnostics.total_energy - diagnostics.exact_total_energy,
        relative_energy_error,
        diagnostics.electric_charge,
        diagnostics.magnetic_charge,
        missing,
        missing,
        missing,
    )

    MPI.Barrier(comm)
    return result
end

function convergence_rate(
    fine_error::Float64,
    coarse_error::Float64,
    fine_h::Float64,
    coarse_h::Float64,
)
    return log(coarse_error / fine_error) / log(coarse_h / fine_h)
end

function add_convergence_rates(
    results::Vector{DistributedConvergenceResult},
)
    rated = DistributedConvergenceResult[]

    for order in sort(unique(result.order for result in results))
        subset = sort(
            filter(result -> result.order == order, results);
            by = result -> result.mesh_level,
        )
        previous = nothing

        for result in subset
            rate_electric = missing
            rate_magnetic = missing
            rate_total = missing
            if previous !== nothing
                rate_electric = convergence_rate(
                    result.l2_electric_error,
                    previous.l2_electric_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_magnetic = convergence_rate(
                    result.l2_magnetic_error,
                    previous.l2_magnetic_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_total = convergence_rate(
                    result.l2_total_error,
                    previous.l2_total_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
            end

            push!(
                rated,
                DistributedConvergenceResult(
                    result.mpi_ranks,
                    result.order,
                    result.esprk_order,
                    result.cubature_order,
                    result.mesh_level,
                    result.cells_per_axis,
                    result.nelements,
                    result.min_owned_elements,
                    result.max_owned_elements,
                    result.characteristic_h,
                    result.dt,
                    result.nsteps,
                    result.elapsed_seconds,
                    result.l2_electric_error,
                    result.l2_magnetic_error,
                    result.l2_total_error,
                    result.relative_total_error,
                    result.linf_electric_error,
                    result.linf_magnetic_error,
                    result.linf_total_error,
                    result.energy_error,
                    result.relative_energy_error,
                    result.electric_charge,
                    result.magnetic_charge,
                    rate_electric,
                    rate_magnetic,
                    rate_total,
                ),
            )
            previous = result
        end
    end
    return rated
end

function formatted_rate(rate::Union{Missing, Float64})
    return ismissing(rate) ? "-" : @sprintf("%.3f", rate)
end

function print_convergence_results(
    results::Vector{DistributedConvergenceResult},
)
    println()
    println("Distributed Poisson-bracket Maxwell convergence")
    println("------------------------------------------------")
    println(
        rpad("N", 4),
        rpad("ESPRK", 7),
        rpad("cells", 7),
        rpad("Ne", 9),
        rpad("owned", 11),
        rpad("h", 12),
        rpad("dt", 12),
        rpad("steps", 7),
        rpad("L2 E", 13),
        rpad("L2 H", 13),
        rpad("L2 total", 13),
        rpad("rate E", 9),
        rpad("rate H", 9),
        rpad("rate", 9),
        "seconds",
    )

    for result in results
        owned =
            "$(result.min_owned_elements):$(result.max_owned_elements)"
        println(
            rpad(string(result.order), 4),
            rpad(string(result.esprk_order), 7),
            rpad(string(result.cells_per_axis), 7),
            rpad(string(result.nelements), 9),
            rpad(owned, 11),
            rpad(@sprintf("%.3e", result.characteristic_h), 12),
            rpad(@sprintf("%.3e", result.dt), 12),
            rpad(string(result.nsteps), 7),
            rpad(@sprintf("%.3e", result.l2_electric_error), 13),
            rpad(@sprintf("%.3e", result.l2_magnetic_error), 13),
            rpad(@sprintf("%.3e", result.l2_total_error), 13),
            rpad(formatted_rate(result.rate_electric), 9),
            rpad(formatted_rate(result.rate_magnetic), 9),
            rpad(formatted_rate(result.rate_total), 9),
            @sprintf("%.3f", result.elapsed_seconds),
        )
    end
end

function convergence_csv_header()
    return (
        "mpi_ranks,order,esprk_order,cubature_order,mesh_level," *
        "cells_per_axis,nelements,min_owned_elements,max_owned_elements," *
        "characteristic_h,dt,nsteps,elapsed_seconds,l2_electric_error," *
        "l2_magnetic_error,l2_total_error,relative_total_error," *
        "linf_electric_error,linf_magnetic_error,linf_total_error," *
        "energy_error,relative_energy_error,electric_charge," *
        "magnetic_charge,rate_electric,rate_magnetic,rate_total"
    )
end

function write_convergence_results(
    path::String,
    results::Vector{DistributedConvergenceResult},
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, convergence_csv_header())
        for result in results
            values = (
                result.mpi_ranks,
                result.order,
                result.esprk_order,
                result.cubature_order,
                result.mesh_level,
                result.cells_per_axis,
                result.nelements,
                result.min_owned_elements,
                result.max_owned_elements,
                result.characteristic_h,
                result.dt,
                result.nsteps,
                result.elapsed_seconds,
                result.l2_electric_error,
                result.l2_magnetic_error,
                result.l2_total_error,
                result.relative_total_error,
                result.linf_electric_error,
                result.linf_magnetic_error,
                result.linf_total_error,
                result.energy_error,
                result.relative_energy_error,
                result.electric_charge,
                result.magnetic_charge,
                ismissing(result.rate_electric) ? "" : result.rate_electric,
                ismissing(result.rate_magnetic) ? "" : result.rate_magnetic,
                ismissing(result.rate_total) ? "" : result.rate_total,
            )
            println(io, join(values, ','))
        end
    end
    return path
end

function convergence_verdict(
    results::Vector{DistributedConvergenceResult},
)
    verdict = true
    messages = String[]

    for order in sort(unique(result.order for result in results))
        subset = sort(
            filter(result -> result.order == order, results);
            by = result -> result.mesh_level,
        )
        errors = [result.l2_total_error for result in subset]
        monotone = all(
            errors[index] < errors[index - 1]
            for index in 2:length(errors)
        )
        verdict &= monotone
        final_rate = subset[end].rate_total
        push!(
            messages,
            "N=$order: monotone=$monotone, final rate=" *
            formatted_rate(final_rate),
        )
    end
    return verdict, messages
end

function run_convergence_study(
    config::DistributedConvergenceConfig,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    if rank == 0
        println("Distributed Poisson-bracket Maxwell convergence study")
        println("-----------------------------------------------------")
        println("MPI ranks:          ", nranks)
        println("DG orders:          ", config.orders)
        println("mesh levels:        ", config.cells_per_axis)
        println("final time:         ", config.final_time)
        println("CFL:                ", config.cfl)
        println("interior jitter:    ", config.jitter)
        println("ESPRK rule:         order N+1, H-first")
        println("cubature rule:      Jaskowiec-Sukumar max(2,2N+4)")
        println("analytical mode:    unit-cube PEC eigenmode")
        println("output:             ", config.output)
        println()
    end

    results = DistributedConvergenceResult[]
    for order in config.orders
        for (level, cells_per_axis) in enumerate(config.cells_per_axis)
            if rank == 0
                @printf(
                    "Running N=%d, ESPRK=%d, level=%d, cells=%d\n",
                    order,
                    order + 1,
                    level,
                    cells_per_axis,
                )
            end
            push!(
                results,
                run_distributed_convergence_case(
                    cells_per_axis,
                    order,
                    level,
                    config,
                    comm,
                ),
            )
        end
    end

    rated_results = add_convergence_rates(results)
    if rank == 0
        print_convergence_results(rated_results)
        write_convergence_results(config.output, rated_results)
        verdict, messages = convergence_verdict(rated_results)
        println()
        println("Convergence checks")
        println("------------------")
        foreach(println, messages)
        println("Overall monotonic convergence: ", verdict ? "PASS" : "FAIL")
        println("Wrote CSV: ", config.output)
    end
    return rated_results
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
