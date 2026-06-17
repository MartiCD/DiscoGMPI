#!/usr/bin/env julia

using Dates
using MPI
using Printf
using DiscoGMPI

const BENCH_PEC_BOUNDARY_ID = 10
const BENCH_TET_FACES = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)

const CSV_COLUMNS = (
    "timestamp",
    "mode",
    "setup",
    "mpi_ranks",
    "cells_per_axis",
    "global_elements",
    "order",
    "nodes_per_element",
    "iterations",
    "warmup",
    "setup_seconds_max",
    "rhs_total_seconds_max",
    "halo_seconds_max",
    "rhs_assembly_seconds_max",
    "ghost_zero_seconds_max",
    "rhs_total_seconds_mean",
    "halo_seconds_mean",
    "rhs_assembly_seconds_mean",
    "owned_elements_min",
    "owned_elements_max",
    "ghost_elements_max",
    "interface_faces_total",
    "interface_faces_max",
    "send_values_max",
    "rhs_calls_per_second",
    "owned_element_rhs_per_second",
)

const CSV_HEADER = join(CSV_COLUMNS, ',')

struct ScalingConfig
    mode::Symbol
    setup::Symbol
    cells::Int
    order::Int
    iterations::Int
    warmup::Int
    epsilon::Float64
    mu::Float64
    cache_dir::String
    rebuild_cache::Bool
    output_dir::String
    csv_path::String
end

function usage(io::IO = stdout)
    println(io, """
Distributed Maxwell MPI scaling benchmark

Usage:
  mpiexec -n <ranks> julia --project=. \\
    benchmark/distributed/benchmark_mpi_scaling.jl [options]

Options:
  --mode=strong|weak     Strong keeps --cells fixed. Weak scales cells by
                         round(cells * ranks^(1/3)). Default: strong
  --setup=cache|root|collective
                         cache prepares/loads rank-local mesh shards.
                         root builds rank-local meshes on rank zero.
                         collective has every rank build from global mesh.
                         Default: cache
  --cells=N              Strong: cells per axis. Weak: cells per rank-scale
                         base. Default: 4
  --order=N              DG order. Default: 2
  --iterations=N         Timed RHS evaluations. Default: 20
  --warmup=N             Untimed RHS evaluations. Default: 3
  --epsilon=X            Electric permittivity. Default: 1.0
  --mu=X                 Magnetic permeability. Default: 1.0
  --cache-dir=PATH       Rank-local mesh cache root. Default:
                         benchmark/distributed/mesh_cache
  --rebuild-cache        Recreate this run's mesh cache.
  --output-dir=PATH      Metadata output root. Default:
                         benchmark/distributed/output
  --csv=PATH             Append CSV_RESULT rows to this CSV on rank zero.
  --csv-header           Print the CSV header and exit.
  --help, -h             Print this message and exit.
""")
end

function parse_symbol_option(value::AbstractString, allowed)
    symbol = Symbol(lowercase(value))
    symbol in allowed ||
        throw(ArgumentError("Expected one of $(join(allowed, ", ")), got $value."))
    return symbol
end

function parse_args(args::Vector{String})
    mode = :strong
    setup = :cache
    cells = 4
    order = 2
    iterations = 20
    warmup = 3
    epsilon = 1.0
    mu = 1.0
    cache_dir = joinpath("benchmark", "distributed", "mesh_cache")
    rebuild_cache = false
    output_dir = joinpath("benchmark", "distributed", "output")
    csv_path = ""

    for arg in args
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--csv-header"
            println(CSV_HEADER)
            exit(0)
        elseif startswith(arg, "--mode=")
            mode = parse_symbol_option(
                split(arg, "=", limit = 2)[2],
                (:strong, :weak),
            )
        elseif startswith(arg, "--setup=")
            setup = parse_symbol_option(
                split(arg, "=", limit = 2)[2],
                (:cache, :root, :collective),
            )
        elseif startswith(arg, "--cells=")
            cells = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--order=")
            order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--iterations=")
            iterations = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--warmup=")
            warmup = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--epsilon=") || startswith(arg, "--eps=")
            epsilon = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cache-dir=")
            cache_dir = abspath(split(arg, "=", limit = 2)[2])
        elseif arg == "--rebuild-cache"
            rebuild_cache = true
        elseif startswith(arg, "--output-dir=")
            output_dir = abspath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--csv=")
            csv_path = abspath(split(arg, "=", limit = 2)[2])
        else
            throw(ArgumentError("Unknown argument: $arg. Use --help."))
        end
    end

    cells >= 1 || throw(ArgumentError("--cells must be positive."))
    order >= 1 || throw(ArgumentError("--order must be positive."))
    iterations >= 1 || throw(ArgumentError("--iterations must be positive."))
    warmup >= 0 || throw(ArgumentError("--warmup must be nonnegative."))
    epsilon > 0.0 || throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 || throw(ArgumentError("--mu must be positive."))

    return ScalingConfig(
        mode,
        setup,
        cells,
        order,
        iterations,
        warmup,
        epsilon,
        mu,
        cache_dir,
        rebuild_cache,
        output_dir,
        csv_path,
    )
end

function node_id(i::Int, j::Int, k::Int, cells::Int)
    return 1 + i + (cells + 1) * (j + (cells + 1) * k)
end

function structured_cube_points(cells::Int)
    points = zeros(Float64, 3, (cells + 1)^3)
    for k in 0:cells, j in 0:cells, i in 0:cells
        node = node_id(i, j, k, cells)
        points[:, node] .= (i / cells, j / cells, k / cells)
    end
    return points
end

function structured_cube_tets(cells::Int)
    tetrahedra = NTuple{4, Int}[]
    for k in 0:(cells - 1), j in 0:(cells - 1), i in 0:(cells - 1)
        v000 = node_id(i, j, k, cells)
        v100 = node_id(i + 1, j, k, cells)
        v010 = node_id(i, j + 1, k, cells)
        v110 = node_id(i + 1, j + 1, k, cells)
        v001 = node_id(i, j, k + 1, cells)
        v101 = node_id(i + 1, j, k + 1, cells)
        v011 = node_id(i, j + 1, k + 1, cells)
        v111 = node_id(i + 1, j + 1, k + 1, cells)
        append!(
            tetrahedra,
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            ),
        )
    end
    return reduce(hcat, collect.(tetrahedra))
end

function sorted_face_key(nodes::NTuple{3, Int})
    return Tuple(sort(collect(nodes)))
end

function boundary_tris(tets::AbstractMatrix{Int})
    counts = Dict{NTuple{3, Int}, Int}()
    oriented = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    for elem in axes(tets, 2), face in BENCH_TET_FACES
        nodes = (tets[face[1], elem], tets[face[2], elem], tets[face[3], elem])
        key = sorted_face_key(nodes)
        counts[key] = get(counts, key, 0) + 1
        oriented[key] = nodes
    end
    faces = [oriented[key] for (key, count) in counts if count == 1]
    return isempty(faces) ? Matrix{Int}(undef, 3, 0) : reduce(hcat, collect.(faces))
end

function structured_cube_mesh(cells::Int)
    points = structured_cube_points(cells)
    tets = structured_cube_tets(cells)
    tris = boundary_tris(tets)
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    boundary_id = zeros(Int, ntets + ntris)
    boundary_id[(ntets + 1):(ntets + ntris)] .= BENCH_PEC_BOUNDARY_ID
    return RawVTUMesh(
        points,
        tets,
        tris,
        collect(1:ntets),
        collect((ntets + 1):(ntets + ntris)),
        Dict{String, Any}("boundary_id" => boundary_id),
    )
end

function x_balanced_partition(mesh::RawVTUMesh, nranks::Int)
    nelements = size(mesh.tets, 2)
    bary_x = Vector{Float64}(undef, nelements)
    for elem in 1:nelements
        bary_x[elem] = sum(mesh.points[1, mesh.tets[:, elem]]) / 4
    end
    order = sortperm(1:nelements; by = elem -> (bary_x[elem], elem))
    partition = Vector{Int}(undef, nelements)
    for (position, elem) in enumerate(order)
        partition[elem] = min((position - 1) * nranks ÷ nelements, nranks - 1)
    end
    return partition
end

function run_cells(config::ScalingConfig, nranks::Int)
    if config.mode == :strong
        return config.cells
    end
    return max(1, round(Int, config.cells * cbrt(nranks)))
end

function case_label(config::ScalingConfig, nranks::Int, cells::Int)
    return "$(config.mode)_$(config.setup)_r$(nranks)_c$(cells)_N$(config.order)"
end

function cache_case_dir(config::ScalingConfig, nranks::Int, cells::Int)
    return joinpath(config.cache_dir, case_label(config, nranks, cells))
end

function output_case_dir(config::ScalingConfig, nranks::Int, cells::Int)
    return joinpath(config.output_dir, case_label(config, nranks, cells))
end

function root_mesh_and_partition(cells::Int, nranks::Int, rank::Int)
    if rank == 0
        mesh = structured_cube_mesh(cells)
        return mesh, x_balanced_partition(mesh, nranks)
    end
    return nothing, nothing
end

function build_benchmark_dg(
    config::ScalingConfig,
    cells::Int,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    cache_dir = cache_case_dir(config, nranks, cells)

    if config.setup == :cache
        manifest = joinpath(cache_dir, "mesh_manifest.toml")
        local_cache_ready = isfile(manifest) && !config.rebuild_cache
        cache_ready =
            MPI.Allreduce(local_cache_ready ? 1 : 0, min, comm) == 1
        if cache_ready
            return build_distributed_dg_from_partition(
                cache_dir,
                config.order;
                comm = comm,
            ), "rank-local cache"
        end

        mesh = structured_cube_mesh(cells)
        partition = x_balanced_partition(mesh, nranks)
        prepare_distributed_mesh_partition_collective(
            mesh,
            partition,
            cache_dir;
            comm = comm,
            metadata = Dict(
                "benchmark" => "mpi_scaling",
                "mode" => string(config.mode),
                "setup" => "collective cache preparation",
                "cells_per_axis" => cells,
                "partition" => "x-balanced",
            ),
        )
        return build_distributed_dg_from_partition(
            cache_dir,
            config.order;
            comm = comm,
        ), "prepared rank-local cache"
    elseif config.setup == :root
        mesh, partition = root_mesh_and_partition(cells, nranks, rank)
        return build_distributed_dg_from_root(
            mesh,
            partition,
            config.order;
            comm = comm,
        ), "root-distributed"
    else
        mesh = structured_cube_mesh(cells)
        partition = x_balanced_partition(mesh, nranks)
        return build_distributed_dg(
            mesh,
            partition,
            config.order;
            comm = comm,
        ), "collective-global-input"
    end
end

function initial_field(distributed_dg)
    return interpolate_maxwell_field(
        distributed_dg,
        (x, y, z) -> (
            sin(pi * x) * sin(pi * y) * sin(pi * z),
            0.25 * cos(pi * x) * sin(pi * y) * sin(pi * z),
            -0.5 * sin(pi * x) * cos(pi * y) * sin(pi * z),
        ),
        (x, y, z) -> (
            0.3 * cos(pi * x) * sin(pi * y) * cos(pi * z),
            -0.2 * sin(pi * x) * cos(pi * y) * cos(pi * z),
            0.4 * sin(pi * x) * sin(pi * y) * cos(pi * z),
        ),
    )
end

function local_partition_metrics(distributed_dg)
    owned = length(distributed_dg.distributed_mesh.partition.owned)
    ghosts = length(distributed_dg.distributed_mesh.partition.ghosts)
    interfaces = sum(
        length(distributed_dg.exchange.faces[neighbor])
        for neighbor in distributed_dg.exchange.neighbors;
        init = 0,
    )
    send_values = sum(
        length(distributed_dg.exchange.send_buffers[neighbor])
        for neighbor in distributed_dg.exchange.neighbors;
        init = 0,
    )
    return (owned, ghosts, interfaces, send_values)
end

function accumulate_profile!(
    sums::Vector{Float64},
    timing,
)
    sums[1] += timing.total_seconds
    sums[2] += timing.halo_exchange_seconds
    sums[3] += timing.rhs_assembly_seconds
    sums[4] += timing.ghost_zero_seconds
    return sums
end

function append_csv_row(path::String, row::String)
    isempty(path) && return
    mkpath(dirname(path))
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, CSV_HEADER)
        println(io, row)
    end
end

function csv_row(values)
    return join(values, ',')
end

function main()
    config = parse_args(ARGS)
    mpi_was_initialized = MPI.Initialized()
    mpi_was_initialized || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    try
        cells = run_cells(config, nranks)

        MPI.Barrier(comm)
        setup_elapsed = @elapsed begin
            distributed_dg, setup_description =
                build_benchmark_dg(config, cells, comm)
        end
        setup_seconds_max = MPI.Allreduce(setup_elapsed, max, comm)

        registry = MaxwellBoundaryRegistry(
            Dict(BENCH_PEC_BOUNDARY_ID => MaxwellBC_PEC),
        )
        formulation = PoissonBracketFormulation()
        U = initial_field(distributed_dg)
        rhs = DiscoGMPI.similar_maxwell_rhs(U)

        for _ in 1:config.warmup
            profile_distributed_maxwell_rhs!(
                rhs,
                U,
                distributed_dg,
                registry,
                formulation;
                ε = config.epsilon,
                μ = config.mu,
            )
        end

        MPI.Barrier(comm)
        local_sums = zeros(Float64, 4)
        for _ in 1:config.iterations
            timing = profile_distributed_maxwell_rhs!(
                rhs,
                U,
                distributed_dg,
                registry,
                formulation;
                ε = config.epsilon,
                μ = config.mu,
            )
            accumulate_profile!(local_sums, timing)
        end

        max_sums = MPI.Allreduce(local_sums, max, comm)
        mean_sums = MPI.Allreduce(local_sums, +, comm) ./ nranks
        local_owned, local_ghosts, local_interfaces, local_send_values =
            local_partition_metrics(distributed_dg)
        owned_min = MPI.Allreduce(local_owned, min, comm)
        owned_max = MPI.Allreduce(local_owned, max, comm)
        ghosts_max = MPI.Allreduce(local_ghosts, max, comm)
        interface_total = MPI.Allreduce(local_interfaces, +, comm)
        interface_max = MPI.Allreduce(local_interfaces, max, comm)
        send_values_max = MPI.Allreduce(local_send_values, max, comm)
        global_elements = MPI.Allreduce(local_owned, +, comm)
        total_time_max = max_sums[1]
        rhs_calls_per_second = config.iterations / total_time_max
        owned_element_rhs_per_second =
            (global_elements * config.iterations) / total_time_max

        metadata_dir = output_case_dir(config, nranks, cells)
        write_distributed_run_metadata(
            metadata_dir,
            distributed_dg;
            configuration = Dict(
                "benchmark" => "mpi_scaling",
                "mode" => string(config.mode),
                "setup" => string(config.setup),
                "setup_description" => setup_description,
                "cells_per_axis" => cells,
                "order" => config.order,
                "iterations" => config.iterations,
                "warmup" => config.warmup,
            ),
            runtime = Dict(
                "setup_seconds_max" => setup_seconds_max,
                "rhs_total_seconds_max" => total_time_max,
                "halo_seconds_max" => max_sums[2],
                "rhs_assembly_seconds_max" => max_sums[3],
                "ghost_zero_seconds_max" => max_sums[4],
                "rhs_calls_per_second" => rhs_calls_per_second,
                "owned_element_rhs_per_second" =>
                    owned_element_rhs_per_second,
            ),
        )

        row = csv_row((
            string(Dates.now(Dates.UTC)),
            config.mode,
            config.setup,
            nranks,
            cells,
            global_elements,
            config.order,
            distributed_dg.dg.ref.Np,
            config.iterations,
            config.warmup,
            setup_seconds_max,
            max_sums[1],
            max_sums[2],
            max_sums[3],
            max_sums[4],
            mean_sums[1],
            mean_sums[2],
            mean_sums[3],
            owned_min,
            owned_max,
            ghosts_max,
            interface_total,
            interface_max,
            send_values_max,
            rhs_calls_per_second,
            owned_element_rhs_per_second,
        ))

        if rank == 0
            println("CSV_RESULT,", row)
            append_csv_row(config.csv_path, row)
            println("Metadata: ", metadata_dir)
        end
    finally
        mpi_was_initialized || MPI.Finalize()
    end
end

main()
