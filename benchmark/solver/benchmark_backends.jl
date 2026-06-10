#!/usr/bin/env julia

using BenchmarkTools
using Statistics
using Printf
using Dates
using Random
using DiscoGMPI

const BENCH_PEC_BOUNDARY_ID = 10
const BENCH_TET_FACES = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)

const CSV_HEADER = (
    "timestamp,formulation,julia_version,threads,t_serial_min,t_serial_med,t_serial_mean," *
    "t_threaded_min,t_threaded_med,t_threaded_mean,serial_allocs," *
    "threaded_allocs,serial_memory,threaded_memory,speedup_min,speedup_med," *
    "speedup_mean,efficiency_med,abs_err,rel_err,correct"
)

struct BackendBenchmarkConfig
    formulation::Symbol
    cells_per_axis::Int
    order::Int
    jitter::Float64
    seed::Int
    eps::Float64
    mu::Float64
    seconds::Float64
    samples::Int
    evals::Int
end

struct BackendBenchmarkModel{F<:AbstractMaxwellDGFormulation}
    dg::DGDiscretization
    registry::MaxwellBoundaryRegistry
    formulation::F
    eps::Float64
    mu::Float64
end

function print_usage()
    println("DiscoGMPI SerialBackend vs ThreadedBackend RHS benchmark")
    println()
    println("Usage:")
    println("  julia --project=. --threads=N benchmark/solver/benchmark_backends.jl [options]")
    println()
    println("Options:")
    println("  --formulation=F  RHS formulation: hw-upwind or pb. Default: hw-upwind")
    println("  --cells=N        Cube cells per axis. Default: 6")
    println("  --order=N        DG polynomial order. Default: 3")
    println("  --jitter=X       Interior-node jitter fraction. Default: 0.05")
    println("  --seed=N         Random seed for deterministic mesh. Default: 1234")
    println("  --eps=X          Electric permittivity. Default: 1.0")
    println("  --mu=X           Magnetic permeability. Default: 1.0")
    println("  --seconds=X      BenchmarkTools seconds per benchmark. Default: 3.0")
    println("  --samples=N      BenchmarkTools sample cap. Default: 10000")
    println("  --evals=N        BenchmarkTools evals per sample. Default: 1")
    println("  --csv-header     Print only the CSV header and exit")
    println("  --help, -h       Print this message and exit")
    println()
    println("Julia threads are controlled outside this script with --threads=N or JULIA_NUM_THREADS.")
end

function parse_args(args)
    config = BackendBenchmarkConfig(
        :hw_upwind,
        6,
        3,
        0.05,
        1234,
        1.0,
        1.0,
        3.0,
        10000,
        1,
    )

    formulation = config.formulation
    cells_per_axis = config.cells_per_axis
    order = config.order
    jitter = config.jitter
    seed = config.seed
    eps_value = config.eps
    mu_value = config.mu
    seconds = config.seconds
    samples = config.samples
    evals = config.evals

    for arg in args
        if arg == "--help" || arg == "-h"
            print_usage()
            exit(0)
        elseif arg == "--csv-header"
            println(CSV_HEADER)
            exit(0)
        elseif startswith(arg, "--formulation=")
            formulation = parse_formulation(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cells=")
            cells_per_axis = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--order=")
            order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--jitter=")
            jitter = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--eps=")
            eps_value = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu_value = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seconds=")
            seconds = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--evals=")
            evals = parse(Int, split(arg, "=", limit = 2)[2])
        else
            error("Unknown argument: $arg. Run with --help for usage.")
        end
    end

    if cells_per_axis < 1
        error("--cells must be positive.")
    end

    if order < 1
        error("--order must be positive.")
    end

    if jitter < 0.0
        error("--jitter must be nonnegative.")
    end

    if eps_value <= 0.0 || mu_value <= 0.0
        error("--eps and --mu must be positive.")
    end

    if seconds <= 0.0
        error("--seconds must be positive.")
    end

    if samples < 1
        error("--samples must be positive.")
    end

    if evals < 1
        error("--evals must be positive.")
    end

    return BackendBenchmarkConfig(
        formulation,
        cells_per_axis,
        order,
        jitter,
        seed,
        eps_value,
        mu_value,
        seconds,
        samples,
        evals,
    )
end

function parse_formulation(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized = replace(normalized, "_" => "-")

    if normalized in ("hw", "hw-upwind", "hesthaven-warburton", "hesthaven-warburton-upwind")
        return :hw_upwind
    elseif normalized in ("pb", "poisson-bracket", "poissonbracket")
        return :poisson_bracket
    else
        error("Unsupported --formulation=$value. Expected hw-upwind or pb.")
    end
end

function formulation_description(kind::Symbol)
    if kind == :hw_upwind
        return "HesthavenWarburtonFormulation / upwind"
    elseif kind == :poisson_bracket
        return "PoissonBracketFormulation / central"
    else
        error("Unsupported formulation kind: $kind.")
    end
end

function formulation_csv_name(kind::Symbol)
    if kind == :hw_upwind
        return "hw-upwind"
    elseif kind == :poisson_bracket
        return "pb"
    else
        error("Unsupported formulation kind: $kind.")
    end
end

function build_formulation(kind::Symbol)
    if kind == :hw_upwind
        return HesthavenWarburtonFormulation(MaxwellFlux_Upwind)
    elseif kind == :poisson_bracket
        return PoissonBracketFormulation()
    else
        error("Unsupported formulation kind: $kind.")
    end
end

function configure_benchmarktools!(config::BackendBenchmarkConfig)
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = config.seconds
    BenchmarkTools.DEFAULT_PARAMETERS.samples = config.samples
    BenchmarkTools.DEFAULT_PARAMETERS.evals = config.evals
    return nothing
end

function node_id(i::Int, j::Int, k::Int, cells_per_axis::Int)
    n = cells_per_axis + 1
    return 1 + i + n * (j + n * k)
end

function build_jittered_cube_points(
    cells_per_axis::Int;
    jitter::Float64,
    seed::Int,
)
    rng = Random.MersenneTwister(seed + cells_per_axis)
    npoints_1d = cells_per_axis + 1
    coords = collect(range(-1.0, 1.0; length = npoints_1d))
    h = 2.0 / cells_per_axis

    points = zeros(Float64, 3, npoints_1d^3)

    for k in 0:cells_per_axis
        for j in 0:cells_per_axis
            for i in 0:cells_per_axis
                id = node_id(i, j, k, cells_per_axis)
                x = coords[i + 1]
                y = coords[j + 1]
                z = coords[k + 1]

                if 0 < i < cells_per_axis &&
                   0 < j < cells_per_axis &&
                   0 < k < cells_per_axis
                    scale = jitter * h
                    x += scale * (2.0 * rand(rng) - 1.0)
                    y += scale * (2.0 * rand(rng) - 1.0)
                    z += scale * (2.0 * rand(rng) - 1.0)
                end

                points[1, id] = x
                points[2, id] = y
                points[3, id] = z
            end
        end
    end

    return points
end

function build_cube_tets(cells_per_axis::Int)
    tets = NTuple{4, Int}[]

    for k in 0:(cells_per_axis - 1)
        for j in 0:(cells_per_axis - 1)
            for i in 0:(cells_per_axis - 1)
                v000 = node_id(i,     j,     k,     cells_per_axis)
                v100 = node_id(i + 1, j,     k,     cells_per_axis)
                v010 = node_id(i,     j + 1, k,     cells_per_axis)
                v110 = node_id(i + 1, j + 1, k,     cells_per_axis)
                v001 = node_id(i,     j,     k + 1, cells_per_axis)
                v101 = node_id(i + 1, j,     k + 1, cells_per_axis)
                v011 = node_id(i,     j + 1, k + 1, cells_per_axis)
                v111 = node_id(i + 1, j + 1, k + 1, cells_per_axis)

                push!(tets, (v000, v100, v110, v111))
                push!(tets, (v000, v110, v010, v111))
                push!(tets, (v000, v010, v011, v111))
                push!(tets, (v000, v011, v001, v111))
                push!(tets, (v000, v001, v101, v111))
                push!(tets, (v000, v101, v100, v111))
            end
        end
    end

    return reduce(hcat, collect.(tets))
end

function sorted_face_key(nodes::NTuple{3, Int})
    s = sort(collect(nodes))
    return (s[1], s[2], s[3])
end

function build_boundary_tris(tets::Matrix{Int})
    counts = Dict{NTuple{3, Int}, Int}()
    oriented_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for e in axes(tets, 2)
        tet = tets[:, e]

        for local_face in 1:4
            ids = BENCH_TET_FACES[local_face]
            nodes = (tet[ids[1]], tet[ids[2]], tet[ids[3]])
            key = sorted_face_key(nodes)

            counts[key] = get(counts, key, 0) + 1
            oriented_nodes[key] = nodes
        end
    end

    tris = NTuple{3, Int}[]

    for (key, count) in counts
        if count == 1
            push!(tris, oriented_nodes[key])
        elseif count != 2
            error("Non-manifold face $key appears $count times.")
        end
    end

    return reduce(hcat, collect.(tris))
end

function build_jittered_pec_cube_mesh(config::BackendBenchmarkConfig)
    points = build_jittered_cube_points(
        config.cells_per_axis;
        jitter = config.jitter,
        seed = config.seed,
    )
    tets = build_cube_tets(config.cells_per_axis)
    tris = build_boundary_tris(tets)

    ntets = size(tets, 2)
    ntris = size(tris, 2)

    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))

    boundary_id = zeros(Int, ntets + ntris)
    boundary_id[tri_cell_ids] .= BENCH_PEC_BOUNDARY_ID

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_id),
    )

    check_mesh_consistency(mesh)

    return mesh
end

function build_model(config::BackendBenchmarkConfig)
    mesh = build_jittered_pec_cube_mesh(config)
    dg = DGDiscretization(mesh, config.order; backend = SerialBackend())
    registry = MaxwellBoundaryRegistry(
        Dict(BENCH_PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )
    formulation = build_formulation(config.formulation)

    return BackendBenchmarkModel(
        dg,
        registry,
        formulation,
        config.eps,
        config.mu,
    )
end

function initial_condition(model::BackendBenchmarkModel)
    Efun = function (x, y, z)
        Ex = sin(pi * x) * cos(0.5 * pi * y) + 0.15 * z
        Ey = cos(0.5 * pi * x) * sin(pi * z) - 0.20 * y
        Ez = sin(0.5 * pi * y) * cos(pi * z) + 0.10 * x
        return Ex, Ey, Ez
    end

    Hfun = function (x, y, z)
        Hx = cos(pi * y) * sin(0.5 * pi * z) - 0.12 * x
        Hy = sin(0.5 * pi * z) * cos(pi * x) + 0.18 * y
        Hz = cos(0.5 * pi * x) * sin(pi * y) - 0.16 * z
        return Hx, Hy, Hz
    end

    return interpolate_maxwell_field(model.dg.mesh, model.dg.ref, Efun, Hfun)
end

function rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    model::BackendBenchmarkModel,
    backend::AbstractBackend,
)
    maxwell_rhs!(
        rhs,
        U,
        model.dg,
        model.registry,
        model.formulation,
        backend;
        ε = model.eps,
        μ = model.mu,
    )

    return rhs
end

function rhs_components(rhs::MaxwellRHS)
    return (
        rhs.rhsEx,
        rhs.rhsEy,
        rhs.rhsEz,
        rhs.rhsHx,
        rhs.rhsHy,
        rhs.rhsHz,
    )
end

function max_abs_rhs(rhs::MaxwellRHS)
    return maximum(maximum(abs, component) for component in rhs_components(rhs))
end

function max_abs_rhs_difference(a::MaxwellRHS, b::MaxwellRHS)
    return maximum(
        maximum(abs.(a_component .- b_component))
        for (a_component, b_component) in zip(rhs_components(a), rhs_components(b))
    )
end

function rhs_isapprox(
    a::MaxwellRHS,
    b::MaxwellRHS;
    rtol::Float64,
    atol::Float64,
)
    return all(
        isapprox(a_component, b_component; rtol = rtol, atol = atol)
        for (a_component, b_component) in zip(rhs_components(a), rhs_components(b))
    )
end

function format_bytes(bytes::Integer)
    units = ("B", "KiB", "MiB", "GiB")
    value = Float64(bytes)
    unit = units[1]

    for candidate in units
        unit = candidate
        if value < 1024.0 || candidate == units[end]
            break
        end
        value /= 1024.0
    end

    return @sprintf("%.3f %s", value, unit)
end

function active_project_string()
    try
        project = Base.active_project()
        return project === nothing ? "unknown" : String(project)
    catch
        return "unknown"
    end
end

function timestamp_string()
    return Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
end

function print_environment(config::BackendBenchmarkConfig)
    println("DiscoGMPI backend RHS benchmark")
    println("------------------------------")
    println("Timestamp:          ", timestamp_string())
    println("Julia version:      ", VERSION)
    println("Julia threads:      ", Base.Threads.nthreads())
    println("Active project:     ", active_project_string())
    println()
    println("Problem")
    println("-------")
    println("cells_per_axis:     ", config.cells_per_axis)
    println("formulation:        ", formulation_description(config.formulation))
    println("DG order:           ", config.order)
    println("interior jitter:    ", config.jitter)
    println("seed:               ", config.seed)
    println("epsilon:            ", config.eps)
    println("mu:                 ", config.mu)
    println("Benchmark seconds:  ", config.seconds)
    println("Benchmark samples:  ", config.samples)
    println("Benchmark evals:    ", config.evals)
    println()
end

function print_model_summary(model::BackendBenchmarkModel, U::MaxwellField)
    println("Model")
    println("-----")
    println("elements:           ", size(model.dg.mesh.tets, 2))
    println("boundary faces:     ", length(model.dg.flux_faces.boundary))
    println("interior faces:     ", length(model.dg.flux_faces.interior))
    println("boundary colors:    ", length(model.dg.flux_faces.boundary_colors))
    println("interior colors:    ", length(model.dg.flux_faces.interior_colors))
    println("nodes per element:  ", model.dg.ref.Np)
    println("field dofs:         ", length(U.Ex) * 6)
    println()
end

function print_trial_report(
    name::AbstractString,
    t_min::Float64,
    t_med::Float64,
    t_mean::Float64,
    allocs::Integer,
    memory::Integer,
)
    @printf("%-18s min:    %.6e s\n", name, t_min)
    @printf("%-18s median: %.6e s\n", "", t_med)
    @printf("%-18s mean:   %.6e s\n", "", t_mean)
    @printf("%-18s allocs: %d\n", "", allocs)
    @printf("%-18s memory: %s (%d bytes)\n", "", format_bytes(memory), memory)
end

function run_benchmark(config::BackendBenchmarkConfig)
    configure_benchmarktools!(config)
    print_environment(config)

    model = build_model(config)
    U = initial_condition(model)
    print_model_summary(model, U)

    rhs_serial = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_threaded = DiscoGMPI.similar_maxwell_rhs(U)

    serial_backend = SerialBackend()
    threaded_backend = ThreadedBackend()

    rhs!(rhs_serial, U, model, serial_backend)
    rhs!(rhs_threaded, U, model, threaded_backend)

    abs_err = max_abs_rhs_difference(rhs_serial, rhs_threaded)
    rel_err = abs_err / max(max_abs_rhs(rhs_serial), eps(eltype(rhs_serial.rhsEx)))
    correct = rhs_isapprox(
        rhs_serial,
        rhs_threaded;
        rtol = 1e-10,
        atol = 1e-12,
    )

    println("Correctness")
    println("-----------")
    @printf("abs_err:            %.6e\n", abs_err)
    @printf("rel_err:            %.6e\n", rel_err)
    println("isapprox:           ", correct)

    if !correct
        println("WARNING: SerialBackend and ThreadedBackend RHS values are not isapprox.")
    end

    println()
    println("Warming up both backends...")
    rhs!(rhs_serial, U, model, serial_backend)
    rhs!(rhs_threaded, U, model, threaded_backend)
    GC.gc()

    println("Benchmarking one RHS evaluation per sample...")
    b_serial = @benchmark rhs!($rhs_serial, $U, $model, $serial_backend)
    GC.gc()
    b_threaded = @benchmark rhs!($rhs_threaded, $U, $model, $threaded_backend)

    serial_min = minimum(b_serial)
    serial_med = median(b_serial)
    serial_mean = mean(b_serial)

    threaded_min = minimum(b_threaded)
    threaded_med = median(b_threaded)
    threaded_mean = mean(b_threaded)

    t_serial_min = serial_min.time * 1e-9
    t_serial_med = serial_med.time * 1e-9
    t_serial_mean = serial_mean.time * 1e-9

    t_threaded_min = threaded_min.time * 1e-9
    t_threaded_med = threaded_med.time * 1e-9
    t_threaded_mean = threaded_mean.time * 1e-9

    serial_allocs = serial_min.allocs
    threaded_allocs = threaded_min.allocs

    serial_memory = serial_min.memory
    threaded_memory = threaded_min.memory

    speedup_min = t_serial_min / t_threaded_min
    speedup_med = t_serial_med / t_threaded_med
    speedup_mean = t_serial_mean / t_threaded_mean
    efficiency_med = speedup_med / Base.Threads.nthreads()

    println()
    println("Results")
    println("-------")
    println("threads:            ", Base.Threads.nthreads())
    print_trial_report(
        "SerialBackend",
        t_serial_min,
        t_serial_med,
        t_serial_mean,
        serial_allocs,
        serial_memory,
    )
    print_trial_report(
        "ThreadedBackend",
        t_threaded_min,
        t_threaded_med,
        t_threaded_mean,
        threaded_allocs,
        threaded_memory,
    )
    @printf("speedup min:        %.6f\n", speedup_min)
    @printf("speedup median:     %.6f\n", speedup_med)
    @printf("speedup mean:       %.6f\n", speedup_mean)
    @printf("efficiency median:  %.6f\n", efficiency_med)
    @printf("abs_err:            %.6e\n", abs_err)
    @printf("rel_err:            %.6e\n", rel_err)

    println()
    println("CSV")
    println("---")
    @printf(
        "CSV_RESULT,%s,%s,%s,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%d,%d,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%s\n",
        timestamp_string(),
        formulation_csv_name(config.formulation),
        string(VERSION),
        Base.Threads.nthreads(),
        t_serial_min,
        t_serial_med,
        t_serial_mean,
        t_threaded_min,
        t_threaded_med,
        t_threaded_mean,
        serial_allocs,
        threaded_allocs,
        serial_memory,
        threaded_memory,
        speedup_min,
        speedup_med,
        speedup_mean,
        efficiency_med,
        abs_err,
        rel_err,
        string(correct),
    )

    return nothing
end

function main(args = ARGS)
    config = parse_args(args)
    run_benchmark(config)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
