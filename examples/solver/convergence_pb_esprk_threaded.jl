#!/usr/bin/env julia

include(joinpath(@__DIR__, "convergence_hw_upwind.jl"))

struct PBConvergenceConfig
    cells_per_axis::Vector{Int}
    orders::Vector{Int}
    final_time::Float64
    cfl::Float64
    backend::Symbol
    jitter::Float64
    seed::Int
    output::String
    eps::Float64
    mu::Float64
end

struct PBConvergenceResult
    order::Int
    esprk_order::Int
    cubature_order::Int
    mesh_level::Int
    cells_per_axis::Int
    nelements::Int
    h::Float64
    dt::Float64
    nsteps::Int
    l2_E::Float64
    l2_H::Float64
    l2_total::Float64
    rel_total::Float64
    linf_E::Float64
    linf_H::Float64
    linf_total::Float64
    rate_E::Union{Missing, Float64}
    rate_H::Union{Missing, Float64}
    rate::Union{Missing, Float64}
end

function parse_pb_convergence_args(args)
    config = PBConvergenceConfig(
        [3, 4, 5, 6],
        [1, 2, 3, 4],
        10.0,
        0.1,
        :threaded,
        0.08,
        1234,
        joinpath("output", "convergence_pb_esprk_threaded.csv"),
        1.0,
        1.0,
    )

    cells_per_axis = config.cells_per_axis
    orders = config.orders
    final_time = config.final_time
    cfl = config.cfl
    backend = config.backend
    jitter = config.jitter
    seed = config.seed
    output = config.output
    eps = config.eps
    mu = config.mu

    for arg in args
        if arg == "--help" || arg == "-h"
            print_pb_convergence_usage()
            exit(0)
        elseif startswith(arg, "--cells=")
            cells_per_axis = parse_int_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--orders=")
            orders = parse_int_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seconds=") ||
               startswith(arg, "--time=") ||
               startswith(arg, "--final-time=")
            final_time = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--esprk=") ||
               startswith(arg, "--psrk=") ||
               startswith(arg, "--rk=")
            parse_pb_esprk_rule(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--backend=")
            backend = parse_backend(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--jitter=")
            jitter = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--eps=")
            eps = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
        else
            error("Unknown argument: $arg. Run with --help for usage.")
        end
    end

    if length(cells_per_axis) != 4
        error(
            "This convergence driver expects exactly four mesh levels. " *
            "Use --cells=3,4,5,6 for the requested validation."
        )
    end

    if any(n -> n < 1, cells_per_axis)
        error("All entries in --cells must be positive.")
    end

    if any(N -> N < 1, orders)
        error("All entries in --orders must be positive.")
    end

    if any(N -> pb_esprk_order(N) > 6, orders)
        error("ESPRK order is N + 1, so spatial orders above 5 are unsupported.")
    end

    if final_time <= 0.0
        error("--seconds must be positive.")
    end

    if cfl <= 0.0
        error("--cfl must be positive.")
    end

    if jitter < 0.0
        error("--jitter must be nonnegative.")
    end

    if eps <= 0.0 || mu <= 0.0
        error("--eps and --mu must be positive.")
    end

    return PBConvergenceConfig(
        cells_per_axis,
        orders,
        final_time,
        cfl,
        backend,
        jitter,
        seed,
        output,
        eps,
        mu,
    )
end

function parse_pb_esprk_rule(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized = replace(normalized, " " => "")
    normalized = replace(normalized, "{" => "")
    normalized = replace(normalized, "}" => "")

    if normalized in ("auto", "order+1", "n+1", "spatialorder+1")
        return nothing
    end

    error(
        "This validation fixes ESPRK per spatial order as order + 1. " *
        "Use --esprk=order+1 or omit --esprk."
    )
end

function print_pb_convergence_usage()
    println("Poisson-bracket Maxwell convergence driver with H-first ESPRK")
    println()
    println("Usage:")
    println("  julia --project=. --threads=N examples/solver/convergence_pb_esprk_threaded.jl [options]")
    println()
    println("Defaults for the requested validation:")
    println("  --cells=3,4,5,6")
    println("  --orders=1,2,3,4")
    println("  --seconds=10")
    println("  ESPRK order is fixed per spatial order as esprk = order + 1")
    println("  Strict continuous errors use cubature order max(2, 2 * order + 4)")
    println()
    println("Options:")
    println("  --cells=a,b,c,d     Four jittered tet cube mesh levels. Default: 3,4,5,6")
    println("  --orders=a,b,c,d    DG polynomial orders. Default: 1,2,3,4")
    println("  --seconds=T         Final physical time. Default: 10")
    println("  --time=T            Alias for --seconds=T")
    println("  --final-time=T      Alias for --seconds=T")
    println("  --cfl=C             CFL factor for dt estimate. Default: 0.1")
    println("  --esprk=order+1     Accepted explicit form of the fixed ESPRK rule")
    println("  --psrk=order+1      Alias for --esprk=order+1")
    println("  --rk=order+1        Alias for --esprk=order+1")
    println("  --backend=B         Backend: serial or threaded. Default: threaded")
    println("  --jitter=J          Interior-node jitter fraction of grid spacing. Default: 0.08")
    println("  --seed=N            Random seed for deterministic meshes. Default: 1234")
    println("  --eps=X             Electric permittivity. Default: 1.0")
    println("  --mu=X              Magnetic permeability. Default: 1.0")
    println("  --output=PATH       CSV output path. Default: output/convergence_pb_esprk_threaded.csv")
    println("                       Also writes PATH with _3dec before the extension")
end

pb_esprk_order(order::Int) = order + 1

function advance_maxwell_pb_esprk!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    esprk_order::Int,
    dt::Float64,
    nsteps::Int,
    eps::Float64,
    mu::Float64,
)
    scheme = explicit_partitioned_symplectic_rk_scheme(
        esprk_order;
        first_partition = :H,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)
    rhs_function! = DiscoGMPI.make_maxwell_rhs_function(
        dg,
        registry,
        formulation;
        ε = eps,
        μ = mu,
    )

    for _ in 1:nsteps
        partitioned_symplectic_rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )
    end

    return U
end

function run_pb_convergence_case(
    cells_per_axis::Int,
    order::Int,
    level::Int,
    config::PBConvergenceConfig,
)
    mesh = build_jittered_pec_cube_mesh(
        cells_per_axis;
        jitter = config.jitter,
        seed = config.seed,
    )
    dg = DGDiscretization(
        mesh,
        order;
        backend = build_backend(config.backend),
    )

    registry = MaxwellBoundaryRegistry(
        Dict(CONV_PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )
    formulation = PoissonBracketFormulation()

    U = exact_maxwell_field(
        mesh,
        dg.ref,
        0.0;
        eps = config.eps,
        mu = config.mu,
    )

    dt_est, _ = estimate_maxwell_dt(
        mesh,
        dg.geometry,
        dg.ref;
        CFL = config.cfl,
        ε = config.eps,
        μ = config.mu,
    )

    nsteps = max(1, ceil(Int, config.final_time / dt_est))
    dt = config.final_time / nsteps
    esprk_order = pb_esprk_order(order)

    advance_maxwell_pb_esprk!(
        U,
        dg,
        registry,
        formulation;
        esprk_order = esprk_order,
        dt = dt,
        nsteps = nsteps,
        eps = config.eps,
        mu = config.mu,
    )

    strict_error = maxwell_continuous_error(
        U,
        mesh,
        dg.ref,
        dg.mappings,
        config.final_time;
        eps = config.eps,
        mu = config.mu,
    )

    return PBConvergenceResult(
        order,
        esprk_order,
        strict_error.cubature_order,
        level,
        cells_per_axis,
        size(mesh.tets, 2),
        characteristic_h(dg),
        dt,
        nsteps,
        strict_error.l2_E,
        strict_error.l2_H,
        strict_error.l2_total,
        strict_error.rel_total,
        strict_error.linf_E,
        strict_error.linf_H,
        strict_error.linf_total,
        missing,
        missing,
        missing,
    )
end

function with_pb_rates(results::Vector{PBConvergenceResult})
    out = PBConvergenceResult[]

    for order in sort(unique(r.order for r in results))
        subset = sort(
            filter(r -> r.order == order, results);
            by = r -> r.mesh_level,
        )

        previous = nothing

        for result in subset
            rate_E = missing
            rate_H = missing
            rate = missing

            if previous !== nothing
                rate_E = convergence_rate(
                    result.l2_E,
                    previous.l2_E,
                    result.h,
                    previous.h,
                )
                rate_H = convergence_rate(
                    result.l2_H,
                    previous.l2_H,
                    result.h,
                    previous.h,
                )
                rate = convergence_rate(
                    result.l2_total,
                    previous.l2_total,
                    result.h,
                    previous.h,
                )
            end

            push!(
                out,
                PBConvergenceResult(
                    result.order,
                    result.esprk_order,
                    result.cubature_order,
                    result.mesh_level,
                    result.cells_per_axis,
                    result.nelements,
                    result.h,
                    result.dt,
                    result.nsteps,
                    result.l2_E,
                    result.l2_H,
                    result.l2_total,
                    result.rel_total,
                    result.linf_E,
                    result.linf_H,
                    result.linf_total,
                    rate_E,
                    rate_H,
                    rate,
                ),
            )

            previous = result
        end
    end

    return out
end

function print_pb_results(results::Vector{PBConvergenceResult})
    println()
    println("Poisson-bracket H-first ESPRK Maxwell convergence")
    println("-------------------------------------------------")
    println(
        rpad("N", 4),
        rpad("ESPRK", 8),
        rpad("Q", 6),
        rpad("level", 8),
        rpad("cells", 8),
        rpad("Ne", 10),
        rpad("h", 14),
        rpad("dt", 14),
        rpad("steps", 8),
        rpad("L2 E", 16),
        rpad("L2 H", 16),
        rpad("L2 total", 16),
        rpad("rel total", 16),
        rpad("Linf E", 16),
        rpad("Linf H", 16),
        rpad("Linf total", 16),
        rpad("rate E", 10),
        rpad("rate H", 10),
        "rate total",
    )

    for result in results
        rate_E_string = ismissing(result.rate_E) ? "-" : @sprintf("%.4f", result.rate_E)
        rate_H_string = ismissing(result.rate_H) ? "-" : @sprintf("%.4f", result.rate_H)
        rate_string = ismissing(result.rate) ? "-" : @sprintf("%.4f", result.rate)
        println(
            rpad(string(result.order), 4),
            rpad(string(result.esprk_order), 8),
            rpad(string(result.cubature_order), 6),
            rpad(string(result.mesh_level), 8),
            rpad(string(result.cells_per_axis), 8),
            rpad(string(result.nelements), 10),
            rpad(@sprintf("%.6e", result.h), 14),
            rpad(@sprintf("%.6e", result.dt), 14),
            rpad(string(result.nsteps), 8),
            rpad(@sprintf("%.6e", result.l2_E), 16),
            rpad(@sprintf("%.6e", result.l2_H), 16),
            rpad(@sprintf("%.6e", result.l2_total), 16),
            rpad(@sprintf("%.6e", result.rel_total), 16),
            rpad(@sprintf("%.6e", result.linf_E), 16),
            rpad(@sprintf("%.6e", result.linf_H), 16),
            rpad(@sprintf("%.6e", result.linf_total), 16),
            rpad(rate_E_string, 10),
            rpad(rate_H_string, 10),
            rate_string,
        )
    end
end

function write_pb_results_csv(
    path::AbstractString,
    results::Vector{PBConvergenceResult},
)
    ensure_parent_dir(path)

    open(path, "w") do io
        println(io, pb_results_csv_header())

        for result in results
            rate_E_string = ismissing(result.rate_E) ? "" : string(result.rate_E)
            rate_H_string = ismissing(result.rate_H) ? "" : string(result.rate_H)
            rate_string = ismissing(result.rate) ? "" : string(result.rate)
            println(
                io,
                join(
                    (
                        result.order,
                        result.esprk_order,
                        result.cubature_order,
                        result.mesh_level,
                        result.cells_per_axis,
                        result.nelements,
                        result.h,
                        result.dt,
                        result.nsteps,
                        result.l2_E,
                        result.l2_H,
                        result.l2_total,
                        result.rel_total,
                        result.linf_E,
                        result.linf_H,
                        result.linf_total,
                        rate_E_string,
                        rate_H_string,
                        rate_string,
                    ),
                    ",",
                ),
            )
        end
    end
end

function pb_results_csv_header()
    return (
        "order,esprk_order,cubature_order,mesh_level,cells_per_axis," *
        "nelements,h,dt,nsteps,l2_E,l2_H,l2_total,rel_total," *
        "linf_E,linf_H,linf_total,rate_E,rate_H,rate_total"
    )
end

function rounded_pb_results_csv_path(path::AbstractString)
    root, ext = splitext(path)

    if isempty(ext)
        return string(path, "_3dec.csv")
    end

    return string(root, "_3dec", ext)
end

function trim_decimal_zeros(s::AbstractString)
    out = String(s)

    if !occursin(".", out)
        return out
    end

    while endswith(out, "0")
        out = out[1:(end - 1)]
    end

    if endswith(out, ".")
        out = out[1:(end - 1)]
    end

    return out
end

function format_scientific_3dec(x::Float64)
    mantissa, exponent = split(@sprintf("%.3e", x), "e")

    return string(
        trim_decimal_zeros(mantissa),
        "e",
        parse(Int, exponent),
    )
end

function format_float_3dec(x::Float64)
    if !isfinite(x)
        return string(x)
    elseif x == 0.0
        return "0"
    end

    ax = abs(x)

    if 0.1 <= ax < 1e4
        return trim_decimal_zeros(@sprintf("%.3f", x))
    else
        return format_scientific_3dec(x)
    end
end

format_float_3dec(x::Union{Missing, Float64}) =
    ismissing(x) ? "" : format_float_3dec(x)

function write_pb_results_csv_3dec(
    path::AbstractString,
    results::Vector{PBConvergenceResult},
)
    ensure_parent_dir(path)

    open(path, "w") do io
        println(io, pb_results_csv_header())

        for result in results
            println(
                io,
                join(
                    (
                        result.order,
                        result.esprk_order,
                        result.cubature_order,
                        result.mesh_level,
                        result.cells_per_axis,
                        result.nelements,
                        format_float_3dec(result.h),
                        format_float_3dec(result.dt),
                        result.nsteps,
                        format_float_3dec(result.l2_E),
                        format_float_3dec(result.l2_H),
                        format_float_3dec(result.l2_total),
                        format_float_3dec(result.rel_total),
                        format_float_3dec(result.linf_E),
                        format_float_3dec(result.linf_H),
                        format_float_3dec(result.linf_total),
                        format_float_3dec(result.rate_E),
                        format_float_3dec(result.rate_H),
                        format_float_3dec(result.rate),
                    ),
                    ",",
                ),
            )
        end
    end
end

function main_pb_convergence(args = ARGS)
    config = parse_pb_convergence_args(args)

    println("Poisson-bracket Maxwell convergence driver")
    println("------------------------------------------")
    println("Formulation:        PoissonBracketFormulation")
    println("Numerical flux:     centered")
    println("Time integrator:    H-first explicit symplectic partitioned RK")
    println("ESPRK rule:         esprk_order = spatial_order + 1")
    println("Error norm:         strict continuous L2 and quadrature-point L∞")
    println("Cubature rule:      Jaskowiec-Sukumar, order max(2, 2N + 4)")
    println("Backend:            ", backend_description(config.backend))
    println("Julia threads:      ", Base.Threads.nthreads())
    println("Orders:             ", config.orders)
    println("Mesh levels:        ", config.cells_per_axis)
    println("Final time:         ", config.final_time)
    println("CFL:                ", config.cfl)
    println("Interior jitter:    ", config.jitter)
    println("PEC boundary id:    ", CONV_PEC_BOUNDARY_ID)
    println()

    if config.backend == :threaded && Base.Threads.nthreads() == 1
        println("WARNING: ThreadedBackend selected with only one Julia thread.")
        println("         Launch with --threads=N or JULIA_NUM_THREADS=N for validation.")
        println()
    end

    results = PBConvergenceResult[]

    for order in config.orders
        esprk_order = pb_esprk_order(order)

        for (level, cells_per_axis) in enumerate(config.cells_per_axis)
            @printf(
                "Running order N=%d, ESPRK=%d, mesh level %d, cells_per_axis=%d\n",
                order,
                esprk_order,
                level,
                cells_per_axis,
            )

            result = run_pb_convergence_case(
                cells_per_axis,
                order,
                level,
                config,
            )
            push!(results, result)
        end
    end

    rated_results = with_pb_rates(results)

    print_pb_results(rated_results)
    write_pb_results_csv(config.output, rated_results)
    rounded_output = rounded_pb_results_csv_path(config.output)
    write_pb_results_csv_3dec(rounded_output, rated_results)

    println()
    println("Wrote CSV: ", config.output)
    println("Wrote 3-decimal CSV: ", rounded_output)

    return rated_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_pb_convergence()
end
