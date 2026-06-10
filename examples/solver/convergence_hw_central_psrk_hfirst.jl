#!/usr/bin/env julia

include(joinpath(@__DIR__, "convergence_hw_upwind.jl"))

function parse_partition(value::AbstractString)
    normalized = replace(uppercase(strip(value)), ":" => "")

    if normalized in ("E", "ELECTRIC")
        return :E
    elseif normalized in ("H", "MAGNETIC")
        return :H
    else
        error("Unsupported first partition: $value. Use E or H.")
    end
end

function parse_central_psrk_args(args)
    config = ConvergenceConfig(
        [2, 3, 4, 5],
        [1, 2, 4],
        0.005,
        0.1,
        2,
        0.08,
        1234,
        joinpath("output", "convergence_hw_central_psrk_hfirst.csv"),
        1.0,
        1.0,
    )

    cells_per_axis = config.cells_per_axis
    orders = config.orders
    final_time = config.final_time
    cfl = config.cfl
    psrk_order = config.rk_order
    first_partition = :H
    jitter = config.jitter
    seed = config.seed
    output = config.output
    eps = config.eps
    mu = config.mu

    for arg in args
        if arg == "--help" || arg == "-h"
            print_central_psrk_usage()
            exit(0)
        elseif startswith(arg, "--cells=")
            cells_per_axis = parse_int_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--orders=")
            orders = parse_int_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--time=") || startswith(arg, "--final-time=")
            final_time = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--psrk=") || startswith(arg, "--rk=")
            psrk_order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--first=") || startswith(arg, "--first-partition=")
            first_partition = parse_partition(split(arg, "=", limit = 2)[2])
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
            "Use --cells=a,b,c,d."
        )
    end

    if any(n -> n < 1, cells_per_axis)
        error("All entries in --cells must be positive.")
    end

    if any(N -> N < 1, orders)
        error("All entries in --orders must be positive.")
    end

    if final_time <= 0.0
        error("--time must be positive.")
    end

    if cfl <= 0.0
        error("--cfl must be positive.")
    end

    if !(psrk_order in 1:6)
        error("--psrk must be an ESPRK order from 1 through 6.")
    end

    if jitter < 0.0
        error("--jitter must be nonnegative.")
    end

    if eps <= 0.0 || mu <= 0.0
        error("--eps and --mu must be positive.")
    end

    return (
        config = ConvergenceConfig(
            cells_per_axis,
            orders,
            final_time,
            cfl,
            psrk_order,
            jitter,
            seed,
            output,
            eps,
            mu,
        ),
        first_partition = first_partition,
    )
end

function print_central_psrk_usage()
    println("Hesthaven-Warburton Maxwell convergence driver with centered flux and H-first PSRK")
    println()
    println("Usage:")
    println("  julia --project=. examples/solver/convergence_hw_central_psrk_hfirst.jl [options]")
    println()
    println("Options:")
    println("  --cells=a,b,c,d     Four jittered tet cube mesh levels. Default: 2,3,4,5")
    println("  --orders=a,b,c      DG polynomial orders. Default: 1,2,4")
    println("  --time=T            Final time. Default: 0.005")
    println("  --cfl=C             CFL factor for dt estimate. Default: 0.1")
    println("  --psrk=N            ESPRK order 1 through 6. Default: 2")
    println("  --rk=N              Alias for --psrk=N")
    println("  --first=E|H         First updated Maxwell partition. Default: H")
    println("  --jitter=J          Interior-node jitter fraction of grid spacing. Default: 0.08")
    println("  --seed=N            Random seed for deterministic meshes. Default: 1234")
    println("  --eps=X             Electric permittivity. Default: 1.0")
    println("  --mu=X              Magnetic permeability. Default: 1.0")
    println("  --output=PATH       CSV output path. Default: output/convergence_hw_central_psrk_hfirst.csv")
end

function advance_maxwell_psrk!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    psrk_order::Int,
    first_partition::Symbol,
    dt::Float64,
    nsteps::Int,
    eps::Float64,
    mu::Float64,
)
    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = first_partition,
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

function run_central_psrk_case(
    cells_per_axis::Int,
    order::Int,
    level::Int,
    config::ConvergenceConfig,
    first_partition::Symbol,
)
    mesh = build_jittered_pec_cube_mesh(
        cells_per_axis;
        jitter = config.jitter,
        seed = config.seed,
    )
    dg = DGDiscretization(mesh, order)

    registry = MaxwellBoundaryRegistry(
        Dict(CONV_PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )
    formulation = HesthavenWarburtonFormulation(MaxwellFlux_Central)

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

    advance_maxwell_psrk!(
        U,
        dg,
        registry,
        formulation;
        psrk_order = config.rk_order,
        first_partition = first_partition,
        dt = dt,
        nsteps = nsteps,
        eps = config.eps,
        mu = config.mu,
    )

    exact = exact_maxwell_field(
        mesh,
        dg.ref,
        config.final_time;
        eps = config.eps,
        mu = config.mu,
    )

    l2_E, l2_H, l2_total, rel_total = maxwell_l2_error(
        U,
        exact,
        dg.ref,
        dg.mappings,
    )

    return ConvergenceResult(
        order,
        level,
        cells_per_axis,
        size(mesh.tets, 2),
        characteristic_h(dg),
        dt,
        nsteps,
        l2_E,
        l2_H,
        l2_total,
        rel_total,
        missing,
        missing,
        missing,
    )
end

function print_central_psrk_results(results::Vector{ConvergenceResult})
    println()
    println("HW centered-flux H-first PSRK Maxwell convergence")
    println("-------------------------------------------------")
    println(
        rpad("N", 4),
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
            rpad(rate_E_string, 10),
            rpad(rate_H_string, 10),
            rate_string,
        )
    end
end

function main_central_psrk(args = ARGS)
    parsed = parse_central_psrk_args(args)
    config = parsed.config
    first_partition = parsed.first_partition

    println("Hesthaven-Warburton Maxwell convergence driver")
    println("----------------------------------------------")
    println("Formulation:        HesthavenWarburtonFormulation")
    println("Numerical flux:     centered")
    println("Time integrator:    explicit partitioned symplectic RK", config.rk_order)
    println("First partition:    ", first_partition)
    println("Orders:             ", config.orders)
    println("Mesh levels:        ", config.cells_per_axis)
    println("Final time:         ", config.final_time)
    println("CFL:                ", config.cfl)
    println("Interior jitter:    ", config.jitter)
    println("PEC boundary id:    ", CONV_PEC_BOUNDARY_ID)
    println()

    results = ConvergenceResult[]

    for order in config.orders
        for (level, cells_per_axis) in enumerate(config.cells_per_axis)
            @printf(
                "Running order N=%d, mesh level %d, cells_per_axis=%d\n",
                order,
                level,
                cells_per_axis,
            )

            result = run_central_psrk_case(
                cells_per_axis,
                order,
                level,
                config,
                first_partition,
            )
            push!(results, result)
        end
    end

    rated_results = with_rates(results)

    print_central_psrk_results(rated_results)
    write_results_csv(config.output, rated_results)

    println()
    println("Wrote CSV: ", config.output)

    return rated_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_central_psrk()
end
