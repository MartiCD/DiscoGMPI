#!/usr/bin/env julia

include(joinpath(@__DIR__, "stationary_pec_cube.jl"))

function parse_pb_esprk_partition(value::AbstractString)
    normalized = replace(uppercase(strip(value)), ":" => "")

    if normalized in ("E", "ELECTRIC")
        return :E
    elseif normalized in ("H", "MAGNETIC")
        return :H
    else
        error("Unsupported first partition: $value. Use E or H.")
    end
end

function parse_pb_esprk_backend(value::AbstractString)
    normalized = lowercase(strip(value))

    if normalized == "serial"
        return :serial
    elseif normalized == "threaded" || normalized == "threads"
        return :threaded
    else
        error("Unsupported --backend=$value. Expected serial or threaded.")
    end
end

function build_pb_esprk_backend(kind::Symbol)
    if kind == :serial
        return SerialBackend()
    elseif kind == :threaded
        return ThreadedBackend()
    else
        error("Unsupported backend kind: $kind.")
    end
end

function pb_esprk_backend_description(kind::Symbol)
    if kind == :serial
        return "serial"
    elseif kind == :threaded
        return "threaded ($(Base.Threads.nthreads()) Julia threads)"
    else
        error("Unsupported backend kind: $kind.")
    end
end

function parse_pb_esprk_args(args)
    cells_per_axis = 2
    order = 1
    nsteps = 5
    esprk_order = 2
    first_partition = :H
    cfl = 0.05
    backend = :serial
    output = joinpath("output", "stationary_pec_cube_pb_esprk_final")

    for arg in args
        if arg == "--help" || arg == "-h"
            print_pb_esprk_usage()
            exit(0)
        elseif startswith(arg, "--cells=")
            cells_per_axis = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--order=")
            order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nsteps=")
            nsteps = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--esprk=") ||
               startswith(arg, "--psrk=") ||
               startswith(arg, "--rk=")
            esprk_order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--first=") ||
               startswith(arg, "--first-partition=")
            first_partition = parse_pb_esprk_partition(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--backend=")
            backend = parse_pb_esprk_backend(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        else
            error("Unknown argument: $arg. Run with --help for usage.")
        end
    end

    if !(esprk_order in 1:6)
        error("--esprk must be an ESPRK order from 1 through 6.")
    end

    if first_partition != :H
        error("PoissonBracketFormulation requires H-first ESPRK. Use --first=H.")
    end

    return (
        cells_per_axis = cells_per_axis,
        order = order,
        nsteps = nsteps,
        esprk_order = esprk_order,
        first_partition = first_partition,
        cfl = cfl,
        backend = backend,
        output = output,
    )
end

function print_pb_esprk_usage()
    println("Stationary Poisson-bracket Maxwell ESPRK driver in [-1,1]^3 with PEC walls")
    println()
    println("Usage:")
    println("  julia --project=. examples/solver/stationary_pec_cube_pb_esprk.jl [options]")
    println()
    println("Options:")
    println("  --cells=N             Structured cube cells per axis before tet split. Default: 2")
    println("  --order=N             DG polynomial order. Default: 1")
    println("  --nsteps=N            ESPRK steps to run. Default: 5")
    println("  --esprk=N             ESPRK order 1 through 6. Default: 2")
    println("  --psrk=N              Alias for --esprk=N")
    println("  --rk=N                Compatibility alias for --esprk=N")
    println("  --first=H             First updated Maxwell partition. PB requires H. Default: H")
    println("  --first-partition=H   Alias for --first=H")
    println("  --cfl=X               CFL factor used for dt estimate. Default: 0.05")
    println("  --backend=B           Backend: serial or threaded. Default: serial")
    println("  --output=PATH         Final VTU output path. Default: output/stationary_pec_cube_pb_esprk_final.vtu")
    println()
    println("For threaded runs, launch Julia with JULIA_NUM_THREADS=N.")
end

function main_pb_esprk(args = ARGS)
    config = parse_pb_esprk_args(args)

    mesh = build_pec_cube_mesh(config.cells_per_axis)
    dg = DGDiscretization(
        mesh,
        config.order;
        backend = build_pb_esprk_backend(config.backend),
    )

    print_topology_summary(mesh, dg.topology)
    print_geometry_summary(dg.geometry)
    print_mapping_summary(mesh, dg.geometry, dg.mappings)
    print_reference_tet_summary(dg.ref)
    print_reference_face_operator_summary(dg.ref, dg.fops)

    registry = MaxwellBoundaryRegistry(
        Dict(PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )

    formulation = PoissonBracketFormulation()

    U = stationary_maxwell_field(mesh, dg.ref)
    U0 = copy_maxwell_field(U)

    rhs = similar_rhs(U)
    maxwell_rhs!(
        rhs,
        U,
        dg,
        registry,
        formulation,
    )

    rhs0 = max_abs_rhs(rhs)
    energy0 = maxwell_energy(U, dg.ref, dg.mappings)
    energy_rate0 = maxwell_energy_rate(U, rhs, dg.ref, dg.mappings)
    tangential_e0 = max_tangential_electric_field(U, dg.flux_faces)

    dt, sizes = estimate_maxwell_dt(
        mesh,
        dg.geometry,
        dg.ref;
        CFL = config.cfl,
    )

    println("Stationary PEC cube Maxwell run with PoissonBracketFormulation + ESPRK")
    println("-----------------------------------------------------------------------")
    println("Domain:              [-1, 1]^3")
    println("Boundary condition:  PEC on all exterior faces, boundary_id = ", PEC_BOUNDARY_ID)
    println("Initial field:       PEC cavity mode with omega = sqrt(3) * pi")
    println("Formulation:         PoissonBracketFormulation")
    println("Time integrator:     Explicit symplectic partitioned Runge-Kutta")
    println("Stage order:         H partition first, then E partition")
    println("Cells per axis:      ", config.cells_per_axis)
    println("Tetrahedra:          ", size(mesh.tets, 2))
    println("Boundary triangles:  ", size(mesh.tris, 2))
    println("DG order:            ", config.order)
    println("Backend:             ", pb_esprk_backend_description(config.backend))
    println("ESPRK order:         ", config.esprk_order)
    println("First partition:     ", config.first_partition)
    println("Initial energy:      ", energy0.total)
    println("Initial dE/dt:       ", energy_rate0)
    println("Initial max |rhs|:   ", rhs0)
    println("Initial max |n x E|: ", tangential_e0)
    println("Estimated dt:        ", dt)
    println()

    run_maxwell_partitioned_symplectic_time_steps!(
        U,
        dg,
        registry,
        formulation;
        psrk_order = config.esprk_order,
        first_partition = config.first_partition,
        dt = dt,
        nsteps = config.nsteps,
        energy_every = max(1, config.nsteps),
    )

    rhs_final = similar_rhs(U)
    maxwell_rhs!(
        rhs_final,
        U,
        dg,
        registry,
        formulation,
    )

    energy1 = maxwell_energy(U, dg.ref, dg.mappings)
    energy_rate1 = maxwell_energy_rate(U, rhs_final, dg.ref, dg.mappings)
    field_delta = max_abs_field_difference(U, U0)
    rhs1 = max_abs_rhs(rhs_final)
    tangential_e1 = max_tangential_electric_field(U, dg.flux_faces)
    final_time = dt * config.nsteps
    saved_files = save_final_fields_vtu(
        config.output,
        mesh,
        dg.ref,
        U;
        time = final_time,
    )

    println()
    println("Poisson-bracket ESPRK diagnostics")
    println("---------------------------------")
    println("Final energy:         ", energy1.total)
    println("Energy drift:         ", energy1.total - energy0.total)
    println("Final dE/dt:          ", energy_rate1)
    println("Max field change:     ", field_delta)
    println("Final max |rhs|:      ", rhs1)
    println("Final max |n x E|:    ", tangential_e1)
    println("Saved final fields:   ", join(saved_files, ", "))

    if field_delta < 1e-12 && rhs1 < 1e-12 && tangential_e1 < 1e-12
        println("Result: PASS")
    else
        println("Result: CHECK TOLERANCES")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_pb_esprk()
end
