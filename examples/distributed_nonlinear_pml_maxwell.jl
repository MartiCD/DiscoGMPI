#!/usr/bin/env julia

using MPI
using DiscoGMPI

include(joinpath(@__DIR__, "meshes", "generate_periodic_box_2x1x1.jl"))

Base.@kwdef struct PMLDemoConfig
    nx::Int = 16
    order::Int = 2
    rk_order::Int = 4
    final_time::Float64 = 1.5
    cfl::Float64 = 0.15
    pml_width::Float64 = 0.5
    sigma_max::Float64 = 12.0
    sigma_degree::Int = 2
    regularization::Float64 = 1e-12
    energy_every::Int = 10
    output_dir::String = joinpath(
        normpath(joinpath(@__DIR__, "..")),
        "output",
        "distributed_nonlinear_pml",
    )
end

function usage(io::IO = stdout)
    println(io, """
Distributed nonlinear PML Maxwell demonstration

Usage:
  mpiexec -n <ranks> julia --project=. \
    examples/distributed_nonlinear_pml_maxwell.jl [options]

Options:
  --nx N                 Structured x cells; six tetrahedra per cell.
                         Default: 16
  --order N              DG polynomial order. Default: 2
  --rk-order N           Explicit RK order in 1:5. Default: 4
  --final-time T         Final simulation time. Default: 1.5
  --cfl C                Maxwell CFL factor. Default: 0.15
  --pml-width W          Layer width at each x boundary. Default: 0.5
  --sigma-max S          Maximum PML damping. Default: 12
  --sigma-degree N       Polynomial damping degree. Default: 2
  --regularization E     Denominator regularization. Default: 1e-12
  --energy-every N       Energy CSV interval. Default: 10
  --output-dir PATH      Output directory.
  --help                 Show this message.

The pulse travels toward +x. Boundaries are absorbing in x and periodic in
y and z. The PML uses the six-field nonlinear equations from Abarbanel,
Gottlieb, and Hesthaven (2006), without auxiliary variables.
""")
end

function option_value(args::Vector{String}, index::Int, name::String)
    argument = args[index]
    prefix = "$name="
    if startswith(argument, prefix)
        return argument[(length(prefix) + 1):end], index
    end
    argument == name || error("Unknown option '$argument'.")
    index < length(args) || error("$name requires a value.")
    return args[index + 1], index + 1
end

function parse_config(args::Vector{String})
    values = Dict{Symbol, Any}(
        field => getfield(PMLDemoConfig(), field)
        for field in fieldnames(PMLDemoConfig)
    )
    option_fields = Dict(
        "--nx" => (:nx, Int),
        "--order" => (:order, Int),
        "--rk-order" => (:rk_order, Int),
        "--final-time" => (:final_time, Float64),
        "--cfl" => (:cfl, Float64),
        "--pml-width" => (:pml_width, Float64),
        "--sigma-max" => (:sigma_max, Float64),
        "--sigma-degree" => (:sigma_degree, Int),
        "--regularization" => (:regularization, Float64),
        "--energy-every" => (:energy_every, Int),
        "--output-dir" => (:output_dir, String),
    )

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            return nothing
        end
        name = nothing
        for candidate in keys(option_fields)
            if argument == candidate ||
               startswith(argument, "$candidate=")
                name = candidate
                break
            end
        end
        name === nothing && error("Unknown option '$argument'.")
        field, type = option_fields[name]
        raw_value, index = option_value(args, index, name)
        values[field] = type == String ? abspath(raw_value) : parse(type, raw_value)
        index += 1
    end

    config = PMLDemoConfig(; values...)
    config.nx >= 1 || error("--nx must be positive.")
    config.order >= 1 || error("--order must be positive.")
    1 <= config.rk_order <= 5 || error("--rk-order must be in 1:5.")
    config.final_time > 0.0 || error("--final-time must be positive.")
    config.cfl > 0.0 || error("--cfl must be positive.")
    0.0 < config.pml_width < 1.0 ||
        error("--pml-width must lie in (0,1).")
    config.sigma_max >= 0.0 || error("--sigma-max must be non-negative.")
    config.sigma_degree >= 1 || error("--sigma-degree must be positive.")
    config.regularization > 0.0 ||
        error("--regularization must be positive.")
    config.energy_every >= 1 || error("--energy-every must be positive.")
    return config
end

function periodic_box_mesh(nx::Int)
    points = build_points(nx)
    tetrahedron_tuples = build_tetrahedra(nx)
    triangle_tuples = boundary_triangles(tetrahedron_tuples)
    boundary_tags = [
        boundary_tag(points, triangle) for triangle in triangle_tuples
    ]
    tetrahedra = reduce(hcat, collect.(tetrahedron_tuples))
    triangles = reduce(hcat, collect.(triangle_tuples))
    ntets = size(tetrahedra, 2)
    ntris = size(triangles, 2)
    boundary_id = zeros(Int, ntets + ntris)
    boundary_id[(ntets + 1):end] .= boundary_tags

    return RawVTUMesh(
        points,
        tetrahedra,
        triangles,
        collect(1:ntets),
        collect((ntets + 1):(ntets + ntris)),
        Dict{String, Any}("boundary_id" => boundary_id),
    )
end

function slab_partition(nx::Int, nranks::Int)
    nx >= nranks ||
        error("--nx=$nx must be at least the MPI rank count $nranks.")
    partition = Vector{Int}(undef, 6 * nx)
    for slab in 0:(nx - 1)
        owner = min(fld(slab * nranks, nx), nranks - 1)
        partition[(6 * slab + 1):(6 * slab + 6)] .= owner
    end
    return partition
end

function run_demo(config::PMLDemoConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_mesh = rank == 0 ? periodic_box_mesh(config.nx) : nothing
    root_partition =
        rank == 0 ? slab_partition(config.nx, nranks) : nothing
    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        config.order;
        comm = comm,
    )

    periodic_specs = (
        DiscoGMPI.PeriodicBoundarySpec(
            3,
            4,
            (0.0, -LY, 0.0),
            :y_periodic,
        ),
        DiscoGMPI.PeriodicBoundarySpec(
            5,
            6,
            (0.0, 0.0, -LZ),
            :z_periodic,
        ),
    )
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        periodic_specs,
    )
    registry = MaxwellBoundaryRegistry(
        Dict(
            1 => MaxwellBC_Absorbing,
            2 => MaxwellBC_Absorbing,
            3 => MaxwellBC_None,
            4 => MaxwellBC_None,
            5 => MaxwellBC_None,
            6 => MaxwellBC_None,
        ),
    )
    formulation = HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

    pulse_center = 0.75
    pulse_width = 0.12
    envelope = x -> exp(-((x - pulse_center) / pulse_width)^2)
    U = interpolate_maxwell_field(
        distributed_dg,
        (x, y, z) -> (0.0, 0.0, envelope(x)),
        (x, y, z) -> (0.0, -envelope(x), 0.0),
    )

    left_interface = config.pml_width
    right_interface = LX - config.pml_width
    sigma_x = (x, y, z) -> max(
        polynomial_pml_sigma(
            x,
            left_interface,
            0.0;
            sigma_max = config.sigma_max,
            degree = config.sigma_degree,
        ),
        polynomial_pml_sigma(
            x,
            right_interface,
            LX;
            sigma_max = config.sigma_max,
            degree = config.sigma_degree,
        ),
    )
    pml = build_maxwell_nonlinear_pml(
        distributed_dg;
        sigma_x = sigma_x,
        regularization = config.regularization,
    )

    local_dt, _ = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = config.cfl,
    )
    estimated_dt = MPI.Allreduce(local_dt, min, comm)
    nsteps = ceil(Int, config.final_time / estimated_dt)
    dt = config.final_time / nsteps
    scheme = explicit_rk_scheme(config.rk_order)
    work = MaxwellRKWorkspace(U, scheme)
    energy0 = distributed_maxwell_energy(U, distributed_dg)

    energy_io = nothing
    if rank == 0
        mkpath(config.output_dir)
        energy_path = joinpath(config.output_dir, "energy.csv")
        energy_io = open(energy_path, "w")
        println(
            energy_io,
            "step,time,electric,magnetic,total,relative_to_initial",
        )
        println(
            energy_io,
            join(
                (
                    0,
                    0.0,
                    energy0.electric,
                    energy0.magnetic,
                    energy0.total,
                    1.0,
                ),
                ',',
            ),
        )
        println("Distributed nonlinear PML Maxwell demonstration")
        println("MPI ranks:       ", nranks)
        println("x cells:         ", config.nx)
        println("tetrahedra:      ", 6 * config.nx)
        println("DG order:        ", config.order)
        println("RK scheme:       ", scheme.name)
        println("PML width:       ", config.pml_width)
        println("sigma max:       ", config.sigma_max)
        println("dt / steps:      ", dt, " / ", nsteps)
        println("initial energy:  ", energy0.total)
        println("output:          ", energy_path)
    end

    for step in 1:nsteps
        distributed_periodic_maxwell_nonlinear_pml_rk_step!(
            U,
            work,
            scheme,
            dt,
            distributed_dg,
            periodic,
            registry,
            formulation,
            pml,
        )

        if step % config.energy_every == 0 || step == nsteps
            energy = distributed_maxwell_energy(U, distributed_dg)
            if rank == 0
                ratio = energy.total / max(energy0.total, eps(Float64))
                println(
                    energy_io,
                    join(
                        (
                            step,
                            step * dt,
                            energy.electric,
                            energy.magnetic,
                            energy.total,
                            ratio,
                        ),
                        ',',
                    ),
                )
                flush(energy_io)
            end
        end
    end

    if rank == 0
        final_energy = distributed_maxwell_energy(U, distributed_dg)
        println(
            "final/initial energy: ",
            final_energy.total / max(energy0.total, eps(Float64)),
        )
        close(energy_io)
    else
        distributed_maxwell_energy(U, distributed_dg)
    end
    return U
end

function main(args::Vector{String})
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    try
        config = parse_config(args)
        if config === nothing
            rank == 0 && usage()
            return
        end
        run_demo(config, comm)
    finally
        initialized_here && !MPI.Finalized() && MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
