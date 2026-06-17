#!/usr/bin/env julia

# MPI convergence study for the periodic Poisson-bracket Maxwell plane wave.
#
# From the DiscoGMPI repository root:
#   mpiexec -n 2 julia --project=. \
#     examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl

include(joinpath(@__DIR__, "distributed_periodic_poisson_bracket_maxwell.jl"))

using Printf

const PERIODIC_CONVERGENCE_RATE_TOLERANCE = 0.5
const PERIODIC_COMPONENT_RATE_ERROR_FLOOR = 1e-12
const PERIODIC_REFERENCE_TET_VOLUME = 4.0 / 3.0

struct DistributedPeriodicConvergenceConfig
    mesh_family::Symbol
    nx_targets::Vector{Int}
    orders::Vector{Int}
    final_time::Float64
    periods::Union{Nothing, Float64}
    cfl::Float64
    cfl_divisor::Float64
    epsilon::Float64
    mu::Float64
    wave_number::Float64
    geo_path::String
    mesh_dir::String
    output::String
end

# struct DistributedPeriodicConvergenceResult
#     mpi_ranks::Int
#     boundary_condition::Symbol
#     order::Int
#     esprk_order::Int
#     cubature_order::Int
#     mesh_level::Int
#     nx_target::Int
#     target_h::Float64
#     nelements::Int
#     min_owned_elements::Int
#     max_owned_elements::Int
#     characteristic_h::Float64
#     h_min::Float64
#     h_max::Float64
#     dt::Float64
#     nsteps::Int
#     elapsed_seconds::Float64
#     l2_electric_error::Float64
#     l2_magnetic_error::Float64
#     l2_total_error::Float64
#     l2_ex_error::Float64
#     l2_ey_error::Float64
#     l2_ez_error::Float64
#     l2_hx_error::Float64
#     l2_hy_error::Float64
#     l2_hz_error::Float64
#     relative_total_error::Float64
#     linf_electric_error::Float64
#     linf_magnetic_error::Float64
#     linf_total_error::Float64
#     energy_error::Float64
#     relative_energy_error::Float64
#     electric_charge::Float64
#     magnetic_charge::Float64
#     rate_electric::Union{Missing, Float64}
#     rate_magnetic::Union{Missing, Float64}
#     rate_total::Union{Missing, Float64}
#     rate_ex::Union{Missing, Float64}
#     rate_ey::Union{Missing, Float64}
#     rate_ez::Union{Missing, Float64}
#     rate_hx::Union{Missing, Float64}
#     rate_hy::Union{Missing, Float64}
#     rate_hz::Union{Missing, Float64}
# end

struct DistributedPeriodicConvergenceResult
    mpi_ranks::Int
    boundary_condition::Symbol
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

    # Rates based on characteristic_h
    rate_electric::Union{Missing, Float64}
    rate_magnetic::Union{Missing, Float64}
    rate_total::Union{Missing, Float64}
    rate_max_electric::Union{Missing, Float64}
    rate_max_magnetic::Union{Missing, Float64}
    rate_max_total::Union{Missing, Float64}

    # Additional aggregate rates based on h_min
    rate_electric_hmin::Union{Missing, Float64}
    rate_magnetic_hmin::Union{Missing, Float64}
    rate_total_hmin::Union{Missing, Float64}

    # Additional aggregate rates based on h_max
    rate_electric_hmax::Union{Missing, Float64}
    rate_magnetic_hmax::Union{Missing, Float64}
    rate_total_hmax::Union{Missing, Float64}

    # Component rates based on characteristic_h
    rate_ex::Union{Missing, Float64}
    rate_ey::Union{Missing, Float64}
    rate_ez::Union{Missing, Float64}
    rate_hx::Union{Missing, Float64}
    rate_hy::Union{Missing, Float64}
    rate_hz::Union{Missing, Float64}
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
                      N, 2N, 4N, and 8N. Default: 2
  --levels=L          Number of uniform h-refinement levels. Default: 4
  --nx-targets=a,b    Explicit NxTarget family. Consecutive entries must
                      double. This overrides --nx-target and --levels.
  --geo=PATH          Parameterized periodic Gmsh geometry for unstructured
                      meshes. Default: examples/meshes/periodic_box_unstructured.geo
  --mesh-dir=PATH     Generated unstructured VTK mesh directory. Default:
                      <output-directory>/periodic_convergence_meshes
  --orders=a,b,c      DG polynomial orders in 2:5. Default: 2,3,4
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
  --output=PATH       Output CSV. Default:
                      output/convergence_distributed_periodic_poisson_bracket.csv
  --help              Show this message.

Method:
  - axis-aligned periodic box from either structured or Gmsh meshes
  - periodic boundary pairs 1<->2, 3<->4, and 5<->6
  - x-directed analytical plane wave
  - structured or unstructured mesh family with target h, h/2, h/4, and h/8
  - PoissonBracketFormulation with centered flux
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
            base_nx_target = parse(Int, split(arg, "=", limit = 2)[2])
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
    length(nx_targets) >= 2 ||
        throw(ArgumentError("At least two mesh levels are required."))
    all(>=(1), nx_targets) ||
        throw(ArgumentError("NxTarget values must be positive."))
    all(
        level -> nx_targets[level + 1] == 2 * nx_targets[level],
        1:(length(nx_targets) - 1),
    ) ||
        throw(
            ArgumentError(
                "NxTarget values must define uniform halving: N,2N,4N,...",
            ),
        )
    if mesh_family == :structured
        all(iseven, nx_targets) ||
            throw(
                ArgumentError(
                    "Structured periodic NxTarget values must be even so ny=nz=nx/2 is integral.",
                ),
            )
    end
    all(order -> 2 <= order <= 5, orders) ||
        throw(
            ArgumentError(
                "--orders must be in 2:5. Order 1 aliases the plane wave on " *
                "coarse structured levels, and ESPRK order is N+1.",
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
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))
    wave_number > 0.0 ||
        throw(ArgumentError("--wave-number must be positive."))
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
        geo_path,
        mesh_dir,
        abspath(output),
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
    command = `$(gmsh) $(config.geo_path) -3 -format vtk -bin 0 -nt 1 -v 1 -setnumber NxTarget $(nx_target) -o $(mesh_path)`
    run(command)
    isfile(mesh_path) ||
        error("Gmsh did not create the expected mesh $mesh_path.")
    return mesh_path, load_periodic_mesh(mesh_path)
end

function periodic_convergence_node_id(
    i::Int,
    j::Int,
    k::Int,
    nx::Int,
    ny::Int,
)
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)
end

function build_structured_periodic_convergence_points(nx::Int)
    nx >= 2 && iseven(nx) ||
        throw(ArgumentError("Structured periodic nx must be even and at least 2."))
    ny = nx ÷ 2
    nz = nx ÷ 2
    points = zeros(Float64, 3, (nx + 1) * (ny + 1) * (nz + 1))

    for k in 0:nz, j in 0:ny, i in 0:nx
        node = periodic_convergence_node_id(i, j, k, nx, ny)
        points[:, node] .= (2.0 * i / nx, j / ny, k / nz)
    end
    return points
end

function build_structured_periodic_convergence_tets(nx::Int)
    nx >= 2 && iseven(nx) ||
        throw(ArgumentError("Structured periodic nx must be even and at least 2."))
    ny = nx ÷ 2
    nz = nx ÷ 2
    tetrahedra = NTuple{4, Int}[]

    for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
        v000 = periodic_convergence_node_id(i, j, k, nx, ny)
        v100 = periodic_convergence_node_id(i + 1, j, k, nx, ny)
        v010 = periodic_convergence_node_id(i, j + 1, k, nx, ny)
        v110 = periodic_convergence_node_id(i + 1, j + 1, k, nx, ny)
        v001 = periodic_convergence_node_id(i, j, k + 1, nx, ny)
        v101 = periodic_convergence_node_id(i + 1, j, k + 1, nx, ny)
        v011 = periodic_convergence_node_id(i, j + 1, k + 1, nx, ny)
        v111 = periodic_convergence_node_id(i + 1, j + 1, k + 1, nx, ny)

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

function build_structured_periodic_convergence_mesh(nx::Int)
    points = build_structured_periodic_convergence_points(nx)
    tets = build_structured_periodic_convergence_tets(nx)
    box = periodic_box(points)
    tolerance = max(1e-10, 1e-10 * maximum(periodic_box_lengths(box)))
    tris = build_boundary_tris(tets)
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))
    boundary_ids = zeros(Int, ntets + ntris)
    for triangle in axes(tris, 2)
        boundary_ids[ntets + triangle] = periodic_boundary_id(
            points,
            @view(tris[:, triangle]);
            box = box,
            tolerance = tolerance,
        )
    end

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

function periodic_balanced_spatial_partition(
    mesh::RawVTUMesh,
    nranks::Int,
)
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

function periodic_distributed_characteristic_h(
    distributed_dg::DistributedDGDiscretization,
)
    local_volume = 0.0
    for elem in distributed_dg.distributed_mesh.partition.owned
        local_volume +=
            PERIODIC_REFERENCE_TET_VOLUME *
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
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    nsteps::Int;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
    cubature_order::Int,
    track_time_l2_maxima::Bool = true,
    compute_time_error_norms::Bool = false,
)
    workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    formulation = PoissonBracketFormulation()
    max_l2_electric = NaN
    max_l2_magnetic = NaN
    max_l2_total = NaN
    previous_time_errors = nothing
    time_l2_squared = zeros(Float64, 9)
    time_linf = zeros(Float64, 9)
    if track_time_l2_maxima
        initial_diagnostics = distributed_quadrature_diagnostics(
            U,
            distributed_dg,
            0.0,
            cubature_order;
            epsilon = epsilon,
            mu = mu,
            wave = wave,
        )
        max_l2_electric = initial_diagnostics.electric_error_l2
        max_l2_magnetic = initial_diagnostics.magnetic_error_l2
        max_l2_total = initial_diagnostics.field_error_l2
    end
    if compute_time_error_norms
        error_workspace = DistributedMaxwellComponentL2Workspace(
            distributed_dg,
            cubature_order,
        )
        previous_time_errors = periodic_space_l2_error_vector(
            error_workspace,
            U,
            distributed_dg,
            0.0,
            cubature_order;
            epsilon = epsilon,
            mu = mu,
            wave = wave,
        )
        time_linf .= previous_time_errors
    end

    for step in 1:nsteps
        distributed_periodic_partitioned_symplectic_rk_step!(
            U,
            workspace,
            scheme,
            dt,
            distributed_dg,
            periodic,
            registry,
            formulation;
            ε = epsilon,
            μ = mu,
        )
        if track_time_l2_maxima
            diagnostics = distributed_quadrature_diagnostics(
                U,
                distributed_dg,
                step * dt,
                cubature_order;
                epsilon = epsilon,
                mu = mu,
                wave = wave,
            )
            max_l2_electric = max(
                max_l2_electric,
                diagnostics.electric_error_l2,
            )
            max_l2_magnetic = max(
                max_l2_magnetic,
                diagnostics.magnetic_error_l2,
            )
            max_l2_total = max(max_l2_total, diagnostics.field_error_l2)
        end
        if compute_time_error_norms
            current_time_errors = periodic_space_l2_error_vector(
                error_workspace,
                U,
                distributed_dg,
                step * dt,
                cubature_order;
                epsilon = epsilon,
                mu = mu,
                wave = wave,
            )
            periodic_time_error_update!(
                time_l2_squared,
                time_linf,
                previous_time_errors,
                current_time_errors,
                dt,
            )
            previous_time_errors = current_time_errors
        end
    end
    return (
        electric = max_l2_electric,
        magnetic = max_l2_magnetic,
        total = max_l2_total,
        time_l2 = compute_time_error_norms ? Tuple(sqrt.(time_l2_squared)) : nothing,
        time_linf = compute_time_error_norms ? Tuple(time_linf) : nothing,
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
    nsteps_override::Union{Nothing, Int} = nothing,
    dt_limit_out = nothing,
    track_time_l2_maxima::Bool = true,
    time_error_norms_out = nothing,
    enforce_cfl_limit::Bool = true,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_partition =
        rank == 0 ?
        periodic_balanced_spatial_partition(root_mesh, nranks) :
        nothing
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
            cubature_order = cubature_order,
            track_time_l2_maxima = track_time_l2_maxima,
            compute_time_error_norms = time_error_norms_out !== nothing,
        )
    end
    if time_error_norms_out !== nothing
        time_error_norms_out[] = (
            l2 = time_l2_maxima[].time_l2,
            linf = time_l2_maxima[].time_linf,
        )
    end
    elapsed = MPI.Allreduce(local_elapsed, max, comm)

    diagnostics = distributed_quadrature_diagnostics(
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
    relative_energy_error =
        (final_energy.total - initial_energy.total) /
        max(initial_energy.total, eps(Float64))
    l2_maxima = track_time_l2_maxima ? time_l2_maxima[] : (
        electric = diagnostics.electric_error_l2,
        magnetic = diagnostics.magnetic_error_l2,
        total = diagnostics.field_error_l2,
    )

    characteristic_h = periodic_distributed_characteristic_h(distributed_dg)
    quality = distributed_mesh_quality_metrics(distributed_dg)

    result = DistributedPeriodicConvergenceResult(
        nranks,
        :periodic,
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
        component_errors...,
        diagnostics.field_relative_error,
        linf_electric,
        linf_magnetic,
        linf_total,
        diagnostics.total_energy - diagnostics.exact_total_energy,
        relative_energy_error,
        diagnostics.electric_charge,
        diagnostics.magnetic_charge,
        # missing,
        # missing,
        # missing,
        # missing,
        # missing,
        # missing,
        # missing,
        # missing,
        # missing,
        missing, # rate_electric
        missing, # rate_magnetic
        missing, # rate_total
        missing, # rate_max_electric
        missing, # rate_max_magnetic
        missing, # rate_max_total
        missing, # rate_electric_hmin
        missing, # rate_magnetic_hmin
        missing, # rate_total_hmin
        missing, # rate_electric_hmax
        missing, # rate_magnetic_hmax
        missing, # rate_total_hmax
        missing, # rate_ex
        missing, # rate_ey
        missing, # rate_ez
        missing, # rate_hx
        missing, # rate_hy
        missing, # rate_hz
    )

    MPI.Barrier(comm)
    return result
end

# function periodic_convergence_rate(
#     fine_error::Float64,
#     coarse_error::Float64,
#     fine_h::Float64,
#     coarse_h::Float64,
# )
#     fine_error > 0.0 && coarse_error > 0.0 || return missing
#     return log(coarse_error / fine_error) / log(coarse_h / fine_h)
# end

function periodic_convergence_rate(
    fine_error::Float64,
    coarse_error::Float64,
    fine_h::Float64,
    coarse_h::Float64,
)
    fine_error > 0.0 && coarse_error > 0.0 || return missing
    fine_h > 0.0 && coarse_h > 0.0 || return missing
    coarse_h != fine_h || return missing

    return log(coarse_error / fine_error) / log(coarse_h / fine_h)
end

function add_periodic_convergence_rates(
    results::Vector{DistributedPeriodicConvergenceResult},
)
    rated = DistributedPeriodicConvergenceResult[]

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
            rate_max_electric = missing
            rate_max_magnetic = missing
            rate_max_total = missing
            rate_electric_hmin = missing
            rate_magnetic_hmin = missing
            rate_total_hmin = missing
            rate_electric_hmax = missing
            rate_magnetic_hmax = missing
            rate_total_hmax = missing
            component_rates = ntuple(_ -> missing, 6)
            if previous !== nothing
                rate_electric = periodic_convergence_rate(
                    result.l2_electric_error,
                    previous.l2_electric_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_magnetic = periodic_convergence_rate(
                    result.l2_magnetic_error,
                    previous.l2_magnetic_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_total = periodic_convergence_rate(
                    result.l2_total_error,
                    previous.l2_total_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_max_electric = periodic_convergence_rate(
                    result.max_l2_electric_error,
                    previous.max_l2_electric_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_max_magnetic = periodic_convergence_rate(
                    result.max_l2_magnetic_error,
                    previous.max_l2_magnetic_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_max_total = periodic_convergence_rate(
                    result.max_l2_total_error,
                    previous.max_l2_total_error,
                    result.characteristic_h,
                    previous.characteristic_h,
                )
                rate_electric_hmin = periodic_convergence_rate(
                    result.l2_electric_error,
                    previous.l2_electric_error,
                    result.h_min,
                    previous.h_min,
                )
                rate_magnetic_hmin = periodic_convergence_rate(
                    result.l2_magnetic_error,
                    previous.l2_magnetic_error,
                    result.h_min,
                    previous.h_min,
                )
                rate_total_hmin = periodic_convergence_rate(
                    result.l2_total_error,
                    previous.l2_total_error,
                    result.h_min,
                    previous.h_min,
                )

                rate_electric_hmax = periodic_convergence_rate(
                    result.l2_electric_error,
                    previous.l2_electric_error,
                    result.h_max,
                    previous.h_max,
                )
                rate_magnetic_hmax = periodic_convergence_rate(
                    result.l2_magnetic_error,
                    previous.l2_magnetic_error,
                    result.h_max,
                    previous.h_max,
                )
                rate_total_hmax = periodic_convergence_rate(
                    result.l2_total_error,
                    previous.l2_total_error,
                    result.h_max,
                    previous.h_max,
                )
                fine_components = (
                    result.l2_ex_error,
                    result.l2_ey_error,
                    result.l2_ez_error,
                    result.l2_hx_error,
                    result.l2_hy_error,
                    result.l2_hz_error,
                )
                coarse_components = (
                    previous.l2_ex_error,
                    previous.l2_ey_error,
                    previous.l2_ez_error,
                    previous.l2_hx_error,
                    previous.l2_hy_error,
                    previous.l2_hz_error,
                )
                component_rates = ntuple(
                    component ->
                        coarse_components[component] > PERIODIC_COMPONENT_RATE_ERROR_FLOOR ?
                        periodic_convergence_rate(
                            fine_components[component],
                            coarse_components[component],
                            result.characteristic_h,
                            previous.characteristic_h,
                        ) :
                        missing,
                    6,
                )
            end

            push!(
                rated,
                DistributedPeriodicConvergenceResult(
                    result.mpi_ranks,
                    result.boundary_condition,
                    result.mesh_family,
                    result.order,
                    result.esprk_order,
                    result.cubature_order,
                    result.mesh_level,
                    result.nx_target,
                    result.target_h,
                    result.nelements,
                    result.min_owned_elements,
                    result.max_owned_elements,
                    result.characteristic_h,
                    result.h_min,
                    result.h_max,
                    result.h_ratio,
                    result.volume_min,
                    result.volume_max,
                    result.volume_total,
                    result.volume_ratio,
                    result.edge_min,
                    result.edge_max,
                    result.edge_ratio,
                    result.mean_ratio_min,
                    result.mean_ratio_avg,
                    result.dt,
                    result.nsteps,
                    result.elapsed_seconds,
                    result.l2_electric_error,
                    result.l2_magnetic_error,
                    result.l2_total_error,
                    result.max_l2_electric_error,
                    result.max_l2_magnetic_error,
                    result.max_l2_total_error,
                    result.l2_ex_error,
                    result.l2_ey_error,
                    result.l2_ez_error,
                    result.l2_hx_error,
                    result.l2_hy_error,
                    result.l2_hz_error,
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
                    rate_max_electric,
                    rate_max_magnetic,
                    rate_max_total,
                    rate_electric_hmin,
                    rate_magnetic_hmin,
                    rate_total_hmin,
                    rate_electric_hmax,
                    rate_magnetic_hmax,
                    rate_total_hmax,
                    component_rates...,
                ),
            )
            previous = result
        end
    end
    return rated
end

function formatted_periodic_rate(rate::Union{Missing, Float64})
    return ismissing(rate) ? "-" : @sprintf("%.3f", rate)
end

function periodic_rate_pass(
    rate::Union{Missing, Float64},
    expected::Float64,
    tolerance::Float64,
)
    ismissing(rate) && return false
    return isfinite(rate) && rate >= expected - tolerance
end

function formatted_periodic_rate_verdict(
    rate::Union{Missing, Float64},
    expected::Float64,
    tolerance::Float64,
)
    return periodic_rate_pass(rate, expected, tolerance) ? "PASS" : "FAIL"
end

function print_periodic_convergence_results(
    results::Vector{DistributedPeriodicConvergenceResult},
)
    println()
    println("Distributed periodic Poisson-bracket Maxwell convergence")
    println("---------------------------------------------------------")
    println(
        rpad("N", 4),
        rpad("ESPRK", 7),
        rpad("mesh", 13),
        rpad("level", 7),
        rpad("NxTarget", 10),
        rpad("Ne", 9),
        rpad("owned", 11),
        rpad("h target", 12),
        rpad("h char", 12),
        rpad("h min", 12),
        rpad("h max", 12),
        rpad("q min", 10),
        rpad("dt", 12),
        rpad("steps", 7),
        rpad("L2 E", 13),
        rpad("L2 H", 13),
        rpad("L2 total", 13),
        rpad("max_t E", 13),
        rpad("max_t H", 13),
        rpad("max_t tot", 13),
        rpad("rate E", 9),
        rpad("rate H", 9),
        rpad("r max E", 9),
        rpad("r max H", 9),
        rpad("rate", 9),
        "seconds",
    )

    for result in results
        owned =
            "$(result.min_owned_elements):$(result.max_owned_elements)"
        println(
            rpad(string(result.order), 4),
            rpad(string(result.esprk_order), 7),
            rpad(string(result.mesh_family), 13),
            rpad(string(result.mesh_level), 7),
            rpad(string(result.nx_target), 10),
            rpad(string(result.nelements), 9),
            rpad(owned, 11),
            rpad(@sprintf("%.3e", result.target_h), 12),
            rpad(@sprintf("%.3e", result.characteristic_h), 12),
            rpad(@sprintf("%.3e", result.h_min), 12),
            rpad(@sprintf("%.3e", result.h_max), 12),
            rpad(@sprintf("%.3f", result.mean_ratio_min), 10),
            rpad(@sprintf("%.3e", result.dt), 12),
            rpad(string(result.nsteps), 7),
            rpad(@sprintf("%.3e", result.l2_electric_error), 13),
            rpad(@sprintf("%.3e", result.l2_magnetic_error), 13),
            rpad(@sprintf("%.3e", result.l2_total_error), 13),
            rpad(@sprintf("%.3e", result.max_l2_electric_error), 13),
            rpad(@sprintf("%.3e", result.max_l2_magnetic_error), 13),
            rpad(@sprintf("%.3e", result.max_l2_total_error), 13),
            rpad(formatted_periodic_rate(result.rate_electric), 9),
            rpad(formatted_periodic_rate(result.rate_magnetic), 9),
            rpad(formatted_periodic_rate(result.rate_max_electric), 9),
            rpad(formatted_periodic_rate(result.rate_max_magnetic), 9),
            rpad(formatted_periodic_rate(result.rate_total), 9),
            @sprintf("%.3f", result.elapsed_seconds),
        )
    end

    println()
    println("Component-wise L2 errors and characteristic-h rates")
    println("----------------------------------------------------")
    println(
        rpad("N", 4),
        rpad("level", 7),
        rpad("L2 Ex", 13),
        rpad("r Ex", 8),
        rpad("L2 Ey", 13),
        rpad("r Ey", 8),
        rpad("L2 Ez", 13),
        rpad("r Ez", 8),
        rpad("L2 Hx", 13),
        rpad("r Hx", 8),
        rpad("L2 Hy", 13),
        rpad("r Hy", 8),
        rpad("L2 Hz", 13),
        "r Hz",
    )
    for result in results
        println(
            rpad(string(result.order), 4),
            rpad(string(result.mesh_level), 7),
            rpad(@sprintf("%.3e", result.l2_ex_error), 13),
            rpad(formatted_periodic_rate(result.rate_ex), 8),
            rpad(@sprintf("%.3e", result.l2_ey_error), 13),
            rpad(formatted_periodic_rate(result.rate_ey), 8),
            rpad(@sprintf("%.3e", result.l2_ez_error), 13),
            rpad(formatted_periodic_rate(result.rate_ez), 8),
            rpad(@sprintf("%.3e", result.l2_hx_error), 13),
            rpad(formatted_periodic_rate(result.rate_hx), 8),
            rpad(@sprintf("%.3e", result.l2_hy_error), 13),
            rpad(formatted_periodic_rate(result.rate_hy), 8),
            rpad(@sprintf("%.3e", result.l2_hz_error), 13),
            formatted_periodic_rate(result.rate_hz),
        )
    end

    println()
    println("Plane-wave transverse leakage")
    println("-----------------------------")
    println(
        rpad("N", 4),
        rpad("level", 7),
        rpad("h char", 12),
        rpad("L2 E transverse", 18),
        "L2 H transverse",
    )
    for result in results
        electric_transverse =
            hypot(result.l2_ex_error, result.l2_ey_error)
        magnetic_transverse =
            hypot(result.l2_hx_error, result.l2_hz_error)
        println(
            rpad(string(result.order), 4),
            rpad(string(result.mesh_level), 7),
            rpad(@sprintf("%.3e", result.characteristic_h), 12),
            rpad(@sprintf("%.3e", electric_transverse), 18),
            @sprintf("%.3e", magnetic_transverse),
        )
    end

    println()
    println("Aggregate convergence rates by mesh-size definition")
    println("---------------------------------------------------")
    println(
        rpad("N", 4),
        rpad("level", 7),
        rpad("E char", 10),
        rpad("E hmin", 10),
        rpad("E hmax", 10),
        rpad("H char", 10),
        rpad("H hmin", 10),
        rpad("H hmax", 10),
        rpad("tot char", 10),
        rpad("tot hmin", 10),
        "tot hmax",
    )

    for result in results
        println(
            rpad(string(result.order), 4),
            rpad(string(result.mesh_level), 7),
            rpad(formatted_periodic_rate(result.rate_electric), 10),
            rpad(formatted_periodic_rate(result.rate_electric_hmin), 10),
            rpad(formatted_periodic_rate(result.rate_electric_hmax), 10),
            rpad(formatted_periodic_rate(result.rate_magnetic), 10),
            rpad(formatted_periodic_rate(result.rate_magnetic_hmin), 10),
            rpad(formatted_periodic_rate(result.rate_magnetic_hmax), 10),
            rpad(formatted_periodic_rate(result.rate_total), 10),
            rpad(formatted_periodic_rate(result.rate_total_hmin), 10),
            formatted_periodic_rate(result.rate_total_hmax),
        )
    end

    first_order = minimum(result.order for result in results)
    mesh_rows = sort(
        filter(result -> result.order == first_order, results);
        by = result -> result.mesh_level,
    )
    println()
    println("Mesh-size ratios used by the rate calculation")
    println("---------------------------------------------")
    for index in 2:length(mesh_rows)
        coarse = mesh_rows[index - 1]
        fine = mesh_rows[index]
        @printf(
            "level %d -> %d: target ratio %.6f, characteristic ratio %.6f, h_min ratio %.6f, h_max ratio %.6f\n",
            coarse.mesh_level,
            fine.mesh_level,
            coarse.target_h / fine.target_h,
            coarse.characteristic_h / fine.characteristic_h,
            coarse.h_min / fine.h_min,
            coarse.h_max / fine.h_max,
        )
    end
end

function periodic_convergence_csv_header()
    return (
        "mpi_ranks,boundary_condition,mesh_family,order,esprk_order,cubature_order,mesh_level," *
        "nx_target,target_h,nelements,min_owned_elements,max_owned_elements," *
        "characteristic_h,h_min,h_max,h_ratio,volume_min,volume_max," *
        "volume_total,volume_ratio,edge_min,edge_max,edge_ratio," *
        "mean_ratio_min,mean_ratio_avg,dt,nsteps,elapsed_seconds,l2_electric_error," *
        "l2_magnetic_error,l2_total_error," *
        "max_l2_electric_error,max_l2_magnetic_error,max_l2_total_error," *
        "l2_ex_error,l2_ey_error,l2_ez_error,l2_hx_error,l2_hy_error,l2_hz_error," *
        "relative_total_error," *
        "linf_electric_error,linf_magnetic_error,linf_total_error," *
        "energy_error,relative_energy_error,electric_charge," *
        # "magnetic_charge,rate_electric,rate_magnetic,rate_total," *
        # "rate_ex,rate_ey,rate_ez,rate_hx,rate_hy,rate_hz"
        "magnetic_charge,rate_electric,rate_magnetic,rate_total," *
        "rate_max_electric,rate_max_magnetic,rate_max_total," *
        "rate_electric_hmin,rate_magnetic_hmin,rate_total_hmin," *
        "rate_electric_hmax,rate_magnetic_hmax,rate_total_hmax," *
        "rate_ex,rate_ey,rate_ez,rate_hx,rate_hy,rate_hz"
    )
end

function write_periodic_convergence_results(
    path::String,
    results::Vector{DistributedPeriodicConvergenceResult},
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, periodic_convergence_csv_header())
        for result in results
            values = (
                result.mpi_ranks,
                result.boundary_condition,
                result.mesh_family,
                result.order,
                result.esprk_order,
                result.cubature_order,
                result.mesh_level,
                result.nx_target,
                result.target_h,
                result.nelements,
                result.min_owned_elements,
                result.max_owned_elements,
                result.characteristic_h,
                result.h_min,
                result.h_max,
                result.h_ratio,
                result.volume_min,
                result.volume_max,
                result.volume_total,
                result.volume_ratio,
                result.edge_min,
                result.edge_max,
                result.edge_ratio,
                result.mean_ratio_min,
                result.mean_ratio_avg,
                result.dt,
                result.nsteps,
                result.elapsed_seconds,
                result.l2_electric_error,
                result.l2_magnetic_error,
                result.l2_total_error,
                result.max_l2_electric_error,
                result.max_l2_magnetic_error,
                result.max_l2_total_error,
                result.l2_ex_error,
                result.l2_ey_error,
                result.l2_ez_error,
                result.l2_hx_error,
                result.l2_hy_error,
                result.l2_hz_error,
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
                ismissing(result.rate_max_electric) ? "" : result.rate_max_electric,
                ismissing(result.rate_max_magnetic) ? "" : result.rate_max_magnetic,
                ismissing(result.rate_max_total) ? "" : result.rate_max_total,
                ismissing(result.rate_electric_hmin) ? "" : result.rate_electric_hmin,
                ismissing(result.rate_magnetic_hmin) ? "" : result.rate_magnetic_hmin,
                ismissing(result.rate_total_hmin) ? "" : result.rate_total_hmin,
                ismissing(result.rate_electric_hmax) ? "" : result.rate_electric_hmax,
                ismissing(result.rate_magnetic_hmax) ? "" : result.rate_magnetic_hmax,
                ismissing(result.rate_total_hmax) ? "" : result.rate_total_hmax,
                ismissing(result.rate_ex) ? "" : result.rate_ex,
                ismissing(result.rate_ey) ? "" : result.rate_ey,
                ismissing(result.rate_ez) ? "" : result.rate_ez,
                ismissing(result.rate_hx) ? "" : result.rate_hx,
                ismissing(result.rate_hy) ? "" : result.rate_hy,
                ismissing(result.rate_hz) ? "" : result.rate_hz,
            )
            println(io, join(values, ','))
        end
    end
    return path
end

function periodic_convergence_verdict(
    results::Vector{DistributedPeriodicConvergenceResult},
)
    verdict = true
    messages = String[]

    for order in sort(unique(result.order for result in results))
        subset = sort(
            filter(result -> result.order == order, results);
            by = result -> result.mesh_level,
        )
        finest = subset[end]
        expected_electric = order + 1.0
        expected_magnetic = Float64(order)
        ez_pass = periodic_rate_pass(
            finest.rate_ez,
            expected_electric,
            PERIODIC_CONVERGENCE_RATE_TOLERANCE,
        )
        hy_pass = periodic_rate_pass(
            finest.rate_hy,
            expected_magnetic,
            PERIODIC_CONVERGENCE_RATE_TOLERANCE,
        )
        verdict &= ez_pass && hy_pass
        push!(
            messages,
            @sprintf(
                "N=%d physical components: Ez rate=%s, target %.1f: %s; Hy rate=%s, target %.1f: %s",
                order,
                formatted_periodic_rate(finest.rate_ez),
                expected_electric,
                formatted_periodic_rate_verdict(
                    finest.rate_ez,
                    expected_electric,
                    PERIODIC_CONVERGENCE_RATE_TOLERANCE,
                ),
                formatted_periodic_rate(finest.rate_hy),
                expected_magnetic,
                formatted_periodic_rate_verdict(
                    finest.rate_hy,
                    expected_magnetic,
                    PERIODIC_CONVERGENCE_RATE_TOLERANCE,
                ),
            ),
        )
        push!(
            messages,
            @sprintf(
                "N=%d aggregate rates: final E=%s, final H=%s; max_t E=%s, max_t H=%s",
                order,
                formatted_periodic_rate(finest.rate_electric),
                formatted_periodic_rate(finest.rate_magnetic),
                formatted_periodic_rate(finest.rate_max_electric),
                formatted_periodic_rate(finest.rate_max_magnetic),
            ),
        )
    end
    return verdict, messages
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
        println("target h ratios:    1, 1/2, 1/4, ...")
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
        println("ESPRK rule:         order N+1, H-first")
        println("cubature rule:      Jaskowiec-Sukumar max(2,2N+4)")
        println("analytical mode:    traveling plane wave along +x")
        println("output:             ", config.output)
        println()
    end

    results = DistributedPeriodicConvergenceResult[]
    root_meshes = Dict{Int, Union{Nothing, RawVTUMesh}}()
    for nx_target in config.nx_targets
        _, root_mesh =
            root_periodic_convergence_mesh(config, nx_target, comm)
        root_meshes[nx_target] = root_mesh
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
                    root_meshes[nx_target],
                ),
            )
        end
    end

    rated_results = add_periodic_convergence_rates(results)
    if rank == 0
        print_periodic_convergence_results(rated_results)
        write_periodic_convergence_results(config.output, rated_results)
        verdict, messages = periodic_convergence_verdict(rated_results)
        println()
        println("Convergence checks")
        println("------------------")
        foreach(println, messages)
        println(
            "Overall expected-rate convergence: ",
            verdict ? "PASS" : "FAIL",
        )
        println("Wrote CSV: ", config.output)
    end
    return rated_results
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
