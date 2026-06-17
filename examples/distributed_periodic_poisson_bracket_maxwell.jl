#!/usr/bin/env julia

# Distributed periodic Maxwell traveling-wave experiment using the
# Poisson-bracket formulation.
#
# From the DiscoGMPI repository root:
#   mpiexec -n 2 julia --project=. \
#     examples/distributed_periodic_poisson_bracket_maxwell.jl --partitions=2

using MPI
using DiscoGMPI
using LinearAlgebra: dot
using Printf
using WriteVTK:
    MeshCell,
    VTKCellTypes,
    VTKCellData,
    VTKFieldData,
    VTKPointData,
    pvtk_grid

include(
    joinpath(
        @__DIR__,
        "..",
        "src",
        "solver",
        "kernels",
        "JaskowiecSukumar.jl",
    ),
)

const DEFAULT_WAVE_NUMBER = 2.0 * pi
const MAGNETIC_AMPLITUDE = 1.0
const TET_FACE_NODE_IDS = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)
const REF_TET_VERTEX_COORDS = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
)

struct ExperimentConfig
    mesh_path::String
    partition_path::String
    partition_count::Int
    repartition::Bool
    distributed_mesh_dir::String
    rebuild_distributed_mesh::Bool
    output_dir::String
    polynomial_order::Int
    esprk_order::Int
    final_time::Float64
    cfl::Float64
    epsilon::Float64
    mu::Float64
    wave_number::Float64
    energy_every::Int
    cubature_order::Int
    paraview_every::Int
    checkpoint_every::Int
    checkpoint_dir::String
    restart_path::String
end

struct PeriodicBox
    lower::NTuple{3, Float64}
    upper::NTuple{3, Float64}
end

struct PlaneWaveParameters
    wave_number::Float64
    angular_frequency::Float64
    magnetic_amplitude::Float64
    x_origin::Float64
end

struct MaxwellQuadratureDiagnostics
    cubature_order::Int
    electric_energy::Float64
    magnetic_energy::Float64
    total_energy::Float64
    exact_electric_energy::Float64
    exact_magnetic_energy::Float64
    exact_total_energy::Float64
    electric_l2::Float64
    exact_electric_l2::Float64
    electric_error_l2::Float64
    electric_relative_error::Float64
    magnetic_l2::Float64
    exact_magnetic_l2::Float64
    magnetic_error_l2::Float64
    magnetic_relative_error::Float64
    field_error_l2::Float64
    field_relative_error::Float64
    energy_density_l2::Float64
    exact_energy_density_l2::Float64
    energy_density_error_l2::Float64
    energy_density_relative_error::Float64
    optical_chirality::Float64
    exact_optical_chirality::Float64
    optical_chirality_error::Float64
    optical_chirality_density_l2::Float64
    exact_optical_chirality_density_l2::Float64
    optical_chirality_density_error_l2::Float64
    electric_charge::Float64
    magnetic_charge::Float64
    exact_electric_charge::Float64
    exact_magnetic_charge::Float64
    linear_momentum_x::Float64
    linear_momentum_y::Float64
    linear_momentum_z::Float64
    exact_linear_momentum_x::Float64
    exact_linear_momentum_y::Float64
    exact_linear_momentum_z::Float64
    angular_momentum_x::Float64
    angular_momentum_y::Float64
    angular_momentum_z::Float64
    exact_angular_momentum_x::Float64
    exact_angular_momentum_y::Float64
    exact_angular_momentum_z::Float64
end

function usage(io::IO = stdout)
    println(io, """
Distributed periodic Poisson-bracket Maxwell experiment

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_periodic_poisson_bracket_maxwell.jl [options]

Options:
  --mesh PATH           Legacy tetrahedral VTK mesh of an axis-aligned box
                        with translationally matching opposite surfaces.
                        Default: examples/meshes/periodic_box_2x1x1_nx4.vtk
  --partitions N        Number of METIS partitions. It must equal the number
                        of MPI ranks launched by mpiexec. Default: MPI ranks.
  --partition PATH      Zero-based METIS element partition.
                        When omitted, rank zero creates one automatically.
  --repartition         Regenerate the METIS partition even if it exists.
  --distributed-mesh-dir PATH
                        Rank-local mesh cache. Each rank independently loads
                        mesh_rankNNNN.bin after the cache is prepared.
                        Default: output/distributed_mesh_cache/<mesh>_ranksN
  --rebuild-distributed-mesh
                        Recreate the rank-local mesh cache from --mesh and
                        the partition before running.
  --output-dir PATH     Output directory.
                        Default: output/distributed_periodic_poisson_bracket
  --order N             DG polynomial order (N >= 2). Order 1 aliases the
                        prescribed sine wave to zero on this mesh. Default: 2
  --esprk-order N       ESPRK order in 1:6. Default: 4
  --final-time T        Simulation end time. Default: 0.25
  --cfl C               CFL used to estimate dt. Default: 0.05
  --epsilon VALUE       Electric permittivity. The supplied analytical mode
                        requires 1.0. Default: 1.0
  --mu VALUE            Magnetic permeability. The supplied analytical mode
                        requires 1.0. Default: 1.0
  --wave-number VALUE   Positive x-directed wave number. It must contain an
                        integer number of wavelengths in the mesh x extent.
                        Default: 2*pi
  --energy-every N      Write energy every N steps. Default: 1
  --cubature-order N    Jaskowiec-Sukumar volume cubature order.
                        Default: max(2, 2 * DG order + 4)
  --paraview-every N    Write a ParaView snapshot every N steps, plus the
                        initial and final states. Default: 10
  --checkpoint-every N  Write a restart checkpoint every N steps.
                        Zero disables periodic checkpoints; the final state
                        is always checkpointed. Default: 0
  --checkpoint-dir PATH Checkpoint root directory.
                        Default: <output-dir>/checkpoints
  --restart PATH        Restart from a checkpoint directory containing
                        checkpoint_manifest.toml. --final-time remains the
                        absolute target time.
  --help                Show this message.

Outputs:
  energy.csv                       DG mass-matrix energy history.
  quadrature_diagnostics.csv       Energy, chirality, L2 errors, charges,
                                   and momenta.
  run_metadata.toml                Run configuration and runtime metadata.
  partition_metadata.csv           Per-rank ownership and halo metadata.
  fields.pvd                       ParaView time-series collection.
  paraview_series.csv              Snapshot step/time index.
  fields/fields_stepNNNNNNNN.pvtu  Parallel field snapshot.
  checkpoints/stepNNNNNNNN/        Per-rank restart checkpoint.
  integration_points_rankNNNN.csv  Final integration-point values per rank.
""")
end

function option_value(args::Vector{String}, i::Int, name::String)
    arg = args[i]
    prefix = name * "="

    if startswith(arg, prefix)
        return arg[(length(prefix) + 1):end], i
    end

    arg == name ||
        throw(ArgumentError("Unknown option '$arg'. Use --help for valid options."))
    i < length(args) ||
        throw(ArgumentError("Option '$name' requires a value."))

    return args[i + 1], i + 1
end

function parse_arguments(
    args::Vector{String},
    nranks::Int,
    repository_root::String,
)
    mesh_path = joinpath(
        repository_root,
        "examples",
        "meshes",
        "periodic_box_2x1x1_nx4.vtk",
    )
    partition_path = ""
    partition_count = nranks
    repartition = false
    distributed_mesh_dir = ""
    rebuild_distributed_mesh = false
    output_dir =
        joinpath(
            repository_root,
            "output",
            "distributed_periodic_poisson_bracket",
        )
    polynomial_order = 2
    esprk_order = 4
    final_time = 0.25
    cfl = 0.05
    epsilon = 1.0
    mu = 1.0
    wave_number = DEFAULT_WAVE_NUMBER
    energy_every = 1
    cubature_order = 0
    paraview_every = 10
    checkpoint_every = 0
    checkpoint_dir = ""
    restart_path = ""

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--mesh")
            value, i = option_value(args, i, "--mesh")
            mesh_path = abspath(value)
        elseif startswith(arg, "--partitions")
            value, i = option_value(args, i, "--partitions")
            partition_count = parse(Int, value)
        elseif startswith(arg, "--partition")
            value, i = option_value(args, i, "--partition")
            if tryparse(Int, value) !== nothing
                throw(
                    ArgumentError(
                        "--partition expects a partition-file path, but got " *
                        "'$value'. To request $value MPI partitions, use " *
                        "--partitions=$value.",
                    ),
                )
            end
            partition_path = abspath(value)
        elseif arg == "--repartition"
            repartition = true
        elseif startswith(arg, "--distributed-mesh-dir")
            value, i = option_value(args, i, "--distributed-mesh-dir")
            distributed_mesh_dir = abspath(value)
        elseif arg == "--rebuild-distributed-mesh"
            rebuild_distributed_mesh = true
        elseif startswith(arg, "--output-dir")
            value, i = option_value(args, i, "--output-dir")
            output_dir = abspath(value)
        elseif startswith(arg, "--order")
            value, i = option_value(args, i, "--order")
            polynomial_order = parse(Int, value)
        elseif startswith(arg, "--esprk-order")
            value, i = option_value(args, i, "--esprk-order")
            esprk_order = parse(Int, value)
        elseif startswith(arg, "--final-time")
            value, i = option_value(args, i, "--final-time")
            final_time = parse(Float64, value)
        elseif startswith(arg, "--cfl")
            value, i = option_value(args, i, "--cfl")
            cfl = parse(Float64, value)
        elseif startswith(arg, "--epsilon")
            value, i = option_value(args, i, "--epsilon")
            epsilon = parse(Float64, value)
        elseif startswith(arg, "--mu")
            value, i = option_value(args, i, "--mu")
            mu = parse(Float64, value)
        elseif startswith(arg, "--wave-number")
            value, i = option_value(args, i, "--wave-number")
            wave_number = parse(Float64, value)
        elseif startswith(arg, "--energy-every")
            value, i = option_value(args, i, "--energy-every")
            energy_every = parse(Int, value)
        elseif startswith(arg, "--cubature-order")
            value, i = option_value(args, i, "--cubature-order")
            cubature_order = parse(Int, value)
        elseif startswith(arg, "--paraview-every")
            value, i = option_value(args, i, "--paraview-every")
            paraview_every = parse(Int, value)
        elseif startswith(arg, "--checkpoint-every")
            value, i = option_value(args, i, "--checkpoint-every")
            checkpoint_every = parse(Int, value)
        elseif startswith(arg, "--checkpoint-dir")
            value, i = option_value(args, i, "--checkpoint-dir")
            checkpoint_dir = abspath(value)
        elseif startswith(arg, "--restart")
            value, i = option_value(args, i, "--restart")
            restart_path = abspath(value)
        else
            throw(ArgumentError("Unknown option '$arg'. Use --help for valid options."))
        end

        i += 1
    end

    partition_count >= 1 ||
        throw(ArgumentError("--partitions must be at least 1."))
    partition_count == nranks ||
        throw(
            ArgumentError(
                "--partitions=$partition_count must equal the MPI rank count " *
                "$nranks. Launch with mpiexec -n $partition_count.",
            ),
        )
    if isempty(partition_path)
        mesh_name = splitext(basename(mesh_path))[1]
        partition_path = joinpath(
            repository_root,
            "output",
            "partitions",
            "$(mesh_name).mesh.epart.$partition_count",
        )
    end
    if isempty(distributed_mesh_dir)
        mesh_name = splitext(basename(mesh_path))[1]
        distributed_mesh_dir = joinpath(
            repository_root,
            "output",
            "distributed_mesh_cache",
            "$(mesh_name)_ranks$partition_count",
        )
    end
    if isempty(checkpoint_dir)
        checkpoint_dir = joinpath(output_dir, "checkpoints")
    end

    polynomial_order >= 2 ||
        throw(
            ArgumentError(
                "--order must be at least 2: for this mesh, all order-1 DG " *
                "nodes lie at zeros of sin(2*pi*x).",
            ),
        )
    1 <= esprk_order <= 6 ||
        throw(ArgumentError("--esprk-order must be in 1:6."))
    final_time > 0.0 ||
        throw(ArgumentError("--final-time must be positive."))
    cfl > 0.0 ||
        throw(ArgumentError("--cfl must be positive."))
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))
    wave_number > 0.0 ||
        throw(ArgumentError("--wave-number must be positive."))
    isapprox(epsilon, 1.0; rtol = 0.0, atol = 1e-14) ||
        throw(ArgumentError("The prescribed analytical solution requires epsilon=1."))
    isapprox(mu, 1.0; rtol = 0.0, atol = 1e-14) ||
        throw(ArgumentError("The prescribed analytical solution requires mu=1."))
    energy_every >= 1 ||
        throw(ArgumentError("--energy-every must be at least 1."))
    cubature_order == 0 || 2 <= cubature_order <= 20 ||
        throw(ArgumentError("--cubature-order must be in 2:20."))
    paraview_every >= 1 ||
        throw(ArgumentError("--paraview-every must be at least 1."))
    checkpoint_every >= 0 ||
        throw(ArgumentError("--checkpoint-every must be non-negative."))

    return ExperimentConfig(
        mesh_path,
        partition_path,
        partition_count,
        repartition,
        distributed_mesh_dir,
        rebuild_distributed_mesh,
        output_dir,
        polynomial_order,
        esprk_order,
        final_time,
        cfl,
        epsilon,
        mu,
        wave_number,
        energy_every,
        cubature_order,
        paraview_every,
        checkpoint_every,
        checkpoint_dir,
        restart_path,
    )
end

function sorted_face_key(nodes::NTuple{3, Int})
    values = sort(collect(nodes))
    return (values[1], values[2], values[3])
end

function build_boundary_tris(tets::Matrix{Int})
    counts = Dict{NTuple{3, Int}, Int}()
    oriented_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for elem in axes(tets, 2)
        tet = @view tets[:, elem]

        for ids in TET_FACE_NODE_IDS
            nodes = (tet[ids[1]], tet[ids[2]], tet[ids[3]])
            key = sorted_face_key(nodes)
            counts[key] = get(counts, key, 0) + 1
            oriented_nodes[key] = nodes
        end
    end

    boundary_faces = NTuple{3, Int}[]

    for (key, count) in counts
        if count == 1
            push!(boundary_faces, oriented_nodes[key])
        elseif count != 2
            error("Non-manifold tetrahedral face $key appears $count times.")
        end
    end

    sort!(boundary_faces; by = sorted_face_key)
    return reduce(hcat, collect.(boundary_faces))
end

function periodic_boundary_id(
    points::Matrix{Float64},
    triangle::AbstractVector{Int};
    box::PeriodicBox,
    tolerance::Float64 = 1e-10,
)
    centroid = (
        sum(@view points[1, triangle]) / 3.0,
        sum(@view points[2, triangle]) / 3.0,
        sum(@view points[3, triangle]) / 3.0,
    )
    abs(centroid[1] - box.lower[1]) <= tolerance && return 1
    abs(centroid[1] - box.upper[1]) <= tolerance && return 2
    abs(centroid[2] - box.lower[2]) <= tolerance && return 3
    abs(centroid[2] - box.upper[2]) <= tolerance && return 4
    abs(centroid[3] - box.lower[3]) <= tolerance && return 5
    abs(centroid[3] - box.upper[3]) <= tolerance && return 6

    error(
        "Boundary triangle $(collect(triangle)) at centroid $centroid is not " *
        "on one of the inferred box planes $(box.lower) to $(box.upper). " *
        "The periodic driver requires an axis-aligned box mesh.",
    )
end

function periodic_box(points::Matrix{Float64})
    size(points, 2) > 0 || error("Cannot infer bounds from an empty mesh.")
    lower_values = vec(minimum(points; dims = 2))
    upper_values = vec(maximum(points; dims = 2))
    lower = Tuple(lower_values)
    upper = Tuple(upper_values)
    lengths = ntuple(dimension -> upper[dimension] - lower[dimension], 3)
    all(>(0.0), lengths) ||
        error("The periodic mesh has degenerate coordinate bounds $lower to $upper.")
    return PeriodicBox(lower, upper)
end

function periodic_box_lengths(box::PeriodicBox)
    return ntuple(
        dimension -> box.upper[dimension] - box.lower[dimension],
        3,
    )
end

function load_periodic_mesh(mesh_path::String)
    points, tets = read_mesh_file_tet_vtk(mesh_path)
    size(tets, 2) > 0 ||
        error("The mesh '$mesh_path' contains no tetrahedra.")

    box = periodic_box(points)
    tolerance = max(1e-10, 1e-10 * maximum(periodic_box_lengths(box)))

    tris = build_boundary_tris(tets)
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))
    boundary_ids = zeros(Int, ntets + ntris)
    for triangle in axes(tris, 2)
        boundary_ids[ntets + triangle] =
            periodic_boundary_id(
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

function exact_periodic_wave_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    electric_scale =
        -wave.wave_number * wave.magnetic_amplitude /
        (wave.angular_frequency * epsilon)
    phase = (x, time_value) ->
        wave.wave_number * (x - wave.x_origin) -
        wave.angular_frequency * time_value

    electric = (x, y, z) ->
        (0.0, 0.0, electric_scale * sin(phase(x, time)))
    magnetic = (x, y, z) ->
        (0.0, wave.magnetic_amplitude * sin(phase(x, time)), 0.0)

    return electric, magnetic
end

function exact_periodic_wave_curl_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    electric_scale =
        -wave.wave_number * wave.magnetic_amplitude /
        (wave.angular_frequency * epsilon)
    phase = (x, time_value) ->
        wave.wave_number * (x - wave.x_origin) -
        wave.angular_frequency * time_value

    curl_electric = (x, y, z) ->
        (
            0.0,
            -electric_scale * wave.wave_number * cos(phase(x, time)),
            0.0,
        )
    curl_magnetic = (x, y, z) ->
        (
            0.0,
            0.0,
            wave.wave_number * wave.magnetic_amplitude * cos(phase(x, time)),
        )

    return curl_electric, curl_magnetic
end

function optical_chirality_density(
    electric::NTuple{3, Float64},
    curl_electric::NTuple{3, Float64},
    magnetic::NTuple{3, Float64},
    curl_magnetic::NTuple{3, Float64};
    epsilon::Float64,
    mu::Float64,
)
    return 0.5 * (
        epsilon * dot(electric, curl_electric) +
        mu * dot(magnetic, curl_magnetic)
    )
end

function periodic_boundary_specs(box::PeriodicBox)
    Lx, Ly, Lz = periodic_box_lengths(box)
    return (
        DiscoGMPI.PeriodicBoundarySpec(
            1,
            2,
            (-Lx, 0.0, 0.0),
            :x_periodic,
        ),
        DiscoGMPI.PeriodicBoundarySpec(
            3,
            4,
            (0.0, -Ly, 0.0),
            :y_periodic,
        ),
        DiscoGMPI.PeriodicBoundarySpec(
            5,
            6,
            (0.0, 0.0, -Lz),
            :z_periodic,
        ),
    )
end

function distributed_periodic_box(
    distributed_dg::DistributedDGDiscretization,
)
    points = distributed_dg.dg.mesh.points
    local_lower = vec(minimum(points; dims = 2))
    local_upper = vec(maximum(points; dims = 2))
    global_lower =
        MPI.Allreduce(local_lower, min, distributed_dg.comm)
    global_upper =
        MPI.Allreduce(local_upper, max, distributed_dg.comm)
    return PeriodicBox(Tuple(global_lower), Tuple(global_upper))
end

function configured_plane_wave(
    config::ExperimentConfig,
    box::PeriodicBox,
)
    Lx = periodic_box_lengths(box)[1]
    periods = config.wave_number * Lx / (2.0 * pi)
    nearest_periods = round(Int, periods)
    isapprox(periods, nearest_periods; rtol = 1e-10, atol = 1e-10) ||
        throw(
            ArgumentError(
                "--wave-number=$(config.wave_number) gives $periods periods " *
                "over the inferred x extent $Lx. A periodic analytical wave " *
                "requires an integer number of periods. For one period use " *
                "--wave-number=$(2.0 * pi / Lx); multiply that value by an " *
                "integer for additional periods.",
            ),
        )
    nearest_periods >= 1 ||
        throw(ArgumentError("The periodic wave must contain at least one x period."))

    angular_frequency =
        config.wave_number / sqrt(config.epsilon * config.mu)
    return PlaneWaveParameters(
        config.wave_number,
        angular_frequency,
        MAGNETIC_AMPLITUDE,
        box.lower[1],
    )
end

function resolved_cubature_order(config::ExperimentConfig)
    order = config.cubature_order == 0 ?
            max(2, 2 * config.polynomial_order + 4) :
            config.cubature_order
    order <= 20 ||
        error(
            "DG order $(config.polynomial_order) requires cubature order " *
            "$order, but Jaskowiec-Sukumar rules are available only through 20.",
        )
    return order
end

function reference_interpolation_matrix(
    ref::ReferenceTet,
    cubature_points::Matrix{Float64},
)
    rq = collect(@view cubature_points[:, 1])
    sq = collect(@view cubature_points[:, 2])
    tq = collect(@view cubature_points[:, 3])
    modal_values =
        DiscoGMPI.orthonormal_vandermonde_tet(rq, sq, tq, ref.basis)
    return modal_values * ref.invV
end

function distributed_quadrature_diagnostics(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )
    exact_curl_electric, exact_curl_magnetic =
        exact_periodic_wave_curl_functions(
            time;
            epsilon = epsilon,
            mu = mu,
            wave = wave,
        )

    mesh = distributed_dg.dg.mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    physical_operators = distributed_dg.dg.physops.elements

    # Energies, field norms/errors, chirality, charges, and momenta.
    local_sums = zeros(Float64, 32)

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view mesh.tets[:, elem]
        jacobian = mappings[elem].absdetJ
        operators = physical_operators[elem]

        @views begin
            Ex = U.Ex[:, elem]
            Ey = U.Ey[:, elem]
            Ez = U.Ez[:, elem]
            Hx = U.Hx[:, elem]
            Hy = U.Hy[:, elem]
            Hz = U.Hz[:, elem]
            divergence_electric =
                operators.Dx * Ex +
                operators.Dy * Ey +
                operators.Dz * Ez
            divergence_magnetic =
                operators.Dx * Hx +
                operators.Dy * Hy +
                operators.Dz * Hz
            curl_electric_x =
                operators.Dy * Ez - operators.Dz * Ey
            curl_electric_y =
                operators.Dz * Ex - operators.Dx * Ez
            curl_electric_z =
                operators.Dx * Ey - operators.Dy * Ex
            curl_magnetic_x =
                operators.Dy * Hz - operators.Dz * Hy
            curl_magnetic_y =
                operators.Dz * Hx - operators.Dx * Hz
            curl_magnetic_z =
                operators.Dx * Hy - operators.Dy * Hx

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
                numerical_Ex = dot(row, Ex)
                numerical_Ey = dot(row, Ey)
                numerical_Ez = dot(row, Ez)
                numerical_Hx = dot(row, Hx)
                numerical_Hy = dot(row, Hy)
                numerical_Hz = dot(row, Hz)

                numerical_electric_squared =
                    numerical_Ex^2 + numerical_Ey^2 + numerical_Ez^2
                numerical_magnetic_squared =
                    numerical_Hx^2 + numerical_Hy^2 + numerical_Hz^2
                exact_electric_squared =
                    exact_Ex^2 + exact_Ey^2 + exact_Ez^2
                exact_magnetic_squared =
                    exact_Hx^2 + exact_Hy^2 + exact_Hz^2

                electric_error_squared =
                    (numerical_Ex - exact_Ex)^2 +
                    (numerical_Ey - exact_Ey)^2 +
                    (numerical_Ez - exact_Ez)^2
                magnetic_error_squared =
                    (numerical_Hx - exact_Hx)^2 +
                    (numerical_Hy - exact_Hy)^2 +
                    (numerical_Hz - exact_Hz)^2

                numerical_electric_energy_density =
                    0.5 * epsilon * numerical_electric_squared
                numerical_magnetic_energy_density =
                    0.5 * mu * numerical_magnetic_squared
                numerical_energy_density =
                    numerical_electric_energy_density +
                    numerical_magnetic_energy_density

                exact_electric_energy_density =
                    0.5 * epsilon * exact_electric_squared
                exact_magnetic_energy_density =
                    0.5 * mu * exact_magnetic_squared
                exact_energy_density =
                    exact_electric_energy_density +
                    exact_magnetic_energy_density

                numerical_electric = (
                    numerical_Ex,
                    numerical_Ey,
                    numerical_Ez,
                )
                numerical_magnetic = (
                    numerical_Hx,
                    numerical_Hy,
                    numerical_Hz,
                )
                numerical_curl_electric = (
                    dot(row, curl_electric_x),
                    dot(row, curl_electric_y),
                    dot(row, curl_electric_z),
                )
                numerical_curl_magnetic = (
                    dot(row, curl_magnetic_x),
                    dot(row, curl_magnetic_y),
                    dot(row, curl_magnetic_z),
                )
                exact_electric_vector = (exact_Ex, exact_Ey, exact_Ez)
                exact_magnetic_vector = (exact_Hx, exact_Hy, exact_Hz)
                exact_curl_electric_vector =
                    exact_curl_electric(x, y, z)
                exact_curl_magnetic_vector =
                    exact_curl_magnetic(x, y, z)
                numerical_optical_chirality =
                    optical_chirality_density(
                        numerical_electric,
                        numerical_curl_electric,
                        numerical_magnetic,
                        numerical_curl_magnetic;
                        epsilon = epsilon,
                        mu = mu,
                    )
                exact_optical_chirality =
                    optical_chirality_density(
                        exact_electric_vector,
                        exact_curl_electric_vector,
                        exact_magnetic_vector,
                        exact_curl_magnetic_vector;
                        epsilon = epsilon,
                        mu = mu,
                    )

                electric_charge_density =
                    epsilon * dot(row, divergence_electric)
                magnetic_charge_density =
                    mu * dot(row, divergence_magnetic)

                momentum_scale = epsilon * mu
                momentum_x =
                    momentum_scale *
                    (numerical_Ey * numerical_Hz -
                     numerical_Ez * numerical_Hy)
                momentum_y =
                    momentum_scale *
                    (numerical_Ez * numerical_Hx -
                     numerical_Ex * numerical_Hz)
                momentum_z =
                    momentum_scale *
                    (numerical_Ex * numerical_Hy -
                     numerical_Ey * numerical_Hx)
                exact_momentum_x =
                    momentum_scale *
                    (exact_Ey * exact_Hz - exact_Ez * exact_Hy)
                exact_momentum_y =
                    momentum_scale *
                    (exact_Ez * exact_Hx - exact_Ex * exact_Hz)
                exact_momentum_z =
                    momentum_scale *
                    (exact_Ex * exact_Hy - exact_Ey * exact_Hx)

                angular_momentum_x = y * momentum_z - z * momentum_y
                angular_momentum_y = z * momentum_x - x * momentum_z
                angular_momentum_z = x * momentum_y - y * momentum_x
                exact_angular_momentum_x =
                    y * exact_momentum_z - z * exact_momentum_y
                exact_angular_momentum_y =
                    z * exact_momentum_x - x * exact_momentum_z
                exact_angular_momentum_z =
                    x * exact_momentum_y - y * exact_momentum_x

                physical_weight = jacobian * cubature_weights[q]

                local_sums[1] +=
                    physical_weight * numerical_electric_energy_density
                local_sums[2] +=
                    physical_weight * numerical_magnetic_energy_density
                local_sums[3] +=
                    physical_weight * exact_electric_energy_density
                local_sums[4] +=
                    physical_weight * exact_magnetic_energy_density
                local_sums[5] +=
                    physical_weight * numerical_electric_squared
                local_sums[6] +=
                    physical_weight * exact_electric_squared
                local_sums[7] +=
                    physical_weight * electric_error_squared
                local_sums[8] +=
                    physical_weight * numerical_magnetic_squared
                local_sums[9] +=
                    physical_weight * exact_magnetic_squared
                local_sums[10] +=
                    physical_weight * magnetic_error_squared
                local_sums[11] +=
                    physical_weight * numerical_energy_density^2
                local_sums[12] +=
                    physical_weight * exact_energy_density^2
                local_sums[13] +=
                    physical_weight *
                    (numerical_energy_density - exact_energy_density)^2
                local_sums[14] +=
                    physical_weight * numerical_optical_chirality
                local_sums[15] +=
                    physical_weight * exact_optical_chirality
                local_sums[16] +=
                    physical_weight * numerical_optical_chirality^2
                local_sums[17] +=
                    physical_weight * exact_optical_chirality^2
                local_sums[18] +=
                    physical_weight *
                    (numerical_optical_chirality -
                     exact_optical_chirality)^2
                local_sums[19] +=
                    physical_weight * electric_charge_density
                local_sums[20] +=
                    physical_weight * magnetic_charge_density
                local_sums[21] += physical_weight * momentum_x
                local_sums[22] += physical_weight * momentum_y
                local_sums[23] += physical_weight * momentum_z
                local_sums[24] += physical_weight * exact_momentum_x
                local_sums[25] += physical_weight * exact_momentum_y
                local_sums[26] += physical_weight * exact_momentum_z
                local_sums[27] += physical_weight * angular_momentum_x
                local_sums[28] += physical_weight * angular_momentum_y
                local_sums[29] += physical_weight * angular_momentum_z
                local_sums[30] +=
                    physical_weight * exact_angular_momentum_x
                local_sums[31] +=
                    physical_weight * exact_angular_momentum_y
                local_sums[32] +=
                    physical_weight * exact_angular_momentum_z
            end
        end
    end

    sums = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    electric_l2 = sqrt(max(sums[5], 0.0))
    exact_electric_l2 = sqrt(max(sums[6], 0.0))
    electric_error_l2 = sqrt(max(sums[7], 0.0))
    magnetic_l2 = sqrt(max(sums[8], 0.0))
    exact_magnetic_l2 = sqrt(max(sums[9], 0.0))
    magnetic_error_l2 = sqrt(max(sums[10], 0.0))
    field_error_l2 =
        sqrt(max(sums[7] + sums[10], 0.0))
    exact_field_l2 =
        sqrt(max(sums[6] + sums[9], 0.0))
    energy_density_l2 = sqrt(max(sums[11], 0.0))
    exact_energy_density_l2 = sqrt(max(sums[12], 0.0))
    energy_density_error_l2 = sqrt(max(sums[13], 0.0))
    optical_chirality_density_l2 = sqrt(max(sums[16], 0.0))
    exact_optical_chirality_density_l2 = sqrt(max(sums[17], 0.0))
    optical_chirality_density_error_l2 = sqrt(max(sums[18], 0.0))

    return MaxwellQuadratureDiagnostics(
        cubature_order,
        sums[1],
        sums[2],
        sums[1] + sums[2],
        sums[3],
        sums[4],
        sums[3] + sums[4],
        electric_l2,
        exact_electric_l2,
        electric_error_l2,
        electric_error_l2 / max(exact_electric_l2, eps(Float64)),
        magnetic_l2,
        exact_magnetic_l2,
        magnetic_error_l2,
        magnetic_error_l2 / max(exact_magnetic_l2, eps(Float64)),
        field_error_l2,
        field_error_l2 / max(exact_field_l2, eps(Float64)),
        energy_density_l2,
        exact_energy_density_l2,
        energy_density_error_l2,
        energy_density_error_l2 /
        max(exact_energy_density_l2, eps(Float64)),
        sums[14],
        sums[15],
        sums[14] - sums[15],
        optical_chirality_density_l2,
        exact_optical_chirality_density_l2,
        optical_chirality_density_error_l2,
        sums[19],
        sums[20],
        0.0,
        0.0,
        sums[21],
        sums[22],
        sums[23],
        sums[24],
        sums[25],
        sums[26],
        sums[27],
        sums[28],
        sums[29],
        sums[30],
        sums[31],
        sums[32],
    )
end

function write_quadrature_diagnostics_row(
    io::IO,
    step::Int,
    time::Float64,
    diagnostics::MaxwellQuadratureDiagnostics,
)
    energy_error =
        diagnostics.total_energy - diagnostics.exact_total_energy
    relative_energy_error =
        energy_error /
        max(abs(diagnostics.exact_total_energy), eps(Float64))

    values = (
        step,
        time,
        diagnostics.cubature_order,
        diagnostics.electric_energy,
        diagnostics.magnetic_energy,
        diagnostics.total_energy,
        diagnostics.exact_electric_energy,
        diagnostics.exact_magnetic_energy,
        diagnostics.exact_total_energy,
        energy_error,
        relative_energy_error,
        diagnostics.electric_l2,
        diagnostics.exact_electric_l2,
        diagnostics.electric_error_l2,
        diagnostics.electric_relative_error,
        diagnostics.magnetic_l2,
        diagnostics.exact_magnetic_l2,
        diagnostics.magnetic_error_l2,
        diagnostics.magnetic_relative_error,
        diagnostics.field_error_l2,
        diagnostics.field_relative_error,
        diagnostics.energy_density_l2,
        diagnostics.exact_energy_density_l2,
        diagnostics.energy_density_error_l2,
        diagnostics.energy_density_relative_error,
        diagnostics.optical_chirality,
        diagnostics.exact_optical_chirality,
        diagnostics.optical_chirality_error,
        diagnostics.optical_chirality_density_l2,
        diagnostics.exact_optical_chirality_density_l2,
        diagnostics.optical_chirality_density_error_l2,
        diagnostics.electric_charge,
        diagnostics.magnetic_charge,
        diagnostics.exact_electric_charge,
        diagnostics.exact_magnetic_charge,
        diagnostics.linear_momentum_x,
        diagnostics.linear_momentum_y,
        diagnostics.linear_momentum_z,
        diagnostics.exact_linear_momentum_x,
        diagnostics.exact_linear_momentum_y,
        diagnostics.exact_linear_momentum_z,
        diagnostics.angular_momentum_x,
        diagnostics.angular_momentum_y,
        diagnostics.angular_momentum_z,
        diagnostics.exact_angular_momentum_x,
        diagnostics.exact_angular_momentum_y,
        diagnostics.exact_angular_momentum_z,
    )
    println(io, join(values, ','))
    flush(io)
    return nothing
end

function reference_vertex_node_ids(ref::ReferenceTet; tol::Float64 = 1e-12)
    ids = Vector{Int}(undef, 4)

    for vertex in 1:4
        rv, sv, tv = REF_TET_VERTEX_COORDS[vertex]
        matches = findall(
            node -> abs(ref.r[node] - rv) < tol &&
                    abs(ref.s[node] - sv) < tol &&
                    abs(ref.t[node] - tv) < tol,
            1:ref.Np,
        )
        length(matches) == 1 ||
            error("Could not identify reference tetrahedron vertex $vertex.")
        ids[vertex] = only(matches)
    end

    return ids
end

function vtk_lagrange_triangle_barycentric_indices(order::Int)
    order >= 0 || throw(ArgumentError("Lagrange order must be non-negative."))
    order == 0 && return NTuple{3, Int}[(0, 0, 0)]

    indices = NTuple{3, Int}[
        (0, 0, order),
        (order, 0, 0),
        (0, order, 0),
    ]
    for offset in 1:(order - 1)
        push!(indices, (offset, 0, order - offset))
    end
    for offset in 1:(order - 1)
        push!(indices, (order - offset, offset, 0))
    end
    for offset in 1:(order - 1)
        push!(indices, (0, order - offset, offset))
    end
    if order >= 3
        for inner in vtk_lagrange_triangle_barycentric_indices(order - 3)
            push!(indices, ntuple(i -> inner[i] + 1, 3))
        end
    end
    return indices
end

function vtk_lagrange_tetra_barycentric_indices(order::Int)
    order >= 0 || throw(ArgumentError("Lagrange order must be non-negative."))
    order == 0 && return NTuple{4, Int}[(0, 0, 0, 0)]

    indices = NTuple{4, Int}[
        (0, 0, 0, order),
        (order, 0, 0, 0),
        (0, order, 0, 0),
        (0, 0, order, 0),
    ]
    edge_vertices = (
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    )
    barycentric_slot = (4, 1, 2, 3)
    for (first_vertex, second_vertex) in edge_vertices
        for offset in 1:(order - 1)
            values = zeros(Int, 4)
            values[barycentric_slot[first_vertex + 1]] = order - offset
            values[barycentric_slot[second_vertex + 1]] = offset
            push!(indices, Tuple(values))
        end
    end

    if order >= 3
        face_vertices = (
            (0, 1, 3),
            (2, 3, 1),
            (0, 3, 2),
            (0, 2, 1),
        )
        face_interior = [
            ntuple(i -> value[i] + 1, 3)
            for value in
                vtk_lagrange_triangle_barycentric_indices(order - 3)
        ]
        for (first_vertex, second_vertex, third_vertex) in face_vertices
            for local_index in face_interior
                values = zeros(Int, 4)
                values[barycentric_slot[first_vertex + 1]] = local_index[3]
                values[barycentric_slot[second_vertex + 1]] = local_index[1]
                values[barycentric_slot[third_vertex + 1]] = local_index[2]
                push!(indices, Tuple(values))
            end
        end
    end

    if order >= 4
        for inner in vtk_lagrange_tetra_barycentric_indices(order - 4)
            push!(indices, ntuple(i -> inner[i] + 1, 4))
        end
    end
    return indices
end

function vtk_lagrange_tetra_node_ids(ref::ReferenceTet)
    ref.N >= 1 ||
        throw(ArgumentError("ParaView Lagrange output requires order >= 1."))
    nodes_by_barycentric = Dict{NTuple{4, Int}, Int}()
    for node in 1:ref.Np
        lambda2 = round(Int, ref.N * (ref.r[node] + 1.0) / 2.0)
        lambda3 = round(Int, ref.N * (ref.s[node] + 1.0) / 2.0)
        lambda4 = round(Int, ref.N * (ref.t[node] + 1.0) / 2.0)
        lambda1 = ref.N - lambda2 - lambda3 - lambda4
        nodes_by_barycentric[
            (lambda2, lambda3, lambda4, lambda1)
        ] = node
    end

    vtk_indices = vtk_lagrange_tetra_barycentric_indices(ref.N)
    length(vtk_indices) == ref.Np ||
        error("Incorrect VTK Lagrange node count for order $(ref.N).")
    all(haskey(nodes_by_barycentric, index) for index in vtk_indices) ||
        error("The DG and VTK Lagrange node sets do not match.")
    return [nodes_by_barycentric[index] for index in vtk_indices]
end

function write_energy_row(
    io::IO,
    step::Int,
    time::Float64,
    energy::DiscoGMPI.MaxwellEnergy,
    initial_total::Float64,
)
    relative_drift =
        (energy.total - initial_total) / max(initial_total, eps(Float64))
    values = (
        step,
        time,
        energy.electric,
        energy.magnetic,
        energy.total,
        relative_drift,
        energy.Ex,
        energy.Ey,
        energy.Ez,
        energy.Hx,
        energy.Hy,
        energy.Hz,
    )
    println(io, join(values, ','))
    flush(io)
    return relative_drift
end

function write_parallel_fields(
    output_basename::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField;
    time::Float64,
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    nranks = MPI.Comm_size(distributed_dg.comm)
    mesh = distributed_dg.dg.mesh
    distributed_mesh = distributed_dg.distributed_mesh
    owned = distributed_mesh.partition.owned
    ref = distributed_dg.dg.ref
    vtk_node_ids = vtk_lagrange_tetra_node_ids(ref)
    nowned = length(owned)
    nodes_per_element = ref.Np

    points = zeros(Float64, 3, nodes_per_element * nowned)
    electric = zeros(Float64, 3, nodes_per_element * nowned)
    magnetic = zeros(Float64, 3, nodes_per_element * nowned)
    exact_electric_values = similar(electric)
    exact_magnetic_values = similar(magnetic)
    cells = Vector{MeshCell}(undef, nowned)
    global_element_ids = Vector{Int}(undef, nowned)
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )

    for (owned_index, local_elem) in enumerate(owned)
        first_node = nodes_per_element * (owned_index - 1) + 1
        cell_nodes = collect(first_node:(first_node + nodes_per_element - 1))
        cells[owned_index] =
            MeshCell(VTKCellTypes.VTK_LAGRANGE_TETRAHEDRON, cell_nodes)
        global_element_ids[owned_index] =
            distributed_mesh.elements.global_ids[local_elem]

        tet_nodes = @view mesh.tets[:, local_elem]
        for vtk_node in 1:nodes_per_element
            output_node = cell_nodes[vtk_node]
            field_node = vtk_node_ids[vtk_node]
            x, y, z = DiscoGMPI.map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[field_node],
                ref.s[field_node],
                ref.t[field_node],
            )
            points[:, output_node] .= (x, y, z)
            electric[:, output_node] .= (
                U.Ex[field_node, local_elem],
                U.Ey[field_node, local_elem],
                U.Ez[field_node, local_elem],
            )
            magnetic[:, output_node] .= (
                U.Hx[field_node, local_elem],
                U.Hy[field_node, local_elem],
                U.Hz[field_node, local_elem],
            )
            exact_electric_values[:, output_node] .= exact_electric(x, y, z)
            exact_magnetic_values[:, output_node] .= exact_magnetic(x, y, z)
        end
    end

    electric_error = electric - exact_electric_values
    magnetic_error = magnetic - exact_magnetic_values
    electric_magnitude = vec(sqrt.(sum(abs2, electric; dims = 1)))
    magnetic_magnitude = vec(sqrt.(sum(abs2, magnetic; dims = 1)))

    return pvtk_grid(
        output_basename,
        points,
        cells;
        part = rank + 1,
        nparts = nranks,
        ismain = rank == 0,
        append = false,
        compress = false,
    ) do vtk
        vtk[
            "ElectricField",
            VTKPointData(),
            component_names = ("Ex", "Ey", "Ez"),
        ] = electric
        vtk[
            "MagneticField",
            VTKPointData(),
            component_names = ("Hx", "Hy", "Hz"),
        ] = magnetic
        vtk[
            "ExactElectricField",
            VTKPointData(),
            component_names = ("ExactEx", "ExactEy", "ExactEz"),
        ] = exact_electric_values
        vtk[
            "ExactMagneticField",
            VTKPointData(),
            component_names = ("ExactHx", "ExactHy", "ExactHz"),
        ] = exact_magnetic_values
        vtk[
            "ElectricFieldError",
            VTKPointData(),
            component_names = ("ErrorEx", "ErrorEy", "ErrorEz"),
        ] = electric_error
        vtk[
            "MagneticFieldError",
            VTKPointData(),
            component_names = ("ErrorHx", "ErrorHy", "ErrorHz"),
        ] = magnetic_error
        vtk["ElectricFieldMagnitude", VTKPointData()] = electric_magnitude
        vtk["MagneticFieldMagnitude", VTKPointData()] = magnetic_magnitude
        vtk["GlobalElementId", VTKCellData()] = global_element_ids
        vtk["OwnerRank", VTKCellData()] = fill(rank, nowned)
        vtk["PolynomialOrder", VTKCellData()] =
            fill(distributed_dg.dg.ref.N, nowned)
        vtk["TimeValue", VTKFieldData()] = time
    end
end

function xml_escape(value::AbstractString)
    escaped = replace(value, '&' => "&amp;")
    escaped = replace(escaped, '<' => "&lt;")
    escaped = replace(escaped, '>' => "&gt;")
    escaped = replace(escaped, '"' => "&quot;")
    return replace(escaped, '\'' => "&apos;")
end

function atomic_output_file(writer::Function, path::String)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            writer(io)
            flush(io)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function read_paraview_series(path::String)
    entries = NamedTuple{(:step, :time, :dataset), Tuple{Int, Float64, String}}[]
    isfile(path) || return entries

    open(path, "r") do io
        first = true
        for line in eachline(io)
            if first
                first = false
                continue
            end
            stripped = strip(line)
            isempty(stripped) && continue
            fields = split(stripped, ','; limit = 3)
            length(fields) == 3 ||
                error("Invalid ParaView series row in $path: $line")
            push!(
                entries,
                (
                    step = parse(Int, fields[1]),
                    time = parse(Float64, fields[2]),
                    dataset = fields[3],
                ),
            )
        end
    end
    return entries
end

function write_paraview_series(
    output_dir::String,
    entries,
)
    entries_by_step = Dict(entry.step => entry for entry in entries)
    sorted_entries =
        sort(collect(values(entries_by_step)); by = entry -> entry.step)
    csv_path = joinpath(output_dir, "paraview_series.csv")
    atomic_output_file(csv_path) do io
        println(io, "step,time,dataset")
        for entry in sorted_entries
            println(io, entry.step, ',', entry.time, ',', entry.dataset)
        end
    end

    pvd_path = joinpath(output_dir, "fields.pvd")
    atomic_output_file(pvd_path) do io
        println(io, "<?xml version=\"1.0\"?>")
        println(
            io,
            "<VTKFile type=\"Collection\" version=\"0.1\" " *
            "byte_order=\"LittleEndian\">",
        )
        println(io, "  <Collection>")
        for entry in sorted_entries
            println(
                io,
                "    <DataSet timestep=\"", entry.time,
                "\" group=\"\" part=\"0\" file=\"",
                xml_escape(entry.dataset),
                "\"/>",
            )
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
    return sorted_entries
end

function write_paraview_snapshot!(
    entries,
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    step::Int,
    time::Float64,
    wave::PlaneWaveParameters;
    epsilon::Float64,
    mu::Float64,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    fields_dir = joinpath(output_dir, "fields")
    collective_root_action(
        distributed_dg.comm,
        "ParaView output directory creation",
    ) do
        mkpath(fields_dir)
    end
    MPI.Barrier(distributed_dg.comm)

    basename = @sprintf("fields_step%08d", step)
    output_basename = joinpath(fields_dir, basename)
    collective_rank_action(
        distributed_dg.comm,
        "Parallel VTK snapshot writing at step $step",
    ) do
        write_parallel_fields(
            output_basename,
            distributed_dg,
            U;
            time = time,
            epsilon = epsilon,
            mu = mu,
            wave = wave,
        )
    end
    MPI.Barrier(distributed_dg.comm)

    root_error = nothing
    if rank == 0
        try
            filter!(entry -> entry.step != step, entries)
            push!(
                entries,
                (
                    step = step,
                    time = time,
                    dataset = joinpath("fields", basename * ".pvtu"),
                ),
            )
            replacement = write_paraview_series(output_dir, entries)
            empty!(entries)
            append!(entries, replacement)
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, distributed_dg.comm; root = 0)
    root_error === nothing ||
        error("ParaView collection writing failed: $root_error")
    MPI.Barrier(distributed_dg.comm)
    return nothing
end

function checkpoint_step_dir(root::String, step::Int)
    return joinpath(root, @sprintf("step%08d", step))
end

function write_latest_checkpoint(
    checkpoint_root::String,
    checkpoint_path::String,
    step::Int,
    time::Float64,
)
    path = joinpath(checkpoint_root, "latest.toml")
    atomic_output_file(path) do io
        println(io, "step = ", step)
        println(io, "time = ", time)
        println(
            io,
            "path = \"",
            replace(relpath(checkpoint_path, checkpoint_root), '"' => "\\\""),
            "\"",
        )
    end
    return path
end

function collectively_write_latest_checkpoint(
    comm::MPI.Comm,
    checkpoint_root::String,
    checkpoint_path::String,
    step::Int,
    time::Float64,
)
    rank = MPI.Comm_rank(comm)
    root_error = nothing
    if rank == 0
        try
            write_latest_checkpoint(
                checkpoint_root,
                checkpoint_path,
                step,
                time,
            )
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, comm; root = 0)
    root_error === nothing ||
        error("Latest-checkpoint pointer writing failed: $root_error")
    MPI.Barrier(comm)
    return nothing
end

function collective_root_action(
    action::Function,
    comm::MPI.Comm,
    context::String,
)
    rank = MPI.Comm_rank(comm)
    root_error = nothing
    if rank == 0
        try
            action()
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, comm; root = 0)
    root_error === nothing || error("$context failed: $root_error")
    return nothing
end

function collective_rank_action(
    action::Function,
    comm::MPI.Comm,
    context::String,
)
    rank = MPI.Comm_rank(comm)
    local_error = nothing
    try
        action()
    catch error
        local_error = sprint(showerror, error)
    end
    errors = MPI.gather(local_error, comm; root = 0)
    message = nothing
    if rank == 0
        failures = [
            "rank $(index - 1): $error"
            for (index, error) in enumerate(errors)
            if error !== nothing
        ]
        if !isempty(failures)
            message = context * " failed:\n" * join(failures, "\n")
        end
    end
    message = MPI.bcast(message, comm; root = 0)
    message === nothing || error(message)
    return nothing
end

function write_final_integration_points(
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    rank_label = lpad(string(rank), 4, '0')
    output_path =
        joinpath(output_dir, "integration_points_rank$rank_label.csv")
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )
    exact_curl_electric, exact_curl_magnetic =
        exact_periodic_wave_curl_functions(
            time;
            epsilon = epsilon,
            mu = mu,
            wave = wave,
        )

    mesh = distributed_dg.dg.mesh
    distributed_mesh = distributed_dg.distributed_mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    physical_operators = distributed_dg.dg.physops.elements

    open(output_path, "w") do io
        println(
            io,
            "global_element,local_element,integration_point,r,s,t,x,y,z," *
            "physical_weight,Ex,Ey,Ez,exact_Ex,exact_Ey,exact_Ez," *
            "Hx,Hy,Hz,exact_Hx,exact_Hy,exact_Hz," *
            "electric_energy_density,magnetic_energy_density," *
            "total_energy_density,exact_total_energy_density," *
            "energy_density_error,optical_chirality_density," *
            "exact_optical_chirality_density," *
            "optical_chirality_density_error,electric_charge_density," *
            "magnetic_charge_density,exact_electric_charge_density," *
            "exact_magnetic_charge_density,momentum_x,momentum_y," *
            "momentum_z,exact_momentum_x,exact_momentum_y," *
            "exact_momentum_z,angular_momentum_x,angular_momentum_y," *
            "angular_momentum_z,exact_angular_momentum_x," *
            "exact_angular_momentum_y,exact_angular_momentum_z",
        )

        for elem in distributed_mesh.partition.owned
            global_elem = distributed_mesh.elements.global_ids[elem]
            tet_nodes = @view mesh.tets[:, elem]
            jacobian = mappings[elem].absdetJ
            operators = physical_operators[elem]

            @views begin
                Ex = U.Ex[:, elem]
                Ey = U.Ey[:, elem]
                Ez = U.Ez[:, elem]
                Hx = U.Hx[:, elem]
                Hy = U.Hy[:, elem]
                Hz = U.Hz[:, elem]
                divergence_electric =
                    operators.Dx * Ex +
                    operators.Dy * Ey +
                    operators.Dz * Ez
                divergence_magnetic =
                    operators.Dx * Hx +
                    operators.Dy * Hy +
                    operators.Dz * Hz
                curl_electric_x =
                    operators.Dy * Ez - operators.Dz * Ey
                curl_electric_y =
                    operators.Dz * Ex - operators.Dx * Ez
                curl_electric_z =
                    operators.Dx * Ey - operators.Dy * Ex
                curl_magnetic_x =
                    operators.Dy * Hz - operators.Dz * Hy
                curl_magnetic_y =
                    operators.Dz * Hx - operators.Dx * Hz
                curl_magnetic_z =
                    operators.Dx * Hy - operators.Dy * Hx

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

                    exact_Ex, exact_Ey, exact_Ez =
                        exact_electric(x, y, z)
                    exact_Hx, exact_Hy, exact_Hz =
                        exact_magnetic(x, y, z)
                    row = view(interpolation, q, :)
                    numerical_Ex = dot(row, Ex)
                    numerical_Ey = dot(row, Ey)
                    numerical_Ez = dot(row, Ez)
                    numerical_Hx = dot(row, Hx)
                    numerical_Hy = dot(row, Hy)
                    numerical_Hz = dot(row, Hz)

                    electric_energy_density =
                        0.5 * epsilon *
                        (
                            numerical_Ex^2 +
                            numerical_Ey^2 +
                            numerical_Ez^2
                        )
                    magnetic_energy_density =
                        0.5 * mu *
                        (
                            numerical_Hx^2 +
                            numerical_Hy^2 +
                            numerical_Hz^2
                        )
                    total_energy_density =
                        electric_energy_density + magnetic_energy_density
                    exact_total_energy_density =
                        0.5 * epsilon *
                        (exact_Ex^2 + exact_Ey^2 + exact_Ez^2) +
                        0.5 * mu *
                        (exact_Hx^2 + exact_Hy^2 + exact_Hz^2)
                    numerical_optical_chirality =
                        optical_chirality_density(
                            (numerical_Ex, numerical_Ey, numerical_Ez),
                            (
                                dot(row, curl_electric_x),
                                dot(row, curl_electric_y),
                                dot(row, curl_electric_z),
                            ),
                            (numerical_Hx, numerical_Hy, numerical_Hz),
                            (
                                dot(row, curl_magnetic_x),
                                dot(row, curl_magnetic_y),
                                dot(row, curl_magnetic_z),
                            );
                            epsilon = epsilon,
                            mu = mu,
                        )
                    exact_optical_chirality =
                        optical_chirality_density(
                            (exact_Ex, exact_Ey, exact_Ez),
                            exact_curl_electric(x, y, z),
                            (exact_Hx, exact_Hy, exact_Hz),
                            exact_curl_magnetic(x, y, z);
                            epsilon = epsilon,
                            mu = mu,
                        )
                    electric_charge_density =
                        epsilon * dot(row, divergence_electric)
                    magnetic_charge_density =
                        mu * dot(row, divergence_magnetic)

                    momentum_scale = epsilon * mu
                    momentum_x =
                        momentum_scale *
                        (numerical_Ey * numerical_Hz -
                         numerical_Ez * numerical_Hy)
                    momentum_y =
                        momentum_scale *
                        (numerical_Ez * numerical_Hx -
                         numerical_Ex * numerical_Hz)
                    momentum_z =
                        momentum_scale *
                        (numerical_Ex * numerical_Hy -
                         numerical_Ey * numerical_Hx)
                    exact_momentum_x =
                        momentum_scale *
                        (exact_Ey * exact_Hz - exact_Ez * exact_Hy)
                    exact_momentum_y =
                        momentum_scale *
                        (exact_Ez * exact_Hx - exact_Ex * exact_Hz)
                    exact_momentum_z =
                        momentum_scale *
                        (exact_Ex * exact_Hy - exact_Ey * exact_Hx)

                    angular_momentum_x =
                        y * momentum_z - z * momentum_y
                    angular_momentum_y =
                        z * momentum_x - x * momentum_z
                    angular_momentum_z =
                        x * momentum_y - y * momentum_x
                    exact_angular_momentum_x =
                        y * exact_momentum_z - z * exact_momentum_y
                    exact_angular_momentum_y =
                        z * exact_momentum_x - x * exact_momentum_z
                    exact_angular_momentum_z =
                        x * exact_momentum_y - y * exact_momentum_x

                    values = (
                        global_elem,
                        elem,
                        q,
                        r,
                        s,
                        t,
                        x,
                        y,
                        z,
                        jacobian * cubature_weights[q],
                        numerical_Ex,
                        numerical_Ey,
                        numerical_Ez,
                        exact_Ex,
                        exact_Ey,
                        exact_Ez,
                        numerical_Hx,
                        numerical_Hy,
                        numerical_Hz,
                        exact_Hx,
                        exact_Hy,
                        exact_Hz,
                        electric_energy_density,
                        magnetic_energy_density,
                        total_energy_density,
                        exact_total_energy_density,
                        total_energy_density - exact_total_energy_density,
                        numerical_optical_chirality,
                        exact_optical_chirality,
                        numerical_optical_chirality -
                        exact_optical_chirality,
                        electric_charge_density,
                        magnetic_charge_density,
                        0.0,
                        0.0,
                        momentum_x,
                        momentum_y,
                        momentum_z,
                        exact_momentum_x,
                        exact_momentum_y,
                        exact_momentum_z,
                        angular_momentum_x,
                        angular_momentum_y,
                        angular_momentum_z,
                        exact_angular_momentum_x,
                        exact_angular_momentum_y,
                        exact_angular_momentum_z,
                    )
                    println(io, join(values, ','))
                end
            end
        end
    end

    return output_path
end

function generate_metis_partition(config::ExperimentConfig)
    mkpath(dirname(config.partition_path))
    if config.partition_count == 1
        _, tetrahedra = read_mesh_file_tet_vtk(config.mesh_path)
        open(config.partition_path, "w") do io
            foreach(_ -> println(io, 0), axes(tetrahedra, 2))
        end
        return config.partition_path
    end

    mesh_name = splitext(basename(config.mesh_path))[1]
    metis_mesh_path =
        joinpath(dirname(config.partition_path), "$mesh_name.mesh")
    write_metis_mesh_from_vtk(config.mesh_path, metis_mesh_path)

    mpmetis = Sys.which("mpmetis")
    mpmetis === nothing &&
        error(
            "Automatic partitioning requires the 'mpmetis' executable. " *
            "Install METIS or pass --partition PATH.",
        )

    command = `$mpmetis $metis_mesh_path $(config.partition_count)`
    run(command)
    generated_path =
        "$metis_mesh_path.epart.$(config.partition_count)"
    isfile(generated_path) ||
        error("mpmetis did not create the expected file $generated_path.")

    if abspath(generated_path) != abspath(config.partition_path)
        cp(generated_path, config.partition_path; force = true)
    end
    return config.partition_path
end

function load_root_inputs(
    config::ExperimentConfig,
    rank::Int,
    nranks::Int,
    comm::MPI.Comm,
)
    mesh = nothing
    partition = nothing
    load_error = nothing

    if rank == 0
        try
            isfile(config.mesh_path) ||
                error("Mesh file not found: $(config.mesh_path)")

            mesh = load_periodic_mesh(config.mesh_path)
            if config.repartition || !isfile(config.partition_path)
                generate_metis_partition(config)
            end
            partition = read_metis_epart(config.partition_path)
            length(partition) == size(mesh.tets, 2) ||
                error(
                    "Partition file $(config.partition_path) has " *
                    "$(length(partition)) entries, but mesh " *
                    "$(config.mesh_path) has $(size(mesh.tets, 2)) " *
                    "tetrahedra. Use a partition generated for this mesh, " *
                    "or regenerate it with --repartition.",
                )
            all(part -> 0 <= part < config.partition_count, partition) ||
                error("Partition entries must be zero-based ranks in 0:$(nranks - 1).")
            all(
                part -> any(==(part), partition),
                0:(config.partition_count - 1),
            ) ||
                error("Every MPI rank must own at least one tetrahedron.")
        catch error
            load_error = sprint(showerror, error)
        end
    end

    load_error = MPI.bcast(load_error, comm; root = 0)
    load_error === nothing || error(load_error)
    return mesh, partition
end

function load_or_prepare_distributed_dg(
    config::ExperimentConfig,
    rank::Int,
    nranks::Int,
    comm::MPI.Comm,
)
    manifest_path =
        joinpath(config.distributed_mesh_dir, "mesh_manifest.toml")
    use_existing = MPI.bcast(
        rank == 0 &&
        isfile(manifest_path) &&
        !config.rebuild_distributed_mesh &&
        !config.repartition,
        comm;
        root = 0,
    )

    if use_existing
        return build_distributed_dg_from_partition(
            config.distributed_mesh_dir,
            config.polynomial_order;
            comm = comm,
        ), "rank-local cache"
    end

    root_mesh, root_partition =
        load_root_inputs(config, rank, nranks, comm)
    prepare_distributed_mesh_partition(
        root_mesh,
        root_partition,
        config.distributed_mesh_dir;
        comm = comm,
        metadata = Dict(
            "source_mesh" => config.mesh_path,
            "source_partition" => config.partition_path,
            "partition_count" => config.partition_count,
            "boundary_condition" => "periodic",
        ),
    )
    return build_distributed_dg_from_partition(
        config.distributed_mesh_dir,
        config.polynomial_order;
        comm = comm,
    ), "prepared rank-local cache"
end

function experiment_configuration(
    config::ExperimentConfig,
    mesh_load_mode::String,
    cubature_order::Int,
    box::PeriodicBox,
    wave::PlaneWaveParameters,
)
    return Dict{String, Any}(
        "mesh_path" => config.mesh_path,
        "partition_path" => config.partition_path,
        "partition_count" => config.partition_count,
        "repartition" => config.repartition,
        "distributed_mesh_dir" => config.distributed_mesh_dir,
        "mesh_load_mode" => mesh_load_mode,
        "output_dir" => config.output_dir,
        "polynomial_order" => config.polynomial_order,
        "esprk_order" => config.esprk_order,
        "final_time" => config.final_time,
        "cfl" => config.cfl,
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "domain_lower" => collect(box.lower),
        "domain_upper" => collect(box.upper),
        "domain_lengths" => collect(periodic_box_lengths(box)),
        "energy_every" => config.energy_every,
        "cubature_order" => cubature_order,
        "paraview_every" => config.paraview_every,
        "checkpoint_every" => config.checkpoint_every,
        "checkpoint_dir" => config.checkpoint_dir,
        "restart_path" => config.restart_path,
        "formulation" => "PoissonBracketFormulation",
        "flux" => "centered",
        "boundary_condition" => "periodic in x, y, and z",
        "periodic_pairs" => "1<->2, 3<->4, 5<->6",
        "analytical_solution" => "traveling wave along +x",
        "optical_chirality_definition" =>
            "0.5*(epsilon*E dot curl(E) + mu*H dot curl(H))",
        "exact_optical_chirality" => 0.0,
        "wave_number" => wave.wave_number,
        "angular_frequency" => wave.angular_frequency,
        "magnetic_amplitude" => wave.magnetic_amplitude,
        "wave_x_origin" => wave.x_origin,
    )
end

function checkpoint_metadata(
    config::ExperimentConfig,
    initial_energy_total::Float64,
    mesh_load_mode::String,
    box::PeriodicBox,
    wave::PlaneWaveParameters,
)
    return Dict{String, Any}(
        "initial_energy_total" => initial_energy_total,
        "target_final_time" => config.final_time,
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "wave_number" => wave.wave_number,
        "angular_frequency" => wave.angular_frequency,
        "wave_x_origin" => wave.x_origin,
        "domain_lower" => collect(box.lower),
        "domain_upper" => collect(box.upper),
        "esprk_order" => config.esprk_order,
        "first_partition" => "H",
        "distributed_mesh_dir" => config.distributed_mesh_dir,
        "mesh_load_mode" => mesh_load_mode,
    )
end

function open_history_file(path::String, restarting::Bool)
    append_existing = restarting && isfile(path) && filesize(path) > 0
    return open(path, append_existing ? "a" : "w"), !append_existing
end

function restart_initial_energy(
    state::DistributedCheckpointState,
    fallback::Float64,
)
    value = get(state.metadata, "initial_energy_total", fallback)
    return value isa Number ? Float64(value) : parse(Float64, string(value))
end

function restart_metadata_number(
    state::DistributedCheckpointState,
    name::String,
)
    haskey(state.metadata, name) || return nothing
    value = state.metadata[name]
    return value isa Number ? Float64(value) : parse(Float64, string(value))
end

function validate_restart_configuration(
    state::DistributedCheckpointState,
    config::ExperimentConfig,
)
    for (name, expected) in (
        ("epsilon", config.epsilon),
        ("mu", config.mu),
        ("wave_number", config.wave_number),
        ("esprk_order", Float64(config.esprk_order)),
    )
        stored = restart_metadata_number(state, name)
        stored === nothing && continue
        isapprox(stored, expected; rtol = 0.0, atol = 1e-14) ||
            throw(
                ArgumentError(
                    "Checkpoint $name=$stored does not match the requested " *
                    "value $expected.",
                ),
            )
    end

    first_partition =
        get(state.metadata, "first_partition", "H")
    string(first_partition) == "H" ||
        throw(
            ArgumentError(
                "The checkpoint does not use the required H-first " *
                "Poisson-bracket integrator.",
            ),
        )
    return nothing
end

function remaining_time_step(
    final_time::Float64,
    start_time::Float64,
    checkpoint_dt::Float64,
)
    remaining = final_time - start_time
    remaining > 0.0 ||
        throw(
            ArgumentError(
                "--final-time ($final_time) must be greater than the " *
                "checkpoint time ($start_time).",
            ),
        )
    ratio = remaining / checkpoint_dt
    nearest = round(Int, ratio)
    if nearest >= 1 && isapprox(ratio, nearest; rtol = 1e-10, atol = 1e-12)
        return checkpoint_dt, nearest
    end
    nsteps = max(1, ceil(Int, ratio))
    return remaining / nsteps, nsteps
end

function run_experiment(config::ExperimentConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    distributed_dg, mesh_load_mode =
        load_or_prepare_distributed_dg(config, rank, nranks, comm)
    box = distributed_periodic_box(distributed_dg)
    wave = configured_plane_wave(config, box)

    cubature_order = resolved_cubature_order(config)
    restarting = !isempty(config.restart_path)
    restart_state = nothing
    U = nothing

    if restarting
        U, restart_state = load_distributed_checkpoint(
            config.restart_path,
            distributed_dg,
        )
        validate_restart_configuration(restart_state, config)
    else
        electric, magnetic = exact_periodic_wave_functions(
            0.0;
            epsilon = config.epsilon,
            mu = config.mu,
            wave = wave,
        )
        U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    end

    registry = MaxwellBoundaryRegistry(
        Dict(boundary_id => MaxwellBC_None for boundary_id in 1:6),
    )
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        periodic_boundary_specs(box),
    )
    formulation = PoissonBracketFormulation()
    scheme = explicit_partitioned_symplectic_rk_scheme(
        config.esprk_order;
        first_partition = :H,
    )
    workspace = MaxwellPartitionedRKWorkspace(U, scheme)

    local_dt, local_sizes = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = config.cfl,
        ε = config.epsilon,
        μ = config.mu,
    )
    estimated_dt = MPI.Allreduce(local_dt, min, comm)
    global_hmin = MPI.Allreduce(local_sizes.hmin, min, comm)
    directed_periodic_faces = MPI.Allreduce(
        length(periodic.faces),
        +,
        comm,
    )
    start_step = restarting ? restart_state.step : 0
    start_time = restarting ? restart_state.time : 0.0
    dt, remaining_steps = if restarting
        remaining_time_step(
            config.final_time,
            start_time,
            restart_state.dt,
        )
    else
        steps = max(1, ceil(Int, config.final_time / estimated_dt))
        config.final_time / steps, steps
    end
    final_step = start_step + remaining_steps

    collective_root_action(
        comm,
        "Output directory creation",
    ) do
        mkpath(config.output_dir)
        mkpath(config.checkpoint_dir)
    end
    MPI.Barrier(comm)

    energy_path = joinpath(config.output_dir, "energy.csv")
    quadrature_path =
        joinpath(config.output_dir, "quadrature_diagnostics.csv")
    energy_io = nothing
    quadrature_io = nothing
    write_history_header = false
    history_open_error = nothing
    if rank == 0
        try
            energy_io, energy_header =
                open_history_file(energy_path, restarting)
            quadrature_io, quadrature_header =
                open_history_file(quadrature_path, restarting)
            energy_header == quadrature_header ||
                error("Energy and quadrature history append states disagree.")
            write_history_header = energy_header
        catch error
            energy_io !== nothing && close(energy_io)
            quadrature_io !== nothing && close(quadrature_io)
            history_open_error = sprint(showerror, error)
        end
    end
    history_open_error =
        MPI.bcast(history_open_error, comm; root = 0)
    history_open_error === nothing ||
        error("Diagnostic history opening failed: $history_open_error")

    current_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    initial_energy_total = restarting ?
                           restart_initial_energy(
        restart_state,
        current_energy.total,
    ) :
                           current_energy.total
    current_quadrature = distributed_quadrature_diagnostics(
        U,
        distributed_dg,
        start_time,
        cubature_order;
        epsilon = config.epsilon,
        mu = config.mu,
        wave = wave,
    )

    collective_root_action(
        comm,
        "Diagnostic history initialization",
    ) do
        if write_history_header
            println(
                energy_io,
                "step,time,electric,magnetic,total,relative_drift," *
                "Ex,Ey,Ez,Hx,Hy,Hz",
            )
            println(
                quadrature_io,
                "step,time,cubature_order,electric_energy,magnetic_energy," *
                "total_energy,exact_electric_energy,exact_magnetic_energy," *
                "exact_total_energy,energy_error,relative_energy_error," *
                "electric_l2,exact_electric_l2,electric_error_l2," *
                "electric_relative_error,magnetic_l2,exact_magnetic_l2," *
                "magnetic_error_l2,magnetic_relative_error,field_error_l2," *
                "field_relative_error,energy_density_l2," *
                "exact_energy_density_l2,energy_density_error_l2," *
                "energy_density_relative_error,optical_chirality," *
                "exact_optical_chirality,optical_chirality_error," *
                "optical_chirality_density_l2," *
                "exact_optical_chirality_density_l2," *
                "optical_chirality_density_error_l2,electric_charge," *
                "magnetic_charge,exact_electric_charge," *
                "exact_magnetic_charge,linear_momentum_x," *
                "linear_momentum_y,linear_momentum_z," *
                "exact_linear_momentum_x,exact_linear_momentum_y," *
                "exact_linear_momentum_z,angular_momentum_x," *
                "angular_momentum_y,angular_momentum_z," *
                "exact_angular_momentum_x,exact_angular_momentum_y," *
                "exact_angular_momentum_z",
            )
            write_energy_row(
                energy_io,
                start_step,
                start_time,
                current_energy,
                initial_energy_total,
            )
            write_quadrature_diagnostics_row(
                quadrature_io,
                start_step,
                start_time,
                current_quadrature,
            )
        end
    end

    if rank == 0
        println("Distributed periodic Poisson-bracket Maxwell experiment")
        println("-------------------------------------------------------")
        println("MPI ranks:            ", nranks)
        println("mesh:                 ", config.mesh_path)
        println("partition:            ", config.partition_path)
        println("partitions:           ", config.partition_count)
        println("distributed mesh:     ", config.distributed_mesh_dir)
        println("mesh load mode:       ", mesh_load_mode)
        println("DG order:             ", config.polynomial_order)
        println("time integrator:      ", scheme.name)
        println("cubature order:       ", cubature_order)
        println("boundary condition:   periodic pairs 1<->2, 3<->4, 5<->6")
        println("periodic face pairs:  ", div(directed_periodic_faces, 2))
        println("domain lower:         ", box.lower)
        println("domain upper:         ", box.upper)
        println("wave number:          ", wave.wave_number)
        println("angular frequency:    ", wave.angular_frequency)
        println("wave x origin:        ", wave.x_origin)
        println("analytical solution:  x_phase = x - wave_x_origin")
        println("                      E_z = -k*H0/(omega*epsilon)")
        println("                            * sin(k*x_phase-omega*t)")
        println("                      H_y = H0*sin(k*x_phase-omega*t)")
        println("optical chirality:    0.5*(epsilon E.curl(E) + mu H.curl(H))")
        println("electric charge:      integral of div(epsilon E)")
        println("magnetic charge:      integral of div(mu H)")
        println("linear momentum:      integral of epsilon*mu*(E x H)")
        println("angular momentum:     about coordinate origin")
        println("global hmin:          ", global_hmin)
        println("estimated dt:         ", estimated_dt)
        println("used dt:              ", dt)
        println("start step/time:      ", start_step, " / ", start_time)
        println("final step/time:      ", final_step, " / ", config.final_time)
        println("remaining steps:      ", remaining_steps)
        println("initial energy:       ", initial_energy_total)
        println("ParaView every:       ", config.paraview_every)
        println("checkpoint every:     ", config.checkpoint_every)
        println("checkpoint directory: ", config.checkpoint_dir)
        if restarting
            println("restart checkpoint:   ", config.restart_path)
        end
        println("output directory:     ", config.output_dir)
    end

    configuration =
        experiment_configuration(
            config,
            mesh_load_mode,
            cubature_order,
            box,
            wave,
        )
    write_distributed_run_metadata(
        config.output_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "running",
            "start_step" => start_step,
            "start_time" => start_time,
            "final_step" => final_step,
            "target_final_time" => config.final_time,
            "remaining_steps" => remaining_steps,
            "estimated_dt" => estimated_dt,
            "used_dt" => dt,
            "global_hmin" => global_hmin,
            "initial_energy_total" => initial_energy_total,
        ),
    )

    series_entries = NamedTuple{
        (:step, :time, :dataset),
        Tuple{Int, Float64, String},
    }[]
    series_read_error = nothing
    if rank == 0 && restarting
        try
            series_entries = read_paraview_series(
                joinpath(config.output_dir, "paraview_series.csv"),
            )
        catch error
            series_read_error = sprint(showerror, error)
        end
    end
    series_read_error = MPI.bcast(series_read_error, comm; root = 0)
    series_read_error === nothing ||
        error("ParaView series loading failed: $series_read_error")
    snapshot_exists = MPI.bcast(
        rank == 0 && any(entry -> entry.step == start_step, series_entries),
        comm;
        root = 0,
    )
    if !snapshot_exists
        write_paraview_snapshot!(
            series_entries,
            config.output_dir,
            distributed_dg,
            U,
            start_step,
            start_time,
            wave;
            epsilon = config.epsilon,
            mu = config.mu,
        )
    end

    checkpoint_info = checkpoint_metadata(
        config,
        initial_energy_total,
        mesh_load_mode,
        box,
        wave,
    )
    local_elapsed = 0.0
    final_checkpoint_path = ""
    final_energy = current_energy
    final_quadrature = current_quadrature
    final_time = start_time
    try
        for local_step in 1:remaining_steps
            step = start_step + local_step
            time = local_step == remaining_steps ?
                   config.final_time :
                   start_time + local_step * dt

            local_elapsed += @elapsed begin
                distributed_periodic_partitioned_symplectic_rk_step!(
                    U,
                    workspace,
                    scheme,
                    dt,
                    distributed_dg,
                    periodic,
                    registry,
                    formulation;
                    ε = config.epsilon,
                    μ = config.mu,
                )
            end

            if step % config.energy_every == 0 || step == final_step
                final_energy = distributed_maxwell_energy(
                    U,
                    distributed_dg;
                    ε = config.epsilon,
                    μ = config.mu,
                )
                final_quadrature = distributed_quadrature_diagnostics(
                    U,
                    distributed_dg,
                    time,
                    cubature_order;
                    epsilon = config.epsilon,
                    mu = config.mu,
                    wave = wave,
                )
                collective_root_action(
                    comm,
                    "Diagnostic history writing at step $step",
                ) do
                    relative_drift = write_energy_row(
                        energy_io,
                        step,
                        time,
                        final_energy,
                        initial_energy_total,
                    )
                    write_quadrature_diagnostics_row(
                        quadrature_io,
                        step,
                        time,
                        final_quadrature,
                    )
                    println(
                        "step ", step, "/", final_step,
                        ", t = ", time,
                        ", energy = ", final_energy.total,
                        ", relative drift = ", relative_drift,
                        ", L2(E error) = ",
                        final_quadrature.electric_error_l2,
                        ", L2(H error) = ",
                        final_quadrature.magnetic_error_l2,
                        ", L2(w error) = ",
                        final_quadrature.energy_density_error_l2,
                        ", chirality = ",
                        final_quadrature.optical_chirality,
                        ", Qe = ", final_quadrature.electric_charge,
                        ", Qm = ", final_quadrature.magnetic_charge,
                    )
                end
            end

            if step % config.paraview_every == 0 || step == final_step
                write_paraview_snapshot!(
                    series_entries,
                    config.output_dir,
                    distributed_dg,
                    U,
                    step,
                    time,
                    wave;
                    epsilon = config.epsilon,
                    mu = config.mu,
                )
            end

            checkpoint_due =
                (config.checkpoint_every > 0 &&
                 step % config.checkpoint_every == 0) ||
                step == final_step
            if checkpoint_due
                checkpoint_path =
                    checkpoint_step_dir(config.checkpoint_dir, step)
                write_distributed_checkpoint(
                    checkpoint_path,
                    U,
                    distributed_dg;
                    step = step,
                    time = time,
                    dt = dt,
                    metadata = checkpoint_info,
                )
                collectively_write_latest_checkpoint(
                    comm,
                    config.checkpoint_dir,
                    checkpoint_path,
                    step,
                    time,
                )
                if step == final_step
                    final_checkpoint_path = checkpoint_path
                end
            end
            final_time = time
        end
    finally
        collective_root_action(
            comm,
            "Diagnostic history closing",
        ) do
            close(energy_io)
            close(quadrature_io)
        end
    end

    collective_rank_action(
        comm,
        "Final integration-point output",
    ) do
        write_final_integration_points(
            config.output_dir,
            distributed_dg,
            U,
            final_time,
            cubature_order;
            epsilon = config.epsilon,
            mu = config.mu,
            wave = wave,
        )
    end
    MPI.Barrier(comm)

    elapsed = MPI.Allreduce(local_elapsed, max, comm)
    write_distributed_run_metadata(
        config.output_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "complete",
            "start_step" => start_step,
            "start_time" => start_time,
            "final_step" => final_step,
            "final_time" => final_time,
            "remaining_steps" => remaining_steps,
            "estimated_dt" => estimated_dt,
            "used_dt" => dt,
            "global_hmin" => global_hmin,
            "integration_wall_seconds" => elapsed,
            "initial_energy_total" => initial_energy_total,
            "final_energy_total" => final_energy.total,
            "relative_energy_drift" => (
                (final_energy.total - initial_energy_total) /
                max(initial_energy_total, eps(Float64))
            ),
            "final_checkpoint" => final_checkpoint_path,
        ),
    )

    if rank == 0
        println()
        println("Energy history:        ", energy_path)
        println("Quadrature diagnostics:", quadrature_path)
        println(
            "Run metadata:          ",
            joinpath(config.output_dir, "run_metadata.toml"),
        )
        println(
            "Partition metadata:    ",
            joinpath(config.output_dir, "partition_metadata.csv"),
        )
        println(
            "Integration points:    ",
            joinpath(config.output_dir, "integration_points_rankNNNN.csv"),
        )
        println(
            "ParaView time series:  ",
            joinpath(config.output_dir, "fields.pvd"),
        )
        println("Final checkpoint:      ", final_checkpoint_path)
        println(
            "Plot diagnostics:      python3 examples/plot_maxwell_energy.py ",
            quadrature_path,
        )
    end

    return nothing
end

function main(args::Vector{String})
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    repository_root = normpath(joinpath(@__DIR__, ".."))

    try
        config = parse_arguments(args, nranks, repository_root)

        if config === nothing
            rank == 0 && usage()
            return nothing
        end

        run_experiment(config, comm)
    catch error
        if rank == 0
            println(stderr, "ERROR: ", sprint(showerror, error))
        end
        rethrow()
    finally
        if initialized_here && !MPI.Finalized()
            MPI.Finalize()
        end
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
