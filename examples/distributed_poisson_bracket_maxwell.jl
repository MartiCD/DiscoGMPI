#!/usr/bin/env julia

# Distributed Maxwell cavity experiment using the Poisson-bracket formulation.
#
# From the DiscoGMPI repository root:
#   mpiexec -n 2 julia --project=. examples/distributed_poisson_bracket_maxwell.jl
#
# The default mesh ships with 2-rank and 4-rank METIS partitions. The legacy
# VTK mesh was generated as a periodic box, but the distributed solver does not
# yet support periodic boundaries. This experiment therefore derives every
# exterior tetrahedral face and applies the selected PEC or PMC condition.

using MPI
using DiscoGMPI
using LinearAlgebra: dot
using Printf
const PEC_BOUNDARY_ID = 10
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
    distributed_mesh_dir::String
    rebuild_distributed_mesh::Bool
    collective_distributed_mesh_prep::Bool
    output_dir::String
    polynomial_order::Int
    esprk_order::Int
    rk_order::Int
    final_time::Float64
    cfl::Float64
    epsilon::Float64
    mu::Float64
    boundary_condition::Symbol
    pml_width::Float64
    pml_sigma_max::Float64
    pml_degree::Int
    pml_a::Float64
    pml_regularization::Float64
    energy_every::Int
    cubature_order::Int
    paraview_every::Int
    checkpoint_every::Int
    checkpoint_dir::String
    restart_path::String
end

function usage(io::IO = stdout)
    println(io, """
Distributed Poisson-bracket Maxwell experiment

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_poisson_bracket_maxwell.jl [options]

Options:
  --mesh PATH           Legacy tetrahedral VTK mesh of the unit cube.
                        Default: examples/meshes/tet_mesh.vtk
  --partition PATH      Zero-based METIS element partition.
                        Default: <mesh stem>.mesh.epart.<MPI ranks>
  --distributed-mesh-dir PATH
                        Rank-local mesh cache. Each rank independently loads
                        mesh_rankNNNN.bin after the cache is prepared.
                        Default: output/distributed_mesh_cache/<mesh>_ranksN
  --rebuild-distributed-mesh
                        Recreate the rank-local mesh cache from --mesh and
                        --partition before running.
  --collective-distributed-mesh-prep
                        Recreate a missing or stale rank-local mesh cache by
                        having every rank read the global mesh and partition,
                        avoiding root-built rank-local mesh packets.
  --output-dir PATH     Output directory.
                        Default: output/distributed_poisson_bracket
  --order N             DG polynomial order (N >= 1). Default: 2
  --esprk-order N       ESPRK order in 1:6. Default: 4
  --rk-order N          Explicit RK order in 1:5 when PML is active.
                        Default: 4
  --final-time T        Simulation end time. Default: 0.25
  --cfl C               CFL used to estimate dt. Default: 0.05
  --epsilon VALUE       Electric permittivity. Default: 1.0
  --mu VALUE            Magnetic permeability. Default: 1.0
  --boundary-condition NAME
                        Exterior boundary condition: pec or pmc.
                        Default: pec
  --pml-width W         Width of a nonlinear PML layer on every side.
                        Zero disables PML. Default: 0
  --pml-sigma-max S     Peak nonlinear PML damping. Zero disables PML.
                        Default: 0
  --pml-degree N        Polynomial PML profile degree. Default: 2
  --pml-a VALUE         Nonlinear PML weighting parameter in (0,1).
                        Default: 0.5
  --pml-regularization VALUE
                        Positive PML denominator regularization.
                        Default: 1e-12
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
    mesh_path = joinpath(repository_root, "examples", "meshes", "tet_mesh.vtk")
    partition_path = ""
    distributed_mesh_dir = ""
    rebuild_distributed_mesh = false
    collective_distributed_mesh_prep = false
    output_dir =
        joinpath(repository_root, "output", "distributed_poisson_bracket")
    polynomial_order = 2
    esprk_order = 4
    rk_order = 4
    final_time = 0.25
    cfl = 0.05
    epsilon = 1.0
    mu = 1.0
    boundary_condition = :pec
    pml_width = 0.0
    pml_sigma_max = 0.0
    pml_degree = 2
    pml_a = 0.5
    pml_regularization = 1e-12
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
        elseif startswith(arg, "--partition")
            value, i = option_value(args, i, "--partition")
            partition_path = abspath(value)
        elseif startswith(arg, "--distributed-mesh-dir")
            value, i = option_value(args, i, "--distributed-mesh-dir")
            distributed_mesh_dir = abspath(value)
        elseif arg == "--rebuild-distributed-mesh"
            rebuild_distributed_mesh = true
        elseif arg == "--collective-distributed-mesh-prep"
            collective_distributed_mesh_prep = true
        elseif startswith(arg, "--output-dir")
            value, i = option_value(args, i, "--output-dir")
            output_dir = abspath(value)
        elseif startswith(arg, "--order")
            value, i = option_value(args, i, "--order")
            polynomial_order = parse(Int, value)
        elseif startswith(arg, "--esprk-order")
            value, i = option_value(args, i, "--esprk-order")
            esprk_order = parse(Int, value)
        elseif startswith(arg, "--rk-order")
            value, i = option_value(args, i, "--rk-order")
            rk_order = parse(Int, value)
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
        elseif startswith(arg, "--boundary-condition")
            value, i = option_value(args, i, "--boundary-condition")
            boundary_condition = Symbol(lowercase(value))
        elseif startswith(arg, "--pml-width")
            value, i = option_value(args, i, "--pml-width")
            pml_width = parse(Float64, value)
        elseif startswith(arg, "--pml-sigma-max")
            value, i = option_value(args, i, "--pml-sigma-max")
            pml_sigma_max = parse(Float64, value)
        elseif startswith(arg, "--pml-degree")
            value, i = option_value(args, i, "--pml-degree")
            pml_degree = parse(Int, value)
        elseif startswith(arg, "--pml-a")
            value, i = option_value(args, i, "--pml-a")
            pml_a = parse(Float64, value)
        elseif startswith(arg, "--pml-regularization")
            value, i = option_value(args, i, "--pml-regularization")
            pml_regularization = parse(Float64, value)
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

    if isempty(partition_path)
        partition_path = splitext(mesh_path)[1] * ".mesh.epart.$nranks"
    end
    if isempty(distributed_mesh_dir)
        mesh_name = splitext(basename(mesh_path))[1]
        distributed_mesh_dir = joinpath(
            repository_root,
            "output",
            "distributed_mesh_cache",
            "$(mesh_name)_ranks$nranks",
        )
    end
    if isempty(checkpoint_dir)
        checkpoint_dir = joinpath(output_dir, "checkpoints")
    end

    polynomial_order >= 1 ||
        throw(ArgumentError("--order must be at least 1."))
    1 <= esprk_order <= 6 ||
        throw(ArgumentError("--esprk-order must be in 1:6."))
    1 <= rk_order <= 5 ||
        throw(ArgumentError("--rk-order must be in 1:5."))
    final_time > 0.0 ||
        throw(ArgumentError("--final-time must be positive."))
    cfl > 0.0 ||
        throw(ArgumentError("--cfl must be positive."))
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))
    boundary_condition in (:pec, :pmc) ||
        throw(ArgumentError("--boundary-condition must be pec or pmc."))
    pml_width >= 0.0 ||
        throw(ArgumentError("--pml-width must be non-negative."))
    pml_sigma_max >= 0.0 ||
        throw(ArgumentError("--pml-sigma-max must be non-negative."))
    pml_degree >= 1 ||
        throw(ArgumentError("--pml-degree must be at least one."))
    0.0 < pml_a < 1.0 ||
        throw(ArgumentError("--pml-a must lie in (0,1)."))
    pml_regularization > 0.0 ||
        throw(ArgumentError("--pml-regularization must be positive."))
    xor(pml_width > 0.0, pml_sigma_max > 0.0) &&
        throw(
            ArgumentError(
                "--pml-width and --pml-sigma-max must both be positive " *
                "to enable PML, or both be zero to disable it.",
            ),
        )
    if pml_width > 0.0 &&
       (!isapprox(epsilon, 1.0) || !isapprox(mu, 1.0))
        throw(
            ArgumentError(
                "The nonlinear PML currently requires --epsilon=1 and --mu=1.",
            ),
        )
    end
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
        distributed_mesh_dir,
        rebuild_distributed_mesh,
        collective_distributed_mesh_prep,
        output_dir,
        polynomial_order,
        esprk_order,
        rk_order,
        final_time,
        cfl,
        epsilon,
        mu,
        boundary_condition,
        pml_width,
        pml_sigma_max,
        pml_degree,
        pml_a,
        pml_regularization,
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

function load_pec_mesh(mesh_path::String)
    points, tets = read_mesh_file_tet_vtk(mesh_path)
    size(tets, 2) > 0 ||
        error("The mesh '$mesh_path' contains no tetrahedra.")

    tolerance = 1e-10
    bounds = [
        (minimum(@view points[dimension, :]),
         maximum(@view points[dimension, :]))
        for dimension in 1:3
    ]
    all(
        bound -> abs(bound[1]) <= tolerance &&
                 abs(bound[2] - 1.0) <= tolerance,
        bounds,
    ) ||
        error(
            "The analytical cavity mode requires a [0,1]^3 mesh; " *
            "coordinate bounds are $bounds.",
        )

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

pml_enabled(config::ExperimentConfig) =
    config.pml_width > 0.0 && config.pml_sigma_max > 0.0

function configured_boundary_kind(config::ExperimentConfig)
    if config.boundary_condition == :pec
        return MaxwellBC_PEC
    elseif config.boundary_condition == :pmc
        return MaxwellBC_PMC
    end
    throw(
        ArgumentError(
            "Unsupported boundary condition $(config.boundary_condition).",
        ),
    )
end

function distributed_mesh_bounds(
    distributed_dg::DistributedDGDiscretization,
)
    points = distributed_dg.dg.mesh.points
    local_minimum = vec(minimum(points; dims = 2))
    local_maximum = vec(maximum(points; dims = 2))
    global_minimum =
        MPI.Allreduce(local_minimum, min, distributed_dg.comm)
    global_maximum =
        MPI.Allreduce(local_maximum, max, distributed_dg.comm)
    return global_minimum, global_maximum
end

function build_configured_nonlinear_pml(
    distributed_dg::DistributedDGDiscretization,
    config::ExperimentConfig,
)
    pml_enabled(config) || return nothing
    lower, upper = distributed_mesh_bounds(distributed_dg)
    lengths = upper .- lower
    minimum(lengths) > 0.0 ||
        throw(ArgumentError("The distributed mesh bounds are degenerate."))
    2.0 * config.pml_width < minimum(lengths) ||
        throw(
            ArgumentError(
                "--pml-width=$(config.pml_width) must be smaller than " *
                "half the shortest domain extent $(minimum(lengths)).",
            ),
        )

    directional_profile = function (coordinate, direction)
        left_interface = lower[direction] + config.pml_width
        right_interface = upper[direction] - config.pml_width
        return max(
            polynomial_pml_sigma(
                coordinate,
                left_interface,
                lower[direction];
                sigma_max = config.pml_sigma_max,
                degree = config.pml_degree,
            ),
            polynomial_pml_sigma(
                coordinate,
                right_interface,
                upper[direction];
                sigma_max = config.pml_sigma_max,
                degree = config.pml_degree,
            ),
        )
    end

    return build_maxwell_nonlinear_pml(
        distributed_dg;
        sigma_x = (x, y, z) -> directional_profile(x, 1),
        sigma_y = (x, y, z) -> directional_profile(y, 2),
        sigma_z = (x, y, z) -> directional_profile(z, 3),
        a = config.pml_a,
        regularization = config.pml_regularization,
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

function distributed_quadrature_diagnostics(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
)
    return distributed_cavity_quadrature_diagnostics(
        U,
        distributed_dg,
        time,
        cubature_order;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
end

function write_parallel_fields(
    output_basename::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField;
    time::Float64,
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    return write_parallel_maxwell_fields(
        output_basename,
        distributed_dg,
        U;
        time = time,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_paraview_snapshot!(
    entries,
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    step::Int,
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    return write_maxwell_paraview_snapshot!(
        entries,
        output_dir,
        distributed_dg,
        U,
        step,
        time;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_final_integration_points(
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    exact_curl_electric, exact_curl_magnetic =
        exact_cavity_mode_curl_functions(
            time;
            epsilon = epsilon,
            mu = mu,
            boundary_condition = boundary_condition,
        )
    return write_maxwell_integration_points(
        output_dir,
        distributed_dg,
        U,
        cubature_order;
        epsilon = epsilon,
        mu = mu,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
        exact_curl_electric = exact_curl_electric,
        exact_curl_magnetic = exact_curl_magnetic,
    )
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
            isfile(config.partition_path) ||
                error(
                    "Partition file not found: $(config.partition_path). " *
                    "Use 2 or 4 MPI ranks with the shipped mesh, or pass " *
                    "--partition PATH.",
                )

            mesh = load_pec_mesh(config.mesh_path)
            partition = read_metis_epart(config.partition_path)
            length(partition) == size(mesh.tets, 2) ||
                error(
                    "Partition has $(length(partition)) entries but the mesh " *
                    "has $(size(mesh.tets, 2)) tetrahedra.",
                )
            all(part -> 0 <= part < nranks, partition) ||
                error("Partition entries must be zero-based ranks in 0:$(nranks - 1).")
            all(part -> any(==(part), partition), 0:(nranks - 1)) ||
                error("Every MPI rank must own at least one tetrahedron.")
        catch error
            load_error = sprint(showerror, error)
        end
    end

    load_error = MPI.bcast(load_error, comm; root = 0)
    load_error === nothing || error(load_error)
    return mesh, partition
end

function validate_partition(
    partition::AbstractVector{<:Integer},
    mesh::RawVTUMesh,
    nranks::Int,
)
    length(partition) == size(mesh.tets, 2) ||
        error(
            "Partition has $(length(partition)) entries but the mesh " *
            "has $(size(mesh.tets, 2)) tetrahedra.",
        )
    all(part -> 0 <= part < nranks, partition) ||
        error("Partition entries must be zero-based ranks in 0:$(nranks - 1).")
    all(part -> any(==(part), partition), 0:(nranks - 1)) ||
        error("Every MPI rank must own at least one tetrahedron.")
    return nothing
end

function load_collective_inputs(
    config::ExperimentConfig,
    nranks::Int,
    comm::MPI.Comm,
)
    local_error = nothing
    mesh = nothing
    partition = nothing

    try
        isfile(config.mesh_path) ||
            error("Mesh file not found: $(config.mesh_path)")
        isfile(config.partition_path) ||
            error(
                "Partition file not found: $(config.partition_path). " *
                "Use 2 or 4 MPI ranks with the shipped mesh, or pass " *
                "--partition PATH.",
            )

        mesh = load_pec_mesh(config.mesh_path)
        partition = read_metis_epart(config.partition_path)
        validate_partition(partition, mesh, nranks)
    catch error
        local_error = sprint(showerror, error)
    end

    errors = MPI.gather(local_error, comm; root = 0)
    message = nothing
    if MPI.Comm_rank(comm) == 0
        failures = [
            "rank $(index - 1): $error"
            for (index, error) in enumerate(errors)
            if error !== nothing
        ]
        !isempty(failures) &&
            (message = "Collective mesh input loading failed:\n" *
                       join(failures, "\n"))
    end
    message = MPI.bcast(message, comm; root = 0)
    message === nothing || error(message)
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
        !config.rebuild_distributed_mesh,
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

    preparation_metadata = Dict(
        "source_mesh" => config.mesh_path,
        "source_partition" => config.partition_path,
        "boundary_condition" => uppercase(string(config.boundary_condition)),
        "collective_distributed_mesh_prep" =>
            config.collective_distributed_mesh_prep,
    )
    mesh_load_mode = if config.collective_distributed_mesh_prep
        mesh, partition = load_collective_inputs(config, nranks, comm)
        prepare_distributed_mesh_partition_collective(
            mesh,
            partition,
            config.distributed_mesh_dir;
            comm = comm,
            metadata = preparation_metadata,
        )
        "collectively prepared rank-local cache"
    else
        root_mesh, root_partition =
            load_root_inputs(config, rank, nranks, comm)
        prepare_distributed_mesh_partition(
            root_mesh,
            root_partition,
            config.distributed_mesh_dir;
            comm = comm,
            metadata = preparation_metadata,
        )
        "root-prepared rank-local cache"
    end
    return build_distributed_dg_from_partition(
        config.distributed_mesh_dir,
        config.polynomial_order;
        comm = comm,
    ), mesh_load_mode
end

function experiment_configuration(
    config::ExperimentConfig,
    mesh_load_mode::String,
    cubature_order::Int,
)
    return Dict{String, Any}(
        "mesh_path" => config.mesh_path,
        "partition_path" => config.partition_path,
        "distributed_mesh_dir" => config.distributed_mesh_dir,
        "collective_distributed_mesh_prep" =>
            config.collective_distributed_mesh_prep,
        "mesh_load_mode" => mesh_load_mode,
        "output_dir" => config.output_dir,
        "polynomial_order" => config.polynomial_order,
        "esprk_order" => config.esprk_order,
        "rk_order" => config.rk_order,
        "final_time" => config.final_time,
        "cfl" => config.cfl,
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "boundary_condition" => string(config.boundary_condition),
        "pml_enabled" => pml_enabled(config),
        "pml_width" => config.pml_width,
        "pml_sigma_max" => config.pml_sigma_max,
        "pml_degree" => config.pml_degree,
        "pml_a" => config.pml_a,
        "pml_regularization" => config.pml_regularization,
        "energy_every" => config.energy_every,
        "cubature_order" => cubature_order,
        "paraview_every" => config.paraview_every,
        "checkpoint_every" => config.checkpoint_every,
        "checkpoint_dir" => config.checkpoint_dir,
        "restart_path" => config.restart_path,
        "formulation" => "PoissonBracketFormulation",
        "flux" => "centered",
        "time_integrator" => (
            pml_enabled(config) ? "explicit RK" : "H-first ESPRK"
        ),
        "analytical_solution" => (
            "unit-cube $(uppercase(string(config.boundary_condition))) eigenmode"
        ),
        "optical_chirality_definition" =>
            "0.5*(epsilon*E dot curl(E) + mu*H dot curl(H))",
        "paraview_point_data" => [
            "ElectricField",
            "MagneticField",
            "ExactElectricField",
            "ExactMagneticField",
            "ElectricFieldError",
            "MagneticFieldError",
            "ElectricFieldMagnitude",
            "MagneticFieldMagnitude",
        ],
        "paraview_cell_data" => [
            "GlobalElementId",
            "OwnerRank",
            "PolynomialOrder",
        ],
        "paraview_field_data" => ["TimeValue"],
    )
end

function checkpoint_metadata(
    config::ExperimentConfig,
    initial_energy_total::Float64,
    mesh_load_mode::String,
)
    return Dict{String, Any}(
        "initial_energy_total" => initial_energy_total,
        "target_final_time" => config.final_time,
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "esprk_order" => config.esprk_order,
        "rk_order" => config.rk_order,
        "first_partition" => pml_enabled(config) ? "none" : "H",
        "boundary_condition" => string(config.boundary_condition),
        "pml_enabled" => pml_enabled(config),
        "pml_width" => config.pml_width,
        "pml_sigma_max" => config.pml_sigma_max,
        "pml_degree" => config.pml_degree,
        "pml_a" => config.pml_a,
        "pml_regularization" => config.pml_regularization,
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

    for (name, expected) in (
        ("boundary_condition", string(config.boundary_condition)),
        ("pml_enabled", string(pml_enabled(config))),
        ("pml_degree", string(config.pml_degree)),
    )
        haskey(state.metadata, name) || continue
        string(state.metadata[name]) == expected ||
            throw(
                ArgumentError(
                    "Checkpoint $name=$(state.metadata[name]) does not " *
                    "match the requested value $expected.",
                ),
            )
    end

    numeric_configuration = if pml_enabled(config)
        (
            ("rk_order", Float64(config.rk_order)),
            ("pml_width", config.pml_width),
            ("pml_sigma_max", config.pml_sigma_max),
            ("pml_a", config.pml_a),
            ("pml_regularization", config.pml_regularization),
        )
    else
        (("esprk_order", Float64(config.esprk_order)),)
    end
    for (name, expected) in numeric_configuration
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

    if !pml_enabled(config)
        first_partition = get(state.metadata, "first_partition", "H")
        string(first_partition) == "H" ||
            throw(
                ArgumentError(
                    "The checkpoint does not use the required H-first " *
                    "Poisson-bracket integrator.",
                ),
            )
    end
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
        electric, magnetic = exact_cavity_mode_functions(
            0.0;
            epsilon = config.epsilon,
            mu = config.mu,
            boundary_condition = config.boundary_condition,
        )
        U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    end

    registry = MaxwellBoundaryRegistry(
        Dict(PEC_BOUNDARY_ID => configured_boundary_kind(config)),
    )
    formulation = PoissonBracketFormulation()
    pml = build_configured_nonlinear_pml(distributed_dg, config)
    scheme = if pml === nothing
        explicit_partitioned_symplectic_rk_scheme(
            config.esprk_order;
            first_partition = :H,
        )
    else
        explicit_rk_scheme(config.rk_order)
    end
    workspace = if pml === nothing
        MaxwellPartitionedRKWorkspace(U, scheme)
    else
        MaxwellRKWorkspace(U, scheme)
    end

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
        boundary_condition = config.boundary_condition,
    )

    collective_root_action(
        comm,
        "Diagnostic history initialization",
    ) do
        if write_history_header
            write_energy_header(energy_io)
            write_quadrature_diagnostics_header(quadrature_io)
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
        println("Distributed Poisson-bracket Maxwell experiment")
        println("----------------------------------------------")
        println("MPI ranks:            ", nranks)
        println("mesh:                 ", config.mesh_path)
        println("partition:            ", config.partition_path)
        println("distributed mesh:     ", config.distributed_mesh_dir)
        println("mesh load mode:       ", mesh_load_mode)
        println("DG order:             ", config.polynomial_order)
        println("time integrator:      ", scheme.name)
        println("cubature order:       ", cubature_order)
        println(
            "boundary condition:   ",
            uppercase(string(config.boundary_condition)),
            " on all exterior faces",
        )
        println(
            "analytical solution:  unit-cube ",
            uppercase(string(config.boundary_condition)),
            " eigenmode",
        )
        println("nonlinear PML:        ", pml === nothing ? "disabled" : "enabled")
        if pml !== nothing
            println("PML width:            ", config.pml_width)
            println("PML sigma max:        ", config.pml_sigma_max)
            println("PML degree:           ", config.pml_degree)
            println("PML a:                ", config.pml_a)
            println("PML regularization:   ", config.pml_regularization)
            println(
                "reference errors:     undamped cavity mode; diagnostic only",
            )
        end
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
        experiment_configuration(config, mesh_load_mode, cubature_order)
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
            start_time;
            epsilon = config.epsilon,
            mu = config.mu,
            boundary_condition = config.boundary_condition,
        )
    end

    checkpoint_info = checkpoint_metadata(
        config,
        initial_energy_total,
        mesh_load_mode,
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
                if pml === nothing
                    distributed_partitioned_symplectic_rk_step!(
                        U,
                        workspace,
                        scheme,
                        dt,
                        distributed_dg,
                        registry,
                        formulation;
                        ε = config.epsilon,
                        μ = config.mu,
                    )
                else
                    distributed_maxwell_nonlinear_pml_rk_step!(
                        U,
                        workspace,
                        scheme,
                        dt,
                        distributed_dg,
                        registry,
                        formulation,
                        pml,
                    )
                end
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
                    boundary_condition = config.boundary_condition,
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
                    time;
                    epsilon = config.epsilon,
                    mu = config.mu,
                    boundary_condition = config.boundary_condition,
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
            boundary_condition = config.boundary_condition,
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
