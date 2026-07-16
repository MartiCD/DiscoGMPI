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
const DEFAULT_WAVE_NUMBER = 2.0 * pi
const MAGNETIC_AMPLITUDE = 1.0
const ExperimentConfig = DistributedPeriodicMaxwellExperimentConfig
const PeriodicBox = AxisAlignedBox

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
  --collective-distributed-mesh-prep
                        Recreate a missing or stale rank-local mesh cache by
                        having every rank read the global mesh and partition,
                        avoiding root-built rank-local mesh packets. If
                        partition generation is needed, rank zero still runs
                        mpmetis once before the collective read.
  --output-dir PATH     Output directory.
                        Default: output/distributed_periodic_poisson_bracket
  --run-root PATH       Create an isolated timestamped run directory below PATH.
                        Mutually exclusive with --output-dir.
  --run-name NAME       Human-readable run-name suffix for --run-root.
  --order N             DG polynomial order (N >= 1). On the default
                        structured mesh, order 1 aliases the prescribed
                        sine wave to zero. Default: 2
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
  --flux NAME           Poisson-bracket surface flux: centered or alternating.
                        Default: centered
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
  diagnostics/energy.csv           DG mass-matrix energy history.
  diagnostics/quadrature_diagnostics.csv
                                   Energy, chirality, L2 errors, charges,
                                   and momenta.
  config/run_layout.toml           Run-directory layout.
  config/resolved_config.toml      Fully resolved run configuration.
  config/run_metadata.toml         Runtime metadata.
  config/partition_metadata.csv    Per-rank ownership and halo metadata.
  input/input_manifest.toml        Mesh, partition, cache, restart inputs.
  status.toml                      Current run status.
  fields.pvd                       ParaView time-series collection.
  paraview_series.csv              Snapshot step/time index.
  fields/fields_stepNNNNNNNN.pvtu  Parallel field snapshot.
  checkpoints/stepNNNNNNNN/        Per-rank restart checkpoint.
  diagnostics/integration_points_rankNNNN.csv
                                   Final integration-point values per rank.
  energy.csv, quadrature_diagnostics.csv, run_metadata.toml, and
  partition_metadata.csv are compatibility symlinks at the run root.
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
    run_stamp::String,
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
    collective_distributed_mesh_prep = false
    output_dir =
        joinpath(
            repository_root,
            "output",
            "distributed_periodic_poisson_bracket",
        )
    output_dir_set = false
    run_root = ""
    run_name = ""
    polynomial_order = 2
    esprk_order = 4
    final_time = 0.25
    cfl = 0.05
    epsilon = 1.0
    mu = 1.0
    wave_number = DEFAULT_WAVE_NUMBER
    flux_kind = MaxwellFlux_Central
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
        elseif arg == "--collective-distributed-mesh-prep"
            collective_distributed_mesh_prep = true
        elseif startswith(arg, "--output-dir")
            value, i = option_value(args, i, "--output-dir")
            output_dir = abspath(value)
            output_dir_set = true
        elseif startswith(arg, "--run-root")
            value, i = option_value(args, i, "--run-root")
            run_root = abspath(value)
        elseif startswith(arg, "--run-name")
            value, i = option_value(args, i, "--run-name")
            run_name = value
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
        elseif startswith(arg, "--flux")
            value, i = option_value(args, i, "--flux")
            flux_kind = parse_maxwell_flux_kind(value)
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

    if !isempty(run_root)
        output_dir_set &&
            throw(ArgumentError("--run-root and --output-dir are mutually exclusive."))
        if isempty(run_name)
            run_name =
                "periodic-poisson-bracket_" *
                maxwell_flux_kind_label(flux_kind) *
                "_p$(polynomial_order)_r$(nranks)"
        end
        output_dir = isolated_run_directory(
            run_root,
            run_name;
            stamp = run_stamp,
        )
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

    polynomial_order >= 1 ||
        throw(ArgumentError("--order must be at least 1."))
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
    flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating) ||
        throw(ArgumentError("--flux must be centered or alternating."))
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
        collective_distributed_mesh_prep,
        output_dir,
        polynomial_order,
        esprk_order,
        final_time,
        cfl,
        epsilon,
        mu,
        wave_number,
        flux_kind,
        energy_every,
        cubature_order,
        paraview_every,
        checkpoint_every,
        checkpoint_dir,
        restart_path,
    )
end

load_periodic_mesh(mesh_path::String) =
    load_periodic_box_mesh(mesh_path)

periodic_box(points::Matrix{Float64}) = infer_axis_aligned_box(points)
periodic_box_lengths(box::PeriodicBox) = axis_aligned_box_lengths(box)
periodic_boundary_specs(box::PeriodicBox) = periodic_box_specs(box)
distributed_periodic_box(distributed_dg::DistributedDGDiscretization) =
    distributed_axis_aligned_box(distributed_dg)

function configured_plane_wave(
    config::ExperimentConfig,
    box::PeriodicBox,
)
    return plane_wave_for_periodic_box(
        box;
        wave_number = config.wave_number,
        epsilon = config.epsilon,
        mu = config.mu,
        magnetic_amplitude = MAGNETIC_AMPLITUDE,
    )
end

resolved_cubature_order(config::ExperimentConfig) =
    resolved_maxwell_cubature_order(
        config.polynomial_order,
        config.cubature_order,
    )

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

function ensure_partition_file(
    config::ExperimentConfig,
    rank::Int,
    comm::MPI.Comm,
)
    local_error = nothing
    if rank == 0
        try
            if config.repartition || !isfile(config.partition_path)
                generate_metis_partition(config)
            end
        catch error
            local_error = sprint(showerror, error)
        end
    end
    local_error = MPI.bcast(local_error, comm; root = 0)
    local_error === nothing || error(local_error)
    MPI.Barrier(comm)
    return config.partition_path
end

function validate_partition(
    partition::AbstractVector{<:Integer},
    mesh::RawVTUMesh,
    config::ExperimentConfig,
    nranks::Int,
)
    return validate_element_partition(
        partition,
        mesh,
        nranks;
        partition_count = config.partition_count,
        partition_path = config.partition_path,
        mesh_path = config.mesh_path,
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

            mesh = load_periodic_mesh(config.mesh_path)
            if config.repartition || !isfile(config.partition_path)
                generate_metis_partition(config)
            end
            partition = read_metis_epart(config.partition_path)
            validate_partition(partition, mesh, config, nranks)
        catch error
            load_error = sprint(showerror, error)
        end
    end

    load_error = MPI.bcast(load_error, comm; root = 0)
    load_error === nothing || error(load_error)
    return mesh, partition
end

function load_collective_inputs(
    config::ExperimentConfig,
    rank::Int,
    nranks::Int,
    comm::MPI.Comm,
)
    ensure_partition_file(config, rank, comm)

    local_error = nothing
    mesh = nothing
    partition = nothing

    try
        isfile(config.mesh_path) ||
            error("Mesh file not found: $(config.mesh_path)")
        mesh = load_periodic_mesh(config.mesh_path)
        partition = read_metis_epart(config.partition_path)
        validate_partition(partition, mesh, config, nranks)
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

    preparation_metadata = Dict(
        "source_mesh" => config.mesh_path,
        "source_partition" => config.partition_path,
        "partition_count" => config.partition_count,
        "boundary_condition" => "periodic",
        "collective_distributed_mesh_prep" =>
            config.collective_distributed_mesh_prep,
    )
    mesh_load_mode = if config.collective_distributed_mesh_prep
        mesh, partition =
            load_collective_inputs(config, rank, nranks, comm)
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
    box::PeriodicBox,
    wave::PlaneWaveParameters,
)
    return Dict{String, Any}(
        "mesh_path" => config.mesh_path,
        "partition_path" => config.partition_path,
        "partition_count" => config.partition_count,
        "repartition" => config.repartition,
        "distributed_mesh_dir" => config.distributed_mesh_dir,
        "collective_distributed_mesh_prep" =>
            config.collective_distributed_mesh_prep,
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
        "flux" => maxwell_flux_kind_label(config.flux_kind),
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
        "flux" => maxwell_flux_kind_label(config.flux_kind),
        "distributed_mesh_dir" => config.distributed_mesh_dir,
        "mesh_load_mode" => mesh_load_mode,
    )
end

function validate_restart_configuration(
    state::DistributedCheckpointState,
    config::ExperimentConfig,
)
    validate_restart_numbers(
        state,
        (
            ("epsilon", config.epsilon),
            ("mu", config.mu),
            ("wave_number", config.wave_number),
            ("esprk_order", Float64(config.esprk_order)),
        ),
    )
    validate_restart_strings(
        state,
        (
            ("first_partition", "H"),
            ("flux", maxwell_flux_kind_label(config.flux_kind)),
        ),
    )
    return nothing
end

function remaining_time_step(
    final_time::Float64,
    start_time::Float64,
    checkpoint_dt::Float64,
)
    plan = restart_time_plan(final_time, start_time, checkpoint_dt)
    return plan.dt, plan.nsteps
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
    formulation = PoissonBracketFormulation(config.flux_kind)
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

    run_paths = prepare_run_directory(config.output_dir, comm)
    checkpoint_dir =
        default_checkpoint_dir(config.output_dir, config.checkpoint_dir, run_paths)

    energy_path = run_diagnostics_path(run_paths, "energy.csv")
    quadrature_path =
        run_diagnostics_path(run_paths, "quadrature_diagnostics.csv")
    history_files = DiagnosticHistoryFiles(nothing, nothing, false)
    history_open_error = nothing
    if rank == 0
        try
            history_files = open_diagnostic_history_files(
                energy_path,
                quadrature_path;
                restarting = restarting,
            )
        catch error
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
    current_quadrature = distributed_periodic_quadrature_diagnostics(
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
        if history_files.write_header
            write_energy_header(history_files.energy_io)
            write_quadrature_diagnostics_header(history_files.quadrature_io)
            write_energy_row(
                history_files.energy_io,
                start_step,
                start_time,
                current_energy,
                initial_energy_total,
            )
            write_quadrature_diagnostics_row(
                history_files.quadrature_io,
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
        println("checkpoint directory: ", checkpoint_dir)
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
    collectively_write_run_provenance(
        run_paths,
        comm;
        configuration = configuration,
        inputs = Dict(
            "mesh_path" => config.mesh_path,
            "partition_path" => config.partition_path,
            "distributed_mesh_dir" => config.distributed_mesh_dir,
            "mesh_load_mode" => mesh_load_mode,
            "checkpoint_dir" => checkpoint_dir,
            "restart_path" => config.restart_path,
        ),
    )
    write_distributed_run_metadata(
        run_paths.config_dir,
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
    collectively_write_run_status(
        run_paths,
        comm,
        "running";
        values = Dict(
            "start_step" => start_step,
            "start_time" => start_time,
            "final_step" => final_step,
            "target_final_time" => config.final_time,
            "checkpoint_dir" => checkpoint_dir,
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
        write_periodic_paraview_snapshot!(
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
                final_quadrature = distributed_periodic_quadrature_diagnostics(
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
                        history_files.energy_io,
                        step,
                        time,
                        final_energy,
                        initial_energy_total,
                    )
                    write_quadrature_diagnostics_row(
                        history_files.quadrature_io,
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
                write_periodic_paraview_snapshot!(
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
                    checkpoint_step_dir(checkpoint_dir, step)
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
                    checkpoint_dir,
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
            close_diagnostic_history_files(history_files)
        end
    end

    collective_rank_action(
        comm,
        "Final integration-point output",
    ) do
        write_periodic_integration_points(
            run_paths.diagnostics_dir,
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
        run_paths.config_dir,
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
    collectively_write_run_status(
        run_paths,
        comm,
        "complete";
        values = Dict(
            "final_step" => final_step,
            "final_time" => final_time,
            "integration_wall_seconds" => elapsed,
            "final_checkpoint" => final_checkpoint_path,
        ),
    )

    if rank == 0
        println()
        println("Energy history:        ", energy_path)
        println("Quadrature diagnostics:", quadrature_path)
        println(
            "Run metadata:          ",
            run_config_path(run_paths, "run_metadata.toml"),
        )
        println(
            "Resolved config:       ",
            run_config_path(run_paths, "resolved_config.toml"),
        )
        println(
            "Input manifest:        ",
            run_input_path(run_paths, "input_manifest.toml"),
        )
        println("Run status:            ", joinpath(run_paths.root, "status.toml"))
        println(
            "Partition metadata:    ",
            run_config_path(run_paths, "partition_metadata.csv"),
        )
        println(
            "Integration points:    ",
            run_diagnostics_path(run_paths, "integration_points_rankNNNN.csv"),
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
        run_stamp = MPI.bcast(rank == 0 ? utc_run_stamp() : "", comm; root = 0)
        config = parse_arguments(args, nranks, repository_root, run_stamp)

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
