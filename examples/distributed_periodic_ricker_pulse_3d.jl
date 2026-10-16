#!/usr/bin/env julia

using MPI
using DiscoGMPI
using LinearAlgebra: cholesky, dot, ldiv!
using Printf

const DEFAULT_RICKER_MESH =
    "examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk"
const PROBE_FORMAT_CSV = "csv"
const PROBE_FORMAT_HDF5 = "hdf5"
const DEFAULT_PROBE_HDF5_CHUNK_ROWS = 8192
const DEFAULT_PRINT_EVERY_WHEN_FLAGGED = 1000
const PROBE_COLUMNS = (
    "step",
    "time",
    "probe_x",
    "probe_y",
    "probe_z",
    "owner_rank",
    "local_element",
    "global_element",
    "local_node",
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
    "exact_Ex",
    "exact_Ey",
    "exact_Ez",
    "exact_Hx",
    "exact_Hy",
    "exact_Hz",
    "electric_error_l2_node",
    "magnetic_error_l2_node",
)
const PROBE_INTEGER_COLUMNS = Set((1, 6, 7, 8, 9))

struct RickerPulseRunConfig
    mesh_path::String
    partition_path::String
    distributed_mesh_dir::String
    rebuild_distributed_mesh::Bool
    output_dir::String
    polynomial_order::Int
    esprk_order::Int
    final_time::Float64
    cfl::Float64
    epsilon::Float64
    mu::Float64
    amplitude::Float64
    frequency::Float64
    xi0::Union{Nothing, Float64}
    direction::NTuple{3, Float64}
    polarization::NTuple{3, Float64}
    periodic_wrap::Bool
    integer_window::Bool
    window_length::Float64
    flux_kind::MaxwellFluxKind
    energy_every::Int
    print_every::Int
    diagnostics_every::Int
    probe_every::Int
    cubature_order::Int
    paraview_every::Int
    probe::Union{Nothing, NTuple{3, Float64}}
    initialization::String
    probe_format::String
    probe_hdf5_chunk_rows::Int
end

struct RickerPulseParameters
    direction::NTuple{3, Float64}
    polarization::NTuple{3, Float64}
    magnetic_polarization::NTuple{3, Float64}
    amplitude::Float64
    frequency::Float64
    epsilon::Float64
    mu::Float64
    impedance::Float64
    speed::Float64
    xi0::Float64
    period_length::Float64
    periodic_wrap::Bool
end

struct ProbeSelection
    owner_rank::Int
    local_element::Int
    local_node::Int
    global_element::Int
    point::NTuple{3, Float64}
    distance::Float64
end

abstract type AbstractProbeWriter end

mutable struct CSVProbeWriter <: AbstractProbeWriter
    io::IO
    path::String
end

mutable struct HDF5ProbeWriter <: AbstractProbeWriter
    hdf5::Module
    file::Any
    dataset::Any
    buffer::Matrix{Float64}
    buffer_count::Int
    rows_written::Int
    path::String
end

usage() = """
Distributed 3D periodic Ricker-pulse Maxwell analysis
-----------------------------------------------------

This driver is the 3D DiscoGMPI analogue of the old 2D Ricker-pulse
dispersion/dissipation script. It initializes a divergence-free Ricker pulse,
advects it in a periodic box with the Poisson-bracket Maxwell operator, and
writes probe and conservation diagnostics.

Usage:
  mpiexec -n 2 julia --project=. examples/distributed_periodic_ricker_pulse_3d.jl [options]

Main options:
  --mesh=PATH                    periodic box VTK mesh
                                  default: $(DEFAULT_RICKER_MESH)
  --partition=PATH               optional element-to-rank partition file
  --distributed-mesh-dir=DIR     rank-local mesh cache
  --rebuild-distributed-mesh     rebuild the rank-local mesh cache
  --output-dir=DIR               output directory
                                  default: output/distributed_periodic_ricker_pulse_3d
  --run-root=DIR                 create an isolated timestamped run directory
  --run-name=NAME                name used with --run-root

Discretization:
  --order=N                      DG polynomial order, N >= 1
  --esprk-order=Q                ESPRK composition order
  --cfl=C                        CFL factor
  --flux=centered|alternating    Poisson-bracket numerical flux

Pulse:
  --frequency=F                  Ricker central frequency parameter f0
  --amplitude=A                  electric-field amplitude
  --epsilon=EPS                  permittivity
  --mu=MU                        permeability
  --direction=DX,DY,DZ           propagation direction
  --polarization=PX,PY,PZ        electric polarization, perpendicular to direction
  --xi0=X                        initial pulse center in direction coordinate
  --no-periodic-wrap             do not wrap the analytical pulse in the periodic direction
  --initialization=l2|interpolation
                                  initial DG field construction; default: l2

Time:
  --final-time=T                 final time
  --integer-window               choose dt so each window has an integer number of steps
  --no-integer-window            use only the CFL estimate and final-time adjustment
  --window-length=T              analysis window length used by --integer-window

Output:
  --probe=X,Y,Z                  requested probe point; nearest DG node is used
  --probe-every=K                write probe history every K steps
  --probe-format=csv|hdf5        probe history format; default: hdf5
  --probe-hdf5-chunk-rows=K      chunk length for HDF5 probe writes
                                  default: $(DEFAULT_PROBE_HDF5_CHUNK_ROWS)
  --energy-every=K               write energy.csv every K steps; 0 disables
  --print-every[=K]              print progress every K steps; 0 disables.
                                  If the value is omitted, K=$(DEFAULT_PRINT_EVERY_WHEN_FLAGGED)
  --diagnostics-every=K          write quadrature_diagnostics.csv every K steps; 0 disables
  --cubature-order=Q             quadrature order for diagnostics; 0 selects max(2,2N+4)
  --paraview-every=K             write fields/fields.pvd every K steps; 0 disables
  --help                         print this message
"""

function parse_option(args::Vector{String}, name::String, default::String)
    prefix = "--$name="
    for arg in args
        startswith(arg, prefix) && return arg[(lastindex(prefix) + 1):end]
    end
    return default
end

has_flag(args::Vector{String}, name::String) = any(==("--$name"), args)

function parse_interval_option(
    args::Vector{String},
    name::String,
    default::Int;
    flag_default::Union{Nothing, Int} = nothing,
)
    if has_flag(args, name)
        flag_default === nothing &&
            throw(ArgumentError("--$name requires a value."))
        return flag_default
    end
    return parse(Int, parse_option(args, name, string(default)))
end

function parse_vector3(value::String, name::String)
    parts = split(value, ',')
    length(parts) == 3 ||
        throw(ArgumentError("--$name must have the form a,b,c."))
    return Tuple(parse(Float64, strip(part)) for part in parts)
end

function parse_optional_vector3(value::String, name::String)
    isempty(strip(value)) && return nothing
    return parse_vector3(value, name)
end

function parse_optional_float(value::String)
    isempty(strip(value)) && return nothing
    return parse(Float64, value)
end

function normalize3(v::NTuple{3, Float64}, name::String)
    nrm = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    nrm > 0.0 || throw(ArgumentError("$name must be non-zero."))
    return (v[1] / nrm, v[2] / nrm, v[3] / nrm)
end

cross3(a::NTuple{3, Float64}, b::NTuple{3, Float64}) = (
    a[2] * b[3] - a[3] * b[2],
    a[3] * b[1] - a[1] * b[3],
    a[1] * b[2] - a[2] * b[1],
)

function parse_ricker_config(
    args::Vector{String},
    nranks::Int,
    run_stamp::String = utc_run_stamp(),
)
    if has_flag(args, "help")
        println(usage())
        return nothing
    end

    run_root = parse_option(args, "run-root", "")
    run_name = parse_option(args, "run-name", "periodic-ricker-pulse-3d")
    output_dir = parse_option(
        args,
        "output-dir",
        "output/distributed_periodic_ricker_pulse_3d",
    )
    if !isempty(run_root)
        output_dir = isolated_run_directory(
            run_root,
            run_name;
            stamp = run_stamp,
        )
    end

    mesh_path = abspath(parse_option(args, "mesh", DEFAULT_RICKER_MESH))
    partition_raw = strip(parse_option(args, "partition", ""))
    partition_path = isempty(partition_raw) ? "" : abspath(partition_raw)
    distributed_mesh_dir = parse_option(args, "distributed-mesh-dir", "")
    if isempty(distributed_mesh_dir)
        mesh_label = splitext(basename(mesh_path))[1]
        distributed_mesh_dir = joinpath(
            "output",
            "distributed_mesh_cache",
            "$(mesh_label)_ricker3d_ranks$(nranks)",
        )
    end

    order = parse(Int, parse_option(args, "order", "2"))
    esprk_order = parse(Int, parse_option(args, "esprk-order", string(order + 1)))
    final_time = parse(Float64, parse_option(args, "final-time", "10.0"))
    cfl = parse(Float64, parse_option(args, "cfl", "0.5"))
    epsilon = parse(Float64, parse_option(args, "epsilon", "1.0"))
    mu = parse(Float64, parse_option(args, "mu", "1.0"))
    amplitude = parse(Float64, parse_option(args, "amplitude", "1.0"))
    frequency = parse(Float64, parse_option(args, "frequency", "1.0"))
    xi0 = parse_optional_float(parse_option(args, "xi0", ""))
    direction = parse_vector3(parse_option(args, "direction", "1,0,0"), "direction")
    polarization =
        parse_vector3(parse_option(args, "polarization", "0,0,1"), "polarization")

    flux_kind = parse_maxwell_flux_kind(parse_option(args, "flux", "centered"))
    flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating) ||
        throw(
            ArgumentError(
                "This Poisson-bracket Ricker driver supports centered and alternating fluxes.",
            ),
        )

    periodic_wrap = !has_flag(args, "no-periodic-wrap")
    integer_window =
        has_flag(args, "integer-window") || !has_flag(args, "no-integer-window")
    window_length = parse(Float64, parse_option(args, "window-length", "1.0"))

    energy_every = parse_interval_option(args, "energy-every", 100)
    print_every = parse_interval_option(
        args,
        "print-every",
        0;
        flag_default = DEFAULT_PRINT_EVERY_WHEN_FLAGGED,
    )
    diagnostics_every = parse_interval_option(args, "diagnostics-every", 0)
    probe_every = parse_interval_option(args, "probe-every", 1)
    cubature_order = parse_interval_option(args, "cubature-order", 0)
    paraview_every = parse_interval_option(args, "paraview-every", 0)
    probe = parse_optional_vector3(parse_option(args, "probe", ""), "probe")
    initialization = lowercase(strip(parse_option(args, "initialization", "l2")))
    probe_format =
        lowercase(strip(parse_option(args, "probe-format", PROBE_FORMAT_HDF5)))
    probe_hdf5_chunk_rows = parse(
        Int,
        parse_option(
            args,
            "probe-hdf5-chunk-rows",
            string(DEFAULT_PROBE_HDF5_CHUNK_ROWS),
        ),
    )

    order >= 1 || throw(ArgumentError("--order must be >= 1."))
    esprk_order >= 1 || throw(ArgumentError("--esprk-order must be >= 1."))
    final_time > 0.0 || throw(ArgumentError("--final-time must be positive."))
    cfl > 0.0 || throw(ArgumentError("--cfl must be positive."))
    epsilon > 0.0 || throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 || throw(ArgumentError("--mu must be positive."))
    frequency > 0.0 || throw(ArgumentError("--frequency must be positive."))
    window_length > 0.0 || throw(ArgumentError("--window-length must be positive."))
    energy_every >= 0 || throw(ArgumentError("--energy-every must be non-negative."))
    print_every >= 0 || throw(ArgumentError("--print-every must be non-negative."))
    diagnostics_every >= 0 ||
        throw(ArgumentError("--diagnostics-every must be non-negative."))
    probe_every >= 0 || throw(ArgumentError("--probe-every must be non-negative."))
    cubature_order >= 0 ||
        throw(ArgumentError("--cubature-order must be non-negative."))
    paraview_every >= 0 ||
        throw(ArgumentError("--paraview-every must be non-negative."))
    initialization in ("l2", "interpolation") ||
        throw(ArgumentError("--initialization must be l2 or interpolation."))
    probe_format in (PROBE_FORMAT_CSV, PROBE_FORMAT_HDF5) ||
        throw(ArgumentError("--probe-format must be csv or hdf5."))
    probe_hdf5_chunk_rows > 0 ||
        throw(ArgumentError("--probe-hdf5-chunk-rows must be positive."))

    return RickerPulseRunConfig(
        mesh_path,
        partition_path,
        abspath(distributed_mesh_dir),
        has_flag(args, "rebuild-distributed-mesh"),
        abspath(output_dir),
        order,
        esprk_order,
        final_time,
        cfl,
        epsilon,
        mu,
        amplitude,
        frequency,
        xi0,
        direction,
        polarization,
        periodic_wrap,
        integer_window,
        window_length,
        flux_kind,
        energy_every,
        print_every,
        diagnostics_every,
        probe_every,
        cubature_order,
        paraview_every,
        probe,
        initialization,
        probe_format,
        probe_hdf5_chunk_rows,
    )
end

function ricker_profile_and_derivative(frequency::Float64, s::Float64)
    a = pi * frequency * s
    expa = exp(-a^2)
    g = (1.0 - 2.0 * a^2) * expa
    dgds = pi * frequency * (-6.0 * a + 4.0 * a^3) * expa
    return g, dgds
end

function direction_axis_if_aligned(direction::NTuple{3, Float64}; tol = 1e-12)
    absdir = (abs(direction[1]), abs(direction[2]), abs(direction[3]))
    value, axis = findmax(absdir)
    value > 1.0 - tol ||
        throw(
            ArgumentError(
                "Periodic analytical wrapping currently requires an axis-aligned direction.",
            ),
        )
    for i in 1:3
        i == axis && continue
        absdir[i] <= tol ||
            throw(
                ArgumentError(
                    "Periodic analytical wrapping currently requires an axis-aligned direction.",
                ),
            )
    end
    return axis
end

function ricker_parameters(config::RickerPulseRunConfig, box::AxisAlignedBox)
    direction = normalize3(config.direction, "direction")
    polarization = normalize3(config.polarization, "polarization")
    abs(dot(direction, polarization)) <= 1e-12 ||
        throw(
            ArgumentError(
                "--polarization must be perpendicular to --direction for a divergence-free pulse.",
            ),
        )

    lengths = axis_aligned_box_lengths(box)
    period_length = Inf
    xi0 = config.xi0
    if config.periodic_wrap
        axis = direction_axis_if_aligned(direction)
        period_length = lengths[axis]
        period_length > 0.0 ||
            throw(ArgumentError("Periodic direction has zero domain length."))
        if xi0 === nothing
            coordinate_min = direction[axis] > 0.0 ? box.lower[axis] : -box.upper[axis]
            xi0 = coordinate_min + 0.25 * period_length
        end
    else
        xi0 = xi0 === nothing ? 0.0 : xi0
    end

    magnetic_polarization = cross3(direction, polarization)
    impedance = sqrt(config.mu / config.epsilon)
    speed = 1.0 / sqrt(config.epsilon * config.mu)

    return RickerPulseParameters(
        direction,
        polarization,
        magnetic_polarization,
        config.amplitude,
        config.frequency,
        config.epsilon,
        config.mu,
        impedance,
        speed,
        xi0::Float64,
        period_length,
        config.periodic_wrap,
    )
end

function wrapped_distance(s::Float64, period::Float64)
    return s - period * round(s / period)
end

function pulse_coordinate(pulse::RickerPulseParameters, x, y, z, time)
    s = pulse.direction[1] * x +
        pulse.direction[2] * y +
        pulse.direction[3] * z -
        pulse.speed * time -
        pulse.xi0
    return pulse.periodic_wrap ? wrapped_distance(s, pulse.period_length) : s
end

function ricker_electric_function(pulse::RickerPulseParameters, time::Float64)
    return function electric(x, y, z)
        s = pulse_coordinate(pulse, x, y, z, time)
        g, _ = ricker_profile_and_derivative(pulse.frequency, s)
        scale = pulse.amplitude * g
        return (
            scale * pulse.polarization[1],
            scale * pulse.polarization[2],
            scale * pulse.polarization[3],
        )
    end
end

function ricker_magnetic_function(pulse::RickerPulseParameters, time::Float64)
    return function magnetic(x, y, z)
        s = pulse_coordinate(pulse, x, y, z, time)
        g, _ = ricker_profile_and_derivative(pulse.frequency, s)
        scale = pulse.amplitude * g / pulse.impedance
        return (
            scale * pulse.magnetic_polarization[1],
            scale * pulse.magnetic_polarization[2],
            scale * pulse.magnetic_polarization[3],
        )
    end
end

function ricker_curl_electric_function(pulse::RickerPulseParameters, time::Float64)
    return function curl_electric(x, y, z)
        s = pulse_coordinate(pulse, x, y, z, time)
        _, dgds = ricker_profile_and_derivative(pulse.frequency, s)
        scale = pulse.amplitude * dgds
        return (
            scale * pulse.magnetic_polarization[1],
            scale * pulse.magnetic_polarization[2],
            scale * pulse.magnetic_polarization[3],
        )
    end
end

function ricker_curl_magnetic_function(pulse::RickerPulseParameters, time::Float64)
    curl_direction = cross3(pulse.direction, pulse.magnetic_polarization)
    return function curl_magnetic(x, y, z)
        s = pulse_coordinate(pulse, x, y, z, time)
        _, dgds = ricker_profile_and_derivative(pulse.frequency, s)
        scale = pulse.amplitude * dgds / pulse.impedance
        return (
            scale * curl_direction[1],
            scale * curl_direction[2],
            scale * curl_direction[3],
        )
    end
end

function exact_functions(pulse::RickerPulseParameters, time::Float64)
    return (
        electric = ricker_electric_function(pulse, time),
        magnetic = ricker_magnetic_function(pulse, time),
        curl_electric = ricker_curl_electric_function(pulse, time),
        curl_magnetic = ricker_curl_magnetic_function(pulse, time),
    )
end

function project_maxwell_field_l2(
    distributed_dg,
    electric,
    magnetic,
    cubature_order::Int,
)
    mesh = distributed_dg.dg.mesh
    ref = distributed_dg.dg.ref
    cubature_points, cubature_weights, nq =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation = reference_interpolation_matrix(ref, cubature_points)
    mass_factor = cholesky(ref.M)
    ntets = size(mesh.tets, 2)

    Ex = zeros(Float64, ref.Np, ntets)
    Ey = zeros(Float64, ref.Np, ntets)
    Ez = zeros(Float64, ref.Np, ntets)
    Hx = zeros(Float64, ref.Np, ntets)
    Hy = zeros(Float64, ref.Np, ntets)
    Hz = zeros(Float64, ref.Np, ntets)

    rhs = zeros(Float64, ref.Np, 6)
    tmp = zeros(Float64, ref.Np)

    for element in 1:ntets
        fill!(rhs, 0.0)
        tet_nodes = @view mesh.tets[:, element]
        for q in 1:nq
            x, y, z = DiscoGMPI.map_to_physical(
                mesh.points,
                tet_nodes,
                cubature_points[q, 1],
                cubature_points[q, 2],
                cubature_points[q, 3],
            )
            ex, ey, ez = electric(x, y, z)
            hx, hy, hz = magnetic(x, y, z)
            weight = cubature_weights[q]
            @inbounds for node in 1:ref.Np
                phi = interpolation[q, node] * weight
                rhs[node, 1] += phi * ex
                rhs[node, 2] += phi * ey
                rhs[node, 3] += phi * ez
                rhs[node, 4] += phi * hx
                rhs[node, 5] += phi * hy
                rhs[node, 6] += phi * hz
            end
        end

        @views begin
            tmp .= rhs[:, 1]
            ldiv!(mass_factor, tmp)
            Ex[:, element] .= tmp
            tmp .= rhs[:, 2]
            ldiv!(mass_factor, tmp)
            Ey[:, element] .= tmp
            tmp .= rhs[:, 3]
            ldiv!(mass_factor, tmp)
            Ez[:, element] .= tmp
            tmp .= rhs[:, 4]
            ldiv!(mass_factor, tmp)
            Hx[:, element] .= tmp
            tmp .= rhs[:, 5]
            ldiv!(mass_factor, tmp)
            Hy[:, element] .= tmp
            tmp .= rhs[:, 6]
            ldiv!(mass_factor, tmp)
            Hz[:, element] .= tmp
        end
    end

    return MaxwellField(Ex, Ey, Ez, Hx, Hy, Hz)
end

function should_write(step::Int, every::Int, final_step::Int)
    every > 0 || return false
    return step == 0 || step == final_step || step % every == 0
end

function load_or_prepare_distributed_dg(config::RickerPulseRunConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    manifest_path = joinpath(config.distributed_mesh_dir, "mesh_manifest.toml")
    use_cached =
        isfile(manifest_path) &&
        !config.rebuild_distributed_mesh &&
        isempty(config.partition_path)

    if use_cached
        return build_distributed_dg_from_partition(
            config.distributed_mesh_dir,
            config.polynomial_order;
            comm = comm,
        ), "cached-rank-local"
    end

    root_mesh = nothing
    root_partition = nothing
    root_error = nothing
    if rank == 0
        try
            root_mesh = load_periodic_box_mesh(config.mesh_path)
            if isempty(config.partition_path)
                root_partition = balanced_spatial_partition(root_mesh, nranks)
            else
                root_partition = read_metis_epart(config.partition_path)
                validate_element_partition(
                    root_partition,
                    root_mesh,
                    nranks;
                    partition_count = nranks,
                    partition_path = config.partition_path,
                    mesh_path = config.mesh_path,
                )
            end
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, comm; root = 0)
    root_error === nothing || error("Ricker-pulse mesh preparation failed: $root_error")

    prepare_distributed_mesh_partition(
        root_mesh,
        root_partition,
        config.distributed_mesh_dir;
        comm = comm,
        root = 0,
        metadata = Dict(
            "source" => "distributed_periodic_ricker_pulse_3d",
            "mesh_path" => config.mesh_path,
            "partition_path" => config.partition_path,
            "partition_count" => nranks,
        ),
    )

    mode = isempty(config.partition_path) ? "balanced-root-partition" : "file-partition"
    return build_distributed_dg_from_partition(
        config.distributed_mesh_dir,
        config.polynomial_order;
        comm = comm,
    ), mode
end

function default_probe_point(config::RickerPulseRunConfig, box::AxisAlignedBox)
    config.probe !== nothing && return config.probe
    return (
        0.5 * (box.lower[1] + box.upper[1]),
        0.5 * (box.lower[2] + box.upper[2]),
        0.5 * (box.lower[3] + box.upper[3]),
    )
end

function local_best_probe(distributed_dg, requested::NTuple{3, Float64}, rank::Int)
    mesh = distributed_dg.dg.mesh
    ref = distributed_dg.dg.ref
    owned = distributed_dg.distributed_mesh.partition.owned
    global_ids = distributed_dg.distributed_mesh.elements.global_ids

    best_distance = Inf
    best_element = 0
    best_node = 0
    best_global = 0
    best_point = (NaN, NaN, NaN)

    for local_element in owned
        tet_nodes = @view mesh.tets[:, local_element]
        for node in 1:ref.Np
            x, y, z = DiscoGMPI.map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[node],
                ref.s[node],
                ref.t[node],
            )
            distance = sqrt(
                (x - requested[1])^2 +
                (y - requested[2])^2 +
                (z - requested[3])^2,
            )
            if distance < best_distance
                best_distance = distance
                best_element = local_element
                best_node = node
                best_global = global_ids[local_element]
                best_point = (x, y, z)
            end
        end
    end

    return ProbeSelection(
        rank,
        best_element,
        best_node,
        best_global,
        best_point,
        best_distance,
    )
end

function select_probe(distributed_dg, requested::NTuple{3, Float64})
    comm = distributed_dg.comm
    rank = MPI.Comm_rank(comm)
    local_probe = local_best_probe(distributed_dg, requested, rank)
    gathered = MPI.gather(local_probe, comm; root = 0)
    selected = nothing
    if rank == 0
        selected = reduce(
            (a, b) -> a.distance <= b.distance ? a : b,
            gathered,
        )
    end
    return MPI.bcast(selected, comm; root = 0)
end

function write_probe_header(io::IO)
    println(io, join(PROBE_COLUMNS, ','))
    return nothing
end

function probe_values(
    U::MaxwellField,
    probe::ProbeSelection,
    exact_electric,
    exact_magnetic,
    rank::Int,
    comm::MPI.Comm,
)
    values = zeros(Float64, 14)
    if rank == probe.owner_rank
        node = probe.local_node
        element = probe.local_element
        electric = (
            U.Ex[node, element],
            U.Ey[node, element],
            U.Ez[node, element],
        )
        magnetic = (
            U.Hx[node, element],
            U.Hy[node, element],
            U.Hz[node, element],
        )
        exact_e = exact_electric(probe.point...)
        exact_h = exact_magnetic(probe.point...)
        values[1:3] .= electric
        values[4:6] .= magnetic
        values[7:9] .= exact_e
        values[10:12] .= exact_h
        values[13] = sqrt(sum((electric[i] - exact_e[i])^2 for i in 1:3))
        values[14] = sqrt(sum((magnetic[i] - exact_h[i])^2 for i in 1:3))
    end
    return MPI.bcast(values, comm; root = probe.owner_rank)
end

function probe_record(
    step::Int,
    time::Float64,
    U::MaxwellField,
    probe::ProbeSelection,
    exact_electric,
    exact_magnetic,
    rank::Int,
    comm::MPI.Comm,
)
    values = probe_values(U, probe, exact_electric, exact_magnetic, rank, comm)
    record = Vector{Float64}(undef, length(PROBE_COLUMNS))
    record[1] = step
    record[2] = time
    record[3] = probe.point[1]
    record[4] = probe.point[2]
    record[5] = probe.point[3]
    record[6] = probe.owner_rank
    record[7] = probe.local_element
    record[8] = probe.global_element
    record[9] = probe.local_node
    record[10:end] .= values
    return record
end

function csv_probe_fields(record::AbstractVector{<:Real})
    fields = Vector{Any}(undef, length(record))
    for i in eachindex(record)
        fields[i] = i in PROBE_INTEGER_COLUMNS ? round(Int, record[i]) : record[i]
    end
    return fields
end

probe_writer_path(writer::CSVProbeWriter) = writer.path
probe_writer_path(writer::HDF5ProbeWriter) = writer.path

function load_hdf5_module()
    try
        @eval import HDF5
        return HDF5
    catch error
        message = sprint(showerror, error)
        throw(
            ArgumentError(
                "HDF5 probe output requires HDF5.jl. " *
                "Run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` " *
                "or use `--probe-format=csv` for short/debug runs. " *
                "Original error: $message",
            ),
        )
    end
end

function open_csv_probe_writer(paths::RunPaths)
    path = run_diagnostics_path(paths, "probe_signal.csv")
    io = open(path, "w")
    write_probe_header(io)
    return CSVProbeWriter(io, path)
end

function open_hdf5_probe_writer(
    paths::RunPaths,
    config::RickerPulseRunConfig,
    probe::ProbeSelection,
    pulse::RickerPulseParameters,
    dt::Float64,
    nsteps::Int,
)
    hdf5 = load_hdf5_module()
    path = run_diagnostics_path(paths, "probe_signal.h5")
    file = nothing
    try
        file = Base.invokelatest(hdf5.h5open, path, "w")
        chunk_rows = min(config.probe_hdf5_chunk_rows, max(nsteps + 1, 1))
        group = Base.invokelatest(hdf5.create_group, file, "probe")
        dataset = Base.invokelatest(
            hdf5.create_dataset,
            group,
            "data",
            Float64,
            ((length(PROBE_COLUMNS), 0), (length(PROBE_COLUMNS), -1));
            chunk = (length(PROBE_COLUMNS), chunk_rows),
        )

        Base.invokelatest(
            hdf5.write_dataset,
            group,
            "columns",
            collect(String, PROBE_COLUMNS),
        )
        Base.invokelatest(hdf5.write_dataset, group, "layout", "columns_by_samples")
        Base.invokelatest(hdf5.write_dataset, group, "probe_point", collect(probe.point))
        Base.invokelatest(hdf5.write_dataset, group, "owner_rank", [probe.owner_rank])
        Base.invokelatest(hdf5.write_dataset, group, "local_element", [probe.local_element])
        Base.invokelatest(hdf5.write_dataset, group, "global_element", [probe.global_element])
        Base.invokelatest(hdf5.write_dataset, group, "local_node", [probe.local_node])
        Base.invokelatest(hdf5.write_dataset, group, "dt", [dt])
        Base.invokelatest(hdf5.write_dataset, group, "nsteps", [nsteps])
        Base.invokelatest(hdf5.write_dataset, group, "probe_every", [config.probe_every])
        Base.invokelatest(hdf5.write_dataset, group, "direction", collect(pulse.direction))
        Base.invokelatest(
            hdf5.write_dataset,
            group,
            "electric_polarization",
            collect(pulse.polarization),
        )
        Base.invokelatest(
            hdf5.write_dataset,
            group,
            "magnetic_polarization",
            collect(pulse.magnetic_polarization),
        )
        Base.invokelatest(hdf5.write_dataset, group, "frequency", [pulse.frequency])
        Base.invokelatest(hdf5.write_dataset, group, "amplitude", [pulse.amplitude])
        Base.invokelatest(hdf5.write_dataset, group, "xi0", [pulse.xi0])
        Base.invokelatest(hdf5.write_dataset, group, "period_length", [pulse.period_length])
        Base.invokelatest(
            hdf5.write_dataset,
            group,
            "periodic_wrap",
            [pulse.periodic_wrap ? 1 : 0],
        )

        return HDF5ProbeWriter(
            hdf5,
            file,
            dataset,
            zeros(Float64, length(PROBE_COLUMNS), chunk_rows),
            0,
            0,
            path,
        )
    catch error
        if file !== nothing
            try
                Base.invokelatest(close, file)
            catch
            end
        end
        rethrow(error)
    end
end

function open_probe_writer(
    paths::RunPaths,
    config::RickerPulseRunConfig,
    probe::ProbeSelection,
    pulse::RickerPulseParameters,
    dt::Float64,
    nsteps::Int,
)
    if config.probe_format == PROBE_FORMAT_CSV
        return open_csv_probe_writer(paths)
    elseif config.probe_format == PROBE_FORMAT_HDF5
        return open_hdf5_probe_writer(paths, config, probe, pulse, dt, nsteps)
    else
        error("Unsupported probe format: $(config.probe_format)")
    end
end

function write_probe_sample!(writer::CSVProbeWriter, record::AbstractVector{<:Real})
    println(writer.io, join(csv_probe_fields(record), ','))
    return nothing
end

function flush_probe_writer!(writer::CSVProbeWriter)
    flush(writer.io)
    return nothing
end

function write_probe_sample!(writer::HDF5ProbeWriter, record::AbstractVector{<:Real})
    writer.buffer_count += 1
    writer.buffer[:, writer.buffer_count] .= record
    if writer.buffer_count == size(writer.buffer, 2)
        flush_probe_writer!(writer)
    end
    return nothing
end

function flush_probe_writer!(writer::HDF5ProbeWriter)
    writer.buffer_count == 0 && return nothing
    old_rows = writer.rows_written
    new_rows = old_rows + writer.buffer_count
    Base.invokelatest(
        writer.hdf5.set_extent_dims,
        writer.dataset,
        (length(PROBE_COLUMNS), new_rows),
    )
    Base.invokelatest(
        setindex!,
        writer.dataset,
        @view(writer.buffer[:, 1:writer.buffer_count]),
        :,
        (old_rows + 1):new_rows,
    )
    writer.rows_written = new_rows
    writer.buffer_count = 0
    return nothing
end

function close_probe_writer!(writer::CSVProbeWriter)
    close(writer.io)
    return nothing
end

function close_probe_writer!(writer::HDF5ProbeWriter)
    flush_probe_writer!(writer)
    Base.invokelatest(close, writer.file)
    return nothing
end

function compute_time_step(
    config::RickerPulseRunConfig,
    estimated_dt::Float64,
)
    if config.integer_window
        windows_float = config.final_time / config.window_length
        windows = round(Int, windows_float)
        abs(windows_float - windows) <= 100 * eps(Float64) * max(1.0, windows_float) ||
            throw(
                ArgumentError(
                    "--integer-window requires --final-time to be an integer multiple of --window-length.",
                ),
            )
        steps_per_window = max(1, ceil(Int, config.window_length / estimated_dt))
        dt = config.window_length / steps_per_window
        return dt, windows * steps_per_window, steps_per_window
    else
        nsteps = max(1, ceil(Int, config.final_time / estimated_dt))
        dt = config.final_time / nsteps
        return dt, nsteps, 0
    end
end

function resolved_cubature_order(config::RickerPulseRunConfig)
    config.cubature_order > 0 && return config.cubature_order
    return max(2, 2 * config.polynomial_order + 4)
end

function print_step_progress(
    step::Int,
    nsteps::Int,
    physical_time::Float64,
    final_time::Float64,
    start_wall::Float64,
)
    elapsed = time() - start_wall
    fraction = step / nsteps
    eta = fraction > 0.0 ? elapsed * (1.0 - fraction) / fraction : Inf
    @printf(
        "  step %d/%d, t = %.8e / %.8e, elapsed = %.2f s, eta = %.2f s\n",
        step,
        nsteps,
        physical_time,
        final_time,
        elapsed,
        eta,
    )
    flush(stdout)
    return nothing
end

function write_partition_metadata(paths::RunPaths, distributed_dg)
    comm = distributed_dg.comm
    rank = MPI.Comm_rank(comm)
    mesh = distributed_dg.dg.mesh
    partition = distributed_dg.distributed_mesh.partition
    local_partition_row = (
        rank = rank,
        owned = length(partition.owned),
        ghosts = length(partition.ghosts),
        local_elements = size(mesh.tets, 2),
        local_points = size(mesh.points, 2),
    )
    gathered = MPI.gather(local_partition_row, comm; root = 0)
    if rank == 0
        open(run_config_path(paths, "partition_metadata.csv"), "w") do io
            println(io, "rank,owned,ghosts,local_elements,local_points")
            for row in gathered
                println(
                    io,
                    join(
                        (
                            row.rank,
                            row.owned,
                            row.ghosts,
                            row.local_elements,
                            row.local_points,
                        ),
                        ',',
                    ),
                )
            end
        end
    end
    MPI.Barrier(comm)
    return nothing
end

function diagnostics_at(
    U::MaxwellField,
    distributed_dg,
    pulse::RickerPulseParameters,
    time::Float64,
    cubature_order::Int,
)
    exact = exact_functions(pulse, time)
    return distributed_maxwell_quadrature_diagnostics(
        U,
        distributed_dg,
        cubature_order;
        epsilon = pulse.epsilon,
        mu = pulse.mu,
        exact_electric = exact.electric,
        exact_magnetic = exact.magnetic,
        exact_curl_electric = exact.curl_electric,
        exact_curl_magnetic = exact.curl_magnetic,
    )
end

function run_ricker_pulse_3d(config::RickerPulseRunConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    distributed_dg, mesh_load_mode = load_or_prepare_distributed_dg(config, comm)
    box = distributed_axis_aligned_box(distributed_dg)
    pulse = ricker_parameters(config, box)
    cubature_order = resolved_cubature_order(config)
    exact0 = exact_functions(pulse, 0.0)
    U = if config.initialization == "l2"
        project_maxwell_field_l2(
            distributed_dg,
            exact0.electric,
            exact0.magnetic,
            cubature_order,
        )
    else
        interpolate_maxwell_field(
            distributed_dg,
            exact0.electric,
            exact0.magnetic,
        )
    end

    registry = MaxwellBoundaryRegistry(
        Dict(boundary_id => MaxwellBC_None for boundary_id in 1:6),
    )
    materials = homogeneous_maxwell_materials(
        size(distributed_dg.dg.mesh.tets, 2);
        epsilon = config.epsilon,
        permeability = config.mu,
    )
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        periodic_box_specs(box);
        materials = materials,
    )
    formulation = PoissonBracketFormulation(config.flux_kind)
    scheme = explicit_partitioned_symplectic_rk_scheme(
        config.esprk_order;
        first_partition = :H,
    )
    workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    rhs_function! = make_distributed_periodic_maxwell_rhs_function(
        distributed_dg,
        periodic,
        registry,
        formulation,
        materials,
    )

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
    dt, nsteps, steps_per_window = compute_time_step(config, estimated_dt)
    final_time = nsteps * dt
    directed_periodic_faces = MPI.Allreduce(length(periodic.faces), +, comm)

    paths = prepare_run_directory(config.output_dir, comm)
    write_partition_metadata(paths, distributed_dg)
    requested_probe = default_probe_point(config, box)
    probe = select_probe(distributed_dg, requested_probe)
    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )

    if rank == 0
        println("Distributed 3D periodic Ricker-pulse Maxwell analysis")
        println("------------------------------------------------------")
        println("MPI ranks:            ", nranks)
        println("mesh:                 ", config.mesh_path)
        println("mesh load mode:       ", mesh_load_mode)
        println("distributed mesh:     ", config.distributed_mesh_dir)
        println("DG / ESPRK order:     ", config.polynomial_order, " / ", config.esprk_order)
        println("time integrator:      ", scheme.name)
        println("flux:                 ", maxwell_flux_kind_label(config.flux_kind))
        println("domain lower:         ", box.lower)
        println("domain upper:         ", box.upper)
        println("periodic face pairs:  ", div(directed_periodic_faces, 2))
        println("direction:            ", pulse.direction)
        println("polarization E:       ", pulse.polarization)
        println("polarization H:       ", pulse.magnetic_polarization)
        println("frequency parameter:  ", pulse.frequency)
        println("amplitude:            ", pulse.amplitude)
        println("initialization:       ", config.initialization)
        println("xi0:                  ", pulse.xi0)
        println("periodic exact wrap:  ", pulse.periodic_wrap)
        println("global hmin:          ", global_hmin)
        println("estimated dt:         ", estimated_dt)
        println("used dt:              ", dt)
        if config.integer_window
            println("steps per window:     ", steps_per_window)
            println("window length:        ", config.window_length)
        end
        println("steps / final time:   ", nsteps, " / ", final_time)
        println("probe requested:      ", requested_probe)
        println("probe actual:         ", probe.point)
        println("probe owner/global el:", probe.owner_rank, " / ", probe.global_element)
        println("initial energy:       ", initial_energy.total)
        println("energy every:         ", config.energy_every)
        println("print every:          ", config.print_every)
        println("probe every:          ", config.probe_every)
        println("probe format:         ", config.probe_format)
        println("quadrature every:     ", config.diagnostics_every)
        println("ParaView every:       ", config.paraview_every)
        println("output directory:     ", paths.root)
    end

    configuration = Dict(
        "problem" => "distributed periodic 3D Ricker pulse",
        "mesh_path" => config.mesh_path,
        "partition_path" => config.partition_path,
        "distributed_mesh_dir" => config.distributed_mesh_dir,
        "mesh_load_mode" => mesh_load_mode,
        "mpi_ranks" => nranks,
        "polynomial_order" => config.polynomial_order,
        "esprk_order" => config.esprk_order,
        "flux" => maxwell_flux_kind_label(config.flux_kind),
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "amplitude" => config.amplitude,
        "frequency" => config.frequency,
        "initialization" => config.initialization,
        "direction" => collect(pulse.direction),
        "electric_polarization" => collect(pulse.polarization),
        "magnetic_polarization" => collect(pulse.magnetic_polarization),
        "xi0" => pulse.xi0,
        "periodic_exact_wrap" => pulse.periodic_wrap,
        "cfl" => config.cfl,
        "estimated_dt" => estimated_dt,
        "dt" => dt,
        "nsteps" => nsteps,
        "print_every" => config.print_every,
        "target_final_time" => config.final_time,
        "actual_final_time" => final_time,
        "global_hmin" => global_hmin,
        "probe_requested" => collect(requested_probe),
        "probe_actual" => collect(probe.point),
        "probe_owner_rank" => probe.owner_rank,
        "probe_global_element" => probe.global_element,
        "probe_format" => config.probe_format,
        "probe_hdf5_chunk_rows" => config.probe_hdf5_chunk_rows,
        "cubature_order" => cubature_order,
    )
    collectively_write_run_provenance(
        paths,
        comm;
        configuration = configuration,
        inputs = Dict(
            "mesh_path" => config.mesh_path,
            "partition_path" => config.partition_path,
            "distributed_mesh_dir" => config.distributed_mesh_dir,
        ),
    )
    write_distributed_run_metadata(
        paths.config_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "running",
            "initial_energy_total" => initial_energy.total,
        ),
    )
    collectively_write_run_status(
        paths,
        comm,
        "running";
        values = Dict(
            "nsteps" => nsteps,
            "dt" => dt,
            "target_final_time" => config.final_time,
            "actual_final_time" => final_time,
        ),
    )

    energy_io = nothing
    diagnostics_io = nothing
    probe_writer = nothing
    output_open_error = nothing
    if rank == 0
        try
            if config.energy_every > 0
                energy_io = open(run_diagnostics_path(paths, "energy.csv"), "w")
                write_energy_header(energy_io)
            end
            if config.diagnostics_every > 0
                diagnostics_io =
                    open(run_diagnostics_path(paths, "quadrature_diagnostics.csv"), "w")
                write_quadrature_diagnostics_header(diagnostics_io)
            end
            if config.probe_every > 0
                probe_writer = open_probe_writer(
                    paths,
                    config,
                    probe,
                    pulse,
                    dt,
                    nsteps,
                )
            end
        catch error
            output_open_error = sprint(showerror, error)
        end
    end
    output_open_error = MPI.bcast(output_open_error, comm; root = 0)
    output_open_error === nothing ||
        error("Diagnostic output opening failed: $output_open_error")

    series_entries = NamedTuple{
        (:step, :time, :dataset),
        Tuple{Int, Float64, String},
    }[]

    function write_outputs(step::Int, time::Float64)
        exact = exact_functions(pulse, time)
        if should_write(step, config.energy_every, nsteps)
            energy = distributed_maxwell_energy(
                U,
                distributed_dg;
                ε = config.epsilon,
                μ = config.mu,
            )
            if rank == 0
                write_energy_row(energy_io, step, time, energy, initial_energy.total)
                flush(energy_io)
            end
        end
        if should_write(step, config.diagnostics_every, nsteps)
            diagnostics =
                diagnostics_at(U, distributed_dg, pulse, time, cubature_order)
            if rank == 0
                write_quadrature_diagnostics_row(
                    diagnostics_io,
                    step,
                    time,
                    diagnostics,
                )
                flush(diagnostics_io)
            end
        end
        if should_write(step, config.probe_every, nsteps)
            record = probe_record(
                step,
                time,
                U,
                probe,
                exact.electric,
                exact.magnetic,
                rank,
                comm,
            )
            if rank == 0
                write_probe_sample!(probe_writer, record)
            end
        end
        if should_write(step, config.paraview_every, nsteps)
            write_maxwell_paraview_snapshot!(
                series_entries,
                paths.root,
                distributed_dg,
                U,
                step,
                time;
                exact_electric = exact.electric,
                exact_magnetic = exact.magnetic,
            )
        end
        return nothing
    end

    write_outputs(0, 0.0)
    start_wall = time()
    for step in 1:nsteps
        partitioned_symplectic_rk_step!(
            U,
            workspace,
            scheme,
            dt,
            rhs_function!,
        )
        physical_time = step * dt
        write_outputs(step, physical_time)
        if rank == 0 && should_write(step, config.print_every, nsteps)
            print_step_progress(
                step,
                nsteps,
                physical_time,
                final_time,
                start_wall,
            )
        end
    end
    elapsed = time() - start_wall

    if rank == 0
        energy_io === nothing || close(energy_io)
        diagnostics_io === nothing || close(diagnostics_io)
        probe_writer === nothing || close_probe_writer!(probe_writer)
    end

    final_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    write_distributed_run_metadata(
        paths.config_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "completed",
            "elapsed_seconds" => elapsed,
            "final_energy_total" => final_energy.total,
            "relative_energy_change" =>
                (final_energy.total - initial_energy.total) /
                max(abs(initial_energy.total), eps(Float64)),
        ),
    )
    collectively_write_run_status(
        paths,
        comm,
        "completed";
        values = Dict(
            "elapsed_seconds" => elapsed,
            "final_energy_total" => final_energy.total,
            "relative_energy_change" =>
                (final_energy.total - initial_energy.total) /
                max(abs(initial_energy.total), eps(Float64)),
        ),
    )

    if rank == 0
        println()
        println("Completed 3D Ricker-pulse run.")
        println("  elapsed seconds:          ", elapsed)
        println("  final energy:             ", final_energy.total)
        println(
            "  relative energy change:   ",
            (final_energy.total - initial_energy.total) /
            max(abs(initial_energy.total), eps(Float64)),
        )
        println("  diagnostics directory:    ", paths.diagnostics_dir)
        if config.probe_every > 0
            println("  probe history:            ", probe_writer_path(probe_writer))
        end
        if config.paraview_every > 0
            println("  ParaView collection:      ", joinpath(paths.fields_dir, "fields.pvd"))
        end
    end

    return nothing
end

function main(args::Vector{String})
    MPI.Init()
    comm = MPI.COMM_WORLD
    config = nothing
    try
        rank = MPI.Comm_rank(comm)
        run_stamp = rank == 0 ? utc_run_stamp() : ""
        run_stamp = MPI.bcast(run_stamp, comm; root = 0)
        config = parse_ricker_config(args, MPI.Comm_size(comm), run_stamp)
        config === nothing && return 0
        run_ricker_pulse_3d(config, comm)
        return 0
    catch error
        message = sprint(showerror, error)
        println(stderr, "ERROR: ", message)
        return 1
    finally
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
