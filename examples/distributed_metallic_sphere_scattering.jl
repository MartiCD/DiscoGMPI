#!/usr/bin/env julia

using MPI
using Printf
using LinearAlgebra
using WriteVTK
using VTKBase
using DiscoGMPI

const DEFAULT_METALLIC_SPHERE_MESH = joinpath(
    @__DIR__,
    "meshes",
    "metallic_sphere_scattering.vtk",
)

const OUTER_PML_BOUNDARY_IDS = 1:6
const METALLIC_SPHERE_PEC_BOUNDARY_ID = 10

Base.@kwdef struct MetallicSphereScatteringConfig
    mesh::String = DEFAULT_METALLIC_SPHERE_MESH
    radius::Float64 = 1.0
    wavelength::Float64 = 1.0
    frequency::Float64 = 0.0
    amplitude::Float64 = 1.0
    order::Int = 2
    esprk_order::Int = 3
    final_time::Float64 = 0.0
    cfl::Float64 = 0.15
    epsilon::Float64 = 1.0
    mu::Float64 = 1.0
    pml_width::Float64 = 1.0
    sigma_max::Float64 = 12.0
    sigma_degree::Int = 2
    flux::MaxwellFluxKind = MaxwellFlux_Central
    pml_a::Float64 = 0.5
    regularization::Float64 = 1.0e-12
    diagnostics_every::Int = 10
    paraview_every::Int = 50
    rcs_enabled::Bool = false
    rcs_start_time::Float64 = -1.0
    rcs_every::Int = 10
    rcs_theta_count::Int = 181
    rcs_theta_min_degrees::Float64 = 0.0
    rcs_theta_max_degrees::Float64 = 180.0
    rcs_phi_degrees::Vector{Float64} = [0.0]
    output_dir::String = joinpath(
        normpath(joinpath(@__DIR__, "..")),
        "output",
        "metallic_sphere_scattering",
    )
end

function usage(io::IO = stdout)
    println(io, """
Distributed metallic-sphere scattering driver

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_metallic_sphere_scattering.jl [options]

Core options:
  --mesh PATH                  Legacy ASCII VTK tetrahedral mesh. Default:
                               examples/meshes/metallic_sphere_scattering.vtk
  --radius R                   Metallic sphere radius used for validation.
                               Default: 1
  --wavelength L               Incident wavelength. Default: 1
  --frequency F                Incident frequency in cycles per time. If F > 0,
                               it overrides wavelength through L = c/F.
                               Default: 0
  --amplitude A                Incident electric-field amplitude. Default: 1
  --order N                    DG polynomial order. Default: 2
  --esprk-order N              ESPRK order in 1:6. Default: 3
  --final-time T               Final simulation time. Default: two periods
  --cfl C                      Maxwell CFL factor. Default: 0.15
  --pml-width W                Cartesian PML width on each box side. Default: 1
  --sigma-max S                Maximum PML damping. Default: 12
  --sigma-degree N             Polynomial PML degree. Default: 2
  --flux centered|alternating|upwind
                               Poisson-bracket numerical flux. Default: centered
  --output-dir PATH            Isolated run output directory.

Additional options:
  --epsilon E                  Constant permittivity. Default: 1
  --mu M                       Constant permeability. Default: 1
  --pml-a A                    Nonlinear PML parameter in (0,1). Default: 0.5
  --regularization R           PML denominator regularization. Default: 1e-12
  --diagnostics-every N        Diagnostics interval in steps. Default: 10
  --paraview-every N           ParaView interval; 0 disables snapshots.
                               Default: 50
  --enable-rcs                 Enable PEC surface-current RCS extraction.
  --disable-rcs                Disable RCS extraction. Default
  --rcs-start-time T           Start Fourier extraction at T. Default: last
                               incident period, or 0 for shorter runs
  --rcs-every N                RCS sampling interval in time steps. Default: 10
  --rcs-theta-count N          Number of polar theta samples. Default: 181
  --rcs-theta-min-degrees A    Minimum theta angle. Default: 0
  --rcs-theta-max-degrees A    Maximum theta angle. Default: 180
  --rcs-phi-degrees LIST       Comma-separated azimuth angles. Default: 0
  --help                       Show this message.

The driver solves for the scattered field. The inner sphere enforces the
incident-aware PEC condition n x E_scat = -n x E_inc. ParaView snapshots write
explicit scattered, incident, and total fields. The RCS diagnostic uses the
metallic sphere PEC surface current J = n x H_total and a time-windowed Fourier
extraction at the incident frequency. The PEC residual diagnostic writes the
sphere-surface norm ||n x E_total|| to diagnostics/pec_boundary_residual.csv.
""")
end

function option_value(args::Vector{String}, index::Int, option::String)
    argument = args[index]
    prefix = "$option="
    if startswith(argument, prefix)
        value = argument[(length(prefix) + 1):end]
        isempty(value) && error("$option requires a value.")
        return value, index
    end
    argument == option || error("Unknown option '$argument'.")
    index < length(args) || error("$option requires a value.")
    return args[index + 1], index + 1
end

function parse_comma_list(value::AbstractString, option::AbstractString)
    entries = [strip(item) for item in split(value, ',')]
    filter!(!isempty, entries)
    isempty(entries) && error("$option requires at least one value.")
    return entries
end

function parse_float_list(value::AbstractString, option::AbstractString)
    return [parse(Float64, item) for item in parse_comma_list(value, option)]
end

function parse_config(args::Vector{String})
    values = Dict{Symbol, Any}(
        field => getfield(MetallicSphereScatteringConfig(), field)
        for field in fieldnames(MetallicSphereScatteringConfig)
    )
    option_fields = Dict(
        "--mesh" => (:mesh, String),
        "--radius" => (:radius, Float64),
        "--wavelength" => (:wavelength, Float64),
        "--frequency" => (:frequency, Float64),
        "--amplitude" => (:amplitude, Float64),
        "--order" => (:order, Int),
        "--esprk-order" => (:esprk_order, Int),
        "--final-time" => (:final_time, Float64),
        "--cfl" => (:cfl, Float64),
        "--epsilon" => (:epsilon, Float64),
        "--mu" => (:mu, Float64),
        "--pml-width" => (:pml_width, Float64),
        "--sigma-max" => (:sigma_max, Float64),
        "--sigma-degree" => (:sigma_degree, Int),
        "--flux" => (:flux, MaxwellFluxKind),
        "--pml-a" => (:pml_a, Float64),
        "--regularization" => (:regularization, Float64),
        "--diagnostics-every" => (:diagnostics_every, Int),
        "--paraview-every" => (:paraview_every, Int),
        "--rcs-start-time" => (:rcs_start_time, Float64),
        "--rcs-every" => (:rcs_every, Int),
        "--rcs-theta-count" => (:rcs_theta_count, Int),
        "--rcs-theta-min-degrees" => (:rcs_theta_min_degrees, Float64),
        "--rcs-theta-max-degrees" => (:rcs_theta_max_degrees, Float64),
        "--rcs-phi-degrees" => (:rcs_phi_degrees, :float_list),
        "--output-dir" => (:output_dir, String),
    )

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            return nothing
        end
        if argument == "--enable-rcs"
            values[:rcs_enabled] = true
            index += 1
            continue
        elseif argument == "--disable-rcs"
            values[:rcs_enabled] = false
            index += 1
            continue
        end
        option = nothing
        for candidate in keys(option_fields)
            if argument == candidate || startswith(argument, "$candidate=")
                option = candidate
                break
            end
        end
        option === nothing && error("Unknown option '$argument'.")

        field, type = option_fields[option]
        raw_value, index = option_value(args, index, option)
        values[field] = if type == String
            field in (:mesh, :output_dir) ? abspath(raw_value) : String(raw_value)
        elseif type == MaxwellFluxKind
            parse_maxwell_flux_kind(raw_value)
        elseif type == :float_list
            parse_float_list(raw_value, option)
        else
            parse(type, raw_value)
        end
        index += 1
    end

    config = MetallicSphereScatteringConfig(; values...)
    config.radius > 0.0 || error("--radius must be positive.")
    config.wavelength > 0.0 || error("--wavelength must be positive.")
    config.frequency >= 0.0 || error("--frequency must be non-negative.")
    config.amplitude != 0.0 || error("--amplitude must be non-zero.")
    config.order >= 1 || error("--order must be positive.")
    1 <= config.esprk_order <= 6 ||
        error("--esprk-order must be in 1:6.")
    config.final_time >= 0.0 ||
        error("--final-time must be non-negative.")
    config.cfl > 0.0 || error("--cfl must be positive.")
    config.epsilon > 0.0 || error("--epsilon must be positive.")
    config.mu > 0.0 || error("--mu must be positive.")
    config.pml_width > 0.0 || error("--pml-width must be positive.")
    config.sigma_max >= 0.0 || error("--sigma-max must be non-negative.")
    config.sigma_degree >= 1 || error("--sigma-degree must be positive.")
    0.0 < config.pml_a < 1.0 || error("--pml-a must lie in (0,1).")
    config.regularization > 0.0 ||
        error("--regularization must be positive.")
    config.diagnostics_every >= 1 ||
        error("--diagnostics-every must be positive.")
    config.paraview_every >= 0 ||
        error("--paraview-every must be non-negative.")
    config.rcs_start_time >= -1.0 ||
        error("--rcs-start-time must be non-negative, or -1 for automatic.")
    config.rcs_every >= 1 || error("--rcs-every must be positive.")
    config.rcs_theta_count >= 1 ||
        error("--rcs-theta-count must be positive.")
    0.0 <= config.rcs_theta_min_degrees <= 180.0 ||
        error("--rcs-theta-min-degrees must lie in [0,180].")
    0.0 <= config.rcs_theta_max_degrees <= 180.0 ||
        error("--rcs-theta-max-degrees must lie in [0,180].")
    config.rcs_theta_min_degrees <= config.rcs_theta_max_degrees ||
        error("--rcs-theta-min-degrees cannot exceed --rcs-theta-max-degrees.")
    isempty(config.rcs_phi_degrees) &&
        error("--rcs-phi-degrees requires at least one value.")
    return config
end

function read_numbers!(io::IO, count::Int, type::Type{T}) where {T}
    values = Vector{T}()
    sizehint!(values, count)
    while length(values) < count
        eof(io) && error("Unexpected end of VTK file while reading data.")
        words = split(strip(readline(io)))
        isempty(words) && continue
        append!(values, parse.(T, words))
    end
    length(values) == count ||
        error("Expected $count entries but read $(length(values)).")
    return values
end

function signed_tet_volume6(
    points::Matrix{Float64},
    tet::NTuple{4, Int},
)
    a = @view points[:, tet[1]]
    b = @view points[:, tet[2]]
    c = @view points[:, tet[3]]
    d = @view points[:, tet[4]]
    bax, bay, baz = b .- a
    cax, cay, caz = c .- a
    dax, day, daz = d .- a
    return (
        bax * (cay * daz - caz * day) -
        bay * (cax * daz - caz * dax) +
        baz * (cax * day - cay * dax)
    )
end

function read_legacy_vtk_scattering_mesh(path::AbstractString)
    isfile(path) || error("Mesh file not found: $path")

    points = zeros(Float64, 3, 0)
    cells = Vector{Vector{Int}}()
    cell_types = Int[]
    cell_data = Dict{String, Any}()
    number_cells = 0

    open(path, "r") do io
        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue
            words = split(line)
            isempty(words) && continue

            if words[1] == "POINTS"
                number_points = parse(Int, words[2])
                raw_points = read_numbers!(io, 3 * number_points, Float64)
                points = reshape(raw_points, 3, number_points)
            elseif words[1] == "CELLS"
                number_cells = parse(Int, words[2])
                cells = Vector{Vector{Int}}(undef, number_cells)
                for cid in 1:number_cells
                    cell_words = split(strip(readline(io)))
                    isempty(cell_words) &&
                        error("Malformed empty VTK cell record.")
                    number_nodes = parse(Int, cell_words[1])
                    nodes = parse.(Int, cell_words[2:end]) .+ 1
                    length(nodes) == number_nodes ||
                        error("Malformed VTK cell $cid in $path.")
                    cells[cid] = nodes
                end
            elseif words[1] == "CELL_TYPES"
                number_types = parse(Int, words[2])
                cell_types = read_numbers!(io, number_types, Int)
            elseif words[1] == "SCALARS"
                number_cells > 0 ||
                    error("SCALARS appeared before CELLS in $path.")
                name = String(words[2])
                lookup = strip(readline(io))
                startswith(lookup, "LOOKUP_TABLE") ||
                    error("Unsupported VTK SCALARS block '$name'.")
                raw_values = read_numbers!(io, number_cells, Float64)
                cell_data[name] = Int.(round.(raw_values))
            end
        end
    end

    size(points, 2) > 0 || error("VTK mesh contains no points: $path")
    length(cells) == length(cell_types) ||
        error("VTK cell and cell-type counts differ in $path.")

    triangles = NTuple{3, Int}[]
    tetrahedra = NTuple{4, Int}[]
    tri_cell_ids = Int[]
    tet_cell_ids = Int[]
    for (cid, (nodes, cell_type)) in enumerate(zip(cells, cell_types))
        if cell_type == 5 && length(nodes) == 3
            push!(triangles, (nodes[1], nodes[2], nodes[3]))
            push!(tri_cell_ids, cid)
        elseif cell_type == 10 && length(nodes) == 4
            tet = (nodes[1], nodes[2], nodes[3], nodes[4])
            volume6 = signed_tet_volume6(points, tet)
            abs(volume6) > 1.0e-14 ||
                error("Mesh contains degenerate tetrahedron $tet.")
            if volume6 < 0.0
                tet = (tet[1], tet[3], tet[2], tet[4])
            end
            push!(tetrahedra, tet)
            push!(tet_cell_ids, cid)
        end
    end

    tets = isempty(tetrahedra) ? Matrix{Int}(undef, 4, 0) :
           reduce(hcat, collect.(tetrahedra))
    tris = isempty(triangles) ? Matrix{Int}(undef, 3, 0) :
           reduce(hcat, collect.(triangles))

    if !haskey(cell_data, "boundary_id")
        if haskey(cell_data, "CellEntityIds")
            cell_data["boundary_id"] = copy(cell_data["CellEntityIds"])
        elseif haskey(cell_data, "gmsh:physical")
            cell_data["boundary_id"] = copy(cell_data["gmsh:physical"])
        else
            error(
                "Mesh has no boundary_id, CellEntityIds, or gmsh:physical " *
                "cell-data array.",
            )
        end
    end

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        cell_data,
    )
    check_mesh_consistency(mesh)
    return mesh
end

function validate_scattering_mesh!(
    mesh::RawVTUMesh,
    config::MetallicSphereScatteringConfig,
)
    boundary_ids = Set(Int.(tri_data(mesh, "boundary_id")))
    missing_outer = setdiff(Set(OUTER_PML_BOUNDARY_IDS), boundary_ids)
    isempty(missing_outer) ||
        error("Mesh is missing outer PML boundary tags: $missing_outer.")
    METALLIC_SPHERE_PEC_BOUNDARY_ID in boundary_ids ||
        error(
            "Mesh is missing PEC metallic sphere boundary tag " *
            "$METALLIC_SPHERE_PEC_BOUNDARY_ID.",
        )

    lower = vec(minimum(mesh.points; dims = 2))
    upper = vec(maximum(mesh.points; dims = 2))
    lengths = upper .- lower
    minimum(lengths) > 0.0 ||
        error("Mesh coordinate bounds are degenerate.")
    2.0 * config.pml_width < minimum(lengths) ||
        error(
            "--pml-width=$(config.pml_width) must be smaller than half " *
            "the shortest mesh extent $(minimum(lengths)).",
        )
    maximum(abs.(lower .+ upper)) <= 0.25 * maximum(lengths) ||
        @warn "The mesh box is not centered near the origin." lower upper
    maximum(sqrt.(sum(abs2, mesh.points; dims = 1))) >= config.radius ||
        error("Mesh bounds do not contain radius $(config.radius).")
    return lower, upper
end

function scattering_boundary_registry()
    kinds = Dict{Int, MaxwellBoundaryKind}()
    for tag in OUTER_PML_BOUNDARY_IDS
        kinds[tag] = MaxwellBC_Absorbing
    end
    kinds[METALLIC_SPHERE_PEC_BOUNDARY_ID] = MaxwellBC_PEC
    return MaxwellBoundaryRegistry(kinds)
end

function resolved_wave_values(config::MetallicSphereScatteringConfig)
    speed = 1.0 / sqrt(config.epsilon * config.mu)
    frequency =
        config.frequency > 0.0 ? config.frequency : speed / config.wavelength
    wavelength =
        config.frequency > 0.0 ? speed / config.frequency : config.wavelength
    period = 1.0 / frequency
    final_time =
        config.final_time > 0.0 ? config.final_time : 2.0 * period
    return (
        speed = speed,
        frequency = frequency,
        wavelength = wavelength,
        period = period,
        angular_frequency = 2.0 * pi * frequency,
        wavenumber = 2.0 * pi / wavelength,
        final_time = final_time,
    )
end

function incident_wave(config::MetallicSphereScatteringConfig)
    wave_values = resolved_wave_values(config)
    wave = IncidentPlaneWaveParameters(
        propagation_direction = (0.0, 0.0, 1.0),
        polarization = (1.0, 0.0, 0.0),
        wavelength = wave_values.wavelength,
        amplitude = config.amplitude,
        epsilon = config.epsilon,
        mu = config.mu,
    )
    return wave, wave_values
end

zero_electric(x, y, z) = (0.0, 0.0, 0.0)
zero_magnetic(x, y, z) = (0.0, 0.0, 0.0)

function write_parallel_scattering_fields(
    output_basename::String,
    distributed_dg::DistributedDGDiscretization,
    scattered::MaxwellField;
    wave::IncidentPlaneWaveParameters,
    time::Float64,
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
    electric_scat = zeros(Float64, 3, nodes_per_element * nowned)
    magnetic_scat = similar(electric_scat)
    electric_inc = similar(electric_scat)
    magnetic_inc = similar(electric_scat)
    electric_total = similar(electric_scat)
    magnetic_total = similar(electric_scat)
    cells = Vector{MeshCell}(undef, nowned)
    global_element_ids = Vector{Int}(undef, nowned)

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
            Einc = incident_electric_plane_wave(x, y, z, time, wave)
            Hinc = incident_magnetic_plane_wave(x, y, z, time, wave)
            Escat = (
                scattered.Ex[field_node, local_elem],
                scattered.Ey[field_node, local_elem],
                scattered.Ez[field_node, local_elem],
            )
            Hscat = (
                scattered.Hx[field_node, local_elem],
                scattered.Hy[field_node, local_elem],
                scattered.Hz[field_node, local_elem],
            )

            points[:, output_node] .= (x, y, z)
            electric_scat[:, output_node] .= Escat
            magnetic_scat[:, output_node] .= Hscat
            electric_inc[:, output_node] .= Einc
            magnetic_inc[:, output_node] .= Hinc
            electric_total[:, output_node] .= (
                Escat[1] + Einc[1],
                Escat[2] + Einc[2],
                Escat[3] + Einc[3],
            )
            magnetic_total[:, output_node] .= (
                Hscat[1] + Hinc[1],
                Hscat[2] + Hinc[2],
                Hscat[3] + Hinc[3],
            )
        end
    end

    electric_scat_magnitude = vec(sqrt.(sum(abs2, electric_scat; dims = 1)))
    magnetic_scat_magnitude = vec(sqrt.(sum(abs2, magnetic_scat; dims = 1)))

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
            "E_scat",
            VTKPointData(),
            component_names = ("E_scat_x", "E_scat_y", "E_scat_z"),
        ] = electric_scat
        vtk[
            "H_scat",
            VTKPointData(),
            component_names = ("H_scat_x", "H_scat_y", "H_scat_z"),
        ] = magnetic_scat
        vtk[
            "E_inc",
            VTKPointData(),
            component_names = ("E_inc_x", "E_inc_y", "E_inc_z"),
        ] = electric_inc
        vtk[
            "H_inc",
            VTKPointData(),
            component_names = ("H_inc_x", "H_inc_y", "H_inc_z"),
        ] = magnetic_inc
        vtk[
            "E_total",
            VTKPointData(),
            component_names = ("E_total_x", "E_total_y", "E_total_z"),
        ] = electric_total
        vtk[
            "H_total",
            VTKPointData(),
            component_names = ("H_total_x", "H_total_y", "H_total_z"),
        ] = magnetic_total
        vtk["E_scat_magnitude", VTKPointData()] = electric_scat_magnitude
        vtk["H_scat_magnitude", VTKPointData()] = magnetic_scat_magnitude
        vtk["GlobalElementId", VTKCellData()] = global_element_ids
        vtk["OwnerRank", VTKCellData()] = fill(rank, nowned)
        vtk["PolynomialOrder", VTKCellData()] =
            fill(distributed_dg.dg.ref.N, nowned)
        vtk["TimeValue", VTKFieldData()] = time
    end
end

function write_scattering_snapshot!(
    entries,
    run_root::AbstractString,
    scattered::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    wave::IncidentPlaneWaveParameters,
    step::Int,
    time::Float64,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    fields_dir = joinpath(String(run_root), "fields")
    collective_root_action(
        distributed_dg.comm,
        "Scattering ParaView output directory creation",
    ) do
        mkpath(fields_dir)
    end
    MPI.Barrier(distributed_dg.comm)

    basename = @sprintf("fields_step%08d", step)
    output_basename = joinpath(fields_dir, basename)
    collective_rank_action(
        distributed_dg.comm,
        "Scattering VTK snapshot writing at step $step",
    ) do
        write_parallel_scattering_fields(
            output_basename,
            distributed_dg,
            scattered;
            wave = wave,
            time = time,
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
            replacement = write_paraview_series(String(run_root), entries)
            empty!(entries)
            append!(entries, replacement)
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, distributed_dg.comm; root = 0)
    root_error === nothing ||
        error("Scattering ParaView collection writing failed: $root_error")
    MPI.Barrier(distributed_dg.comm)
    return nothing
end

function write_diagnostics_header(io::IO)
    println(
        io,
        "step,time,scattered_energy_electric,scattered_energy_magnetic," *
        "scattered_energy_total,electric_charge,magnetic_charge," *
        "linear_momentum_x,linear_momentum_y,linear_momentum_z," *
        "angular_momentum_x,angular_momentum_y,angular_momentum_z",
    )
end

function write_diagnostics_row(
    io::IO,
    step::Int,
    time::Float64,
    invariants::MaxwellInvariantDiagnostics,
)
    energy = invariants.energy
    P = invariants.linear_momentum
    L = invariants.angular_momentum
    println(
        io,
        join(
            (
                step,
                @sprintf("%.16e", time),
                @sprintf("%.16e", energy.electric),
                @sprintf("%.16e", energy.magnetic),
                @sprintf("%.16e", energy.total),
                @sprintf("%.16e", invariants.electric_charge),
                @sprintf("%.16e", invariants.magnetic_charge),
                @sprintf("%.16e", P[1]),
                @sprintf("%.16e", P[2]),
                @sprintf("%.16e", P[3]),
                @sprintf("%.16e", L[1]),
                @sprintf("%.16e", L[2]),
                @sprintf("%.16e", L[3]),
            ),
            ',',
        ),
    )
end

"""
    RCSObservationDirection

One far-field RCS observation direction. The driver stores the spherical
direction `rhat` and the two transverse basis vectors used to project the
reconstructed far field into `Etheta` and `Ephi`.
"""
struct RCSObservationDirection
    theta_degrees::Float64
    phi_degrees::Float64
    direction::NTuple{3, Float64}
    e_theta::NTuple{3, Float64}
    e_phi::NTuple{3, Float64}
end

"""
    RCSFaceSample

Owned quadrature data on one PEC sphere face. The same face samples are reused
by the RCS surface-current integral and the `n x E_total` PEC residual.
"""
struct RCSFaceSample
    elem::Int
    nodes::Vector{Int}
    normal::NTuple{3, Float64}
    weights::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
end

"""
    RCSWorkspace

State for time-windowed RCS accumulation. `accum` stores the rank-local complex
Fourier integrals of `J_s = n x H_total`; `sample_buffer` is the per-time
surface integral before applying the temporal phase and Hann window.
"""
mutable struct RCSWorkspace
    observations::Vector{RCSObservationDirection}
    faces::Vector{RCSFaceSample}
    accum::Matrix{ComplexF64}
    sample_buffer::Matrix{ComplexF64}
    sample_count::Int
    window_sum::Float64
    start_time::Float64
    end_time::Float64
    omega::Float64
    wavenumber::Float64
    impedance::Float64
    incident_amplitude::Float64
end

function dot3(a::NTuple{3, <:Number}, b::NTuple{3, <:Number})
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

function cross3(
    a::NTuple{3, <:Number},
    b::NTuple{3, <:Number},
)
    return (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

struct PECBoundaryResidualDiagnostics
    surface_area::Float64
    integral::Float64
    l2_norm::Float64
    rms_norm::Float64
    relative_rms_norm::Float64
    max_pointwise::Float64
    relative_max_pointwise::Float64
end

struct PECBoundaryResidualWorkspace
    faces::Vector{RCSFaceSample}
    incident_amplitude::Float64
end

"""
    rcs_observation_directions(config)

Build the requested `(theta, phi)` sampling grid and the associated spherical
basis. Angles are stored in degrees for output and in Cartesian form for the
near-to-far phase factor.
"""
function rcs_observation_directions(config::MetallicSphereScatteringConfig)
    theta_values = if config.rcs_theta_count == 1
        [config.rcs_theta_min_degrees]
    else
        collect(
            range(
                config.rcs_theta_min_degrees,
                config.rcs_theta_max_degrees;
                length = config.rcs_theta_count,
            ),
        )
    end

    observations = RCSObservationDirection[]
    for phi_degrees in config.rcs_phi_degrees
        phi = deg2rad(phi_degrees)
        for theta_degrees in theta_values
            theta = deg2rad(theta_degrees)
            sinθ = sin(theta)
            cosθ = cos(theta)
            sinφ = sin(phi)
            cosφ = cos(phi)
            direction = (sinθ * cosφ, sinθ * sinφ, cosθ)
            e_theta = (cosθ * cosφ, cosθ * sinφ, -sinθ)
            e_phi = (-sinφ, cosφ, 0.0)
            push!(
                observations,
                RCSObservationDirection(
                    theta_degrees,
                    phi_degrees,
                    direction,
                    e_theta,
                    e_phi,
                ),
            )
        end
    end
    return observations
end

function rcs_start_time(
    config::MetallicSphereScatteringConfig,
    wave_values,
)
    start_time = config.rcs_start_time >= 0.0 ?
                 config.rcs_start_time :
                 max(0.0, wave_values.final_time - wave_values.period)
    start_time <= wave_values.final_time ||
        error("--rcs-start-time cannot exceed the final simulation time.")
    return start_time
end

"""
    rcs_time_window(time, start_time, end_time)

Hann window used for Fourier extraction at the incident frequency. Samples
before `start_time` are ignored; if `end_time <= start_time`, a unit window is
used to support one-step smoke tests.
"""
function rcs_time_window(time::Float64, start_time::Float64, end_time::Float64)
    if end_time <= start_time
        return 1.0
    end
    if time < start_time || time > end_time
        return 0.0
    end
    ξ = clamp((time - start_time) / (end_time - start_time), 0.0, 1.0)
    return 0.5 * (1.0 - cos(2.0 * pi * ξ))
end

function build_pec_sphere_face_samples(
    distributed_dg::DistributedDGDiscretization;
    context::AbstractString = "PEC sphere surface diagnostic",
)
    dg = distributed_dg.dg
    mesh = dg.mesh
    ref = dg.ref
    owned = Set(distributed_dg.distributed_mesh.partition.owned)
    samples = RCSFaceSample[]

    for ff in dg.flux_faces.boundary
        ff.boundary_id == METALLIC_SPHERE_PEC_BOUNDARY_ID || continue
        ff.trace.elem in owned || continue

        face = ff.trace.face
        nodes = copy(ff.trace.nodes)
        mass_ones =
            dg.fops.face_mass[face] * ones(Float64, length(nodes))
        physical_scale = ff.area / DiscoGMPI.reference_face_area(face)
        weights = physical_scale .* mass_ones
        n = (-ff.normal[1], -ff.normal[2], -ff.normal[3])

        x = Vector{Float64}(undef, length(nodes))
        y = similar(x)
        z = similar(x)
        tet_nodes = @view mesh.tets[:, ff.trace.elem]
        for (i, node) in enumerate(nodes)
            x[i], y[i], z[i] = DiscoGMPI.map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[node],
                ref.s[node],
                ref.t[node],
            )
        end

        push!(
            samples,
            RCSFaceSample(ff.trace.elem, nodes, n, weights, x, y, z),
        )
    end

    global_faces = MPI.Allreduce(length(samples), +, distributed_dg.comm)
    global_faces > 0 ||
        error(
            "$context needs owned faces with boundary tag " *
            "$METALLIC_SPHERE_PEC_BOUNDARY_ID, but none were found.",
        )
    return samples
end

function build_rcs_face_samples(distributed_dg::DistributedDGDiscretization)
    return build_pec_sphere_face_samples(
        distributed_dg;
        context = "RCS",
    )
end

function PECBoundaryResidualWorkspace(
    distributed_dg::DistributedDGDiscretization,
    incident_amplitude::Real,
)
    return PECBoundaryResidualWorkspace(
        build_pec_sphere_face_samples(
            distributed_dg;
            context = "PEC boundary residual",
        ),
        abs(Float64(incident_amplitude)),
    )
end

function pec_boundary_residual(
    workspace::PECBoundaryResidualWorkspace,
    U::MaxwellField,
    wave::IncidentPlaneWaveParameters,
    time::Float64,
    comm::MPI.Comm,
)
    local_area = 0.0
    local_integral = 0.0
    local_max_pointwise = 0.0

    for face in workspace.faces
        for q in eachindex(face.nodes)
            node = face.nodes[q]
            elem = face.elem
            x = face.x[q]
            y = face.y[q]
            z = face.z[q]
            Einc = incident_electric_plane_wave(x, y, z, time, wave)
            Etotal = (
                U.Ex[node, elem] + Einc[1],
                U.Ey[node, elem] + Einc[2],
                U.Ez[node, elem] + Einc[3],
            )
            tangent = cross3(face.normal, Etotal)
            magnitude2 = dot3(tangent, tangent)
            weight = face.weights[q]
            local_area += weight
            local_integral += weight * magnitude2
            local_max_pointwise =
                max(local_max_pointwise, sqrt(max(magnitude2, 0.0)))
        end
    end

    surface_area = MPI.Allreduce(local_area, +, comm)
    integral = MPI.Allreduce(local_integral, +, comm)
    max_pointwise = MPI.Allreduce(local_max_pointwise, max, comm)
    l2_norm = sqrt(max(integral, 0.0))
    rms_norm = surface_area > 0.0 ? sqrt(max(integral, 0.0) / surface_area) : Inf
    amplitude_scale = max(workspace.incident_amplitude, eps(Float64))
    return PECBoundaryResidualDiagnostics(
        surface_area,
        integral,
        l2_norm,
        rms_norm,
        rms_norm / amplitude_scale,
        max_pointwise,
        max_pointwise / amplitude_scale,
    )
end

function write_pec_boundary_residual_header(io::IO)
    println(
        io,
        "step,time,surface_area,integral,l2_norm,rms_norm," *
        "relative_rms_norm,max_pointwise,relative_max_pointwise",
    )
    return nothing
end

function write_pec_boundary_residual_row(
    io::IO,
    step::Int,
    time::Float64,
    residual::PECBoundaryResidualDiagnostics,
)
    println(
        io,
        join(
            (
                step,
                @sprintf("%.16e", time),
                @sprintf("%.16e", residual.surface_area),
                @sprintf("%.16e", residual.integral),
                @sprintf("%.16e", residual.l2_norm),
                @sprintf("%.16e", residual.rms_norm),
                @sprintf("%.16e", residual.relative_rms_norm),
                @sprintf("%.16e", residual.max_pointwise),
                @sprintf("%.16e", residual.relative_max_pointwise),
            ),
            ',',
        ),
    )
    return nothing
end

function RCSWorkspace(
    distributed_dg::DistributedDGDiscretization,
    config::MetallicSphereScatteringConfig,
    wave_values,
)
    observations = rcs_observation_directions(config)
    isempty(observations) && error("RCS needs at least one observation angle.")
    faces = build_rcs_face_samples(distributed_dg)
    ndirections = length(observations)
    return RCSWorkspace(
        observations,
        faces,
        zeros(ComplexF64, 3, ndirections),
        zeros(ComplexF64, 3, ndirections),
        0,
        0.0,
        rcs_start_time(config, wave_values),
        wave_values.final_time,
        wave_values.angular_frequency,
        wave_values.wavenumber,
        sqrt(config.mu / config.epsilon),
        abs(config.amplitude),
    )
end

"""
    accumulate_rcs_sample!(workspace, U, wave, time)

Accumulate one windowed Fourier sample of the PEC surface current
`J_s = n x (H_scat + H_inc)` over owned sphere faces for every observation
direction.
"""
function accumulate_rcs_sample!(
    workspace::RCSWorkspace,
    U::MaxwellField,
    wave::IncidentPlaneWaveParameters,
    time::Float64,
)
    window = rcs_time_window(time, workspace.start_time, workspace.end_time)
    window > 0.0 || return workspace

    fill!(workspace.sample_buffer, 0.0 + 0.0im)
    for face in workspace.faces
        for q in eachindex(face.nodes)
            node = face.nodes[q]
            elem = face.elem
            x = face.x[q]
            y = face.y[q]
            z = face.z[q]
            Hinc = incident_magnetic_plane_wave(x, y, z, time, wave)
            Htotal = (
                U.Hx[node, elem] + Hinc[1],
                U.Hy[node, elem] + Hinc[2],
                U.Hz[node, elem] + Hinc[3],
            )
            current = cross3(face.normal, Htotal)
            weight = face.weights[q]
            point = (x, y, z)

            for (direction_id, observation) in
                enumerate(workspace.observations)
                phase = cis(
                    -workspace.wavenumber *
                    dot3(observation.direction, point),
                )
                factor = weight * phase
                workspace.sample_buffer[1, direction_id] +=
                    factor * current[1]
                workspace.sample_buffer[2, direction_id] +=
                    factor * current[2]
                workspace.sample_buffer[3, direction_id] +=
                    factor * current[3]
            end
        end
    end

    time_phase = cis(workspace.omega * time)
    workspace.accum .+= window .* time_phase .* workspace.sample_buffer
    workspace.window_sum += window
    workspace.sample_count += 1
    return workspace
end

function global_rcs_accumulator(workspace::RCSWorkspace, comm::MPI.Comm)
    ndirections = length(workspace.observations)
    local_values = Vector{Float64}(undef, 6 * ndirections)
    index = 1
    for direction_id in 1:ndirections
        for component in 1:3
            value = workspace.accum[component, direction_id]
            local_values[index] = real(value)
            local_values[index + 1] = imag(value)
            index += 2
        end
    end
    global_values = MPI.Allreduce(local_values, +, comm)

    accum = Matrix{ComplexF64}(undef, 3, ndirections)
    index = 1
    for direction_id in 1:ndirections
        for component in 1:3
            accum[component, direction_id] =
                complex(global_values[index], global_values[index + 1])
            index += 2
        end
    end
    return accum
end

"""
    far_field_from_surface_current(current_integral, observation, workspace)

Map the time-harmonic PEC surface-current integral to the transverse far-field
amplitude used by `sigma = 4*pi*|E_infinity|^2/|E_inc|^2`.
"""
function far_field_from_surface_current(
    current_integral::NTuple{3, ComplexF64},
    observation::RCSObservationDirection,
    workspace::RCSWorkspace,
)
    rhat = observation.direction
    projection = dot3(rhat, current_integral)
    transverse = (
        rhat[1] * projection - current_integral[1],
        rhat[2] * projection - current_integral[2],
        rhat[3] * projection - current_integral[3],
    )
    scale = im * workspace.wavenumber * workspace.impedance / (4.0 * pi)
    return (
        scale * transverse[1],
        scale * transverse[2],
        scale * transverse[3],
    )
end

function write_rcs_results(
    run_paths::RunPaths,
    comm::MPI.Comm,
    workspace::Union{Nothing, RCSWorkspace},
)
    workspace === nothing && return nothing

    rank = MPI.Comm_rank(comm)
    rcs_path = run_diagnostics_path(run_paths, "rcs.csv")
    global_accum = global_rcs_accumulator(workspace, comm)
    sample_count = MPI.Allreduce(workspace.sample_count, max, comm)
    window_sum = MPI.Allreduce(workspace.window_sum, max, comm)

    collective_root_action(comm, "RCS diagnostics writing") do
        open(rcs_path, "w") do io
            println(
                io,
                "theta_degrees,phi_degrees,dir_x,dir_y,dir_z," *
                "abs_E_infinity,abs_Etheta,abs_Ephi," *
                "rcs,rcs_db,real_Etheta,imag_Etheta," *
                "real_Ephi,imag_Ephi,samples,window_sum," *
                "rcs_start_time,rcs_end_time",
            )

            if sample_count == 0 || window_sum <= 0.0
                @warn(
                    "RCS was enabled but no nonzero windowed samples were accumulated.",
                )
                return nothing
            end

            phasor_scale = 2.0 / window_sum
            for (direction_id, observation) in
                enumerate(workspace.observations)
                current_integral = (
                    phasor_scale * global_accum[1, direction_id],
                    phasor_scale * global_accum[2, direction_id],
                    phasor_scale * global_accum[3, direction_id],
                )
                far_field = far_field_from_surface_current(
                    current_integral,
                    observation,
                    workspace,
                )
                Etheta = dot3(observation.e_theta, far_field)
                Ephi = dot3(observation.e_phi, far_field)
                abs_Etheta = abs(Etheta)
                abs_Ephi = abs(Ephi)
                abs_E = sqrt(abs2(Etheta) + abs2(Ephi))
                rcs =
                    4.0 * pi * abs2(abs_E) /
                    max(workspace.incident_amplitude^2, eps(Float64))
                rcs_db = rcs > 0.0 ? 10.0 * log10(rcs) : -Inf
                direction = observation.direction
                println(
                    io,
                    join(
                        (
                            @sprintf("%.16e", observation.theta_degrees),
                            @sprintf("%.16e", observation.phi_degrees),
                            @sprintf("%.16e", direction[1]),
                            @sprintf("%.16e", direction[2]),
                            @sprintf("%.16e", direction[3]),
                            @sprintf("%.16e", abs_E),
                            @sprintf("%.16e", abs_Etheta),
                            @sprintf("%.16e", abs_Ephi),
                            @sprintf("%.16e", rcs),
                            @sprintf("%.16e", rcs_db),
                            @sprintf("%.16e", real(Etheta)),
                            @sprintf("%.16e", imag(Etheta)),
                            @sprintf("%.16e", real(Ephi)),
                            @sprintf("%.16e", imag(Ephi)),
                            sample_count,
                            @sprintf("%.16e", window_sum),
                            @sprintf("%.16e", workspace.start_time),
                            @sprintf("%.16e", workspace.end_time),
                        ),
                        ',',
                    ),
                )
            end
        end
    end
    rank == 0 && println("  RCS diagnostics:          ", rcs_path)
    return rcs_path
end

function timed_partitioned_symplectic_rk_step!(
    U::MaxwellField,
    work::MaxwellPartitionedRKWorkspace,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    rhs_function!::Function,
    boundary_data::MaxwellBoundaryData,
    start_time::Float64,
)
    DiscoGMPI.validate_partitioned_symplectic_rk_scheme(scheme)
    first_partition =
        DiscoGMPI.normalize_maxwell_partition(scheme.first_partition)
    second_partition =
        DiscoGMPI.complementary_maxwell_partition(first_partition)

    first_time = start_time
    second_time = start_time
    for stage in 1:DiscoGMPI.num_stages(scheme)
        first_weight = scheme.first_weights[stage]
        if first_weight != 0.0
            set_boundary_data_time!(boundary_data, first_time)
            rhs_function!(work.rhs, U)
            DiscoGMPI.add_scaled_partition_rhs_to_field!(
                U,
                work.rhs,
                first_partition,
                dt * first_weight,
            )
            first_time += dt * first_weight
        end

        second_weight = scheme.second_weights[stage]
        if second_weight != 0.0
            set_boundary_data_time!(boundary_data, second_time)
            rhs_function!(work.rhs, U)
            DiscoGMPI.add_scaled_partition_rhs_to_field!(
                U,
                work.rhs,
                second_partition,
                dt * second_weight,
            )
            second_time += dt * second_weight
        end
    end

    set_boundary_data_time!(boundary_data, start_time + dt)
    return U
end

function config_dictionary(
    config::MetallicSphereScatteringConfig;
    wave_values,
    mesh_lower,
    mesh_upper,
    rcs_start_time::Float64,
    dt::Float64,
    estimated_dt::Float64,
    steps::Int,
    global_hmin::Float64,
    nranks::Int,
)
    return Dict{String, Any}(
        "driver" => "distributed_metallic_sphere_scattering",
        "mesh" => config.mesh,
        "radius" => config.radius,
        "incident_propagation_direction" => [0.0, 0.0, 1.0],
        "incident_polarization" => [1.0, 0.0, 0.0],
        "wavelength" => wave_values.wavelength,
        "frequency" => wave_values.frequency,
        "angular_frequency" => wave_values.angular_frequency,
        "wavenumber" => wave_values.wavenumber,
        "amplitude" => config.amplitude,
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "order" => config.order,
        "esprk_order" => config.esprk_order,
        "final_time" => wave_values.final_time,
        "cfl" => config.cfl,
        "dt" => dt,
        "estimated_dt" => estimated_dt,
        "steps" => steps,
        "global_hmin" => global_hmin,
        "flux" => maxwell_flux_kind_label(config.flux),
        "pml_width" => config.pml_width,
        "sigma_max" => config.sigma_max,
        "sigma_degree" => config.sigma_degree,
        "pml_a" => config.pml_a,
        "regularization" => config.regularization,
        "diagnostics_every" => config.diagnostics_every,
        "paraview_every" => config.paraview_every,
        "rcs_enabled" => config.rcs_enabled,
        "rcs_start_time" => rcs_start_time,
        "rcs_every" => config.rcs_every,
        "rcs_theta_count" => config.rcs_theta_count,
        "rcs_theta_min_degrees" => config.rcs_theta_min_degrees,
        "rcs_theta_max_degrees" => config.rcs_theta_max_degrees,
        "rcs_phi_degrees" => config.rcs_phi_degrees,
        "mpi_ranks" => nranks,
        "mesh_lower" => collect(mesh_lower),
        "mesh_upper" => collect(mesh_upper),
        "outer_boundary_ids" => collect(OUTER_PML_BOUNDARY_IDS),
        "pec_sphere_boundary_id" => METALLIC_SPHERE_PEC_BOUNDARY_ID,
        "solution_state" => "scattered field",
        "paraview_primary_fields" => "total field = scattered + incident",
    )
end

function run_experiment(config::MetallicSphereScatteringConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    root_mesh = nothing
    root_partition = nothing
    root_lower = zeros(Float64, 3)
    root_upper = zeros(Float64, 3)
    if rank == 0
        root_mesh = read_legacy_vtk_scattering_mesh(config.mesh)
        root_lower, root_upper = validate_scattering_mesh!(root_mesh, config)
        root_partition = balanced_spatial_partition(root_mesh, nranks)
    end
    mesh_lower = Tuple(MPI.bcast(root_lower, comm; root = 0))
    mesh_upper = Tuple(MPI.bcast(root_upper, comm; root = 0))

    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        config.order;
        comm = comm,
        boundary_tag_name = "boundary_id",
    )
    box = distributed_axis_aligned_box(distributed_dg)
    pml = build_six_sided_nonlinear_pml(
        distributed_dg;
        width = config.pml_width,
        sigma_max = config.sigma_max,
        degree = config.sigma_degree,
        a = config.pml_a,
        regularization = config.regularization,
    )
    pml === nothing &&
        error("PML construction returned nothing; sigma_max and width must be positive.")

    wave, wave_values = incident_wave(config)
    final_time = wave_values.final_time
    incident_electric = (x, y, z, t) ->
        incident_electric_plane_wave(x, y, z, t, wave)
    boundary_data = incident_pec_boundary_data(
        METALLIC_SPHERE_PEC_BOUNDARY_ID,
        incident_electric;
        time = 0.0,
    )

    registry = scattering_boundary_registry()
    formulation = PoissonBracketFormulation(config.flux)
    U = interpolate_maxwell_field(
        distributed_dg,
        zero_electric,
        zero_magnetic,
    )
    materials = homogeneous_maxwell_materials(
        size(U.Ex, 2);
        epsilon = config.epsilon,
        permeability = config.mu,
    )
    rhs_function! = make_distributed_maxwell_nonlinear_pml_rhs_function(
        distributed_dg,
        registry,
        formulation,
        materials,
        pml,
        boundary_data,
    )
    scheme = explicit_partitioned_symplectic_rk_scheme(
        config.esprk_order;
        first_partition = :H,
    )
    rk_workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    rcs_workspace =
        config.rcs_enabled ? RCSWorkspace(distributed_dg, config, wave_values) :
        nothing
    pec_residual_workspace = PECBoundaryResidualWorkspace(
        distributed_dg,
        abs(config.amplitude),
    )
    resolved_rcs_start_time =
        rcs_workspace === nothing ? -1.0 : rcs_workspace.start_time

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
    steps = max(1, ceil(Int, final_time / estimated_dt))
    dt = final_time / steps

    run_paths = prepare_run_directory(config.output_dir, comm)
    diagnostics_path = run_diagnostics_path(
        run_paths,
        "scattering_diagnostics.csv",
    )
    pec_residual_path = run_diagnostics_path(
        run_paths,
        "pec_boundary_residual.csv",
    )
    configuration = config_dictionary(
        config;
        wave_values = wave_values,
        mesh_lower = mesh_lower,
        mesh_upper = mesh_upper,
        rcs_start_time = resolved_rcs_start_time,
        dt = dt,
        estimated_dt = estimated_dt,
        steps = steps,
        global_hmin = global_hmin,
        nranks = nranks,
    )

    initial_invariants = distributed_maxwell_invariants(
        U,
        distributed_dg,
        materials,
    )
    initial_pec_residual = pec_boundary_residual(
        pec_residual_workspace,
        U,
        wave,
        0.0,
        comm,
    )
    collectively_write_run_provenance(
        run_paths,
        comm;
        configuration = configuration,
        inputs = Dict(
            "mesh_path" => config.mesh,
            "partition" => "generated balanced spatial partition",
            "outer_boundary_ids" => collect(OUTER_PML_BOUNDARY_IDS),
            "pec_sphere_boundary_id" => METALLIC_SPHERE_PEC_BOUNDARY_ID,
        ),
    )
    write_distributed_run_metadata(
        run_paths.config_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "running",
            "initial_scattered_energy" => initial_invariants.energy.total,
            "initial_pec_boundary_residual_l2" =>
                initial_pec_residual.l2_norm,
            "initial_pec_boundary_residual_relative_rms" =>
                initial_pec_residual.relative_rms_norm,
            "estimated_dt" => estimated_dt,
            "used_dt" => dt,
            "steps" => steps,
        ),
    )
    collectively_write_run_status(
        run_paths,
        comm,
        "running";
        values = Dict(
            "final_step" => steps,
            "target_final_time" => final_time,
            "dt" => dt,
        ),
    )

    diagnostics_io = nothing
    pec_residual_io = nothing
    open_error = nothing
    if rank == 0
        try
            diagnostics_io = open(diagnostics_path, "w")
            write_diagnostics_header(diagnostics_io)
            pec_residual_io = open(pec_residual_path, "w")
            write_pec_boundary_residual_header(pec_residual_io)
        catch error
            open_error = sprint(showerror, error)
        end
    end
    open_error = MPI.bcast(open_error, comm; root = 0)
    open_error === nothing ||
        error("Could not open diagnostics file: $open_error")

    series_entries = NamedTuple{
        (:step, :time, :dataset),
        Tuple{Int, Float64, String},
    }[]
    collective_root_action(comm, "Initial scattering diagnostics writing") do
        write_diagnostics_row(
            diagnostics_io,
            0,
            0.0,
            initial_invariants,
        )
        write_pec_boundary_residual_row(
            pec_residual_io,
            0,
            0.0,
            initial_pec_residual,
        )
        flush(diagnostics_io)
        flush(pec_residual_io)
    end
    if config.paraview_every > 0
        write_scattering_snapshot!(
            series_entries,
            run_paths.root,
            U,
            distributed_dg,
            wave,
            0,
            0.0,
        )
    end

    if rank == 0
        println("Distributed metallic-sphere scattering")
        println("--------------------------------------")
        println("MPI ranks:          ", nranks)
        println("mesh:               ", config.mesh)
        println("domain lower:       ", box.lower)
        println("domain upper:       ", box.upper)
        println("radius:             ", config.radius)
        println("wavelength:         ", wave_values.wavelength)
        println("frequency:          ", wave_values.frequency)
        println("DG / ESPRK order:   ", config.order, " / ", config.esprk_order)
        println("flux:               ", maxwell_flux_kind_label(config.flux))
        println("PML width:          ", config.pml_width)
        println("sigma max/degree:   ", config.sigma_max, " / ", config.sigma_degree)
        println("dt / steps:         ", dt, " / ", steps)
        println("output:             ", run_paths.root)
    end

    final_invariants = initial_invariants
    final_pec_residual = initial_pec_residual
    max_pec_residual_l2 = initial_pec_residual.l2_norm
    max_pec_residual_relative_rms = initial_pec_residual.relative_rms_norm
    max_pec_residual_pointwise = initial_pec_residual.max_pointwise
    local_elapsed = 0.0
    try
        for step in 1:steps
            step_start_time = (step - 1) * dt
            time = step == steps ? final_time : step * dt
            local_elapsed += @elapsed begin
                timed_partitioned_symplectic_rk_step!(
                    U,
                    rk_workspace,
                    scheme,
                    dt,
                    rhs_function!,
                    boundary_data,
                    step_start_time,
                )
            end
            set_boundary_data_time!(boundary_data, time)

            if rcs_workspace !== nothing &&
               (step % config.rcs_every == 0 || step == steps)
                accumulate_rcs_sample!(rcs_workspace, U, wave, time)
            end

            diagnostics_due =
                step % config.diagnostics_every == 0 || step == steps
            if diagnostics_due
                final_invariants = distributed_maxwell_invariants(
                    U,
                    distributed_dg,
                    materials,
                )
                final_pec_residual = pec_boundary_residual(
                    pec_residual_workspace,
                    U,
                    wave,
                    time,
                    comm,
                )
                max_pec_residual_l2 =
                    max(max_pec_residual_l2, final_pec_residual.l2_norm)
                max_pec_residual_relative_rms = max(
                    max_pec_residual_relative_rms,
                    final_pec_residual.relative_rms_norm,
                )
                max_pec_residual_pointwise = max(
                    max_pec_residual_pointwise,
                    final_pec_residual.max_pointwise,
                )
                collective_root_action(
                    comm,
                    "Scattering diagnostics writing at step $step",
                ) do
                    write_diagnostics_row(
                        diagnostics_io,
                        step,
                        time,
                        final_invariants,
                    )
                    write_pec_boundary_residual_row(
                        pec_residual_io,
                        step,
                        time,
                        final_pec_residual,
                    )
                    println(
                        "  step ", step, "/", steps,
                        ", t = ", time,
                        ", scattered energy = ",
                        final_invariants.energy.total,
                        ", PEC residual relative RMS = ",
                        final_pec_residual.relative_rms_norm,
                    )
                    flush(diagnostics_io)
                    flush(pec_residual_io)
                end
            end

            if config.paraview_every > 0 &&
               (step % config.paraview_every == 0 || step == steps)
                write_scattering_snapshot!(
                    series_entries,
                    run_paths.root,
                    U,
                    distributed_dg,
                    wave,
                    step,
                    time,
                )
            end
        end
    finally
        collective_root_action(comm, "Scattering diagnostics closing") do
            diagnostics_io === nothing || close(diagnostics_io)
            pec_residual_io === nothing || close(pec_residual_io)
        end
    end

    elapsed = MPI.Allreduce(local_elapsed, max, comm)
    write_rcs_results(run_paths, comm, rcs_workspace)
    write_distributed_run_metadata(
        run_paths.config_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "complete",
            "elapsed_seconds" => elapsed,
            "final_scattered_energy" => final_invariants.energy.total,
            "final_pec_boundary_residual_l2" =>
                final_pec_residual.l2_norm,
            "final_pec_boundary_residual_rms" =>
                final_pec_residual.rms_norm,
            "final_pec_boundary_residual_relative_rms" =>
                final_pec_residual.relative_rms_norm,
            "final_pec_boundary_residual_max_pointwise" =>
                final_pec_residual.max_pointwise,
            "final_pec_boundary_residual_relative_max_pointwise" =>
                final_pec_residual.relative_max_pointwise,
            "max_sampled_pec_boundary_residual_l2" =>
                max_pec_residual_l2,
            "max_sampled_pec_boundary_residual_relative_rms" =>
                max_pec_residual_relative_rms,
            "max_sampled_pec_boundary_residual_pointwise" =>
                max_pec_residual_pointwise,
            "final_electric_charge" => final_invariants.electric_charge,
            "final_magnetic_charge" => final_invariants.magnetic_charge,
            "final_linear_momentum" =>
                collect(final_invariants.linear_momentum),
            "final_angular_momentum" =>
                collect(final_invariants.angular_momentum),
            "rcs_enabled" => config.rcs_enabled,
            "rcs_samples" =>
                rcs_workspace === nothing ? 0 : rcs_workspace.sample_count,
            "rcs_window_sum" =>
                rcs_workspace === nothing ? 0.0 : rcs_workspace.window_sum,
        ),
    )
    collectively_write_run_status(
        run_paths,
        comm,
        "complete";
        values = Dict(
            "elapsed_seconds" => elapsed,
            "final_step" => steps,
            "final_time" => final_time,
            "final_scattered_energy" => final_invariants.energy.total,
            "final_pec_boundary_residual_relative_rms" =>
                final_pec_residual.relative_rms_norm,
        ),
    )

    if rank == 0
        println("Completed metallic-sphere scattering run.")
        println("  elapsed seconds:         ", elapsed)
        println(
            "  final scattered energy:  ",
            final_invariants.energy.total,
        )
        println("  diagnostics:             ", diagnostics_path)
        println("  PEC residual diagnostics:", pec_residual_path)
    end

    return (
        output_dir = run_paths.root,
        diagnostics_path = diagnostics_path,
        pec_residual_path = pec_residual_path,
        final_time = final_time,
        steps = steps,
        dt = dt,
        final_scattered_energy = final_invariants.energy.total,
        final_pec_boundary_residual_l2 = final_pec_residual.l2_norm,
        final_pec_boundary_residual_rms = final_pec_residual.rms_norm,
        final_pec_boundary_residual_relative_rms =
            final_pec_residual.relative_rms_norm,
        final_pec_boundary_residual_max_pointwise =
            final_pec_residual.max_pointwise,
        final_pec_boundary_residual_relative_max_pointwise =
            final_pec_residual.relative_max_pointwise,
        max_sampled_pec_boundary_residual_l2 = max_pec_residual_l2,
        max_sampled_pec_boundary_residual_relative_rms =
            max_pec_residual_relative_rms,
        max_sampled_pec_boundary_residual_pointwise =
            max_pec_residual_pointwise,
    )
end

function main(args::Vector{String})
    if any(argument -> argument == "--help" || argument == "-h", args)
        usage()
        return nothing
    end

    MPI.Init()
    comm = MPI.COMM_WORLD
    config = nothing
    try
        config = parse_config(args)
        if config === nothing
            MPI.Comm_rank(comm) == 0 && usage()
            return nothing
        end
        run_experiment(config, comm)
    catch error
        message = sprint(showerror, error)
        println(stderr, "ERROR: ", message)
        rethrow()
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
