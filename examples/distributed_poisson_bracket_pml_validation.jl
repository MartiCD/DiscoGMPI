#!/usr/bin/env julia

using MPI
using Printf
using LinearAlgebra
using DiscoGMPI

const DEFAULT_PML_VALIDATION_MESH = joinpath(
    @__DIR__,
    "meshes",
    "periodic_box_structured_nx4_ny2_nz2.vtk",
)

Base.@kwdef struct PMLValidationConfig
    mesh_paths::Vector{String} = [DEFAULT_PML_VALIDATION_MESH]
    order::Int = 2
    esprk_order::Int = 3
    final_time::Float64 = 3.0
    cfl::Float64 = 0.15
    epsilon::Float64 = 1.0
    mu::Float64 = 1.0
    fluxes::Vector{MaxwellFluxKind} = MaxwellFluxKind[MaxwellFlux_Central]
    outer_boundary::Symbol = :pec
    pml_widths::Vector{Float64} = [0.5]
    sigma_maxes::Vector{Float64} = [12.0]
    sigma_degrees::Vector{Int} = [2]
    pml_a::Float64 = 0.5
    regularization::Float64 = 1e-12
    pulse_center::Float64 = 0.75
    pulse_width::Float64 = 0.12
    pulse_amplitude::Float64 = 1.0
    central_elements_per_wavelength::Float64 = 4.0
    central_wavelength::Float64 = 0.5
    central_frequency::Float64 = 0.0
    cubature_order::Int = 0
    energy_every::Int = 10
    paraview_every::Int = 50
    reflection_start::Float64 = -1.0
    left_monitor_max::Float64 = 0.75
    right_monitor_min::Float64 = 1.25
    output_dir::String = joinpath(
        normpath(joinpath(@__DIR__, "..")),
        "output",
        "poisson_bracket_pml_validation",
    )
end

struct PMLValidationCase
    mesh_path::String
    flux_kind::MaxwellFluxKind
    pml_width::Float64
    sigma_max::Float64
    sigma_degree::Int
    name::String
    output_dir::String
end

struct PMLReflectionDiagnostics
    total_energy::Float64
    left_pml_energy::Float64
    right_pml_energy::Float64
    interior_energy::Float64
    left_monitor_energy::Float64
    right_monitor_energy::Float64
end

struct PMLPulseSpectrum
    x_spacing::Float64
    x_intervals::Int
    elements_per_wavelength::Float64
    wavelength::Float64
    wavenumber::Float64
    angular_frequency::Float64
    frequency::Float64
    source::String
end

mutable struct PMLReflectionWorkspace
    cubature_order::Int
    cubature_points::Matrix{Float64}
    cubature_weights::Vector{Float64}
    interpolation::Matrix{Float64}
    local_sums::Vector{Float64}
end

function PMLReflectionWorkspace(
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int,
)
    cubature_points, cubature_weights, _ =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    return PMLReflectionWorkspace(
        cubature_order,
        Matrix{Float64}(cubature_points),
        Vector{Float64}(cubature_weights),
        Matrix{Float64}(interpolation),
        zeros(Float64, 6),
    )
end

function pml_validation_usage(io::IO = stdout)
    println(io, """
Production Poisson-bracket/ESPRK nonlinear PML validation driver

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_poisson_bracket_pml_validation.jl [options]

Core options:
  --mesh PATH                  Single VTK mesh path. Default:
                               examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk
  --meshes LIST                Comma-separated mesh sweep.
  --order N                    DG polynomial order. Default: 2
  --esprk-order N              ESPRK order in 1:6. Default: order + 1
  --final-time T               Final simulation time. Default: 3.0
  --cfl C                      Maxwell CFL factor. Default: 0.15
  --epsilon E                  Constant permittivity. Default: 1
  --mu M                       Constant permeability. Default: 1
  --flux centered|alternating  Single Poisson-bracket flux. Default: centered
  --fluxes LIST                Comma-separated flux sweep, e.g. centered,alternating
  --outer-boundary pec|pmc|absorbing
                               Boundary condition on the two x outer faces.
                               Default: pec

PML and pulse options:
  --pml-width W                Single x-layer width at each boundary.
  --pml-widths LIST            Comma-separated width sweep. Default: 0.5
  --sigma-max S                Single maximum damping value.
  --sigma-maxes LIST           Comma-separated sigma sweep. Default: 12
  --sigma-degree N             Single polynomial damping degree.
  --sigma-degrees LIST         Comma-separated degree sweep. Default: 2
  --pml-a A                    Nonlinear PML parameter in (0,1). Default: 0.5
  --regularization R           PML denominator regularization. Default: 1e-12
  --pulse-center X             Gaussian pulse center. Default: 0.75
  --pulse-width W              Gaussian pulse width. Default: 0.12
  --pulse-amplitude A          Magnetic pulse amplitude. Default: 1
  --central-elements-per-wavelength R
                               Mesh-derived central wavelength uses R x-elements
                               when both central wavelength and frequency are 0.
                               Default: 4
  --central-wavelength L       Central wavelength. Default: 0.5
  --central-frequency F        Override central frequency in cycles per time.
                               Default: 0, inferred from central wavelength

Diagnostics and output:
  --cubature-order N           Reflection diagnostic cubature. Default: 2p+4
  --energy-every N             CSV diagnostic interval. Default: 10
  --paraview-every N           ParaView interval; 0 disables snapshots.
                               Default: 50
  --reflection-start T         Start time for max reflected-energy tracking.
                               Default: estimated from the right PML interface
  --left-monitor-max X         Left monitor region x <= X. Default: 0.75
  --right-monitor-min X        Right monitor region x >= X. Default: 1.25
  --output-dir PATH            Sweep output root.
  --help                       Show this message.

The validation problem launches a right-going sine-modulated Gaussian Maxwell
pulse on an axis-aligned cuboid mesh. The y and z faces are periodic; the x
faces are physical outer boundaries covered by nonlinear PML layers. Each sweep
case writes an isolated run directory with diagnostics, ParaView output,
resolved configuration, run metadata, and partition metadata.
""")
end

function pml_option_value(args::Vector{String}, index::Int, option::String)
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

function parse_int_list(value::AbstractString, option::AbstractString)
    return [parse(Int, item) for item in parse_comma_list(value, option)]
end

function parse_flux_list(value::AbstractString, option::AbstractString)
    return [
        parse_maxwell_flux_kind(item)
        for item in parse_comma_list(value, option)
    ]
end

function parse_mesh_path_list(value::AbstractString, option::AbstractString)
    return [
        abspath(path)
        for path in parse_comma_list(value, option)
    ]
end

function parse_outer_boundary(value::AbstractString)
    name = lowercase(strip(value))
    if name == "pec"
        return :pec
    elseif name == "pmc"
        return :pmc
    elseif name in ("absorbing", "abc")
        return :absorbing
    end
    error("--outer-boundary must be pec, pmc, or absorbing.")
end

function outer_boundary_kind(boundary::Symbol)
    boundary == :pec && return MaxwellBC_PEC
    boundary == :pmc && return MaxwellBC_PMC
    boundary == :absorbing && return MaxwellBC_Absorbing
    error("Unsupported outer boundary '$boundary'.")
end

function parse_config(args::Vector{String})
    values = Dict{Symbol, Any}(
        field => getfield(PMLValidationConfig(), field)
        for field in fieldnames(PMLValidationConfig)
    )
    esprk_order_provided = false

    scalar_options = Dict(
        "--order" => (:order, Int),
        "--esprk-order" => (:esprk_order, Int),
        "--final-time" => (:final_time, Float64),
        "--cfl" => (:cfl, Float64),
        "--epsilon" => (:epsilon, Float64),
        "--mu" => (:mu, Float64),
        "--pml-a" => (:pml_a, Float64),
        "--regularization" => (:regularization, Float64),
        "--pulse-center" => (:pulse_center, Float64),
        "--pulse-width" => (:pulse_width, Float64),
        "--pulse-amplitude" => (:pulse_amplitude, Float64),
        "--central-elements-per-wavelength" =>
            (:central_elements_per_wavelength, Float64),
        "--central-wavelength" => (:central_wavelength, Float64),
        "--central-frequency" => (:central_frequency, Float64),
        "--cubature-order" => (:cubature_order, Int),
        "--energy-every" => (:energy_every, Int),
        "--paraview-every" => (:paraview_every, Int),
        "--reflection-start" => (:reflection_start, Float64),
        "--left-monitor-max" => (:left_monitor_max, Float64),
        "--right-monitor-min" => (:right_monitor_min, Float64),
        "--output-dir" => (:output_dir, String),
    )

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            pml_validation_usage()
            return nothing
        elseif argument == "--mesh" || startswith(argument, "--mesh=")
            raw, index = pml_option_value(args, index, "--mesh")
            values[:mesh_paths] = [abspath(raw)]
        elseif argument == "--meshes" || startswith(argument, "--meshes=")
            raw, index = pml_option_value(args, index, "--meshes")
            values[:mesh_paths] = parse_mesh_path_list(raw, "--meshes")
        elseif argument == "--flux" || startswith(argument, "--flux=")
            raw, index = pml_option_value(args, index, "--flux")
            values[:fluxes] = MaxwellFluxKind[parse_maxwell_flux_kind(raw)]
        elseif argument == "--fluxes" || startswith(argument, "--fluxes=")
            raw, index = pml_option_value(args, index, "--fluxes")
            values[:fluxes] = parse_flux_list(raw, "--fluxes")
        elseif argument == "--outer-boundary" ||
               startswith(argument, "--outer-boundary=") ||
               argument == "--boundary-condition" ||
               startswith(argument, "--boundary-condition=")
            option = startswith(argument, "--boundary-condition") ?
                     "--boundary-condition" : "--outer-boundary"
            raw, index = pml_option_value(args, index, option)
            values[:outer_boundary] = parse_outer_boundary(raw)
        elseif argument == "--pml-width" ||
               startswith(argument, "--pml-width=")
            raw, index = pml_option_value(args, index, "--pml-width")
            values[:pml_widths] = [parse(Float64, raw)]
        elseif argument == "--pml-widths" ||
               startswith(argument, "--pml-widths=")
            raw, index = pml_option_value(args, index, "--pml-widths")
            values[:pml_widths] = parse_float_list(raw, "--pml-widths")
        elseif argument == "--sigma-max" ||
               startswith(argument, "--sigma-max=")
            raw, index = pml_option_value(args, index, "--sigma-max")
            values[:sigma_maxes] = [parse(Float64, raw)]
        elseif argument == "--sigma-maxes" ||
               startswith(argument, "--sigma-maxes=")
            raw, index = pml_option_value(args, index, "--sigma-maxes")
            values[:sigma_maxes] = parse_float_list(raw, "--sigma-maxes")
        elseif argument == "--sigma-degree" ||
               startswith(argument, "--sigma-degree=")
            raw, index = pml_option_value(args, index, "--sigma-degree")
            values[:sigma_degrees] = [parse(Int, raw)]
        elseif argument == "--sigma-degrees" ||
               startswith(argument, "--sigma-degrees=")
            raw, index = pml_option_value(args, index, "--sigma-degrees")
            values[:sigma_degrees] = parse_int_list(raw, "--sigma-degrees")
        else
            matched = nothing
            for option in keys(scalar_options)
                if argument == option || startswith(argument, "$option=")
                    matched = option
                    break
                end
            end
            matched === nothing &&
                error("Unknown option '$argument'. Use --help for usage.")
            field, type = scalar_options[matched]
            raw, index = pml_option_value(args, index, matched)
            values[field] =
                type == String ? abspath(raw) : parse(type, raw)
            esprk_order_provided |= field == :esprk_order
        end
        index += 1
    end

    if !esprk_order_provided
        values[:esprk_order] = values[:order] + 1
    end

    config = PMLValidationConfig(; values...)
    validate_config(config)
    return config
end

function validate_config(config::PMLValidationConfig)
    isempty(config.mesh_paths) && error("At least one mesh path is required.")
    for mesh_path in config.mesh_paths
        isfile(mesh_path) || error("Mesh file not found: $mesh_path")
    end
    config.order >= 1 || error("--order must be positive.")
    1 <= config.esprk_order <= 6 ||
        error("--esprk-order must be between 1 and 6.")
    config.final_time > 0.0 || error("--final-time must be positive.")
    config.cfl > 0.0 || error("--cfl must be positive.")
    config.epsilon > 0.0 || error("--epsilon must be positive.")
    config.mu > 0.0 || error("--mu must be positive.")
    isempty(config.fluxes) && error("At least one flux must be selected.")
    for flux in config.fluxes
        flux in (MaxwellFlux_Central, MaxwellFlux_Alternating) ||
            error("PoissonBracket PML validation supports centered and alternating fluxes.")
        if flux == MaxwellFlux_Alternating && config.outer_boundary != :pec
            error(
                "PoissonBracket alternating flux currently requires " *
                "--outer-boundary=pec.",
            )
        end
    end
    isempty(config.pml_widths) && error("At least one PML width is required.")
    for width in config.pml_widths
        width > 0.0 || error("--pml-width values must be positive.")
    end
    isempty(config.sigma_maxes) && error("At least one sigma value is required.")
    all(sigma -> sigma >= 0.0, config.sigma_maxes) ||
        error("--sigma-max values must be non-negative.")
    isempty(config.sigma_degrees) &&
        error("At least one sigma polynomial degree is required.")
    all(degree -> degree >= 1, config.sigma_degrees) ||
        error("--sigma-degree values must be positive.")
    0.0 < config.pml_a < 1.0 || error("--pml-a must lie in (0,1).")
    config.regularization > 0.0 ||
        error("--regularization must be positive.")
    isfinite(config.pulse_center) ||
        error("--pulse-center must be finite.")
    config.pulse_width > 0.0 || error("--pulse-width must be positive.")
    isfinite(config.pulse_amplitude) ||
        error("--pulse-amplitude must be finite.")
    config.central_elements_per_wavelength > 0.0 ||
        error("--central-elements-per-wavelength must be positive.")
    config.central_wavelength >= 0.0 ||
        error("--central-wavelength must be non-negative.")
    config.central_frequency >= 0.0 ||
        error("--central-frequency must be non-negative.")
    if config.central_wavelength > 0.0 && config.central_frequency > 0.0
        error(
            "Specify at most one of --central-wavelength and " *
            "--central-frequency.",
        )
    end
    config.cubature_order >= 0 ||
        error("--cubature-order must be non-negative.")
    config.energy_every >= 1 || error("--energy-every must be positive.")
    config.paraview_every >= 0 ||
        error("--paraview-every must be non-negative.")
    config.reflection_start >= -1.0 ||
        error("--reflection-start must be non-negative or left at -1.")
    isfinite(config.left_monitor_max) ||
        error("--left-monitor-max must be finite.")
    isfinite(config.right_monitor_min) ||
        error("--right-monitor-min must be finite.")
    return nothing
end

function validate_case_geometry(
    config::PMLValidationConfig,
    case::PMLValidationCase,
    box::AxisAlignedBox,
)
    lengths = axis_aligned_box_lengths(box)
    case.pml_width < 0.5 * lengths[1] ||
        error(
            "PML width $(case.pml_width) must be smaller than half the " *
            "mesh x extent $(lengths[1]) for $(case.mesh_path).",
        )
    x_min = box.lower[1]
    x_max = box.upper[1]
    x_min < config.pulse_center < x_max ||
        error("--pulse-center must lie inside mesh x bounds $x_min to $x_max.")
    config.pulse_center > x_min + case.pml_width ||
        error("--pulse-center must be to the right of the left PML layer.")
    config.pulse_center < x_max - case.pml_width ||
        error("--pulse-center must be to the left of the right PML layer.")
    x_min <= config.left_monitor_max <= x_max ||
        error("--left-monitor-max must lie inside mesh x bounds $x_min to $x_max.")
    x_min <= config.right_monitor_min <= x_max ||
        error("--right-monitor-min must lie inside mesh x bounds $x_min to $x_max.")
    return nothing
end

function pml_periodic_specs(box::AxisAlignedBox)
    _, Ly, Lz = axis_aligned_box_lengths(box)
    return (
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

function pml_boundary_registry(config::PMLValidationConfig)
    outer = outer_boundary_kind(config.outer_boundary)
    return MaxwellBoundaryRegistry(
        Dict(
            1 => outer,
            2 => outer,
            3 => MaxwellBC_None,
            4 => MaxwellBC_None,
            5 => MaxwellBC_None,
            6 => MaxwellBC_None,
        ),
    )
end

function pml_number_token(value::Real)
    return replace(
        @sprintf("%.6g", Float64(value)),
        "." => "p",
        "-" => "m",
        "+" => "",
    )
end

function pml_mesh_token(mesh_path::AbstractString)
    stem = splitext(basename(String(mesh_path)))[1]
    return sanitize_run_name(stem)
end

function pml_case_name(
    mesh_path::AbstractString,
    flux_kind::MaxwellFluxKind,
    pml_width::Float64,
    sigma_max::Float64,
    sigma_degree::Int,
)
    return join(
        (
            "pb-pml",
            "mesh=$(pml_mesh_token(mesh_path))",
            "flux=$(maxwell_flux_kind_label(flux_kind))",
            "w=$(pml_number_token(pml_width))",
            "s=$(pml_number_token(sigma_max))",
            "m=$sigma_degree",
        ),
        "_",
    )
end

function planned_cases(config::PMLValidationConfig)
    cases = PMLValidationCase[]
    for mesh_path in config.mesh_paths
        for flux in config.fluxes
            for width in config.pml_widths
                for sigma_max in config.sigma_maxes
                    for degree in config.sigma_degrees
                        name = pml_case_name(
                            mesh_path,
                            flux,
                            width,
                            sigma_max,
                            degree,
                        )
                        push!(
                            cases,
                            PMLValidationCase(
                                mesh_path,
                                flux,
                                width,
                                sigma_max,
                                degree,
                                name,
                                joinpath(config.output_dir, name),
                            ),
                        )
                    end
                end
            end
        end
    end
    return cases
end

resolved_cubature_order(config::PMLValidationConfig) =
    config.cubature_order > 0 ? config.cubature_order :
    max(2, 2 * config.order + 4)

function mesh_x_spacing(mesh::RawVTUMesh)
    box = infer_axis_aligned_box(mesh.points)
    x_length = box.upper[1] - box.lower[1]
    tolerance = max(1.0e-10, 1.0e-10 * max(x_length, 1.0))
    x_values = sort(collect(@view mesh.points[1, :]))
    planes = Float64[]
    for x in x_values
        if isempty(planes) || abs(x - planes[end]) > tolerance
            push!(planes, x)
        end
    end
    length(planes) >= 2 ||
        error("Cannot infer x mesh spacing from fewer than two x planes.")
    spacings = diff(planes)
    positive_spacings = filter(>(tolerance), spacings)
    isempty(positive_spacings) &&
        error("Cannot infer a positive x mesh spacing.")
    return minimum(positive_spacings), length(planes) - 1
end

function pulse_spectrum(
    config::PMLValidationConfig,
    x_spacing::Float64,
    x_intervals::Int,
)
    wave_speed = 1.0 / sqrt(config.epsilon * config.mu)
    if config.central_frequency > 0.0
        frequency = config.central_frequency
        angular_frequency = 2.0 * pi * frequency
        wavenumber = angular_frequency / wave_speed
        wavelength = 2.0 * pi / wavenumber
        elements_per_wavelength = wavelength / x_spacing
        source = "central_frequency"
    elseif config.central_wavelength > 0.0
        wavelength = config.central_wavelength
        wavenumber = 2.0 * pi / wavelength
        angular_frequency = wave_speed * wavenumber
        frequency = angular_frequency / (2.0 * pi)
        elements_per_wavelength = wavelength / x_spacing
        source = "central_wavelength"
    else
        elements_per_wavelength = config.central_elements_per_wavelength
        wavelength = elements_per_wavelength * x_spacing
        wavenumber = 2.0 * pi / wavelength
        angular_frequency = wave_speed * wavenumber
        frequency = angular_frequency / (2.0 * pi)
        source = "mesh_x_spacing"
    end

    return PMLPulseSpectrum(
        x_spacing,
        x_intervals,
        elements_per_wavelength,
        wavelength,
        wavenumber,
        angular_frequency,
        frequency,
        source,
    )
end

function initial_pulse_functions(
    config::PMLValidationConfig,
    spectrum::PMLPulseSpectrum,
)
    impedance = sqrt(config.mu / config.epsilon)
    envelope = x ->
        config.pulse_amplitude *
        exp(-((x - config.pulse_center) / config.pulse_width)^2) *
        sin(spectrum.wavenumber * (x - config.pulse_center))
    electric = (x, y, z) -> (0.0, 0.0, impedance * envelope(x))
    magnetic = (x, y, z) -> (0.0, -envelope(x), 0.0)
    return electric, magnetic
end

pml_zero_electric(x, y, z) = (0.0, 0.0, 0.0)
pml_zero_magnetic(x, y, z) = (0.0, 0.0, 0.0)

function pml_sigma_function(case::PMLValidationCase, box::AxisAlignedBox)
    left_boundary = box.lower[1]
    right_boundary = box.upper[1]
    left_interface = left_boundary + case.pml_width
    right_interface = right_boundary - case.pml_width
    return (x, y, z) -> max(
        polynomial_pml_sigma(
            x,
            left_interface,
            left_boundary;
            sigma_max = case.sigma_max,
            degree = case.sigma_degree,
        ),
        polynomial_pml_sigma(
            x,
            right_interface,
            right_boundary;
            sigma_max = case.sigma_max,
            degree = case.sigma_degree,
        ),
    )
end

function default_reflection_start(
    config::PMLValidationConfig,
    case::PMLValidationCase,
    box::AxisAlignedBox,
)
    config.reflection_start >= 0.0 && return config.reflection_start
    wave_speed = 1.0 / sqrt(config.epsilon * config.mu)
    right_interface = box.upper[1] - case.pml_width
    outgoing_distance = max(right_interface - config.pulse_center, 0.0)
    return 2.0 * outgoing_distance / wave_speed
end

function distributed_pml_reflection_diagnostics!(
    workspace::PMLReflectionWorkspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization;
    epsilon::Float64,
    mu::Float64,
    x_min::Float64,
    x_max::Float64,
    pml_width::Float64,
    left_monitor_max::Float64,
    right_monitor_min::Float64,
)
    mesh = distributed_dg.dg.mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    owned = distributed_dg.distributed_mesh.partition.owned
    local_sums = workspace.local_sums
    fill!(local_sums, 0.0)

    left_pml_limit = x_min + pml_width
    right_pml_limit = x_max - pml_width
    tolerance = 100 * eps(Float64)

    for elem in owned
        tet_nodes = @view mesh.tets[:, elem]
        jacobian = mappings[elem].absdetJ
        @views begin
            Ex = U.Ex[:, elem]
            Ey = U.Ey[:, elem]
            Ez = U.Ez[:, elem]
            Hx = U.Hx[:, elem]
            Hy = U.Hy[:, elem]
            Hz = U.Hz[:, elem]

            for q in axes(workspace.cubature_points, 1)
                r = workspace.cubature_points[q, 1]
                s = workspace.cubature_points[q, 2]
                t = workspace.cubature_points[q, 3]
                x, _, _ = DiscoGMPI.map_to_physical(
                    mesh.points,
                    tet_nodes,
                    r,
                    s,
                    t,
                )
                row = view(workspace.interpolation, q, :)
                eqx = dot(row, Ex)
                eqy = dot(row, Ey)
                eqz = dot(row, Ez)
                hqx = dot(row, Hx)
                hqy = dot(row, Hy)
                hqz = dot(row, Hz)
                density =
                    0.5 * epsilon * (eqx^2 + eqy^2 + eqz^2) +
                    0.5 * mu * (hqx^2 + hqy^2 + hqz^2)
                contribution =
                    workspace.cubature_weights[q] * jacobian * density

                local_sums[1] += contribution
                if x <= left_pml_limit + tolerance
                    local_sums[2] += contribution
                elseif x >= right_pml_limit - tolerance
                    local_sums[3] += contribution
                else
                    local_sums[4] += contribution
                end
                if x <= left_monitor_max + tolerance
                    local_sums[5] += contribution
                end
                if x >= right_monitor_min - tolerance
                    local_sums[6] += contribution
                end
            end
        end
    end

    sums = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    return PMLReflectionDiagnostics(
        sums[1],
        sums[2],
        sums[3],
        sums[4],
        sums[5],
        sums[6],
    )
end

function write_reflection_header(io::IO)
    println(
        io,
        "step,time,total_energy,relative_total_energy," *
        "left_pml_energy,right_pml_energy,pml_energy,interior_energy," *
        "left_monitor_energy,right_monitor_energy," *
        "max_left_monitor_after_reflection,reflection_ratio",
    )
    return nothing
end

function write_reflection_row(
    io::IO,
    step::Int,
    time::Float64,
    diagnostics::PMLReflectionDiagnostics,
    initial_total_energy::Float64,
    max_left_monitor_after_reflection::Float64,
)
    relative_total =
        diagnostics.total_energy / max(initial_total_energy, eps(Float64))
    reflection_ratio =
        max_left_monitor_after_reflection /
        max(initial_total_energy, eps(Float64))
    println(
        io,
        join(
            (
                step,
                time,
                diagnostics.total_energy,
                relative_total,
                diagnostics.left_pml_energy,
                diagnostics.right_pml_energy,
                diagnostics.left_pml_energy + diagnostics.right_pml_energy,
                diagnostics.interior_energy,
                diagnostics.left_monitor_energy,
                diagnostics.right_monitor_energy,
                max_left_monitor_after_reflection,
                reflection_ratio,
            ),
            ',',
        ),
    )
    return reflection_ratio
end

function config_dictionary(
    config::PMLValidationConfig,
    case::PMLValidationCase;
    cubature_order::Int,
    dt::Float64,
    steps::Int,
    reflection_start::Float64,
    box::AxisAlignedBox,
    spectrum::PMLPulseSpectrum,
)
    return Dict{String, Any}(
        "driver" => "distributed_poisson_bracket_pml_validation",
        "mesh_path" => case.mesh_path,
        "mesh_name" => splitext(basename(case.mesh_path))[1],
        "domain_lower" => collect(box.lower),
        "domain_upper" => collect(box.upper),
        "domain_lengths" => collect(axis_aligned_box_lengths(box)),
        "order" => config.order,
        "esprk_order" => config.esprk_order,
        "final_time" => config.final_time,
        "cfl" => config.cfl,
        "epsilon" => config.epsilon,
        "mu" => config.mu,
        "formulation" => "PoissonBracket",
        "flux" => maxwell_flux_kind_label(case.flux_kind),
        "outer_boundary" => string(config.outer_boundary),
        "periodic_boundaries" => "y,z",
        "pml_width" => case.pml_width,
        "sigma_max" => case.sigma_max,
        "sigma_degree" => case.sigma_degree,
        "pml_a" => config.pml_a,
        "regularization" => config.regularization,
        "pulse_center" => config.pulse_center,
        "pulse_width" => config.pulse_width,
        "pulse_amplitude" => config.pulse_amplitude,
        "pulse_carrier" => "sine-modulated Gaussian",
        "central_elements_per_wavelength" =>
            spectrum.elements_per_wavelength,
        "central_wavelength" => spectrum.wavelength,
        "central_wavenumber" => spectrum.wavenumber,
        "central_angular_frequency" => spectrum.angular_frequency,
        "central_frequency" => spectrum.frequency,
        "central_frequency_source" => spectrum.source,
        "mesh_x_spacing" => spectrum.x_spacing,
        "mesh_x_intervals" => spectrum.x_intervals,
        "cubature_order" => cubature_order,
        "energy_every" => config.energy_every,
        "paraview_every" => config.paraview_every,
        "reflection_start" => reflection_start,
        "left_monitor_max" => config.left_monitor_max,
        "right_monitor_min" => config.right_monitor_min,
        "dt" => dt,
        "steps" => steps,
    )
end

function run_validation_case(
    config::PMLValidationConfig,
    case::PMLValidationCase,
    comm::MPI.Comm,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_mesh = nothing
    root_partition = nothing
    root_x_spacing = 0.0
    root_x_intervals = 0
    if rank == 0
        root_mesh = load_periodic_box_mesh(case.mesh_path)
        root_x_spacing, root_x_intervals = mesh_x_spacing(root_mesh)
        root_partition = balanced_spatial_partition(root_mesh, nranks)
    end
    x_spacing = MPI.bcast(root_x_spacing, comm; root = 0)
    x_intervals = MPI.bcast(root_x_intervals, comm; root = 0)
    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        config.order;
        comm = comm,
    )
    box = distributed_axis_aligned_box(distributed_dg)
    validate_case_geometry(config, case, box)
    spectrum = pulse_spectrum(config, x_spacing, x_intervals)

    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        pml_periodic_specs(box),
    )
    registry = pml_boundary_registry(config)
    formulation = PoissonBracketFormulation(case.flux_kind)
    electric, magnetic = initial_pulse_functions(config, spectrum)
    U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    pml = build_maxwell_nonlinear_pml(
        distributed_dg;
        sigma_x = pml_sigma_function(case, box),
        sigma_y = (x, y, z) -> 0.0,
        sigma_z = (x, y, z) -> 0.0,
        a = config.pml_a,
        regularization = config.regularization,
    )

    scheme = explicit_partitioned_symplectic_rk_scheme(
        config.esprk_order;
        first_partition = :H,
    )
    rk_workspace = MaxwellPartitionedRKWorkspace(U, scheme)
    cubature_order = resolved_cubature_order(config)
    reflection_workspace = PMLReflectionWorkspace(
        distributed_dg,
        cubature_order,
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
    steps = max(1, ceil(Int, config.final_time / estimated_dt))
    dt = config.final_time / steps
    reflection_start = default_reflection_start(config, case, box)

    run_paths = prepare_run_directory(case.output_dir, comm)
    energy_path = run_diagnostics_path(run_paths, "energy.csv")
    reflection_path =
        run_diagnostics_path(run_paths, "reflection_diagnostics.csv")
    configuration = config_dictionary(
        config,
        case;
        cubature_order = cubature_order,
        dt = dt,
        steps = steps,
        reflection_start = reflection_start,
        box = box,
        spectrum = spectrum,
    )

    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    initial_reflection = distributed_pml_reflection_diagnostics!(
        reflection_workspace,
        U,
        distributed_dg;
        epsilon = config.epsilon,
        mu = config.mu,
        x_min = box.lower[1],
        x_max = box.upper[1],
        pml_width = case.pml_width,
        left_monitor_max = config.left_monitor_max,
        right_monitor_min = config.right_monitor_min,
    )

    collectively_write_run_provenance(
        run_paths,
        comm;
        configuration = configuration,
        inputs = Dict(
            "mesh_path" => case.mesh_path,
            "partition" => "generated balanced spatial partition",
            "case_name" => case.name,
            "case_output_dir" => run_paths.root,
        ),
    )
    write_distributed_run_metadata(
        run_paths.config_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "running",
            "mpi_ranks" => nranks,
            "estimated_dt" => estimated_dt,
            "used_dt" => dt,
            "steps" => steps,
            "global_hmin" => global_hmin,
            "initial_energy_total" => initial_energy.total,
            "initial_reflection_total_energy" =>
                initial_reflection.total_energy,
            "central_frequency" => spectrum.frequency,
            "central_wavelength" => spectrum.wavelength,
            "central_wavenumber" => spectrum.wavenumber,
        ),
    )
    collectively_write_run_status(
        run_paths,
        comm,
        "running";
        values = Dict(
            "case" => case.name,
            "final_step" => steps,
            "target_final_time" => config.final_time,
            "dt" => dt,
        ),
    )

    energy_io = nothing
    reflection_io = nothing
    history_open_error = nothing
    if rank == 0
        try
            energy_io = open(energy_path, "w")
            reflection_io = open(reflection_path, "w")
            write_energy_header(energy_io)
            write_reflection_header(reflection_io)
        catch error
            history_open_error = sprint(showerror, error)
        end
    end
    history_open_error = MPI.bcast(history_open_error, comm; root = 0)
    history_open_error === nothing ||
        error("Could not open PML diagnostics: $history_open_error")

    series_entries = NamedTuple{
        (:step, :time, :dataset),
        Tuple{Int, Float64, String},
    }[]
    max_left_monitor_after_reflection = 0.0
    initial_reflection_ratio = 0.0
    collective_root_action(comm, "Initial PML diagnostics writing") do
        write_energy_row(energy_io, 0, 0.0, initial_energy, initial_energy.total)
        initial_reflection_ratio = write_reflection_row(
            reflection_io,
            0,
            0.0,
            initial_reflection,
            initial_reflection.total_energy,
            max_left_monitor_after_reflection,
        )
        flush(energy_io)
        flush(reflection_io)
    end

    if config.paraview_every > 0
        write_maxwell_paraview_snapshot!(
            series_entries,
            run_paths.root,
            distributed_dg,
            U,
            0,
            0.0;
            exact_electric = pml_zero_electric,
            exact_magnetic = pml_zero_magnetic,
        )
    end

    if rank == 0
        println("PML validation case: ", case.name)
        println("  MPI ranks:          ", nranks)
        println("  DG / ESPRK order:   ", config.order, " / ", config.esprk_order)
        println("  flux:               ", maxwell_flux_kind_label(case.flux_kind))
        println("  outer boundary:     ", config.outer_boundary)
        println("  mesh:               ", case.mesh_path)
        println("  domain lower:       ", box.lower)
        println("  domain upper:       ", box.upper)
        println("  x spacing/intervals:", spectrum.x_spacing, " / ", spectrum.x_intervals)
        println("  central wavelength: ", spectrum.wavelength)
        println("  central wavenumber: ", spectrum.wavenumber)
        println("  central frequency:  ", spectrum.frequency)
        println("  PML width:          ", case.pml_width)
        println("  sigma max/degree:   ", case.sigma_max, " / ", case.sigma_degree)
        println("  dt / steps:         ", dt, " / ", steps)
        println("  reflection start:   ", reflection_start)
        println("  output:             ", run_paths.root)
    end

    final_energy = initial_energy
    final_reflection = initial_reflection
    local_elapsed = 0.0
    try
        for step in 1:steps
            time = step == steps ? config.final_time : step * dt
            local_elapsed += @elapsed begin
                distributed_periodic_maxwell_nonlinear_pml_partitioned_symplectic_rk_step!(
                    U,
                    rk_workspace,
                    scheme,
                    dt,
                    distributed_dg,
                    periodic,
                    registry,
                    formulation,
                    pml;
                    ε = config.epsilon,
                    μ = config.mu,
                )
            end

            diagnostics_due =
                step % config.energy_every == 0 || step == steps
            if diagnostics_due
                final_energy = distributed_maxwell_energy(
                    U,
                    distributed_dg;
                    ε = config.epsilon,
                    μ = config.mu,
                )
                final_reflection = distributed_pml_reflection_diagnostics!(
                    reflection_workspace,
                    U,
                    distributed_dg;
                    epsilon = config.epsilon,
                    mu = config.mu,
                    x_min = box.lower[1],
                    x_max = box.upper[1],
                    pml_width = case.pml_width,
                    left_monitor_max = config.left_monitor_max,
                    right_monitor_min = config.right_monitor_min,
                )
                if time >= reflection_start
                    max_left_monitor_after_reflection = max(
                        max_left_monitor_after_reflection,
                        final_reflection.left_monitor_energy,
                    )
                end
                collective_root_action(
                    comm,
                    "PML diagnostics writing at step $step",
                ) do
                    relative_energy_drift = write_energy_row(
                        energy_io,
                        step,
                        time,
                        final_energy,
                        initial_energy.total,
                    )
                    reflection_ratio = write_reflection_row(
                        reflection_io,
                        step,
                        time,
                        final_reflection,
                        initial_reflection.total_energy,
                        max_left_monitor_after_reflection,
                    )
                    println(
                        "  step ", step, "/", steps,
                        ", t = ", time,
                        ", energy = ", final_energy.total,
                        ", relative drift = ", relative_energy_drift,
                        ", reflection ratio = ", reflection_ratio,
                    )
                    flush(energy_io)
                    flush(reflection_io)
                end
            end

            if config.paraview_every > 0 &&
               (step % config.paraview_every == 0 || step == steps)
                write_maxwell_paraview_snapshot!(
                    series_entries,
                    run_paths.root,
                    distributed_dg,
                    U,
                    step,
                    time;
                    exact_electric = pml_zero_electric,
                    exact_magnetic = pml_zero_magnetic,
                )
            end
        end
    finally
        collective_root_action(comm, "PML diagnostics closing") do
            energy_io === nothing || close(energy_io)
            reflection_io === nothing || close(reflection_io)
        end
    end

    elapsed = MPI.Allreduce(local_elapsed, max, comm)
    final_energy_ratio =
        final_energy.total / max(initial_energy.total, eps(Float64))
    reflection_ratio =
        max_left_monitor_after_reflection /
        max(initial_reflection.total_energy, eps(Float64))
    write_distributed_run_metadata(
        run_paths.config_dir,
        distributed_dg;
        configuration = configuration,
        runtime = Dict(
            "status" => "complete",
            "mpi_ranks" => nranks,
            "estimated_dt" => estimated_dt,
            "used_dt" => dt,
            "steps" => steps,
            "global_hmin" => global_hmin,
            "integration_wall_seconds" => elapsed,
            "initial_energy_total" => initial_energy.total,
            "final_energy_total" => final_energy.total,
            "final_energy_ratio" => final_energy_ratio,
            "initial_reflection_total_energy" =>
                initial_reflection.total_energy,
            "final_reflection_total_energy" =>
                final_reflection.total_energy,
            "max_left_monitor_after_reflection" =>
                max_left_monitor_after_reflection,
            "reflection_ratio" => reflection_ratio,
            "initial_reflection_ratio" => initial_reflection_ratio,
            "central_frequency" => spectrum.frequency,
            "central_wavelength" => spectrum.wavelength,
            "central_wavenumber" => spectrum.wavenumber,
        ),
    )
    collectively_write_run_status(
        run_paths,
        comm,
        "complete";
        values = Dict(
            "case" => case.name,
            "final_step" => steps,
            "final_time" => config.final_time,
            "integration_wall_seconds" => elapsed,
            "final_energy_ratio" => final_energy_ratio,
            "reflection_ratio" => reflection_ratio,
        ),
    )

    return (
        case = case.name,
        mesh = splitext(basename(case.mesh_path))[1],
        mesh_path = case.mesh_path,
        flux = maxwell_flux_kind_label(case.flux_kind),
        pml_width = case.pml_width,
        sigma_max = case.sigma_max,
        sigma_degree = case.sigma_degree,
        central_frequency = spectrum.frequency,
        central_wavelength = spectrum.wavelength,
        central_wavenumber = spectrum.wavenumber,
        central_elements_per_wavelength = spectrum.elements_per_wavelength,
        final_energy_ratio = final_energy_ratio,
        reflection_ratio = reflection_ratio,
        max_left_monitor_after_reflection =
            max_left_monitor_after_reflection,
        final_total_energy = final_energy.total,
        initial_total_energy = initial_energy.total,
        dt = dt,
        steps = steps,
        output_dir = run_paths.root,
    )
end

function write_sweep_summary(path::AbstractString, summaries)
    atomic_output_file(String(path)) do io
        println(
            io,
            "case,mesh,mesh_path,flux,pml_width,sigma_max,sigma_degree," *
            "central_frequency,central_wavelength,central_wavenumber," *
            "central_elements_per_wavelength," *
            "final_energy_ratio,reflection_ratio," *
            "max_left_monitor_after_reflection,final_total_energy," *
            "initial_total_energy,dt,steps,output_dir",
        )
        for summary in summaries
            println(
                io,
                join(
                    (
                        summary.case,
                        summary.mesh,
                        summary.mesh_path,
                        summary.flux,
                        summary.pml_width,
                        summary.sigma_max,
                        summary.sigma_degree,
                        summary.central_frequency,
                        summary.central_wavelength,
                        summary.central_wavenumber,
                        summary.central_elements_per_wavelength,
                        summary.final_energy_ratio,
                        summary.reflection_ratio,
                        summary.max_left_monitor_after_reflection,
                        summary.final_total_energy,
                        summary.initial_total_energy,
                        summary.dt,
                        summary.steps,
                        summary.output_dir,
                    ),
                    ',',
                ),
            )
        end
    end
    return String(path)
end

function run_validation_sweep(config::PMLValidationConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    cases = planned_cases(config)
    if rank == 0
        mkpath(config.output_dir)
        println("Poisson-bracket nonlinear PML validation sweep")
        println("---------------------------------------------")
        println("MPI ranks:          ", nranks)
        println("cases:              ", length(cases))
        println("meshes:             ", join(config.mesh_paths, ", "))
        println("DG order:           ", config.order)
        println("ESPRK order:        ", config.esprk_order)
        println("final time:         ", config.final_time)
        println("CFL:                ", config.cfl)
        println("outer boundary:     ", config.outer_boundary)
        println("output root:        ", config.output_dir)
    end

    summaries = Any[]
    for (case_index, case) in enumerate(cases)
        if rank == 0
            println()
            println("Running case ", case_index, "/", length(cases), ": ", case.name)
        end
        push!(summaries, run_validation_case(config, case, comm))
        MPI.Barrier(comm)
    end

    if rank == 0
        summary_path = joinpath(config.output_dir, "sweep_summary.csv")
        write_sweep_summary(summary_path, summaries)
        println()
        println("PML validation sweep complete.")
        println("Summary CSV: ", summary_path)
    end
    return summaries
end

function main(args::Vector{String})
    MPI.Init()
    try
        config = parse_config(args)
        config === nothing && return
        run_validation_sweep(config, MPI.COMM_WORLD)
    finally
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
