#!/usr/bin/env julia

using MPI
using Printf
using TOML

include(joinpath(@__DIR__, "distributed_pec_almond_scattering.jl"))

const AUSTIN_ALMOND_THETA_DEGREES = 90.0
const DEFAULT_AUSTIN_ALMOND_MONOSTATIC_OUTPUT = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "validation_sequence",
    "pec_almond_austin_monostatic_rcs",
)
const DEFAULT_AUSTIN_ALMOND_REFERENCE_DIR = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "benchmark",
    "AustinCEMBenchmarks",
    "Problem_III-Almonds",
    "IIIA-Reference Data",
)

Base.@kwdef struct AustinAlmondMonostaticConfig
    output_dir::String = DEFAULT_AUSTIN_ALMOND_MONOSTATIC_OUTPUT
    phi_degrees::Vector{Float64} = collect(0.0:0.5:180.0)
    polarizations::Vector{String} = ["H", "V"]
    max_final_relative_rms::Float64 = 1.0e-1
    max_sampled_relative_rms::Float64 = 1.0e-1
    reference_dir::String = DEFAULT_AUSTIN_ALMOND_REFERENCE_DIR
    reference_kind::String = "ref"
    frequency_key::String = "fx1"
    fail_on_residual_failure::Bool = false
    collect_only::Bool = false
    case_only::Bool = false
    scattering_args::Vector{String} = String[]
end

function austin_monostatic_usage(io::IO = stdout)
    println(io, """
Austin PEC almond monostatic RCS driver

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_pec_almond_austin_monostatic_rcs.jl \\
    [Austin options] [scattering options]

Austin benchmark options:
  --phi-degrees LIST           Incident azimuths phi_i in degrees. Default:
                               0:0.5:180, matching the Austin IIIA files.
  --polarization H|V|both|LIST Run HH, VV, or both polarizations. Default: both
  --max-final-relative-rms V   Final PEC residual gate on ||n x E_total|| RMS.
                               Default: 1e-1
  --max-sampled-relative-rms V Gate on max sampled PEC residual RMS.
                               Default: 1e-1
  --reference-dir PATH         Austin reference-data directory. Default:
                               benchmark/.../Problem_III-Almonds/
                               IIIA-Reference Data
  --reference-kind ref|meas-m1|meas-m2
                               Reference source. Default: ref
  --frequency-key fx1|fx2|fx3|fx4
                               Austin frequency key. Default: fx1
  --fail-on-residual-failure   Abort the sweep as soon as a case fails the PEC
                               residual gate. Default: keep sweeping but do not
                               compute benchmark errors for failed cases.
  --collect-only               Do not run Maxwell. Scan existing case
                               directories and rebuild the aggregate Austin
                               comparison CSV from per-case summaries or, when
                               needed, from case diagnostics.
  --case-only                  Run the selected case(s) and write their
                               per-case summaries, but do not rebuild the
                               shared comparison CSV. Intended for Slurm arrays.
  --output-dir PATH            Sweep output directory. Default:
                               output/validation_sequence/
                               pec_almond_austin_monostatic_rcs

Scattering options:
  All options accepted by distributed_pec_almond_scattering.jl are accepted,
  including --mesh, --frequency-hz, --order, --esprk-order, --final-time,
  --rcs-start-time, --rcs-every, --cfl, --flux, and --pml-width.

Convention:
  The Austin files report monostatic backscatter with theta_i = 90 deg and
  varying incident azimuth phi_i. For each phi_i this driver sets the observer
  direction rhat(theta=90, phi_i), propagates the incident plane wave along
  -rhat, and samples the RCS in the backscatter direction +rhat. H means
  phi-polarized incident/received field; V means theta-polarized
  incident/received field.

Outputs:
  <output-dir>/diagnostics/austin_monostatic_rcs_comparison.csv
  <output-dir>/cases/phi_<phi>_<H|V>/diagnostics/
      austin_monostatic_case_summary.csv
  <output-dir>/cases/phi_<phi>_<H|V>/...

Reference/error columns are populated only for rows whose PEC residual gate
passes.
""")
end

function austin_option_value(args::Vector{String}, index::Int, option::String)
    return option_value(args, index, option)
end

function parse_phi_degrees(value::AbstractString)
    stripped = strip(value)
    if occursin(":", stripped)
        parts = [parse(Float64, strip(item)) for item in split(stripped, ":")]
        length(parts) == 3 ||
            error("--phi-degrees range syntax is start:step:stop.")
        start, step, stop = parts
        step > 0.0 || error("--phi-degrees range step must be positive.")
        stop >= start || error("--phi-degrees range stop must exceed start.")
        values = Float64[]
        current = start
        while current <= stop + 10.0 * eps(Float64) * max(1.0, abs(stop))
            push!(values, min(current, stop))
            current += step
        end
        return values
    end
    return parse_float_list(stripped, "--phi-degrees")
end

function parse_austin_polarizations(value::AbstractString)
    lowered = lowercase(strip(value))
    if lowered == "both"
        return ["H", "V"]
    end
    entries = parse_comma_list(value, "--polarization")
    polarizations = String[]
    for entry in entries
        pol = uppercase(strip(entry))
        pol in ("H", "V") ||
            error("--polarization entries must be H, V, or both.")
        pol in polarizations || push!(polarizations, pol)
    end
    isempty(polarizations) && error("--polarization requires H, V, or both.")
    return polarizations
end

function forward_scattering_option!(
    scattering_args::Vector{String},
    args::Vector{String},
    index::Int,
)
    push!(scattering_args, args[index])
    if !occursin("=", args[index]) &&
       index < length(args) &&
       !startswith(args[index + 1], "--")
        push!(scattering_args, args[index + 1])
        return index + 1
    end
    return index
end

function parse_austin_monostatic_config(args::Vector{String})
    values = Dict{Symbol, Any}(
        field => getfield(AustinAlmondMonostaticConfig(), field)
        for field in fieldnames(AustinAlmondMonostaticConfig)
    )
    scattering_args = String[]

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            austin_monostatic_usage()
            println()
            pec_almond_usage()
            return nothing
        elseif argument == "--phi-degrees" ||
               startswith(argument, "--phi-degrees=")
            raw, index =
                austin_option_value(args, index, "--phi-degrees")
            values[:phi_degrees] = parse_phi_degrees(raw)
        elseif argument == "--polarization" ||
               startswith(argument, "--polarization=")
            raw, index =
                austin_option_value(args, index, "--polarization")
            values[:polarizations] = parse_austin_polarizations(raw)
        elseif argument == "--max-final-relative-rms" ||
               startswith(argument, "--max-final-relative-rms=")
            raw, index = austin_option_value(
                args,
                index,
                "--max-final-relative-rms",
            )
            values[:max_final_relative_rms] = parse(Float64, raw)
        elseif argument == "--max-sampled-relative-rms" ||
               startswith(argument, "--max-sampled-relative-rms=")
            raw, index = austin_option_value(
                args,
                index,
                "--max-sampled-relative-rms",
            )
            values[:max_sampled_relative_rms] = parse(Float64, raw)
        elseif argument == "--reference-dir" ||
               startswith(argument, "--reference-dir=")
            raw, index =
                austin_option_value(args, index, "--reference-dir")
            values[:reference_dir] = abspath(raw)
        elseif argument == "--reference-kind" ||
               startswith(argument, "--reference-kind=")
            raw, index =
                austin_option_value(args, index, "--reference-kind")
            values[:reference_kind] = String(raw)
        elseif argument == "--frequency-key" ||
               startswith(argument, "--frequency-key=")
            raw, index =
                austin_option_value(args, index, "--frequency-key")
            values[:frequency_key] = String(raw)
        elseif argument == "--output-dir" ||
               startswith(argument, "--output-dir=")
            raw, index = austin_option_value(args, index, "--output-dir")
            values[:output_dir] = abspath(raw)
        elseif argument == "--fail-on-residual-failure"
            values[:fail_on_residual_failure] = true
        elseif argument == "--collect-only"
            values[:collect_only] = true
        elseif argument == "--case-only"
            values[:case_only] = true
        else
            index = forward_scattering_option!(scattering_args, args, index)
        end
        index += 1
    end

    values[:scattering_args] = scattering_args
    config = AustinAlmondMonostaticConfig(; values...)
    isempty(config.phi_degrees) &&
        error("--phi-degrees requires at least one value.")
    all(phi -> 0.0 <= phi <= 180.0, config.phi_degrees) ||
        error("--phi-degrees values must lie in [0,180].")
    config.max_final_relative_rms >= 0.0 ||
        error("--max-final-relative-rms must be non-negative.")
    config.max_sampled_relative_rms >= 0.0 ||
        error("--max-sampled-relative-rms must be non-negative.")
    config.reference_kind in ("ref", "meas-m1", "meas-m2") ||
        error("--reference-kind must be ref, meas-m1, or meas-m2.")
    config.frequency_key in ("fx1", "fx2", "fx3", "fx4") ||
        error("--frequency-key must be fx1, fx2, fx3, or fx4.")
    config.collect_only && config.case_only &&
        error("--collect-only and --case-only are mutually exclusive.")
    return config
end

function spherical_basis(theta_degrees::Float64, phi_degrees::Float64)
    theta = deg2rad(theta_degrees)
    phi = deg2rad(phi_degrees)
    sinθ = sin(theta)
    cosθ = cos(theta)
    sinφ = sin(phi)
    cosφ = cos(phi)
    rhat = (sinθ * cosφ, sinθ * sinφ, cosθ)
    e_theta = (cosθ * cosφ, cosθ * sinφ, -sinθ)
    e_phi = (-sinφ, cosφ, 0.0)
    return rhat, e_theta, e_phi
end

function austin_incident_direction(phi_degrees::Float64)
    rhat, _, _ = spherical_basis(AUSTIN_ALMOND_THETA_DEGREES, phi_degrees)
    return (-rhat[1], -rhat[2], -rhat[3])
end

function austin_incident_polarization(phi_degrees::Float64, polarization::String)
    _, e_theta, e_phi = spherical_basis(
        AUSTIN_ALMOND_THETA_DEGREES,
        phi_degrees,
    )
    return polarization == "H" ? e_phi : e_theta
end

function vector3_cli(vector::NTuple{3, Float64})
    return join((@sprintf("%.17g", value) for value in vector), ",")
end

function angle_token(angle::Float64)
    token = @sprintf("%.6f", angle)
    token = replace(token, "-" => "m", "." => "p")
    return replace(token, r"p?0+$" => "")
end

function reference_file_path(config::AustinAlmondMonostaticConfig, pol::String)
    prefix = if config.reference_kind == "ref"
        "ref"
    elseif config.reference_kind == "meas-m1"
        "meas.m1"
    else
        "meas.m2"
    end
    return joinpath(
        config.reference_dir,
        "$(prefix)_rcs.III.A.s1.$(config.frequency_key).$(pol).txt",
    )
end

function read_reference_rows(path::AbstractString)
    isfile(path) || error("Austin reference file not found: $path")
    rows = NamedTuple[]
    for (line_number, line) in enumerate(eachline(path))
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue
        fields = split(stripped)
        length(fields) >= 4 ||
            error("Malformed reference row $line_number in $path.")
        push!(
            rows,
            (
                frequency_hz = parse(Float64, fields[1]),
                theta_degrees = parse(Float64, fields[2]),
                phi_degrees = parse(Float64, fields[3]),
                rcs_dbsm = parse(Float64, fields[4]),
            ),
        )
    end
    isempty(rows) && error("Reference file contains no rows: $path")
    return sort(rows; by = row -> row.phi_degrees)
end

function interpolate_reference_dbsm(rows, phi_degrees::Float64)
    phi_degrees < rows[1].phi_degrees && return NaN
    phi_degrees > rows[end].phi_degrees && return NaN
    for index in 1:(length(rows) - 1)
        left = rows[index]
        right = rows[index + 1]
        if abs(phi_degrees - left.phi_degrees) <= 1.0e-10
            return left.rcs_dbsm
        elseif left.phi_degrees <= phi_degrees <= right.phi_degrees
            width = right.phi_degrees - left.phi_degrees
            width == 0.0 && return left.rcs_dbsm
            weight = (phi_degrees - left.phi_degrees) / width
            return (1.0 - weight) * left.rcs_dbsm + weight * right.rcs_dbsm
        end
    end
    abs(phi_degrees - rows[end].phi_degrees) <= 1.0e-10 &&
        return rows[end].rcs_dbsm
    return NaN
end

function read_one_rcs_row(path::AbstractString)
    isfile(path) || error("RCS CSV not found: $path")
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    length(lines) == 2 ||
        error("Expected exactly one RCS data row in $path; found $(length(lines) - 1).")
    header = String.(split(lines[1], ","))
    fields = String.(split(lines[2], ",", keepempty = true))
    length(fields) == length(header) ||
        error("Malformed RCS row in $path.")
    return Dict(header[i] => fields[i] for i in eachindex(header))
end

function safe_db(value::Float64)
    return value > 0.0 ? 10.0 * log10(value) : NaN
end

function component_name(pol::String)
    return pol == "H" ? "phi" : "theta"
end

function component_rcs(
    row::AbstractDict{<:AbstractString, <:AbstractString},
    pol::String,
    amplitude::Float64,
)
    value = if pol == "H"
        parse(Float64, row["abs_Ephi"])
    else
        parse(Float64, row["abs_Etheta"])
    end
    return 4.0 * pi * value^2 / max(amplitude^2, eps(Float64))
end

function build_monostatic_case_config(
    config::AustinAlmondMonostaticConfig,
    phi_degrees::Float64,
    pol::String,
)
    incident_direction = austin_incident_direction(phi_degrees)
    incident_polarization = austin_incident_polarization(phi_degrees, pol)
    case_dir = joinpath(
        config.output_dir,
        "cases",
        "phi_$(angle_token(phi_degrees))_$(pol)",
    )
    enforced_args = [
        "--scatterer-name=PEC almond Austin monostatic $(pol)",
        "--enable-rcs",
        "--rcs-theta-count=1",
        "--rcs-theta-min-degrees=$(AUSTIN_ALMOND_THETA_DEGREES)",
        "--rcs-theta-max-degrees=$(AUSTIN_ALMOND_THETA_DEGREES)",
        "--rcs-phi-degrees=$(@sprintf("%.16g", phi_degrees))",
        "--incident-direction=$(vector3_cli(incident_direction))",
        "--incident-polarization=$(vector3_cli(incident_polarization))",
        "--output-dir=$(case_dir)",
    ]
    return parse_config(
        vcat(
            pec_almond_default_args(),
            translate_pec_almond_args(config.scattering_args),
            enforced_args,
        ),
    )
end

const AUSTIN_MONOSTATIC_CASE_SUMMARY_FILE =
    "austin_monostatic_case_summary.csv"
const AUSTIN_MONOSTATIC_SUMMARY_COLUMNS = (
    "status",
    "message",
    "polarization",
    "component",
    "frequency_key",
    "reference_kind",
    "incident_theta_degrees",
    "incident_phi_degrees",
    "observation_theta_degrees",
    "observation_phi_degrees",
    "incident_dir_x",
    "incident_dir_y",
    "incident_dir_z",
    "incident_pol_x",
    "incident_pol_y",
    "incident_pol_z",
    "rcs_component",
    "rcs_component_dbsm",
    "rcs_component_over_lambda2",
    "rcs_component_over_lambda2_db",
    "reference_rcs_dbsm",
    "reference_rcs_over_lambda2_db",
    "error_db",
    "samples",
    "window_sum",
    "final_pec_relative_rms",
    "max_sampled_pec_relative_rms",
    "final_time",
    "steps",
    "dt",
    "case_output_dir",
    "reference_file",
)

Base.@kwdef struct AustinMonostaticCaseRecord
    status::String
    message::String
    polarization::String
    component::String
    frequency_key::String
    reference_kind::String
    incident_theta_degrees::Float64
    incident_phi_degrees::Float64
    observation_theta_degrees::Float64
    observation_phi_degrees::Float64
    incident_dir_x::Float64
    incident_dir_y::Float64
    incident_dir_z::Float64
    incident_pol_x::Float64
    incident_pol_y::Float64
    incident_pol_z::Float64
    rcs_component::Float64
    rcs_component_dbsm::Float64
    rcs_component_over_lambda2::Float64
    rcs_component_over_lambda2_db::Float64
    reference_rcs_dbsm::Float64
    reference_rcs_over_lambda2_db::Float64
    error_db::Float64
    samples::Int
    window_sum::Float64
    final_pec_relative_rms::Float64
    max_sampled_pec_relative_rms::Float64
    final_time::Float64
    steps::Int
    dt::Float64
    case_output_dir::String
    reference_file::String
end

function write_monostatic_header(io::IO)
    println(io, join(AUSTIN_MONOSTATIC_SUMMARY_COLUMNS, ","))
end

function csv_value(value)
    if value isa AbstractString
        return replace(value, '"' => "\"\"")
    elseif value isa Integer
        return string(value)
    elseif value isa Float64
        return isfinite(value) ? @sprintf("%.16e", value) : "NaN"
    else
        return string(value)
    end
end

function write_monostatic_row(io::IO, values)
    println(io, join((csv_value(value) for value in values), ","))
end

function record_values(record::AustinMonostaticCaseRecord)
    return (
        record.status,
        record.message,
        record.polarization,
        record.component,
        record.frequency_key,
        record.reference_kind,
        record.incident_theta_degrees,
        record.incident_phi_degrees,
        record.observation_theta_degrees,
        record.observation_phi_degrees,
        record.incident_dir_x,
        record.incident_dir_y,
        record.incident_dir_z,
        record.incident_pol_x,
        record.incident_pol_y,
        record.incident_pol_z,
        record.rcs_component,
        record.rcs_component_dbsm,
        record.rcs_component_over_lambda2,
        record.rcs_component_over_lambda2_db,
        record.reference_rcs_dbsm,
        record.reference_rcs_over_lambda2_db,
        record.error_db,
        record.samples,
        record.window_sum,
        record.final_pec_relative_rms,
        record.max_sampled_pec_relative_rms,
        record.final_time,
        record.steps,
        record.dt,
        record.case_output_dir,
        record.reference_file,
    )
end

function write_monostatic_record(io::IO, record::AustinMonostaticCaseRecord)
    write_monostatic_row(io, record_values(record))
end

function case_summary_path(case_output_dir::AbstractString)
    return joinpath(
        case_output_dir,
        "diagnostics",
        AUSTIN_MONOSTATIC_CASE_SUMMARY_FILE,
    )
end

function write_case_summary(
    path::AbstractString,
    record::AustinMonostaticCaseRecord,
)
    mkpath(dirname(path))
    open(path, "w") do io
        write_monostatic_header(io)
        write_monostatic_record(io, record)
    end
    return path
end

function write_comparison_summary(
    path::AbstractString,
    records::Vector{AustinMonostaticCaseRecord},
)
    mkpath(dirname(path))
    open(path, "w") do io
        write_monostatic_header(io)
        for record in records
            write_monostatic_record(io, record)
        end
    end
    return path
end

function parse_summary_float(
    row::AbstractDict{<:AbstractString, <:AbstractString},
    key::AbstractString,
)
    return parse(Float64, row[key])
end

function parse_summary_int(
    row::AbstractDict{<:AbstractString, <:AbstractString},
    key::AbstractString,
)
    return parse(Int, row[key])
end

function record_from_summary_row(
    row::AbstractDict{<:AbstractString, <:AbstractString},
)
    return AustinMonostaticCaseRecord(
        status = String(row["status"]),
        message = String(row["message"]),
        polarization = String(row["polarization"]),
        component = String(row["component"]),
        frequency_key = String(row["frequency_key"]),
        reference_kind = String(row["reference_kind"]),
        incident_theta_degrees =
            parse_summary_float(row, "incident_theta_degrees"),
        incident_phi_degrees =
            parse_summary_float(row, "incident_phi_degrees"),
        observation_theta_degrees =
            parse_summary_float(row, "observation_theta_degrees"),
        observation_phi_degrees =
            parse_summary_float(row, "observation_phi_degrees"),
        incident_dir_x = parse_summary_float(row, "incident_dir_x"),
        incident_dir_y = parse_summary_float(row, "incident_dir_y"),
        incident_dir_z = parse_summary_float(row, "incident_dir_z"),
        incident_pol_x = parse_summary_float(row, "incident_pol_x"),
        incident_pol_y = parse_summary_float(row, "incident_pol_y"),
        incident_pol_z = parse_summary_float(row, "incident_pol_z"),
        rcs_component = parse_summary_float(row, "rcs_component"),
        rcs_component_dbsm = parse_summary_float(row, "rcs_component_dbsm"),
        rcs_component_over_lambda2 =
            parse_summary_float(row, "rcs_component_over_lambda2"),
        rcs_component_over_lambda2_db =
            parse_summary_float(row, "rcs_component_over_lambda2_db"),
        reference_rcs_dbsm = parse_summary_float(row, "reference_rcs_dbsm"),
        reference_rcs_over_lambda2_db =
            parse_summary_float(row, "reference_rcs_over_lambda2_db"),
        error_db = parse_summary_float(row, "error_db"),
        samples = parse_summary_int(row, "samples"),
        window_sum = parse_summary_float(row, "window_sum"),
        final_pec_relative_rms =
            parse_summary_float(row, "final_pec_relative_rms"),
        max_sampled_pec_relative_rms =
            parse_summary_float(row, "max_sampled_pec_relative_rms"),
        final_time = parse_summary_float(row, "final_time"),
        steps = parse_summary_int(row, "steps"),
        dt = parse_summary_float(row, "dt"),
        case_output_dir = String(row["case_output_dir"]),
        reference_file = String(row["reference_file"]),
    )
end

function read_case_summary(path::AbstractString)
    row = read_one_rcs_row(path)
    for column in AUSTIN_MONOSTATIC_SUMMARY_COLUMNS
        haskey(row, column) || error("Case summary $path is missing $column.")
    end
    return record_from_summary_row(row)
end

function read_resolved_case_configuration(case_output_dir::AbstractString)
    path = joinpath(case_output_dir, "config", "resolved_config.toml")
    isfile(path) || return Dict{String, Any}()
    parsed = TOML.parsefile(path)
    configuration = get(parsed, "configuration", Dict{String, Any}())
    return configuration isa Dict ? configuration : Dict{String, Any}()
end

function config_float(configuration, key::AbstractString, default::Float64)
    value = get(configuration, key, default)
    value === nothing && return default
    return Float64(value)
end

function config_int(configuration, key::AbstractString, default::Int)
    value = get(configuration, key, default)
    value === nothing && return default
    return Int(value)
end

function read_pec_residual_summary(path::AbstractString)
    isfile(path) || error("PEC residual CSV not found: $path")
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    length(lines) >= 2 || error("PEC residual CSV contains no data rows: $path")
    header = String.(split(lines[1], ","))
    relative_index = findfirst(==("relative_rms_norm"), header)
    relative_index === nothing &&
        error("PEC residual CSV $path has no relative_rms_norm column.")
    values = Float64[]
    for line in lines[2:end]
        fields = String.(split(line, ",", keepempty = true))
        length(fields) == length(header) ||
            error("Malformed PEC residual row in $path.")
        push!(values, parse(Float64, fields[relative_index]))
    end
    return (final = values[end], maximum = maximum(values))
end

function build_monostatic_case_record(
    config::AustinAlmondMonostaticConfig,
    case_config,
    phi_degrees::Float64,
    pol::String;
    final_pec_relative_rms::Float64,
    max_sampled_pec_relative_rms::Float64,
    final_time::Float64,
    steps::Int,
    dt::Float64,
)
    rcs_path = joinpath(case_config.output_dir, "diagnostics", "rcs.csv")
    rcs_row = read_one_rcs_row(rcs_path)
    wave_values = resolved_wave_values(case_config)
    sigma = component_rcs(rcs_row, pol, abs(case_config.amplitude))
    sigma_dbsm = safe_db(sigma)
    sigma_over_lambda2 = sigma / max(wave_values.wavelength^2, eps(Float64))
    sigma_over_lambda2_db = safe_db(sigma_over_lambda2)

    residual_pass =
        final_pec_relative_rms <= config.max_final_relative_rms &&
        max_sampled_pec_relative_rms <= config.max_sampled_relative_rms
    reference_path = reference_file_path(config, pol)
    rows = read_reference_rows(reference_path)
    reference_dbsm = interpolate_reference_dbsm(rows, phi_degrees)
    reference_frequency =
        sum(row.frequency_hz for row in rows) / length(rows)
    reference_wavelength = PEC_ALMOND_C0 / reference_frequency
    reference_norm_db = reference_dbsm - 20.0 * log10(reference_wavelength)
    error_db = sigma_over_lambda2_db - reference_norm_db
    message = "PEC residual gate passed"
    status = "PASS"
    if !residual_pass
        status = "RESIDUAL_FAIL"
        message = "PEC residual gate failed; benchmark error is diagnostic only"
    end

    incident_direction = case_config.incident_direction
    incident_polarization = case_config.incident_polarization
    return AustinMonostaticCaseRecord(
        status = status,
        message = message,
        polarization = pol,
        component = component_name(pol),
        frequency_key = config.frequency_key,
        reference_kind = config.reference_kind,
        incident_theta_degrees = AUSTIN_ALMOND_THETA_DEGREES,
        incident_phi_degrees = phi_degrees,
        observation_theta_degrees =
            parse(Float64, rcs_row["theta_degrees"]),
        observation_phi_degrees = parse(Float64, rcs_row["phi_degrees"]),
        incident_dir_x = incident_direction[1],
        incident_dir_y = incident_direction[2],
        incident_dir_z = incident_direction[3],
        incident_pol_x = incident_polarization[1],
        incident_pol_y = incident_polarization[2],
        incident_pol_z = incident_polarization[3],
        rcs_component = sigma,
        rcs_component_dbsm = sigma_dbsm,
        rcs_component_over_lambda2 = sigma_over_lambda2,
        rcs_component_over_lambda2_db = sigma_over_lambda2_db,
        reference_rcs_dbsm = reference_dbsm,
        reference_rcs_over_lambda2_db = reference_norm_db,
        error_db = error_db,
        samples = parse(Int, rcs_row["samples"]),
        window_sum = parse(Float64, rcs_row["window_sum"]),
        final_pec_relative_rms = final_pec_relative_rms,
        max_sampled_pec_relative_rms = max_sampled_pec_relative_rms,
        final_time = final_time,
        steps = steps,
        dt = dt,
        case_output_dir = case_config.output_dir,
        reference_file = reference_path,
    )
end

function recover_monostatic_case_record(
    config::AustinAlmondMonostaticConfig,
    case_config,
    phi_degrees::Float64,
    pol::String,
)
    residuals = read_pec_residual_summary(
        joinpath(case_config.output_dir, "diagnostics", "pec_boundary_residual.csv"),
    )
    resolved = read_resolved_case_configuration(case_config.output_dir)
    return build_monostatic_case_record(
        config,
        case_config,
        phi_degrees,
        pol;
        final_pec_relative_rms = residuals.final,
        max_sampled_pec_relative_rms = residuals.maximum,
        final_time = config_float(resolved, "final_time", NaN),
        steps = config_int(resolved, "steps", 0),
        dt = config_float(resolved, "dt", NaN),
    )
end

function collect_monostatic_case_records(config::AustinAlmondMonostaticConfig)
    records = AustinMonostaticCaseRecord[]
    recovered = String[]
    missing = String[]
    for phi_degrees in config.phi_degrees
        for pol in config.polarizations
            case_config =
                build_monostatic_case_config(config, phi_degrees, pol)
            path = case_summary_path(case_config.output_dir)
            if isfile(path)
                push!(records, read_case_summary(path))
            elseif isfile(joinpath(case_config.output_dir, "diagnostics", "rcs.csv"))
                record = recover_monostatic_case_record(
                    config,
                    case_config,
                    phi_degrees,
                    pol,
                )
                write_case_summary(path, record)
                push!(records, record)
                push!(recovered, path)
            else
                push!(missing, case_config.output_dir)
            end
        end
    end
    isempty(records) &&
        error("No Austin monostatic case summaries found under $(config.output_dir).")
    if !isempty(recovered)
        println("Recovered missing case summaries:")
        for path in recovered
            println("  ", path)
        end
    end
    if !isempty(missing)
        println("Missing Austin monostatic case outputs:")
        for path in missing
            println("  ", path)
        end
    end
    return records
end

function run_austin_monostatic(config::AustinAlmondMonostaticConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    reference_error = nothing
    if rank == 0
        try
            for pol in config.polarizations
                read_reference_rows(reference_file_path(config, pol))
            end
        catch error
            reference_error = sprint(showerror, error)
        end
    end
    reference_error = MPI.bcast(reference_error, comm; root = 0)
    reference_error === nothing ||
        error("Austin reference validation failed: $reference_error")

    if rank == 0
        mkpath(joinpath(config.output_dir, "diagnostics"))
        mkpath(joinpath(config.output_dir, "cases"))
    end
    MPI.Barrier(comm)

    comparison_path = joinpath(
        config.output_dir,
        "diagnostics",
        "austin_monostatic_rcs_comparison.csv",
    )
    if rank == 0
        println("Austin PEC almond monostatic RCS sweep")
        println("--------------------------------------")
        println("output:              ", config.output_dir)
        println("phi count:           ", length(config.phi_degrees))
        println("polarizations:       ", join(config.polarizations, ","))
        println("residual gate final: ", config.max_final_relative_rms)
        println("residual gate max:   ", config.max_sampled_relative_rms)
        println("collect only:        ", config.collect_only)
        println("case only:           ", config.case_only)
    end

    if !config.collect_only
        for phi_degrees in config.phi_degrees
            for pol in config.polarizations
                case_config =
                    build_monostatic_case_config(config, phi_degrees, pol)
                if rank == 0
                    println()
                    println(
                        "Running Austin monostatic case: phi = ",
                        phi_degrees,
                        " deg, pol = ",
                        pol,
                    )
                end
                summary = run_experiment(case_config, comm)
                residual_pass =
                    summary.final_pec_boundary_residual_relative_rms <=
                    config.max_final_relative_rms &&
                    summary.max_sampled_pec_boundary_residual_relative_rms <=
                    config.max_sampled_relative_rms

                case_error = nothing
                if rank == 0
                    try
                        record = build_monostatic_case_record(
                            config,
                            case_config,
                            phi_degrees,
                            pol;
                            final_pec_relative_rms =
                                summary.final_pec_boundary_residual_relative_rms,
                            max_sampled_pec_relative_rms =
                                summary.max_sampled_pec_boundary_residual_relative_rms,
                            final_time = summary.final_time,
                            steps = summary.steps,
                            dt = summary.dt,
                        )
                        write_case_summary(
                            case_summary_path(summary.output_dir),
                            record,
                        )
                    catch error
                        case_error = sprint(showerror, error)
                    end
                end
                case_error = MPI.bcast(case_error, comm; root = 0)
                case_error === nothing ||
                    error("Austin case postprocessing failed: $case_error")

                if !residual_pass && config.fail_on_residual_failure
                    error(
                        "PEC residual gate failed for phi=$(phi_degrees), " *
                        "polarization=$(pol).",
                    )
                end
            end
        end
    end

    if config.case_only
        rank == 0 && println("Case-only run complete; aggregate CSV was not modified.")
        return comparison_path
    end

    if rank == 0
        records = collect_monostatic_case_records(config)
        write_comparison_summary(comparison_path, records)
        println()
        println("Austin monostatic sweep complete.")
        println("case summaries:  ", length(records))
        println("comparison CSV: ", comparison_path)
    end
    return comparison_path
end

function austin_monostatic_main(args::Vector{String})
    if any(argument -> argument == "--help" || argument == "-h", args)
        austin_monostatic_usage()
        return nothing
    end

    MPI.Init()
    comm = MPI.COMM_WORLD
    try
        config = parse_austin_monostatic_config(args)
        config === nothing && return nothing
        run_austin_monostatic(config, comm)
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
    austin_monostatic_main(ARGS)
end
