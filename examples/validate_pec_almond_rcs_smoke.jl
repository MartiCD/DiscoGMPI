#!/usr/bin/env julia

# Smoke validation for the PEC almond RCS path. This gate intentionally does
# not compare against an analytical solution; the almond needs benchmark
# reference data rather than Mie theory. The purpose here is to verify that the
# near-to-far pipeline writes a finite, positive RCS table at the requested
# observation angles.

using Dates
using MPI
using Printf

include(joinpath(@__DIR__, "distributed_pec_almond_scattering.jl"))

const DEFAULT_PEC_ALMOND_RCS_SMOKE_OUTPUT = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "validation_sequence",
    "pec_almond_rcs_smoke",
)

Base.@kwdef struct PECAlmondRCSSmokeGateConfig
    scattering_config::MetallicSphereScatteringConfig
    min_max_rcs::Float64 = 0.0
    min_samples::Int = 1
    min_window_sum::Float64 = 0.0
    angle_tolerance_degrees::Float64 = 1.0e-8
end

function pec_almond_rcs_smoke_usage(io::IO = stdout)
    println(io, """
PEC almond RCS smoke validation gate

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/validate_pec_almond_rcs_smoke.jl [gate options] [scattering options]

Gate options:
  --min-max-rcs V              Require max numerical RCS >= V. Default: 0
  --min-samples N              Require every RCS row to have at least N
                               accumulated samples. Default: 1
  --min-window-sum V           Require every RCS row to have window_sum > V.
                               Default: 0
  --angle-tolerance-degrees V  Tolerance for matching requested theta/phi
                               angles in rcs.csv. Default: 1e-8
  --help                       Show this message.

Default scattering options:
  --mesh=examples/meshes/pec_almond_scattering_step_fx1_epw4.vtk
  --order=1 --esprk-order=2
  --final-time=1e-6
  --enable-rcs --rcs-start-time=0.0 --rcs-every=1
  --rcs-theta-count=9 --rcs-phi-degrees=0
  --diagnostics-every=1 --paraview-every=0
  --output-dir=output/validation_sequence/pec_almond_rcs_smoke

All options accepted by distributed_pec_almond_scattering.jl can be passed here
and override the defaults above, including --frequency-hz. This smoke gate only
checks RCS table integrity: finite values, positive RCS, positive accumulation
metadata, and coverage of the requested observation angles.
""")
end

function pec_almond_rcs_option_value(
    args::Vector{String},
    index::Int,
    option::String,
)
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

function default_pec_almond_rcs_smoke_args()
    return vcat(
        pec_almond_default_args(),
        [
            "--order=1",
            "--esprk-order=2",
            "--final-time=1e-6",
            "--enable-rcs",
            "--rcs-start-time=0.0",
            "--rcs-every=1",
            "--rcs-theta-count=9",
            "--rcs-phi-degrees=0",
            "--diagnostics-every=1",
            "--paraview-every=0",
            "--output-dir=$(DEFAULT_PEC_ALMOND_RCS_SMOKE_OUTPUT)",
        ],
    )
end

valid_nonnegative_threshold(value::Float64) =
    !isnan(value) && value >= 0.0

function parse_pec_almond_rcs_smoke_gate_config(args::Vector{String})
    min_max_rcs = 0.0
    min_samples = 1
    min_window_sum = 0.0
    angle_tolerance_degrees = 1.0e-8
    scattering_args = String[]

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            pec_almond_rcs_smoke_usage()
            println()
            pec_almond_usage()
            return nothing
        elseif argument == "--min-max-rcs" ||
               startswith(argument, "--min-max-rcs=")
            raw, index =
                pec_almond_rcs_option_value(args, index, "--min-max-rcs")
            min_max_rcs = parse(Float64, raw)
        elseif argument == "--min-samples" ||
               startswith(argument, "--min-samples=")
            raw, index =
                pec_almond_rcs_option_value(args, index, "--min-samples")
            min_samples = parse(Int, raw)
        elseif argument == "--min-window-sum" ||
               startswith(argument, "--min-window-sum=")
            raw, index =
                pec_almond_rcs_option_value(args, index, "--min-window-sum")
            min_window_sum = parse(Float64, raw)
        elseif argument == "--angle-tolerance-degrees" ||
               startswith(argument, "--angle-tolerance-degrees=")
            raw, index = pec_almond_rcs_option_value(
                args,
                index,
                "--angle-tolerance-degrees",
            )
            angle_tolerance_degrees = parse(Float64, raw)
        else
            push!(scattering_args, argument)
        end
        index += 1
    end

    valid_nonnegative_threshold(min_max_rcs) ||
        error("--min-max-rcs must be non-negative.")
    min_samples >= 1 || error("--min-samples must be positive.")
    valid_nonnegative_threshold(min_window_sum) ||
        error("--min-window-sum must be non-negative.")
    valid_nonnegative_threshold(angle_tolerance_degrees) ||
        error("--angle-tolerance-degrees must be non-negative.")

    scattering_config = parse_config(
        vcat(
            default_pec_almond_rcs_smoke_args(),
            translate_pec_almond_args(scattering_args),
        ),
    )
    scattering_config === nothing && return nothing
    scattering_config.rcs_enabled ||
        error("PEC almond RCS smoke validation requires RCS enabled.")
    return PECAlmondRCSSmokeGateConfig(
        scattering_config = scattering_config,
        min_max_rcs = min_max_rcs,
        min_samples = min_samples,
        min_window_sum = min_window_sum,
        angle_tolerance_degrees = angle_tolerance_degrees,
    )
end

struct AlmondRCSRow
    theta_degrees::Float64
    phi_degrees::Float64
    abs_E_infinity::Float64
    abs_Etheta::Float64
    abs_Ephi::Float64
    rcs::Float64
    rcs_db::Float64
    samples::Int
    window_sum::Float64
end

function read_simple_csv(path::AbstractString)
    isfile(path) || error("CSV file not found: $path")
    lines = readlines(path)
    isempty(lines) && error("CSV file is empty: $path")
    header = split(lines[1], ",")
    rows = Vector{Dict{String, String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ",", keepempty = true)
        length(fields) == length(header) ||
            error("Malformed CSV row in $path.")
        push!(rows, Dict(header[i] => fields[i] for i in eachindex(header)))
    end
    return rows
end

function parse_almond_rcs_rows(path::AbstractString)
    rows = AlmondRCSRow[]
    for row in read_simple_csv(path)
        push!(
            rows,
            AlmondRCSRow(
                parse(Float64, row["theta_degrees"]),
                parse(Float64, row["phi_degrees"]),
                parse(Float64, row["abs_E_infinity"]),
                parse(Float64, row["abs_Etheta"]),
                parse(Float64, row["abs_Ephi"]),
                parse(Float64, row["rcs"]),
                parse(Float64, row["rcs_db"]),
                parse(Int, row["samples"]),
                parse(Float64, row["window_sum"]),
            ),
        )
    end
    isempty(rows) && error("RCS CSV contains no data rows: $path")
    return rows
end

function finite_max(values)
    finite_values = filter(isfinite, collect(values))
    isempty(finite_values) && return Inf
    return maximum(finite_values)
end

function finite_min(values)
    finite_values = filter(isfinite, collect(values))
    isempty(finite_values) && return Inf
    return minimum(finite_values)
end

function expected_rcs_angle_pairs(config::MetallicSphereScatteringConfig)
    return [
        (theta = obs.theta_degrees, phi = obs.phi_degrees)
        for obs in rcs_observation_directions(config)
    ]
end

function has_angle_pair(
    rows::Vector{AlmondRCSRow},
    theta::Float64,
    phi::Float64,
    tolerance::Float64,
)
    return any(rows) do row
        abs(row.theta_degrees - theta) <= tolerance &&
            abs(row.phi_degrees - phi) <= tolerance
    end
end

function missing_rcs_angle_pairs(
    rows::Vector{AlmondRCSRow},
    expected_pairs;
    tolerance::Float64,
)
    missing = NamedTuple[]
    for pair in expected_pairs
        has_angle_pair(rows, pair.theta, pair.phi, tolerance) && continue
        push!(missing, pair)
    end
    return missing
end

function almond_rcs_metrics(
    rows::Vector{AlmondRCSRow},
    expected_pairs;
    tolerance::Float64,
)
    missing = missing_rcs_angle_pairs(
        rows,
        expected_pairs;
        tolerance = tolerance,
    )
    finite_row(row) = all(
        isfinite,
        (
            row.abs_E_infinity,
            row.abs_Etheta,
            row.abs_Ephi,
            row.rcs,
            row.rcs_db,
            row.window_sum,
        ),
    )
    return (
        rows = length(rows),
        expected_rows = length(expected_pairs),
        finite_rows = count(finite_row, rows),
        positive_rows = count(row -> row.rcs > 0.0, rows),
        max_rcs = finite_max(row.rcs for row in rows),
        min_rcs = finite_min(row.rcs for row in rows),
        min_samples = minimum(row.samples for row in rows),
        max_samples = maximum(row.samples for row in rows),
        min_window_sum = minimum(row.window_sum for row in rows),
        max_window_sum = maximum(row.window_sum for row in rows),
        missing_angles = missing,
        missing_angle_count = length(missing),
    )
end

function pec_almond_rcs_smoke_record(
    summary,
    config::PECAlmondRCSSmokeGateConfig,
)
    rcs_path = joinpath(summary.output_dir, "diagnostics", "rcs.csv")
    rows = parse_almond_rcs_rows(rcs_path)
    expected_pairs = expected_rcs_angle_pairs(config.scattering_config)
    metrics = almond_rcs_metrics(
        rows,
        expected_pairs;
        tolerance = config.angle_tolerance_degrees,
    )

    row_count_pass = metrics.rows == metrics.expected_rows
    coverage_pass = metrics.missing_angle_count == 0
    finite_pass = metrics.finite_rows == metrics.rows
    positive_pass = metrics.positive_rows == metrics.rows
    sample_pass = metrics.min_samples >= config.min_samples
    window_pass = metrics.min_window_sum > config.min_window_sum
    max_rcs_pass = metrics.max_rcs >= config.min_max_rcs
    passed = row_count_pass &&
             coverage_pass &&
             finite_pass &&
             positive_pass &&
             sample_pass &&
             window_pass &&
             max_rcs_pass
    message = if passed
        "PEC almond RCS smoke gate passed"
    elseif !row_count_pass
        "RCS CSV row count does not match requested angles"
    elseif !coverage_pass
        "RCS CSV does not cover requested angles"
    elseif !finite_pass
        "RCS CSV contains non-finite values"
    elseif !positive_pass
        "RCS CSV contains non-positive RCS values"
    elseif !sample_pass
        "RCS accumulation has too few samples"
    elseif !window_pass
        "RCS accumulation has insufficient window sum"
    else
        "max numerical RCS is below threshold"
    end
    return (
        status = passed ? "PASS" : "FAIL",
        output_dir = summary.output_dir,
        rcs_path = rcs_path,
        final_time = summary.final_time,
        steps = summary.steps,
        dt = summary.dt,
        final_scattered_energy = summary.final_scattered_energy,
        metrics = metrics,
        min_max_rcs = config.min_max_rcs,
        min_samples = config.min_samples,
        min_window_sum = config.min_window_sum,
        angle_tolerance_degrees = config.angle_tolerance_degrees,
        row_count_pass = row_count_pass,
        coverage_pass = coverage_pass,
        finite_pass = finite_pass,
        positive_pass = positive_pass,
        sample_pass = sample_pass,
        window_pass = window_pass,
        max_rcs_pass = max_rcs_pass,
        message = message,
    )
end

function almond_rcs_json_escape(value)
    text = string(value)
    text = replace(text, '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return "\"" * text * "\""
end

function almond_rcs_json_number(value::Real)
    value_float = Float64(value)
    return isfinite(value_float) ? string(value_float) :
           almond_rcs_json_escape(value)
end

function write_pec_almond_rcs_smoke_summary_csv(
    path::AbstractString,
    record,
)
    mkpath(dirname(path))
    metrics = record.metrics
    open(path, "w") do io
        println(
            io,
            "status,output_dir,rcs_path,final_time,steps,dt," *
            "final_scattered_energy,rows,expected_rows,finite_rows," *
            "positive_rows,missing_angle_count,min_rcs,max_rcs," *
            "min_max_rcs,min_samples,max_samples,min_samples_threshold," *
            "min_window_sum,max_window_sum,min_window_sum_threshold," *
            "angle_tolerance_degrees,row_count_pass,coverage_pass," *
            "finite_pass,positive_pass,sample_pass,window_pass," *
            "max_rcs_pass,message",
        )
        println(
            io,
            join(
                (
                    record.status,
                    record.output_dir,
                    record.rcs_path,
                    record.final_time,
                    record.steps,
                    record.dt,
                    record.final_scattered_energy,
                    metrics.rows,
                    metrics.expected_rows,
                    metrics.finite_rows,
                    metrics.positive_rows,
                    metrics.missing_angle_count,
                    metrics.min_rcs,
                    metrics.max_rcs,
                    record.min_max_rcs,
                    metrics.min_samples,
                    metrics.max_samples,
                    record.min_samples,
                    metrics.min_window_sum,
                    metrics.max_window_sum,
                    record.min_window_sum,
                    record.angle_tolerance_degrees,
                    record.row_count_pass,
                    record.coverage_pass,
                    record.finite_pass,
                    record.positive_pass,
                    record.sample_pass,
                    record.window_pass,
                    record.max_rcs_pass,
                    record.message,
                ),
                ',',
            ),
        )
    end
    return String(path)
end

function write_pec_almond_rcs_smoke_summary_json(
    path::AbstractString,
    record,
)
    mkpath(dirname(path))
    metrics = record.metrics
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"disco-gmpi-pec-almond-rcs-smoke/v1\",")
        println(io, "  \"generated_at\": ", almond_rcs_json_escape(Dates.now()), ",")
        println(io, "  \"overall_status\": ", almond_rcs_json_escape(record.status), ",")
        println(io, "  \"output_dir\": ", almond_rcs_json_escape(record.output_dir), ",")
        println(io, "  \"rcs_path\": ", almond_rcs_json_escape(record.rcs_path), ",")
        println(io, "  \"final_time\": ", almond_rcs_json_number(record.final_time), ",")
        println(io, "  \"steps\": ", record.steps, ",")
        println(io, "  \"metrics\": {")
        println(io, "    \"rows\": ", metrics.rows, ",")
        println(io, "    \"expected_rows\": ", metrics.expected_rows, ",")
        println(io, "    \"finite_rows\": ", metrics.finite_rows, ",")
        println(io, "    \"positive_rows\": ", metrics.positive_rows, ",")
        println(io, "    \"missing_angle_count\": ", metrics.missing_angle_count, ",")
        println(io, "    \"min_rcs\": ", almond_rcs_json_number(metrics.min_rcs), ",")
        println(io, "    \"max_rcs\": ", almond_rcs_json_number(metrics.max_rcs), ",")
        println(io, "    \"min_samples\": ", metrics.min_samples, ",")
        println(io, "    \"max_samples\": ", metrics.max_samples, ",")
        println(io, "    \"min_window_sum\": ", almond_rcs_json_number(metrics.min_window_sum), ",")
        println(io, "    \"max_window_sum\": ", almond_rcs_json_number(metrics.max_window_sum))
        println(io, "  },")
        println(io, "  \"thresholds\": {")
        println(io, "    \"min_max_rcs\": ", almond_rcs_json_number(record.min_max_rcs), ",")
        println(io, "    \"min_samples\": ", record.min_samples, ",")
        println(io, "    \"min_window_sum\": ", almond_rcs_json_number(record.min_window_sum), ",")
        println(io, "    \"angle_tolerance_degrees\": ",
                almond_rcs_json_number(record.angle_tolerance_degrees))
        println(io, "  },")
        println(io, "  \"message\": ", almond_rcs_json_escape(record.message))
        println(io, "}")
    end
    return String(path)
end

function print_pec_almond_rcs_smoke_summary(record)
    metrics = record.metrics
    println()
    println("PEC almond RCS smoke validation")
    println("-------------------------------")
    println("status:                    ", record.status)
    println("RCS CSV:                   ", record.rcs_path)
    println(
        "rows / expected:          ",
        metrics.rows,
        " / ",
        metrics.expected_rows,
    )
    println(
        "finite / positive rows:   ",
        metrics.finite_rows,
        " / ",
        metrics.positive_rows,
    )
    println(
        "min/max RCS:              ",
        @sprintf("%.6e", metrics.min_rcs),
        " / ",
        @sprintf("%.6e", metrics.max_rcs),
    )
    println(
        "samples min/max:          ",
        metrics.min_samples,
        " / ",
        metrics.max_samples,
    )
    println(
        "window sum min/max:       ",
        @sprintf("%.6e", metrics.min_window_sum),
        " / ",
        @sprintf("%.6e", metrics.max_window_sum),
    )
    println("missing requested angles: ", metrics.missing_angle_count)
    println("message:                  ", record.message)
    return nothing
end

function pec_almond_rcs_smoke_main(args::Vector{String})
    config = parse_pec_almond_rcs_smoke_gate_config(args)
    config === nothing && return nothing

    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        summary = run_experiment(config.scattering_config, comm)
        record = pec_almond_rcs_smoke_record(summary, config)
        failed = record.status == "FAIL"
        global_failed = MPI.Allreduce(failed ? 1 : 0, max, comm)

        if rank == 0
            csv_path = joinpath(
                config.scattering_config.output_dir,
                "pec_almond_rcs_smoke_summary.csv",
            )
            json_path = joinpath(
                config.scattering_config.output_dir,
                "pec_almond_rcs_smoke_summary.json",
            )
            write_pec_almond_rcs_smoke_summary_csv(csv_path, record)
            write_pec_almond_rcs_smoke_summary_json(json_path, record)
            print_pec_almond_rcs_smoke_summary(record)
            println()
            println("Wrote validation CSV:  ", csv_path)
            println("Wrote validation JSON: ", json_path)
        end
        MPI.Barrier(comm)
        global_failed == 0 ||
            error("PEC almond RCS smoke validation failed.")
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        pec_almond_rcs_smoke_main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    end
end
