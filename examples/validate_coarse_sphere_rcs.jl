#!/usr/bin/env julia

# Coarse-mesh RCS validation gate for the metallic PEC sphere. The gate runs
# the scattering driver with RCS enabled, checks that a nonzero finite RCS table
# was produced, and writes a coarse comparison against the exact PEC Mie series.

using Dates
using MPI
using Printf

include(joinpath(@__DIR__, "distributed_metallic_sphere_scattering.jl"))

const DEFAULT_COARSE_RCS_MESH = joinpath(
    @__DIR__,
    "meshes",
    "metallic_sphere_scattering_validation.vtk",
)

const DEFAULT_COARSE_RCS_OUTPUT = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "validation_sequence",
    "coarse_sphere_rcs",
)

Base.@kwdef struct CoarseSphereRCSGateConfig
    scattering_config::MetallicSphereScatteringConfig
    min_max_rcs::Float64 = 1.0e-12
    max_absolute_normalized_error::Float64 = Inf
    max_rms_relative_error::Float64 = Inf
    max_db_error::Float64 = Inf
    mie_terms::Int = 0
end

function coarse_sphere_rcs_usage(io::IO = stdout)
    println(io, """
Coarse-mesh PEC sphere RCS validation gate

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/validate_coarse_sphere_rcs.jl [gate options] [scattering options]

Gate options:
  --min-max-rcs V              Require max numerical RCS >= V. Default: 1e-12
  --max-absolute-normalized-error V
                               Optional max |sigma_h-sigma_Mie|/(pi a^2).
                               Default: Inf
  --max-rms-relative-error V   Optional RMS relative-error threshold.
                               Default: Inf
  --max-db-error V             Optional max dB-error threshold. Default: Inf
  --mie-terms N                Mie-series terms. Default: automatic
  --help                       Show this message.

Default scattering options:
  --mesh=examples/meshes/metallic_sphere_scattering_validation.vtk
  --order=1 --esprk-order=2
  --final-time=1.0
  --enable-rcs --rcs-start-time=0.0 --rcs-every=5
  --rcs-theta-count=37 --rcs-phi-degrees=0
  --paraview-every=0
  --output-dir=output/validation_sequence/coarse_sphere_rcs

This is a coarse release-gate diagnostic. It verifies that the RCS path runs,
produces finite nonzero values, and records loose Mie-comparison metrics. Use
the optional error thresholds above to make the gate stricter.
""")
end

function coarse_rcs_option_value(
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

function default_coarse_sphere_rcs_args()
    return [
        "--mesh=$(DEFAULT_COARSE_RCS_MESH)",
        "--order=1",
        "--esprk-order=2",
        "--final-time=1.0",
        "--enable-rcs",
        "--rcs-start-time=0.0",
        "--rcs-every=5",
        "--rcs-theta-count=37",
        "--rcs-phi-degrees=0",
        "--paraview-every=0",
        "--output-dir=$(DEFAULT_COARSE_RCS_OUTPUT)",
    ]
end

function valid_nonnegative_threshold(value::Float64)
    return !isnan(value) && value >= 0.0
end

function parse_coarse_sphere_rcs_gate_config(args::Vector{String})
    min_max_rcs = 1.0e-12
    max_absolute_normalized_error = Inf
    max_rms_relative_error = Inf
    max_db_error = Inf
    mie_terms = 0
    scattering_args = String[]

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            coarse_sphere_rcs_usage()
            println()
            usage()
            return nothing
        elseif argument == "--min-max-rcs" ||
               startswith(argument, "--min-max-rcs=")
            raw, index = coarse_rcs_option_value(args, index, "--min-max-rcs")
            min_max_rcs = parse(Float64, raw)
        elseif argument == "--max-absolute-normalized-error" ||
               startswith(argument, "--max-absolute-normalized-error=")
            raw, index = coarse_rcs_option_value(
                args,
                index,
                "--max-absolute-normalized-error",
            )
            max_absolute_normalized_error = parse(Float64, raw)
        elseif argument == "--max-rms-relative-error" ||
               startswith(argument, "--max-rms-relative-error=")
            raw, index = coarse_rcs_option_value(
                args,
                index,
                "--max-rms-relative-error",
            )
            max_rms_relative_error = parse(Float64, raw)
        elseif argument == "--max-db-error" ||
               startswith(argument, "--max-db-error=")
            raw, index = coarse_rcs_option_value(args, index, "--max-db-error")
            max_db_error = parse(Float64, raw)
        elseif argument == "--mie-terms" ||
               startswith(argument, "--mie-terms=")
            raw, index = coarse_rcs_option_value(args, index, "--mie-terms")
            mie_terms = parse(Int, raw)
        else
            push!(scattering_args, argument)
        end
        index += 1
    end

    valid_nonnegative_threshold(min_max_rcs) ||
        error("--min-max-rcs must be non-negative.")
    valid_nonnegative_threshold(max_absolute_normalized_error) ||
        error("--max-absolute-normalized-error must be non-negative.")
    valid_nonnegative_threshold(max_rms_relative_error) ||
        error("--max-rms-relative-error must be non-negative.")
    valid_nonnegative_threshold(max_db_error) ||
        error("--max-db-error must be non-negative.")
    mie_terms >= 0 || error("--mie-terms must be non-negative.")

    scattering_config =
        parse_config(vcat(default_coarse_sphere_rcs_args(), scattering_args))
    scattering_config === nothing && return nothing
    scattering_config.rcs_enabled ||
        error("Coarse RCS validation requires RCS enabled.")
    return CoarseSphereRCSGateConfig(
        scattering_config = scattering_config,
        min_max_rcs = min_max_rcs,
        max_absolute_normalized_error = max_absolute_normalized_error,
        max_rms_relative_error = max_rms_relative_error,
        max_db_error = max_db_error,
        mie_terms = mie_terms,
    )
end

struct CoarseRCSRow
    theta_degrees::Float64
    phi_degrees::Float64
    numerical_rcs::Float64
    numerical_rcs_db::Float64
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

function parse_rcs_rows(path::AbstractString)
    rows = CoarseRCSRow[]
    for row in read_simple_csv(path)
        push!(
            rows,
            CoarseRCSRow(
                parse(Float64, row["theta_degrees"]),
                parse(Float64, row["phi_degrees"]),
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

automatic_mie_terms(size_parameter::Float64) =
    max(8, ceil(Int, size_parameter + 4.0 * cbrt(size_parameter) + 2.0))

# Lightweight Julia Mie evaluator used by the release gate. The Python
# postprocessor is the preferred plotting path, but keeping this local avoids a
# SciPy dependency for the MPI validation summary.
function spherical_bessel_jy(size_parameter::Float64, terms::Int)
    x = size_parameter
    x > 0.0 || error("Mie size parameter must be positive.")
    j = zeros(Float64, terms + 1)
    y = zeros(Float64, terms + 1)
    j[1] = sin(x) / x
    y[1] = -cos(x) / x
    if terms >= 1
        j[2] = sin(x) / x^2 - cos(x) / x
        y[2] = -cos(x) / x^2 - sin(x) / x
    end
    for n in 1:(terms - 1)
        j[n + 2] = ((2.0 * n + 1.0) / x) * j[n + 1] - j[n]
        y[n + 2] = ((2.0 * n + 1.0) / x) * y[n + 1] - y[n]
    end
    return j, y
end

function pec_mie_coefficients(size_parameter::Float64, terms::Int)
    j, y = spherical_bessel_jy(size_parameter, terms)
    electric = Vector{ComplexF64}(undef, terms)
    magnetic = Vector{ComplexF64}(undef, terms)
    x = size_parameter
    for n in 1:terms
        jn = j[n + 1]
        yn = y[n + 1]
        jn_prime = j[n] - ((n + 1.0) / x) * jn
        yn_prime = y[n] - ((n + 1.0) / x) * yn
        hn = complex(jn, yn)
        hn_prime = complex(jn_prime, yn_prime)
        psi = x * jn
        psi_prime = jn + x * jn_prime
        xi = x * hn
        xi_prime = hn + x * hn_prime
        electric[n] = -psi_prime / xi_prime
        magnetic[n] = -psi / xi
    end
    return electric, magnetic
end

function angular_functions(mu::Float64, terms::Int)
    pi_values = Vector{Float64}(undef, terms)
    tau_values = Vector{Float64}(undef, terms)
    terms <= 0 && return pi_values, tau_values

    # Standard Mie angular functions:
    # pi_1 = 1, pi_2 = 3*mu,
    # pi_n = ((2n - 1)/(n - 1))*mu*pi_{n-1} - (n/(n - 1))*pi_{n-2}.
    pi_values[1] = 1.0
    if terms > 1
        pi_values[2] = 3.0 * mu
        for n in 3:terms
            pi_values[n] =
                ((2.0 * n - 1.0) / (n - 1.0)) * mu * pi_values[n - 1] -
                (n / (n - 1.0)) * pi_values[n - 2]
        end
    end

    for n in 1:terms
        pi_previous = n == 1 ? 0.0 : pi_values[n - 1]
        tau_values[n] = n * mu * pi_values[n] - (n + 1.0) * pi_previous
    end
    return pi_values, tau_values
end

function mie_amplitudes(
    theta_degrees::Float64,
    electric::Vector{ComplexF64},
    magnetic::Vector{ComplexF64},
)
    terms = length(electric)
    theta = deg2rad(theta_degrees)
    mu = cos(theta)
    pi_values, tau_values = angular_functions(mu, terms)
    s1 = 0.0 + 0.0im
    s2 = 0.0 + 0.0im
    for n in 1:terms
        weight = (2.0 * n + 1.0) / (n * (n + 1.0))
        s1 += weight * (electric[n] * pi_values[n] + magnetic[n] * tau_values[n])
        s2 += weight * (electric[n] * tau_values[n] + magnetic[n] * pi_values[n])
    end
    return s1, s2
end

function exact_pec_sphere_rcs(
    theta_degrees::Float64,
    phi_degrees::Float64,
    wavenumber::Float64,
    electric::Vector{ComplexF64},
    magnetic::Vector{ComplexF64},
)
    s1, s2 = mie_amplitudes(theta_degrees, electric, magnetic)
    phi = deg2rad(phi_degrees)
    e_theta = s2 * cos(phi) / wavenumber
    e_phi = -s1 * sin(phi) / wavenumber
    abs_e_theta = abs(e_theta)
    abs_e_phi = abs(e_phi)
    abs_e = sqrt(abs2(abs_e_theta) + abs2(abs_e_phi))
    return 4.0 * pi * abs2(abs_e)
end

function db(value::Float64)
    return value > 0.0 ? 10.0 * log10(value) : -Inf
end

function compare_rcs_rows(
    rows::Vector{CoarseRCSRow};
    radius::Float64,
    wavelength::Float64,
    terms::Int,
)
    wavenumber = 2.0 * pi / wavelength
    size_parameter = wavenumber * radius
    resolved_terms =
        terms > 0 ? terms : automatic_mie_terms(size_parameter)
    electric, magnetic = pec_mie_coefficients(size_parameter, resolved_terms)
    normalization = pi * radius^2
    comparison = NamedTuple[]
    for row in rows
        mie_rcs = exact_pec_sphere_rcs(
            row.theta_degrees,
            row.phi_degrees,
            wavenumber,
            electric,
            magnetic,
        )
        numerical = row.numerical_rcs
        absolute_error = abs(numerical - mie_rcs)
        relative_error = absolute_error / max(abs(mie_rcs), eps(Float64))
        numerical_db = isfinite(row.numerical_rcs_db) ? row.numerical_rcs_db :
                       db(numerical)
        mie_db = db(mie_rcs)
        push!(
            comparison,
            (
                theta_degrees = row.theta_degrees,
                phi_degrees = row.phi_degrees,
                rcs_numeric = numerical,
                rcs_mie = mie_rcs,
                absolute_error = absolute_error,
                relative_error = relative_error,
                rcs_numeric_db = numerical_db,
                rcs_mie_db = mie_db,
                db_error = abs(numerical_db - mie_db),
                rcs_numeric_over_pi_a2 = numerical / normalization,
                rcs_mie_over_pi_a2 = mie_rcs / normalization,
                absolute_error_over_pi_a2 = absolute_error / normalization,
                samples = row.samples,
                window_sum = row.window_sum,
                mie_terms = resolved_terms,
                size_parameter = size_parameter,
            ),
        )
    end
    return comparison
end

function finite_max(values)
    finite_values = filter(isfinite, collect(values))
    isempty(finite_values) && return Inf
    return maximum(finite_values)
end

function finite_rms(values)
    finite_values = filter(isfinite, collect(values))
    isempty(finite_values) && return Inf
    return sqrt(sum(abs2, finite_values) / length(finite_values))
end

function coarse_rcs_metrics(rows, comparison)
    numerical_values = [row.numerical_rcs for row in rows]
    samples = [row.samples for row in rows]
    window_sums = [row.window_sum for row in rows]
    return (
        rows = length(rows),
        finite_rows = count(row -> isfinite(row.numerical_rcs), rows),
        positive_rows = count(row -> row.numerical_rcs > 0.0, rows),
        max_numerical_rcs = finite_max(numerical_values),
        min_samples = minimum(samples),
        max_samples = maximum(samples),
        min_window_sum = minimum(window_sums),
        max_window_sum = maximum(window_sums),
        max_absolute_normalized_error =
            finite_max(row.absolute_error_over_pi_a2 for row in comparison),
        max_relative_error =
            finite_max(row.relative_error for row in comparison),
        rms_relative_error =
            finite_rms(row.relative_error for row in comparison),
        max_db_error = finite_max(row.db_error for row in comparison),
        mie_terms = isempty(comparison) ? 0 : comparison[1].mie_terms,
        size_parameter = isempty(comparison) ? NaN : comparison[1].size_parameter,
    )
end

function write_coarse_rcs_comparison_csv(path::AbstractString, comparison)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "theta_degrees,phi_degrees,rcs_numeric,rcs_mie," *
            "absolute_error,relative_error,rcs_numeric_db,rcs_mie_db," *
            "db_error,rcs_numeric_over_pi_a2,rcs_mie_over_pi_a2," *
            "absolute_error_over_pi_a2,samples,window_sum",
        )
        for row in comparison
            println(
                io,
                join(
                    (
                        @sprintf("%.16e", row.theta_degrees),
                        @sprintf("%.16e", row.phi_degrees),
                        @sprintf("%.16e", row.rcs_numeric),
                        @sprintf("%.16e", row.rcs_mie),
                        @sprintf("%.16e", row.absolute_error),
                        @sprintf("%.16e", row.relative_error),
                        @sprintf("%.16e", row.rcs_numeric_db),
                        @sprintf("%.16e", row.rcs_mie_db),
                        @sprintf("%.16e", row.db_error),
                        @sprintf("%.16e", row.rcs_numeric_over_pi_a2),
                        @sprintf("%.16e", row.rcs_mie_over_pi_a2),
                        @sprintf("%.16e", row.absolute_error_over_pi_a2),
                        row.samples,
                        @sprintf("%.16e", row.window_sum),
                    ),
                    ',',
                ),
            )
        end
    end
    return String(path)
end

function coarse_rcs_json_escape(value)
    text = string(value)
    text = replace(text, '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return "\"" * text * "\""
end

function coarse_rcs_json_number(value::Real)
    value_float = Float64(value)
    return isfinite(value_float) ? string(value_float) :
           coarse_rcs_json_escape(value)
end

function coarse_sphere_rcs_record(
    summary,
    config::CoarseSphereRCSGateConfig,
)
    # The default gate is a pipeline check: finite rows, positive samples, and
    # nonzero RCS. Strict Mie-error thresholds are opt-in because a coarse or
    # one-step run is not a quantitative RCS validation.
    rcs_path = joinpath(summary.output_dir, "diagnostics", "rcs.csv")
    rows = parse_rcs_rows(rcs_path)
    comparison = compare_rcs_rows(
        rows;
        radius = config.scattering_config.radius,
        wavelength = resolved_wave_values(config.scattering_config).wavelength,
        terms = config.mie_terms,
    )
    metrics = coarse_rcs_metrics(rows, comparison)

    finite_pass = metrics.finite_rows == metrics.rows
    sample_pass = metrics.min_samples > 0 && metrics.min_window_sum > 0.0
    nonzero_pass = metrics.max_numerical_rcs >= config.min_max_rcs
    normalized_error_pass =
        metrics.max_absolute_normalized_error <=
        config.max_absolute_normalized_error
    relative_error_pass =
        metrics.rms_relative_error <= config.max_rms_relative_error
    db_error_pass = metrics.max_db_error <= config.max_db_error
    passed = finite_pass &&
             sample_pass &&
             nonzero_pass &&
             normalized_error_pass &&
             relative_error_pass &&
             db_error_pass
    message = if passed
        "coarse sphere RCS gate passed"
    elseif !finite_pass
        "RCS CSV contains non-finite values"
    elseif !sample_pass
        "RCS accumulation has no positive samples or window sum"
    elseif !nonzero_pass
        "max numerical RCS is below threshold"
    elseif !normalized_error_pass
        "normalized absolute RCS error exceeds threshold"
    elseif !relative_error_pass
        "RMS relative RCS error exceeds threshold"
    else
        "RCS dB error exceeds threshold"
    end
    comparison_path = joinpath(
        summary.output_dir,
        "diagnostics",
        "coarse_mie_rcs_comparison.csv",
    )
    return (
        status = passed ? "PASS" : "FAIL",
        output_dir = summary.output_dir,
        rcs_path = rcs_path,
        comparison_path = comparison_path,
        final_time = summary.final_time,
        steps = summary.steps,
        dt = summary.dt,
        final_scattered_energy = summary.final_scattered_energy,
        metrics = metrics,
        min_max_rcs = config.min_max_rcs,
        max_absolute_normalized_error_threshold =
            config.max_absolute_normalized_error,
        max_rms_relative_error_threshold = config.max_rms_relative_error,
        max_db_error_threshold = config.max_db_error,
        finite_pass = finite_pass,
        sample_pass = sample_pass,
        nonzero_pass = nonzero_pass,
        normalized_error_pass = normalized_error_pass,
        relative_error_pass = relative_error_pass,
        db_error_pass = db_error_pass,
        comparison = comparison,
        message = message,
    )
end

function write_coarse_sphere_rcs_summary_csv(path::AbstractString, record)
    mkpath(dirname(path))
    metrics = record.metrics
    open(path, "w") do io
        println(
            io,
            "status,output_dir,rcs_path,comparison_path,final_time,steps,dt," *
            "final_scattered_energy,rows,finite_rows,positive_rows," *
            "max_numerical_rcs,min_max_rcs,min_samples,max_samples," *
            "min_window_sum,max_window_sum,mie_terms,size_parameter," *
            "max_absolute_normalized_error," *
            "max_absolute_normalized_error_threshold," *
            "max_relative_error,rms_relative_error," *
            "max_rms_relative_error_threshold,max_db_error," *
            "max_db_error_threshold,finite_pass,sample_pass,nonzero_pass," *
            "normalized_error_pass,relative_error_pass,db_error_pass,message",
        )
        println(
            io,
            join(
                (
                    record.status,
                    record.output_dir,
                    record.rcs_path,
                    record.comparison_path,
                    record.final_time,
                    record.steps,
                    record.dt,
                    record.final_scattered_energy,
                    metrics.rows,
                    metrics.finite_rows,
                    metrics.positive_rows,
                    metrics.max_numerical_rcs,
                    record.min_max_rcs,
                    metrics.min_samples,
                    metrics.max_samples,
                    metrics.min_window_sum,
                    metrics.max_window_sum,
                    metrics.mie_terms,
                    metrics.size_parameter,
                    metrics.max_absolute_normalized_error,
                    record.max_absolute_normalized_error_threshold,
                    metrics.max_relative_error,
                    metrics.rms_relative_error,
                    record.max_rms_relative_error_threshold,
                    metrics.max_db_error,
                    record.max_db_error_threshold,
                    record.finite_pass,
                    record.sample_pass,
                    record.nonzero_pass,
                    record.normalized_error_pass,
                    record.relative_error_pass,
                    record.db_error_pass,
                    record.message,
                ),
                ',',
            ),
        )
    end
    return String(path)
end

function write_coarse_sphere_rcs_summary_json(path::AbstractString, record)
    mkpath(dirname(path))
    metrics = record.metrics
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"disco-gmpi-coarse-sphere-rcs/v1\",")
        println(io, "  \"generated_at\": ", coarse_rcs_json_escape(Dates.now()), ",")
        println(io, "  \"overall_status\": ", coarse_rcs_json_escape(record.status), ",")
        println(io, "  \"output_dir\": ", coarse_rcs_json_escape(record.output_dir), ",")
        println(io, "  \"rcs_path\": ", coarse_rcs_json_escape(record.rcs_path), ",")
        println(io, "  \"comparison_path\": ", coarse_rcs_json_escape(record.comparison_path), ",")
        println(io, "  \"final_time\": ", coarse_rcs_json_number(record.final_time), ",")
        println(io, "  \"steps\": ", record.steps, ",")
        println(io, "  \"metrics\": {")
        println(io, "    \"rows\": ", metrics.rows, ",")
        println(io, "    \"finite_rows\": ", metrics.finite_rows, ",")
        println(io, "    \"positive_rows\": ", metrics.positive_rows, ",")
        println(io, "    \"max_numerical_rcs\": ", coarse_rcs_json_number(metrics.max_numerical_rcs), ",")
        println(io, "    \"min_samples\": ", metrics.min_samples, ",")
        println(io, "    \"min_window_sum\": ", coarse_rcs_json_number(metrics.min_window_sum), ",")
        println(io, "    \"mie_terms\": ", metrics.mie_terms, ",")
        println(io, "    \"size_parameter\": ", coarse_rcs_json_number(metrics.size_parameter), ",")
        println(io, "    \"max_absolute_normalized_error\": ",
                coarse_rcs_json_number(metrics.max_absolute_normalized_error), ",")
        println(io, "    \"rms_relative_error\": ",
                coarse_rcs_json_number(metrics.rms_relative_error), ",")
        println(io, "    \"max_relative_error\": ",
                coarse_rcs_json_number(metrics.max_relative_error), ",")
        println(io, "    \"max_db_error\": ",
                coarse_rcs_json_number(metrics.max_db_error))
        println(io, "  },")
        println(io, "  \"thresholds\": {")
        println(io, "    \"min_max_rcs\": ", coarse_rcs_json_number(record.min_max_rcs), ",")
        println(io, "    \"max_absolute_normalized_error\": ",
                coarse_rcs_json_number(record.max_absolute_normalized_error_threshold), ",")
        println(io, "    \"max_rms_relative_error\": ",
                coarse_rcs_json_number(record.max_rms_relative_error_threshold), ",")
        println(io, "    \"max_db_error\": ",
                coarse_rcs_json_number(record.max_db_error_threshold))
        println(io, "  },")
        println(io, "  \"message\": ", coarse_rcs_json_escape(record.message))
        println(io, "}")
    end
    return String(path)
end

function print_coarse_sphere_rcs_summary(record)
    metrics = record.metrics
    println()
    println("Coarse-mesh PEC sphere RCS validation")
    println("-------------------------------------")
    println("status:                    ", record.status)
    println("RCS CSV:                   ", record.rcs_path)
    println("Mie comparison CSV:        ", record.comparison_path)
    println("rows / finite / positive:  ",
            metrics.rows, " / ", metrics.finite_rows, " / ", metrics.positive_rows)
    println("max numerical RCS:         ",
            @sprintf("%.6e", metrics.max_numerical_rcs),
            " >= ",
            @sprintf("%.6e", record.min_max_rcs))
    println("max |error|/(pi a^2):      ",
            @sprintf("%.6e", metrics.max_absolute_normalized_error))
    println("RMS relative error:        ",
            @sprintf("%.6e", metrics.rms_relative_error))
    println("max dB error:              ",
            @sprintf("%.6e", metrics.max_db_error))
    println("message:                   ", record.message)
    return nothing
end

function coarse_sphere_rcs_main(args::Vector{String})
    config = parse_coarse_sphere_rcs_gate_config(args)
    config === nothing && return nothing

    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        summary = run_experiment(config.scattering_config, comm)
        record = coarse_sphere_rcs_record(summary, config)
        failed = record.status == "FAIL"
        global_failed = MPI.Allreduce(failed ? 1 : 0, max, comm)

        if rank == 0
            write_coarse_rcs_comparison_csv(
                record.comparison_path,
                record.comparison,
            )
            csv_path = joinpath(
                config.scattering_config.output_dir,
                "coarse_sphere_rcs_summary.csv",
            )
            json_path = joinpath(
                config.scattering_config.output_dir,
                "coarse_sphere_rcs_summary.json",
            )
            write_coarse_sphere_rcs_summary_csv(csv_path, record)
            write_coarse_sphere_rcs_summary_json(json_path, record)
            print_coarse_sphere_rcs_summary(record)
            println()
            println("Wrote validation CSV:  ", csv_path)
            println("Wrote validation JSON: ", json_path)
        end
        MPI.Barrier(comm)
        global_failed == 0 ||
            error("Coarse-mesh PEC sphere RCS validation failed.")
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        coarse_sphere_rcs_main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    end
end
