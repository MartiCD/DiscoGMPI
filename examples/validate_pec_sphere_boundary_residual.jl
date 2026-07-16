#!/usr/bin/env julia

# Release-gate validation for the incident-aware PEC sphere boundary condition.
# The metric is the sphere-surface L2 norm ||n x E_total||, where
# E_total = E_scat + E_inc.

using Dates
using MPI
using Printf

include(joinpath(@__DIR__, "distributed_metallic_sphere_scattering.jl"))

const DEFAULT_PEC_SPHERE_RESIDUAL_MESH = joinpath(
    @__DIR__,
    "meshes",
    "metallic_sphere_scattering_validation.vtk",
)

const DEFAULT_PEC_SPHERE_RESIDUAL_OUTPUT = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "validation_sequence",
    "pec_sphere_boundary_residual",
)

Base.@kwdef struct PECSphereBoundaryResidualGateConfig
    scattering_config::MetallicSphereScatteringConfig
    max_final_relative_rms::Float64 = 1.0e-1
    max_final_l2::Float64 = Inf
    max_final_relative_max::Float64 = Inf
    max_sampled_relative_rms::Float64 = Inf
end

function pec_sphere_boundary_residual_usage(io::IO = stdout)
    println(io, """
PEC sphere boundary-residual validation gate

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/validate_pec_sphere_boundary_residual.jl [gate options] [scattering options]

Gate options:
  --max-final-relative-rms R   Maximum allowed final RMS residual divided by
                               incident amplitude. Default: 1e-1
  --max-final-l2 R             Maximum allowed final surface L2 residual.
                               Default: Inf
  --max-final-relative-max R   Maximum allowed final pointwise residual divided
                               by incident amplitude. Default: Inf
  --max-sampled-relative-rms R Maximum allowed sampled RMS residual divided by
                               incident amplitude. This includes startup unless
                               diagnostics are delayed externally. Default: Inf
  --help                       Show this message.

Default scattering options:
  --mesh=examples/meshes/metallic_sphere_scattering_validation.vtk
  --disable-rcs --paraview-every=0
  --output-dir=output/validation_sequence/pec_sphere_boundary_residual

All options accepted by distributed_metallic_sphere_scattering.jl can be passed
here and override the defaults above. The gate writes
pec_sphere_boundary_residual_summary.csv and .json in the selected output
directory, and it exits with nonzero status if any enabled threshold fails.
""")
end

function pec_gate_option_value(args::Vector{String}, index::Int, option::String)
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

function default_pec_sphere_boundary_residual_args()
    return [
        "--mesh=$(DEFAULT_PEC_SPHERE_RESIDUAL_MESH)",
        "--disable-rcs",
        "--paraview-every=0",
        "--output-dir=$(DEFAULT_PEC_SPHERE_RESIDUAL_OUTPUT)",
    ]
end

function valid_nonnegative_threshold(value::Float64)
    return !isnan(value) && value >= 0.0
end

function parse_pec_sphere_boundary_residual_gate_config(args::Vector{String})
    max_final_relative_rms = 1.0e-1
    max_final_l2 = Inf
    max_final_relative_max = Inf
    max_sampled_relative_rms = Inf
    scattering_args = String[]

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            pec_sphere_boundary_residual_usage()
            println()
            usage()
            return nothing
        elseif argument == "--max-final-relative-rms" ||
               startswith(argument, "--max-final-relative-rms=")
            raw, index = pec_gate_option_value(
                args,
                index,
                "--max-final-relative-rms",
            )
            max_final_relative_rms = parse(Float64, raw)
        elseif argument == "--max-final-l2" ||
               startswith(argument, "--max-final-l2=")
            raw, index = pec_gate_option_value(args, index, "--max-final-l2")
            max_final_l2 = parse(Float64, raw)
        elseif argument == "--max-final-relative-max" ||
               startswith(argument, "--max-final-relative-max=")
            raw, index = pec_gate_option_value(
                args,
                index,
                "--max-final-relative-max",
            )
            max_final_relative_max = parse(Float64, raw)
        elseif argument == "--max-sampled-relative-rms" ||
               startswith(argument, "--max-sampled-relative-rms=")
            raw, index = pec_gate_option_value(
                args,
                index,
                "--max-sampled-relative-rms",
            )
            max_sampled_relative_rms = parse(Float64, raw)
        else
            push!(scattering_args, argument)
        end
        index += 1
    end

    valid_nonnegative_threshold(max_final_relative_rms) ||
        error("--max-final-relative-rms must be non-negative.")
    valid_nonnegative_threshold(max_final_l2) ||
        error("--max-final-l2 must be non-negative.")
    valid_nonnegative_threshold(max_final_relative_max) ||
        error("--max-final-relative-max must be non-negative.")
    valid_nonnegative_threshold(max_sampled_relative_rms) ||
        error("--max-sampled-relative-rms must be non-negative.")

    scattering_config =
        parse_config(vcat(default_pec_sphere_boundary_residual_args(), scattering_args))
    scattering_config === nothing && return nothing
    return PECSphereBoundaryResidualGateConfig(
        scattering_config = scattering_config,
        max_final_relative_rms = max_final_relative_rms,
        max_final_l2 = max_final_l2,
        max_final_relative_max = max_final_relative_max,
        max_sampled_relative_rms = max_sampled_relative_rms,
    )
end

function pec_residual_json_escape(value)
    text = string(value)
    text = replace(text, '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return "\"" * text * "\""
end

function pec_residual_json_number(value::Real)
    value_float = Float64(value)
    return isfinite(value_float) ? string(value_float) :
           pec_residual_json_escape(value)
end

function pec_sphere_boundary_residual_record(
    summary,
    config::PECSphereBoundaryResidualGateConfig,
)
    finite_metrics = all(
        isfinite,
        (
            summary.final_pec_boundary_residual_l2,
            summary.final_pec_boundary_residual_relative_rms,
            summary.final_pec_boundary_residual_relative_max_pointwise,
            summary.max_sampled_pec_boundary_residual_relative_rms,
        ),
    )
    final_relative_rms_pass =
        finite_metrics &&
        summary.final_pec_boundary_residual_relative_rms <=
        config.max_final_relative_rms
    final_l2_pass =
        finite_metrics &&
        summary.final_pec_boundary_residual_l2 <= config.max_final_l2
    final_relative_max_pass =
        finite_metrics &&
        summary.final_pec_boundary_residual_relative_max_pointwise <=
        config.max_final_relative_max
    sampled_relative_rms_pass =
        finite_metrics &&
        summary.max_sampled_pec_boundary_residual_relative_rms <=
        config.max_sampled_relative_rms
    passed = finite_metrics &&
             final_relative_rms_pass &&
             final_l2_pass &&
             final_relative_max_pass &&
             sampled_relative_rms_pass
    message = if passed
        "PEC sphere boundary residual gate passed"
    elseif !finite_metrics
        "non-finite PEC boundary residual metric"
    elseif !final_relative_rms_pass
        "final relative RMS residual exceeds threshold"
    elseif !final_l2_pass
        "final L2 residual exceeds threshold"
    elseif !final_relative_max_pass
        "final relative pointwise residual exceeds threshold"
    else
        "sampled relative RMS residual exceeds threshold"
    end
    return (
        status = passed ? "PASS" : "FAIL",
        output_dir = summary.output_dir,
        diagnostics_path = summary.diagnostics_path,
        pec_residual_path = summary.pec_residual_path,
        final_time = summary.final_time,
        steps = summary.steps,
        dt = summary.dt,
        final_scattered_energy = summary.final_scattered_energy,
        final_l2 = summary.final_pec_boundary_residual_l2,
        max_final_l2 = config.max_final_l2,
        final_rms = summary.final_pec_boundary_residual_rms,
        final_relative_rms =
            summary.final_pec_boundary_residual_relative_rms,
        max_final_relative_rms = config.max_final_relative_rms,
        final_max_pointwise =
            summary.final_pec_boundary_residual_max_pointwise,
        final_relative_max_pointwise =
            summary.final_pec_boundary_residual_relative_max_pointwise,
        max_final_relative_max = config.max_final_relative_max,
        max_sampled_l2 = summary.max_sampled_pec_boundary_residual_l2,
        max_sampled_relative_rms =
            summary.max_sampled_pec_boundary_residual_relative_rms,
        max_sampled_relative_rms_threshold =
            config.max_sampled_relative_rms,
        max_sampled_pointwise =
            summary.max_sampled_pec_boundary_residual_pointwise,
        finite_metrics = finite_metrics,
        final_relative_rms_pass = final_relative_rms_pass,
        final_l2_pass = final_l2_pass,
        final_relative_max_pass = final_relative_max_pass,
        sampled_relative_rms_pass = sampled_relative_rms_pass,
        message = message,
    )
end

function write_pec_sphere_boundary_residual_csv(
    path::AbstractString,
    record,
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "status,output_dir,diagnostics_path,pec_residual_path," *
            "final_time,steps,dt,final_scattered_energy,final_l2," *
            "max_final_l2,final_rms,final_relative_rms," *
            "max_final_relative_rms,final_max_pointwise," *
            "final_relative_max_pointwise,max_final_relative_max," *
            "max_sampled_l2,max_sampled_relative_rms," *
            "max_sampled_relative_rms_threshold,max_sampled_pointwise," *
            "finite_metrics,final_relative_rms_pass,final_l2_pass," *
            "final_relative_max_pass,sampled_relative_rms_pass,message",
        )
        println(
            io,
            join(
                (
                    record.status,
                    record.output_dir,
                    record.diagnostics_path,
                    record.pec_residual_path,
                    record.final_time,
                    record.steps,
                    record.dt,
                    record.final_scattered_energy,
                    record.final_l2,
                    record.max_final_l2,
                    record.final_rms,
                    record.final_relative_rms,
                    record.max_final_relative_rms,
                    record.final_max_pointwise,
                    record.final_relative_max_pointwise,
                    record.max_final_relative_max,
                    record.max_sampled_l2,
                    record.max_sampled_relative_rms,
                    record.max_sampled_relative_rms_threshold,
                    record.max_sampled_pointwise,
                    record.finite_metrics,
                    record.final_relative_rms_pass,
                    record.final_l2_pass,
                    record.final_relative_max_pass,
                    record.sampled_relative_rms_pass,
                    record.message,
                ),
                ',',
            ),
        )
    end
    return String(path)
end

function write_pec_sphere_boundary_residual_json(
    path::AbstractString,
    record,
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"disco-gmpi-pec-sphere-boundary-residual/v1\",")
        println(io, "  \"generated_at\": ", pec_residual_json_escape(Dates.now()), ",")
        println(io, "  \"overall_status\": ", pec_residual_json_escape(record.status), ",")
        println(io, "  \"output_dir\": ", pec_residual_json_escape(record.output_dir), ",")
        println(io, "  \"pec_residual_path\": ", pec_residual_json_escape(record.pec_residual_path), ",")
        println(io, "  \"final_time\": ", pec_residual_json_number(record.final_time), ",")
        println(io, "  \"steps\": ", record.steps, ",")
        println(io, "  \"metrics\": {")
        println(io, "    \"final_l2\": ", pec_residual_json_number(record.final_l2), ",")
        println(io, "    \"final_rms\": ", pec_residual_json_number(record.final_rms), ",")
        println(io, "    \"final_relative_rms\": ", pec_residual_json_number(record.final_relative_rms), ",")
        println(io, "    \"final_max_pointwise\": ", pec_residual_json_number(record.final_max_pointwise), ",")
        println(io, "    \"final_relative_max_pointwise\": ", pec_residual_json_number(record.final_relative_max_pointwise), ",")
        println(io, "    \"max_sampled_l2\": ", pec_residual_json_number(record.max_sampled_l2), ",")
        println(io, "    \"max_sampled_relative_rms\": ", pec_residual_json_number(record.max_sampled_relative_rms))
        println(io, "  },")
        println(io, "  \"thresholds\": {")
        println(io, "    \"max_final_l2\": ", pec_residual_json_number(record.max_final_l2), ",")
        println(io, "    \"max_final_relative_rms\": ", pec_residual_json_number(record.max_final_relative_rms), ",")
        println(io, "    \"max_final_relative_max\": ", pec_residual_json_number(record.max_final_relative_max), ",")
        println(io, "    \"max_sampled_relative_rms\": ", pec_residual_json_number(record.max_sampled_relative_rms_threshold))
        println(io, "  },")
        println(io, "  \"message\": ", pec_residual_json_escape(record.message))
        println(io, "}")
    end
    return String(path)
end

function print_pec_sphere_boundary_residual_summary(record)
    println()
    println("PEC sphere boundary-residual validation")
    println("---------------------------------------")
    println("status:                 ", record.status)
    println("final ||n x E_total||:  ", @sprintf("%.6e", record.final_l2))
    println("final relative RMS:     ",
            @sprintf("%.6e", record.final_relative_rms),
            " <= ",
            @sprintf("%.6e", record.max_final_relative_rms))
    println("final relative max:     ",
            @sprintf("%.6e", record.final_relative_max_pointwise),
            " <= ",
            @sprintf("%.6e", record.max_final_relative_max))
    println("residual history:       ", record.pec_residual_path)
    println("message:                ", record.message)
    return nothing
end

function pec_sphere_boundary_residual_main(args::Vector{String})
    config = parse_pec_sphere_boundary_residual_gate_config(args)
    config === nothing && return nothing

    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        summary = run_experiment(config.scattering_config, comm)
        record = pec_sphere_boundary_residual_record(summary, config)
        failed = record.status == "FAIL"
        global_failed = MPI.Allreduce(failed ? 1 : 0, max, comm)

        if rank == 0
            csv_path = joinpath(
                config.scattering_config.output_dir,
                "pec_sphere_boundary_residual_summary.csv",
            )
            json_path = joinpath(
                config.scattering_config.output_dir,
                "pec_sphere_boundary_residual_summary.json",
            )
            write_pec_sphere_boundary_residual_csv(csv_path, record)
            write_pec_sphere_boundary_residual_json(json_path, record)
            print_pec_sphere_boundary_residual_summary(record)
            println()
            println("Wrote validation CSV:  ", csv_path)
            println("Wrote validation JSON: ", json_path)
        end
        MPI.Barrier(comm)
        global_failed == 0 ||
            error("PEC sphere boundary-residual validation failed.")
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        pec_sphere_boundary_residual_main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    end
end
