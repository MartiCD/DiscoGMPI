#!/usr/bin/env julia

# Release-gate validation for an empty-domain incident wave absorbed by the
# nonlinear Poisson-bracket/ESPRK PML path. This script intentionally reuses
# distributed_poisson_bracket_pml_validation.jl so the validation gate and the
# production sweep exercise the same solver implementation.

using Dates
using MPI
using Printf

include(joinpath(@__DIR__, "distributed_poisson_bracket_pml_validation.jl"))

const DEFAULT_EMPTY_DOMAIN_INCIDENT_PML_MESH = joinpath(
    @__DIR__,
    "meshes",
    "periodic_box_structured_nx16_ny8_nz8.vtk",
)

const DEFAULT_EMPTY_DOMAIN_INCIDENT_PML_OUTPUT = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "validation_sequence",
    "empty_domain_incident_pml",
)

Base.@kwdef struct EmptyDomainIncidentPMLGateConfig
    pml_config::PMLValidationConfig
    max_reflection_ratio::Float64 = 1.0e-2
    max_final_energy_ratio::Float64 = 2.5e-1
    max_energy_growth::Float64 = 1.0e-10
end

function empty_domain_incident_pml_usage(io::IO = stdout)
    println(io, """
Empty-domain incident-wave nonlinear PML validation gate

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/validate_empty_domain_incident_pml.jl [gate options] [PML options]

Gate options:
  --max-reflection-ratio R     Maximum allowed reflected monitor energy ratio.
                               Default: 1e-2
  --max-final-energy-ratio R   Maximum allowed final total energy divided by
                               initial total energy. Default: 2.5e-1
  --max-energy-growth R        Allowed numerical energy growth above the
                               initial energy ratio. Default: 1e-10
  --help                       Show this message.

Default PML validation options:
  --mesh=examples/meshes/periodic_box_structured_nx16_ny8_nz8.vtk
  --order=2 --esprk-order=3 --final-time=3.0 --cfl=0.15
  --central-wavelength=0.5 --flux=centered
  --pml-width=0.5 --sigma-max=12 --sigma-degree=2
  --energy-every=10 --paraview-every=50
  --output-dir=output/validation_sequence/empty_domain_incident_pml

Any option accepted by distributed_poisson_bracket_pml_validation.jl can be
passed here and overrides the defaults above. The validation launches a
right-going sine-modulated Gaussian pulse in an otherwise empty periodic
y/z cuboid and measures the reflected energy in the left monitor after the
pulse reaches the right PML.
""")
end

function empty_gate_option_value(
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

function default_empty_domain_incident_pml_args()
    return [
        "--mesh=$(DEFAULT_EMPTY_DOMAIN_INCIDENT_PML_MESH)",
        "--order=2",
        "--esprk-order=3",
        "--final-time=3.0",
        "--cfl=0.15",
        "--central-wavelength=0.5",
        "--flux=centered",
        "--pml-width=0.5",
        "--sigma-max=12.0",
        "--sigma-degree=2",
        "--energy-every=10",
        "--paraview-every=50",
        "--output-dir=$(DEFAULT_EMPTY_DOMAIN_INCIDENT_PML_OUTPUT)",
    ]
end

function parse_empty_domain_incident_pml_config(args::Vector{String})
    max_reflection_ratio = 1.0e-2
    max_final_energy_ratio = 2.5e-1
    max_energy_growth = 1.0e-10
    pml_args = String[]

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            empty_domain_incident_pml_usage()
            println()
            pml_validation_usage()
            return nothing
        elseif argument == "--max-reflection-ratio" ||
               startswith(argument, "--max-reflection-ratio=")
            raw, index = empty_gate_option_value(
                args,
                index,
                "--max-reflection-ratio",
            )
            max_reflection_ratio = parse(Float64, raw)
        elseif argument == "--max-final-energy-ratio" ||
               startswith(argument, "--max-final-energy-ratio=")
            raw, index = empty_gate_option_value(
                args,
                index,
                "--max-final-energy-ratio",
            )
            max_final_energy_ratio = parse(Float64, raw)
        elseif argument == "--max-energy-growth" ||
               startswith(argument, "--max-energy-growth=")
            raw, index = empty_gate_option_value(
                args,
                index,
                "--max-energy-growth",
            )
            max_energy_growth = parse(Float64, raw)
        else
            push!(pml_args, argument)
        end
        index += 1
    end

    isfinite(max_reflection_ratio) && max_reflection_ratio >= 0.0 ||
        error("--max-reflection-ratio must be finite and non-negative.")
    isfinite(max_final_energy_ratio) && max_final_energy_ratio >= 0.0 ||
        error("--max-final-energy-ratio must be finite and non-negative.")
    isfinite(max_energy_growth) && max_energy_growth >= 0.0 ||
        error("--max-energy-growth must be finite and non-negative.")

    pml_config = parse_config(vcat(default_empty_domain_incident_pml_args(), pml_args))
    pml_config === nothing && return nothing
    return EmptyDomainIncidentPMLGateConfig(
        pml_config = pml_config,
        max_reflection_ratio = max_reflection_ratio,
        max_final_energy_ratio = max_final_energy_ratio,
        max_energy_growth = max_energy_growth,
    )
end

function empty_domain_json_escape(value)
    text = string(value)
    text = replace(text, '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return "\"" * text * "\""
end

function empty_domain_incident_pml_records(
    summaries,
    config::EmptyDomainIncidentPMLGateConfig,
)
    records = NamedTuple[]
    for summary in summaries
        finite_metrics = all(
            isfinite,
            (
                summary.reflection_ratio,
                summary.final_energy_ratio,
                summary.initial_total_energy,
                summary.final_total_energy,
                summary.max_left_monitor_after_reflection,
            ),
        )
        initial_positive = finite_metrics && summary.initial_total_energy > 0.0
        reflection_pass =
            finite_metrics &&
            summary.reflection_ratio <= config.max_reflection_ratio
        final_energy_pass =
            finite_metrics &&
            summary.final_energy_ratio <= config.max_final_energy_ratio
        no_growth_pass =
            finite_metrics &&
            summary.final_energy_ratio <= 1.0 + config.max_energy_growth
        passed = finite_metrics &&
                 initial_positive &&
                 reflection_pass &&
                 final_energy_pass &&
                 no_growth_pass
        message = if passed
            "empty-domain incident-wave PML gate passed"
        elseif !finite_metrics
            "non-finite reflection or energy metric"
        elseif !initial_positive
            "initial total energy is not positive"
        elseif !reflection_pass
            "reflection ratio exceeds threshold"
        elseif !final_energy_pass
            "final energy ratio exceeds threshold"
        else
            "total energy grew beyond tolerance"
        end
        push!(
            records,
            (
                case = summary.case,
                status = passed ? "PASS" : "FAIL",
                mesh = summary.mesh,
                mesh_path = summary.mesh_path,
                flux = summary.flux,
                pml_width = summary.pml_width,
                sigma_max = summary.sigma_max,
                sigma_degree = summary.sigma_degree,
                central_frequency = summary.central_frequency,
                central_wavelength = summary.central_wavelength,
                central_wavenumber = summary.central_wavenumber,
                central_elements_per_wavelength =
                    summary.central_elements_per_wavelength,
                reflection_ratio = summary.reflection_ratio,
                max_reflection_ratio = config.max_reflection_ratio,
                final_energy_ratio = summary.final_energy_ratio,
                max_final_energy_ratio = config.max_final_energy_ratio,
                max_energy_growth = config.max_energy_growth,
                max_left_monitor_after_reflection =
                    summary.max_left_monitor_after_reflection,
                initial_total_energy = summary.initial_total_energy,
                final_total_energy = summary.final_total_energy,
                dt = summary.dt,
                steps = summary.steps,
                output_dir = summary.output_dir,
                finite_metrics = finite_metrics,
                initial_positive = initial_positive,
                reflection_pass = reflection_pass,
                final_energy_pass = final_energy_pass,
                no_growth_pass = no_growth_pass,
                message = message,
            ),
        )
    end
    return records
end

function write_empty_domain_incident_pml_csv(path::AbstractString, records)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "case,status,mesh,mesh_path,flux,pml_width,sigma_max," *
            "sigma_degree,central_frequency,central_wavelength," *
            "central_wavenumber,central_elements_per_wavelength," *
            "reflection_ratio,max_reflection_ratio,final_energy_ratio," *
            "max_final_energy_ratio,max_energy_growth," *
            "max_left_monitor_after_reflection,initial_total_energy," *
            "final_total_energy,dt,steps,output_dir,finite_metrics," *
            "initial_positive,reflection_pass,final_energy_pass," *
            "no_growth_pass,message",
        )
        for record in records
            println(
                io,
                join(
                    (
                        record.case,
                        record.status,
                        record.mesh,
                        record.mesh_path,
                        record.flux,
                        record.pml_width,
                        record.sigma_max,
                        record.sigma_degree,
                        record.central_frequency,
                        record.central_wavelength,
                        record.central_wavenumber,
                        record.central_elements_per_wavelength,
                        record.reflection_ratio,
                        record.max_reflection_ratio,
                        record.final_energy_ratio,
                        record.max_final_energy_ratio,
                        record.max_energy_growth,
                        record.max_left_monitor_after_reflection,
                        record.initial_total_energy,
                        record.final_total_energy,
                        record.dt,
                        record.steps,
                        record.output_dir,
                        record.finite_metrics,
                        record.initial_positive,
                        record.reflection_pass,
                        record.final_energy_pass,
                        record.no_growth_pass,
                        record.message,
                    ),
                    ',',
                ),
            )
        end
    end
    return String(path)
end

function write_empty_domain_incident_pml_json(path::AbstractString, records)
    mkpath(dirname(path))
    overall_status =
        any(record -> record.status == "FAIL", records) ? "FAIL" : "PASS"
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"disco-gmpi-empty-domain-incident-pml/v1\",")
        println(io, "  \"generated_at\": ", empty_domain_json_escape(Dates.now()), ",")
        println(io, "  \"overall_status\": ", empty_domain_json_escape(overall_status), ",")
        println(io, "  \"cases\": [")
        for (index, record) in enumerate(records)
            comma = index == length(records) ? "" : ","
            println(io, "    {")
            println(io, "      \"case\": ", empty_domain_json_escape(record.case), ",")
            println(io, "      \"status\": ", empty_domain_json_escape(record.status), ",")
            println(io, "      \"mesh\": ", empty_domain_json_escape(record.mesh), ",")
            println(io, "      \"mesh_path\": ", empty_domain_json_escape(record.mesh_path), ",")
            println(io, "      \"flux\": ", empty_domain_json_escape(record.flux), ",")
            println(io, "      \"pml_width\": ", record.pml_width, ",")
            println(io, "      \"sigma_max\": ", record.sigma_max, ",")
            println(io, "      \"sigma_degree\": ", record.sigma_degree, ",")
            println(io, "      \"central_frequency\": ", record.central_frequency, ",")
            println(io, "      \"central_wavelength\": ", record.central_wavelength, ",")
            println(io, "      \"reflection_ratio\": ", record.reflection_ratio, ",")
            println(io, "      \"max_reflection_ratio\": ", record.max_reflection_ratio, ",")
            println(io, "      \"final_energy_ratio\": ", record.final_energy_ratio, ",")
            println(io, "      \"max_final_energy_ratio\": ", record.max_final_energy_ratio, ",")
            println(io, "      \"max_energy_growth\": ", record.max_energy_growth, ",")
            println(io, "      \"dt\": ", record.dt, ",")
            println(io, "      \"steps\": ", record.steps, ",")
            println(io, "      \"output_dir\": ", empty_domain_json_escape(record.output_dir), ",")
            println(io, "      \"message\": ", empty_domain_json_escape(record.message))
            println(io, "    }", comma)
        end
        println(io, "  ]")
        println(io, "}")
    end
    return String(path)
end

function print_empty_domain_incident_pml_summary(records)
    println()
    println("Empty-domain incident-wave PML validation")
    println("-----------------------------------------")
    for record in records
        println("case:                  ", record.case)
        println("  status:              ", record.status)
        println("  flux:                ", record.flux)
        println("  reflection ratio:    ",
                @sprintf("%.6e", record.reflection_ratio),
                " <= ",
                @sprintf("%.6e", record.max_reflection_ratio))
        println("  final energy ratio:  ",
                @sprintf("%.6e", record.final_energy_ratio),
                " <= ",
                @sprintf("%.6e", record.max_final_energy_ratio))
        println("  output:              ", record.output_dir)
        println("  message:             ", record.message)
    end
    return nothing
end

function empty_domain_incident_pml_main(args::Vector{String})
    config = parse_empty_domain_incident_pml_config(args)
    config === nothing && return nothing

    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        summaries = run_validation_sweep(config.pml_config, comm)
        records = empty_domain_incident_pml_records(summaries, config)
        failed = any(record -> record.status == "FAIL", records)
        global_failed = MPI.Allreduce(failed ? 1 : 0, max, comm)

        if rank == 0
            csv_path = joinpath(
                config.pml_config.output_dir,
                "empty_domain_incident_pml_summary.csv",
            )
            json_path = joinpath(
                config.pml_config.output_dir,
                "empty_domain_incident_pml_summary.json",
            )
            write_empty_domain_incident_pml_csv(csv_path, records)
            write_empty_domain_incident_pml_json(json_path, records)
            print_empty_domain_incident_pml_summary(records)
            println()
            println("Wrote validation CSV:  ", csv_path)
            println("Wrote validation JSON: ", json_path)
        end
        MPI.Barrier(comm)
        global_failed == 0 ||
            error("Empty-domain incident-wave PML validation failed.")
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        empty_domain_incident_pml_main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    end
end
