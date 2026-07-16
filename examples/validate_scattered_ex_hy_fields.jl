#!/usr/bin/env julia

# Qualitative validation gate for metallic-sphere scattered fields. The gate
# runs the scattering driver, forces ParaView output, and verifies that the
# final PVTU snapshot exposes the scattered E/H vector arrays needed to inspect
# E_scat,x and H_scat,y in ParaView.

using Dates
using MPI
using Printf

include(joinpath(@__DIR__, "distributed_metallic_sphere_scattering.jl"))

const DEFAULT_SCATTERED_EX_HY_MESH = joinpath(
    @__DIR__,
    "meshes",
    "metallic_sphere_scattering_validation.vtk",
)

const DEFAULT_SCATTERED_EX_HY_OUTPUT = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "validation_sequence",
    "scattered_ex_hy_fields",
)

Base.@kwdef struct ScatteredExHyFieldsGateConfig
    scattering_config::MetallicSphereScatteringConfig
    min_final_scattered_energy::Float64 = 1.0e-12
    require_paraview::Bool = true
end

function scattered_ex_hy_fields_usage(io::IO = stdout)
    println(io, """
Qualitative scattered Ex/Hy field validation gate

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/validate_scattered_ex_hy_fields.jl [gate options] [scattering options]

Gate options:
  --min-final-scattered-energy E
                               Minimum final scattered-field energy required
                               to avoid accepting a zero-field output.
                               Default: 1e-12
  --allow-missing-paraview     Do not fail if fields.pvd or final PVTU arrays
                               are missing. Default: require ParaView output
  --help                       Show this message.

Default scattering options:
  --mesh=examples/meshes/metallic_sphere_scattering_validation.vtk
  --final-time=0.25
  --disable-rcs
  --paraview-every=1000000000
  --output-dir=output/validation_sequence/scattered_ex_hy_fields

This gate is qualitative: it prepares the final scattered-field dataset for
manual inspection. Open fields.pvd in ParaView and display the X component of
E_scat and the Y component of H_scat. All options accepted by
distributed_metallic_sphere_scattering.jl can be passed here and override the
defaults above.
""")
end

function scattered_gate_option_value(
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

function default_scattered_ex_hy_fields_args()
    return [
        "--mesh=$(DEFAULT_SCATTERED_EX_HY_MESH)",
        "--final-time=0.25",
        "--disable-rcs",
        "--paraview-every=1000000000",
        "--output-dir=$(DEFAULT_SCATTERED_EX_HY_OUTPUT)",
    ]
end

function parse_scattered_ex_hy_fields_gate_config(args::Vector{String})
    min_final_scattered_energy = 1.0e-12
    require_paraview = true
    scattering_args = String[]

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            scattered_ex_hy_fields_usage()
            println()
            usage()
            return nothing
        elseif argument == "--min-final-scattered-energy" ||
               startswith(argument, "--min-final-scattered-energy=")
            raw, index = scattered_gate_option_value(
                args,
                index,
                "--min-final-scattered-energy",
            )
            min_final_scattered_energy = parse(Float64, raw)
        elseif argument == "--allow-missing-paraview"
            require_paraview = false
        else
            push!(scattering_args, argument)
        end
        index += 1
    end

    isfinite(min_final_scattered_energy) &&
        min_final_scattered_energy >= 0.0 ||
        error("--min-final-scattered-energy must be finite and non-negative.")

    scattering_config =
        parse_config(vcat(default_scattered_ex_hy_fields_args(), scattering_args))
    scattering_config === nothing && return nothing
    return ScatteredExHyFieldsGateConfig(
        scattering_config = scattering_config,
        min_final_scattered_energy = min_final_scattered_energy,
        require_paraview = require_paraview,
    )
end

function scattered_fields_json_escape(value)
    text = string(value)
    text = replace(text, '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return "\"" * text * "\""
end

function scattered_fields_json_number(value::Real)
    value_float = Float64(value)
    return isfinite(value_float) ? string(value_float) :
           scattered_fields_json_escape(value)
end

function final_dataset_from_pvd(pvd_path::AbstractString)
    isfile(pvd_path) || return ""
    text = read(pvd_path, String)
    matches = collect(eachmatch(Regex("file=\"([^\"]+\\.pvtu)\""), text))
    isempty(matches) && return ""
    return String(matches[end].captures[1])
end

function paraview_scattered_field_info(output_dir::AbstractString)
    pvd_path = joinpath(String(output_dir), "fields.pvd")
    final_dataset = final_dataset_from_pvd(pvd_path)
    final_pvtu_path =
        isempty(final_dataset) ? "" : joinpath(String(output_dir), final_dataset)
    pvtu_text = isfile(final_pvtu_path) ? read(final_pvtu_path, String) : ""
    has_e_scat = occursin("Name=\"E_scat\"", pvtu_text)
    has_h_scat = occursin("Name=\"H_scat\"", pvtu_text)
    return (
        pvd_path = pvd_path,
        pvd_exists = isfile(pvd_path),
        final_dataset = final_dataset,
        final_pvtu_path = final_pvtu_path,
        final_pvtu_exists = isfile(final_pvtu_path),
        has_e_scat = has_e_scat,
        has_h_scat = has_h_scat,
        paraview_ok =
            isfile(pvd_path) && isfile(final_pvtu_path) &&
            has_e_scat && has_h_scat,
    )
end

function scattered_ex_hy_fields_record(
    summary,
    config::ScatteredExHyFieldsGateConfig,
)
    field_info = paraview_scattered_field_info(summary.output_dir)
    finite_energy = isfinite(summary.final_scattered_energy)
    energy_pass =
        finite_energy &&
        summary.final_scattered_energy >= config.min_final_scattered_energy
    paraview_pass = !config.require_paraview || field_info.paraview_ok
    passed = finite_energy && energy_pass && paraview_pass
    message = if passed
        "qualitative scattered Ex/Hy field gate passed"
    elseif !finite_energy
        "final scattered energy is non-finite"
    elseif !energy_pass
        "final scattered energy is below threshold"
    elseif !field_info.pvd_exists
        "fields.pvd is missing"
    elseif !field_info.final_pvtu_exists
        "final PVTU snapshot is missing"
    elseif !field_info.has_e_scat
        "final PVTU snapshot is missing E_scat"
    elseif !field_info.has_h_scat
        "final PVTU snapshot is missing H_scat"
    else
        "ParaView output check failed"
    end
    return (
        status = passed ? "PASS" : "FAIL",
        output_dir = summary.output_dir,
        diagnostics_path = summary.diagnostics_path,
        pec_residual_path = summary.pec_residual_path,
        pvd_path = field_info.pvd_path,
        final_dataset = field_info.final_dataset,
        final_pvtu_path = field_info.final_pvtu_path,
        final_time = summary.final_time,
        steps = summary.steps,
        dt = summary.dt,
        final_scattered_energy = summary.final_scattered_energy,
        min_final_scattered_energy = config.min_final_scattered_energy,
        require_paraview = config.require_paraview,
        pvd_exists = field_info.pvd_exists,
        final_pvtu_exists = field_info.final_pvtu_exists,
        has_e_scat = field_info.has_e_scat,
        has_h_scat = field_info.has_h_scat,
        paraview_ok = field_info.paraview_ok,
        energy_pass = energy_pass,
        recommended_ex_component = "E_scat_x",
        recommended_hy_component = "H_scat_y",
        message = message,
    )
end

function write_scattered_ex_hy_fields_csv(path::AbstractString, record)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "status,output_dir,diagnostics_path,pec_residual_path," *
            "pvd_path,final_dataset,final_pvtu_path,final_time,steps,dt," *
            "final_scattered_energy,min_final_scattered_energy," *
            "require_paraview,pvd_exists,final_pvtu_exists,has_e_scat," *
            "has_h_scat,paraview_ok,energy_pass," *
            "recommended_ex_component,recommended_hy_component,message",
        )
        println(
            io,
            join(
                (
                    record.status,
                    record.output_dir,
                    record.diagnostics_path,
                    record.pec_residual_path,
                    record.pvd_path,
                    record.final_dataset,
                    record.final_pvtu_path,
                    record.final_time,
                    record.steps,
                    record.dt,
                    record.final_scattered_energy,
                    record.min_final_scattered_energy,
                    record.require_paraview,
                    record.pvd_exists,
                    record.final_pvtu_exists,
                    record.has_e_scat,
                    record.has_h_scat,
                    record.paraview_ok,
                    record.energy_pass,
                    record.recommended_ex_component,
                    record.recommended_hy_component,
                    record.message,
                ),
                ',',
            ),
        )
    end
    return String(path)
end

function write_scattered_ex_hy_fields_json(path::AbstractString, record)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"disco-gmpi-scattered-ex-hy-fields/v1\",")
        println(io, "  \"generated_at\": ", scattered_fields_json_escape(Dates.now()), ",")
        println(io, "  \"overall_status\": ", scattered_fields_json_escape(record.status), ",")
        println(io, "  \"output_dir\": ", scattered_fields_json_escape(record.output_dir), ",")
        println(io, "  \"fields_pvd\": ", scattered_fields_json_escape(record.pvd_path), ",")
        println(io, "  \"final_pvtu\": ", scattered_fields_json_escape(record.final_pvtu_path), ",")
        println(io, "  \"final_time\": ", scattered_fields_json_number(record.final_time), ",")
        println(io, "  \"steps\": ", record.steps, ",")
        println(io, "  \"final_scattered_energy\": ",
                scattered_fields_json_number(record.final_scattered_energy), ",")
        println(io, "  \"min_final_scattered_energy\": ",
                scattered_fields_json_number(record.min_final_scattered_energy), ",")
        println(io, "  \"checks\": {")
        println(io, "    \"pvd_exists\": ", record.pvd_exists, ",")
        println(io, "    \"final_pvtu_exists\": ", record.final_pvtu_exists, ",")
        println(io, "    \"has_e_scat\": ", record.has_e_scat, ",")
        println(io, "    \"has_h_scat\": ", record.has_h_scat, ",")
        println(io, "    \"paraview_ok\": ", record.paraview_ok, ",")
        println(io, "    \"energy_pass\": ", record.energy_pass)
        println(io, "  },")
        println(io, "  \"recommended_components\": {")
        println(io, "    \"ex\": ", scattered_fields_json_escape(record.recommended_ex_component), ",")
        println(io, "    \"hy\": ", scattered_fields_json_escape(record.recommended_hy_component))
        println(io, "  },")
        println(io, "  \"message\": ", scattered_fields_json_escape(record.message))
        println(io, "}")
    end
    return String(path)
end

function print_scattered_ex_hy_fields_summary(record)
    println()
    println("Qualitative scattered Ex/Hy field validation")
    println("--------------------------------------------")
    println("status:                  ", record.status)
    println("final scattered energy:  ",
            @sprintf("%.6e", record.final_scattered_energy),
            " >= ",
            @sprintf("%.6e", record.min_final_scattered_energy))
    println("fields.pvd:              ", record.pvd_path)
    println("final PVTU:              ", record.final_pvtu_path)
    println("ParaView arrays:         E_scat=", record.has_e_scat,
            ", H_scat=", record.has_h_scat)
    println("inspect components:      ",
            record.recommended_ex_component,
            ", ",
            record.recommended_hy_component)
    println("message:                 ", record.message)
    return nothing
end

function scattered_ex_hy_fields_main(args::Vector{String})
    config = parse_scattered_ex_hy_fields_gate_config(args)
    config === nothing && return nothing

    MPI.Init()
    try
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        summary = run_experiment(config.scattering_config, comm)
        record = scattered_ex_hy_fields_record(summary, config)
        failed = record.status == "FAIL"
        global_failed = MPI.Allreduce(failed ? 1 : 0, max, comm)

        if rank == 0
            csv_path = joinpath(
                config.scattering_config.output_dir,
                "scattered_ex_hy_fields_summary.csv",
            )
            json_path = joinpath(
                config.scattering_config.output_dir,
                "scattered_ex_hy_fields_summary.json",
            )
            write_scattered_ex_hy_fields_csv(csv_path, record)
            write_scattered_ex_hy_fields_json(json_path, record)
            print_scattered_ex_hy_fields_summary(record)
            println()
            println("Wrote validation CSV:  ", csv_path)
            println("Wrote validation JSON: ", json_path)
        end
        MPI.Barrier(comm)
        global_failed == 0 ||
            error("Qualitative scattered Ex/Hy field validation failed.")
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        scattered_ex_hy_fields_main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    end
end
