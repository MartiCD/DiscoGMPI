#!/usr/bin/env julia

# End-to-end validation matrix for DiscoGMPI distributed Maxwell runs.
#
# This script intentionally reuses the public example drivers and validates
# their CSV outputs. It checks convergence rates, rank-to-rank reproducibility,
# and the diagnostic schema for energy, charge, momentum, and optical chirality.
#
# Example:
#   julia --project=. examples/validate_distributed_maxwell_matrix.jl \
#     --profile=smoke --cases=cavity-pec,cavity-pmc --ranks=1,2

using Dates
using Printf

const VALIDATION_REPOSITORY_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_RATE_TOLERANCE = 0.5

const CAVITY_CONVERGENCE_SCRIPT =
    joinpath(VALIDATION_REPOSITORY_ROOT, "examples",
             "convergence_distributed_poisson_bracket_maxwell.jl")
const PERIODIC_CONVERGENCE_SCRIPT =
    joinpath(VALIDATION_REPOSITORY_ROOT, "examples",
             "convergence_distributed_periodic_poisson_bracket_maxwell.jl")
const CAVITY_DRIVER_SCRIPT =
    joinpath(VALIDATION_REPOSITORY_ROOT, "examples",
             "distributed_poisson_bracket_maxwell.jl")
const PERIODIC_DRIVER_SCRIPT =
    joinpath(VALIDATION_REPOSITORY_ROOT, "examples",
             "distributed_periodic_poisson_bracket_maxwell.jl")

const MPI_TEST_SPECS = (
    (name = "face-permutation",
     script = joinpath(VALIDATION_REPOSITORY_ROOT, "test",
                       "test_mpi_face_permutation.jl"),
     ranks = (2,)),
    (name = "distributed-maxwell",
     script = joinpath(VALIDATION_REPOSITORY_ROOT, "test",
                       "test_mpi_distributed_maxwell.jl"),
     ranks = (2, 4)),
    (name = "metis-partition",
     script = joinpath(VALIDATION_REPOSITORY_ROOT, "test",
                       "test_mpi_metis_partition.jl"),
     ranks = (2, 4)),
    (name = "physical-coverage",
     script = joinpath(VALIDATION_REPOSITORY_ROOT, "test",
                       "test_mpi_physical_coverage.jl"),
     ranks = (2, 4)),
    (name = "nonlinear-pml",
     script = joinpath(VALIDATION_REPOSITORY_ROOT, "test",
                       "test_mpi_nonlinear_pml.jl"),
     ranks = (2,)),
)

const CAVITY_MESH =
    joinpath(VALIDATION_REPOSITORY_ROOT, "examples", "meshes", "tet_mesh.vtk")
const PERIODIC_MESH =
    joinpath(VALIDATION_REPOSITORY_ROOT, "examples", "meshes",
             "periodic_box_2x1x1_nx4.vtk")

const VALIDATION_CASE_ALIASES = Dict(
    "cavity-pec" => :cavity_pec,
    "pec" => :cavity_pec,
    "cavity_pec" => :cavity_pec,
    "cavity-pmc" => :cavity_pmc,
    "pmc" => :cavity_pmc,
    "cavity_pmc" => :cavity_pmc,
    "periodic" => :periodic,
    "periodic-plane-wave" => :periodic,
    "periodic_plane_wave" => :periodic,
    "pml" => :pml,
    "cavity-pml" => :pml,
    "cavity_pml" => :pml,
)

const INVARIANT_REQUIRED_COLUMNS = [
    "step",
    "time",
    "electric_error_l2",
    "magnetic_error_l2",
    "field_error_l2",
    "energy_error",
    "relative_energy_error",
    "optical_chirality",
    "exact_optical_chirality",
    "optical_chirality_error",
    "optical_chirality_density_error_l2",
    "electric_charge",
    "magnetic_charge",
    "exact_electric_charge",
    "exact_magnetic_charge",
    "linear_momentum_x",
    "linear_momentum_y",
    "linear_momentum_z",
    "exact_linear_momentum_x",
    "exact_linear_momentum_y",
    "exact_linear_momentum_z",
    "angular_momentum_x",
    "angular_momentum_y",
    "angular_momentum_z",
    "exact_angular_momentum_x",
    "exact_angular_momentum_y",
    "exact_angular_momentum_z",
]

const FIELD_ERROR_COLUMNS = [
    "electric_error_l2",
    "magnetic_error_l2",
    "field_error_l2",
]

const EXACT_PAIR_COLUMNS = [
    ("electric_charge", "exact_electric_charge"),
    ("magnetic_charge", "exact_magnetic_charge"),
    ("optical_chirality", "exact_optical_chirality"),
    ("linear_momentum_x", "exact_linear_momentum_x"),
    ("linear_momentum_y", "exact_linear_momentum_y"),
    ("linear_momentum_z", "exact_linear_momentum_z"),
    ("angular_momentum_x", "exact_angular_momentum_x"),
    ("angular_momentum_y", "exact_angular_momentum_y"),
    ("angular_momentum_z", "exact_angular_momentum_z"),
]

Base.@kwdef struct ValidationConfig
    profile::Symbol = :smoke
    cases::Vector{Symbol} = [:cavity_pec, :cavity_pmc, :periodic, :pml]
    ranks::Vector{Int} = [1, 2, 4]
    orders::Vector{Int} = [2]
    cavity_cells::Vector{Int} = [1, 2, 3]
    periodic_nx_targets::Vector{Int} = [2, 4]
    convergence_final_time::Float64 = 0.05
    invariant_final_time::Float64 = 0.02
    cfl::Float64 = 0.2
    output_dir::String =
        joinpath(VALIDATION_REPOSITORY_ROOT, "output", "validation_matrix")
    mpiexec::String = something(Sys.which("mpiexec"), "mpiexec")
    julia::String = Base.julia_cmd().exec[1]
    comparison::Symbol = :tolerance
    rate_policy::Symbol = :report
    rate_tolerance::Float64 = DEFAULT_RATE_TOLERANCE
    rtol::Float64 = 1.0e-10
    atol::Float64 = 1.0e-11
    diagnostic_error_max::Float64 = 1.0e-2
    field_error_max::Float64 = Inf
    energy_drift_max::Float64 = 1.0e-7
    pml_width::Float64 = 0.2
    pml_sigma_max::Float64 = 8.0
    run_convergence::Bool = true
    run_invariants::Bool = true
    run_mpi_tests::Bool = true
    dry_run::Bool = false
end

struct ValidationCheck
    case_name::String
    category::String
    ranks::String
    metric::String
    value::String
    expected::String
    tolerance::String
    status::Symbol
    message::String
end

function profile_defaults(profile::Symbol)
    if profile == :smoke
        return ValidationConfig()
    elseif profile == :standard
        return ValidationConfig(
            profile = :standard,
            orders = [2, 3],
            cavity_cells = [2, 3, 4, 5],
            periodic_nx_targets = [2, 4, 8],
            convergence_final_time = 0.25,
            invariant_final_time = 0.05,
            cfl = 0.08,
            rate_policy = :fail,
            diagnostic_error_max = 1.0e-4,
            energy_drift_max = 1.0e-6,
        )
    elseif profile == :strict
        return ValidationConfig(
            profile = :strict,
            orders = [3, 4],
            cavity_cells = [2, 3, 4, 5],
            periodic_nx_targets = [2, 4, 8, 16],
            convergence_final_time = 0.5,
            invariant_final_time = 0.1,
            cfl = 0.05,
            rate_policy = :fail,
            diagnostic_error_max = 1.0e-6,
            energy_drift_max = 1.0e-7,
        )
    else
        throw(ArgumentError("--profile must be smoke, standard, or strict."))
    end
end

function parse_integer_list(value::AbstractString)
    values = [
        parse(Int, strip(entry))
        for entry in split(value, ",")
        if !isempty(strip(entry))
    ]
    isempty(values) &&
        throw(ArgumentError("Expected a comma-separated integer list."))
    return values
end

function parse_case_list(value::AbstractString)
    cases = Symbol[]
    for entry in split(value, ",")
        key = lowercase(strip(entry))
        isempty(key) && continue
        haskey(VALIDATION_CASE_ALIASES, key) ||
            throw(ArgumentError("Unknown validation case '$entry'."))
        push!(cases, VALIDATION_CASE_ALIASES[key])
    end
    isempty(cases) &&
        throw(ArgumentError("Expected at least one validation case."))
    return cases
end

function parse_bool(value::AbstractString)
    key = lowercase(strip(value))
    if key in ("1", "true", "yes", "on")
        return true
    elseif key in ("0", "false", "no", "off")
        return false
    end
    throw(ArgumentError("Expected a boolean value, got '$value'."))
end

function parse_symbol_option(value::AbstractString, allowed::Tuple)
    symbol = Symbol(lowercase(strip(value)))
    symbol in allowed ||
        throw(ArgumentError("Expected one of $(join(allowed, ", ")), got '$value'."))
    return symbol
end

function print_validation_usage(io::IO = stdout)
    println(io, """
DiscoGMPI distributed Maxwell validation matrix

Usage:
  julia --project=. examples/validate_distributed_maxwell_matrix.jl [options]

Options:
  --profile=NAME              smoke, standard, or strict. Default: smoke
  --cases=a,b                 cavity-pec,cavity-pmc,periodic,pml by default
  --ranks=a,b                 MPI rank counts. Default: 1,2,4
  --orders=a,b                DG orders. Default depends on --profile
  --cavity-cells=a,b          Structured cavity convergence levels
  --periodic-nx-targets=a,b   Periodic Gmsh convergence levels
  --final-time=T              Convergence final time
  --invariant-final-time=T    Production-driver final time
  --cfl=C                     CFL for all launched drivers
  --output-dir=PATH           Validation output directory
  --mpiexec=PATH              MPI launcher. Default: first mpiexec in PATH
  --julia=PATH                Julia executable. Default: current Julia
  --comparison=MODE           tolerance or bitwise. Default: tolerance
  --rate-policy=MODE          report or fail. Default: smoke reports,
                              standard/strict fail
  --rate-tolerance=X          Expected-rate tolerance. Default: 0.5
  --rtol=X                    Rank-comparison relative tolerance. Default: 1e-10
  --atol=X                    Rank-comparison absolute tolerance. Default: 1e-11
  --diagnostic-error-max=X    Max numerical-vs-exact invariant diagnostic
                              error. Default depends on --profile
  --field-error-max=X         Optional max field L2 error. Default: Inf
  --energy-drift-max=X        Max conservative relative energy drift.
                              Default depends on --profile
  --pml-width=X               PML smoke layer width. Default: 0.2
  --pml-sigma-max=X           PML smoke peak damping. Default: 8.0
  --run-convergence=BOOL      Enable convergence-rate runs. Default: true
  --run-invariants=BOOL       Enable production diagnostic runs. Default: true
  --run-mpi-tests=BOOL        Enable standalone MPI regression tests. Default: true
  --dry-run                   Print commands and expected checks only
  --help                      Show this message

Checks:
  - PEC/PMC cavity convergence: aggregate E rate N+1 and H rate N.
  - Periodic plane wave convergence: aggregate E rate N+1, aggregate H rate N,
    active Ez rate N+1, and active Hy rate N.
  - Rank equivalence: rank-1 CSV outputs are compared with every other rank
    count, using the selected comparison mode.
  - Invariant diagnostics: production-driver CSVs must contain finite E/H
    errors, energy, charge, linear and angular momentum, and optical chirality.
  - PML is a smoke/dissipation case: it skips exact invariant targets and
    instead checks finite diagnostics plus non-increasing total energy.
  - Standalone MPI regression tests are launched from this same orchestrator.
  - The official machine-readable summary is validation_summary.json.
""")
end

function parse_validation_arguments(args::Vector{String})
    profile = :smoke
    for arg in args
        if startswith(arg, "--profile=")
            profile =
                parse_symbol_option(split(arg, "=", limit = 2)[2],
                                    (:smoke, :standard, :strict))
        end
    end

    config = profile_defaults(profile)
    for arg in args
        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--profile=")
            continue
        elseif startswith(arg, "--cases=")
            config = ValidationConfig(; pairs(merge_config(config,
                cases = parse_case_list(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--ranks=")
            config = ValidationConfig(; pairs(merge_config(config,
                ranks = parse_integer_list(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--orders=")
            config = ValidationConfig(; pairs(merge_config(config,
                orders = parse_integer_list(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--cavity-cells=") ||
               startswith(arg, "--cells=")
            config = ValidationConfig(; pairs(merge_config(config,
                cavity_cells = parse_integer_list(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--periodic-nx-targets=") ||
               startswith(arg, "--nx-targets=")
            config = ValidationConfig(; pairs(merge_config(config,
                periodic_nx_targets = parse_integer_list(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--final-time=")
            config = ValidationConfig(; pairs(merge_config(config,
                convergence_final_time =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--invariant-final-time=")
            config = ValidationConfig(; pairs(merge_config(config,
                invariant_final_time =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--cfl=")
            config = ValidationConfig(; pairs(merge_config(config,
                cfl = parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--output-dir=")
            config = ValidationConfig(; pairs(merge_config(config,
                output_dir = abspath(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--mpiexec=")
            config = ValidationConfig(; pairs(merge_config(config,
                mpiexec = split(arg, "=", limit = 2)[2]))...)
        elseif startswith(arg, "--julia=")
            config = ValidationConfig(; pairs(merge_config(config,
                julia = split(arg, "=", limit = 2)[2]))...)
        elseif startswith(arg, "--comparison=")
            config = ValidationConfig(; pairs(merge_config(config,
                comparison = parse_symbol_option(
                    split(arg, "=", limit = 2)[2],
                    (:tolerance, :bitwise))))...)
        elseif startswith(arg, "--rate-policy=")
            config = ValidationConfig(; pairs(merge_config(config,
                rate_policy = parse_symbol_option(
                    split(arg, "=", limit = 2)[2],
                    (:report, :fail))))...)
        elseif startswith(arg, "--rate-tolerance=")
            config = ValidationConfig(; pairs(merge_config(config,
                rate_tolerance =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--rtol=")
            config = ValidationConfig(; pairs(merge_config(config,
                rtol = parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--atol=")
            config = ValidationConfig(; pairs(merge_config(config,
                atol = parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--diagnostic-error-max=")
            config = ValidationConfig(; pairs(merge_config(config,
                diagnostic_error_max =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--field-error-max=")
            config = ValidationConfig(; pairs(merge_config(config,
                field_error_max =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--energy-drift-max=")
            config = ValidationConfig(; pairs(merge_config(config,
                energy_drift_max =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--pml-width=")
            config = ValidationConfig(; pairs(merge_config(config,
                pml_width =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--pml-sigma-max=")
            config = ValidationConfig(; pairs(merge_config(config,
                pml_sigma_max =
                    parse(Float64, split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--run-convergence=")
            config = ValidationConfig(; pairs(merge_config(config,
                run_convergence =
                    parse_bool(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--run-invariants=")
            config = ValidationConfig(; pairs(merge_config(config,
                run_invariants =
                    parse_bool(split(arg, "=", limit = 2)[2])))...)
        elseif startswith(arg, "--run-mpi-tests=")
            config = ValidationConfig(; pairs(merge_config(config,
                run_mpi_tests =
                    parse_bool(split(arg, "=", limit = 2)[2])))...)
        elseif arg == "--dry-run"
            config = ValidationConfig(; pairs(merge_config(config,
                dry_run = true))...)
        else
            throw(ArgumentError("Unknown argument '$arg'. Run with --help."))
        end
    end

    validate_config(config)
    return config
end

function merge_config(config::ValidationConfig; kwargs...)
    values = Dict{Symbol, Any}()
    for field in fieldnames(ValidationConfig)
        values[field] = getfield(config, field)
    end
    for (key, value) in kwargs
        values[key] = value
    end
    return values
end

function validate_config(config::ValidationConfig)
    isempty(config.cases) && throw(ArgumentError("At least one case is required."))
    isempty(config.ranks) && throw(ArgumentError("At least one rank is required."))
    isempty(config.orders) && throw(ArgumentError("At least one order is required."))
    all(rank -> rank >= 1, config.ranks) ||
        throw(ArgumentError("Rank counts must be positive."))
    1 in config.ranks ||
        throw(ArgumentError("Rank-comparison validation requires rank 1."))
    all(order -> order >= 1, config.orders) ||
        throw(ArgumentError("Orders must be positive."))
    if :periodic in config.cases
        all(order -> order >= 2, config.orders) ||
            throw(ArgumentError("Periodic validation requires orders >= 2."))
    end
    length(config.cavity_cells) >= 2 ||
        throw(ArgumentError("At least two cavity cell levels are required."))
    length(config.periodic_nx_targets) >= 2 ||
        throw(ArgumentError("At least two periodic NxTarget levels are required."))
    config.convergence_final_time > 0.0 ||
        throw(ArgumentError("--final-time must be positive."))
    config.invariant_final_time > 0.0 ||
        throw(ArgumentError("--invariant-final-time must be positive."))
    config.cfl > 0.0 ||
        throw(ArgumentError("--cfl must be positive."))
    config.rate_tolerance >= 0.0 ||
        throw(ArgumentError("--rate-tolerance must be non-negative."))
    config.rtol >= 0.0 ||
        throw(ArgumentError("--rtol must be non-negative."))
    config.atol >= 0.0 ||
        throw(ArgumentError("--atol must be non-negative."))
    config.diagnostic_error_max >= 0.0 ||
        throw(ArgumentError("--diagnostic-error-max must be non-negative."))
    config.field_error_max >= 0.0 ||
        throw(ArgumentError("--field-error-max must be non-negative."))
    config.energy_drift_max >= 0.0 ||
        throw(ArgumentError("--energy-drift-max must be non-negative."))
    config.pml_width > 0.0 ||
        throw(ArgumentError("--pml-width must be positive."))
    config.pml_sigma_max > 0.0 ||
        throw(ArgumentError("--pml-sigma-max must be positive."))
    return config
end

function case_label(case::Symbol)
    case == :cavity_pec && return "cavity-pec"
    case == :cavity_pmc && return "cavity-pmc"
    case == :periodic && return "periodic"
    case == :pml && return "pml"
    return string(case)
end

case_has_convergence(case::Symbol) = case != :pml
case_has_exact_invariants(case::Symbol) = case != :pml
case_is_pml(case::Symbol) = case == :pml

function boundary_condition(case::Symbol)
    case == :cavity_pec && return "pec"
    case == :cavity_pmc && return "pmc"
    throw(ArgumentError("Case $(case) has no exterior boundary condition."))
end

function comma_join(values)
    return join(string.(values), ",")
end

function shell_quote(value::AbstractString)
    isempty(value) && return "''"
    if occursin(r"[^A-Za-z0-9_@%+=:,./-]", value)
        return "'" * replace(value, "'" => "'\\''") * "'"
    end
    return value
end

function display_command(command::Cmd)
    return join(shell_quote.(command.exec), " ")
end

function validation_command(
    config::ValidationConfig,
    ranks::Int,
    script::String,
    args::Vector{String},
)
    return `$(config.mpiexec) -n $(ranks) $(config.julia) --project=$(VALIDATION_REPOSITORY_ROOT) $(script) $(args)`
end

function run_or_print(config::ValidationConfig, command::Cmd, description::String)
    println()
    println(description)
    println(display_command(command))
    config.dry_run && return true
    try
        run(command)
        return true
    catch error
        println(stderr, "Command failed: ", sprint(showerror, error))
        return false
    end
end

function vtk_tetrahedron_count(path::AbstractString)
    isfile(path) || error("VTK file not found: $path")
    in_cells = false
    remaining = 0
    tetrahedra = 0
    for line in eachline(path)
        fields = split(strip(line))
        isempty(fields) && continue
        if !in_cells && fields[1] == "CELLS"
            length(fields) >= 3 ||
                error("Malformed VTK CELLS line in $path.")
            remaining = parse(Int, fields[2])
            in_cells = true
            continue
        end
        if in_cells && remaining > 0
            parse(Int, fields[1]) == 4 && (tetrahedra += 1)
            remaining -= 1
            remaining == 0 && break
        end
    end
    tetrahedra > 0 ||
        error("No tetrahedra found in VTK file $path.")
    return tetrahedra
end

function write_balanced_partition(path::AbstractString, ntets::Int, ranks::Int)
    ntets >= ranks ||
        throw(ArgumentError("Need at least one tetrahedron per MPI rank."))
    mkpath(dirname(path))
    open(path, "w") do io
        for elem in 1:ntets
            owner = min(div((elem - 1) * ranks, ntets), ranks - 1)
            println(io, owner)
        end
    end
    return path
end

function partition_path(config::ValidationConfig, case::Symbol, ranks::Int)
    mesh = case == :periodic ? PERIODIC_MESH : CAVITY_MESH
    path = joinpath(
        config.output_dir,
        "partitions",
        "$(case_label(case))_ranks$(ranks).epart",
    )
    if !config.dry_run
        write_balanced_partition(path, vtk_tetrahedron_count(mesh), ranks)
    end
    return path
end

function convergence_output_path(config::ValidationConfig, case::Symbol, ranks::Int)
    return joinpath(
        config.output_dir,
        case_label(case),
        "convergence_ranks$(ranks).csv",
    )
end

function invariant_output_dir(config::ValidationConfig, case::Symbol, ranks::Int)
    return joinpath(
        config.output_dir,
        case_label(case),
        "invariants_ranks$(ranks)",
    )
end

function distributed_mesh_dir(config::ValidationConfig, case::Symbol, ranks::Int)
    return joinpath(
        config.output_dir,
        "distributed_mesh_cache",
        "$(case_label(case))_ranks$(ranks)",
    )
end

function convergence_command(config::ValidationConfig, case::Symbol, ranks::Int)
    output = convergence_output_path(config, case, ranks)
    if case in (:cavity_pec, :cavity_pmc)
        args = [
            "--mesh-family=structured",
            "--cells=$(comma_join(config.cavity_cells))",
            "--orders=$(comma_join(config.orders))",
            "--final-time=$(config.convergence_final_time)",
            "--cfl=$(config.cfl)",
            "--boundary-condition=$(boundary_condition(case))",
            "--output=$(output)",
        ]
        return validation_command(config, ranks, CAVITY_CONVERGENCE_SCRIPT, args)
    elseif case == :periodic
        args = [
            "--nx-targets=$(comma_join(config.periodic_nx_targets))",
            "--orders=$(comma_join(config.orders))",
            "--final-time=$(config.convergence_final_time)",
            "--cfl=$(config.cfl)",
            "--output=$(output)",
        ]
        return validation_command(config, ranks, PERIODIC_CONVERGENCE_SCRIPT, args)
    end
    throw(ArgumentError("Unknown validation case $case."))
end

function invariant_command(config::ValidationConfig, case::Symbol, ranks::Int)
    output_dir = invariant_output_dir(config, case, ranks)
    partition = partition_path(config, case, ranks)
    order = case == :periodic ? max(2, minimum(config.orders)) : minimum(config.orders)
    esprk_order = min(order + 1, 6)
    common_args = [
        "--partition=$(partition)",
        "--distributed-mesh-dir=$(distributed_mesh_dir(config, case, ranks))",
        "--rebuild-distributed-mesh",
        "--output-dir=$(output_dir)",
        "--order=$(order)",
        "--esprk-order=$(esprk_order)",
        "--final-time=$(config.invariant_final_time)",
        "--cfl=$(config.cfl)",
        "--energy-every=1",
        "--paraview-every=1000000000",
        "--checkpoint-every=0",
    ]

    if case in (:cavity_pec, :cavity_pmc)
        args = vcat(
            ["--mesh=$(CAVITY_MESH)",
             "--boundary-condition=$(boundary_condition(case))"],
            common_args,
        )
        return validation_command(config, ranks, CAVITY_DRIVER_SCRIPT, args)
    elseif case == :periodic
        args = vcat(
            ["--mesh=$(PERIODIC_MESH)", "--partitions=$(ranks)"],
            common_args,
        )
        return validation_command(config, ranks, PERIODIC_DRIVER_SCRIPT, args)
    elseif case == :pml
        args = vcat(
            ["--mesh=$(CAVITY_MESH)",
             "--boundary-condition=pec",
             "--pml-width=$(config.pml_width)",
             "--pml-sigma-max=$(config.pml_sigma_max)",
             "--pml-degree=2"],
            common_args,
        )
        return validation_command(config, ranks, CAVITY_DRIVER_SCRIPT, args)
    end
    throw(ArgumentError("Unknown validation case $case."))
end

function read_simple_csv(path::AbstractString)
    isfile(path) || error("CSV file not found: $path")
    lines = readlines(path)
    isempty(lines) && error("CSV file is empty: $path")
    header = split(lines[1], ",")
    rows = Vector{Dict{String, String}}()
    for (line_number, line) in enumerate(lines[2:end])
        isempty(strip(line)) && continue
        fields = split(line, ",", keepempty = true)
        length(fields) == length(header) ||
            error("CSV $path line $(line_number + 1) has $(length(fields)) fields; expected $(length(header)).")
        push!(rows, Dict(header[i] => fields[i] for i in eachindex(header)))
    end
    return header, rows
end

function parse_optional_float(value::AbstractString)
    isempty(strip(value)) && return missing
    parsed = tryparse(Float64, strip(value))
    parsed === nothing &&
        throw(ArgumentError("Value '$value' is not a floating-point number."))
    return parsed
end

function parse_required_float(row::Dict{String, String}, column::String)
    haskey(row, column) ||
        throw(ArgumentError("Missing column '$column'."))
    parsed = parse_optional_float(row[column])
    ismissing(parsed) &&
        throw(ArgumentError("Missing numeric value in column '$column'."))
    return parsed
end

function parse_required_int(row::Dict{String, String}, column::String)
    haskey(row, column) ||
        throw(ArgumentError("Missing column '$column'."))
    parsed = tryparse(Int, strip(row[column]))
    parsed === nothing &&
        throw(ArgumentError("Column '$column' has non-integer value '$(row[column])'."))
    return parsed
end

function push_check!(
    checks::Vector{ValidationCheck},
    case_name::String,
    category::String,
    ranks,
    metric::String,
    value,
    expected,
    tolerance,
    status::Symbol,
    message::String,
)
    push!(
        checks,
        ValidationCheck(
            case_name,
            category,
            string(ranks),
            metric,
            string(value),
            string(expected),
            string(tolerance),
            status,
            message,
        ),
    )
end

function push_command_check!(
    checks::Vector{ValidationCheck},
    case_name::String,
    category::String,
    ranks,
    metric::String,
    passed::Bool,
    command::Cmd,
)
    push_check!(
        checks,
        case_name,
        category,
        ranks,
        metric,
        passed ? "completed" : "failed",
        "exit code 0",
        "",
        passed ? :PASS : :FAIL,
        passed ? "Command completed successfully." :
                 "Command failed: $(display_command(command))",
    )
end

function status_from_pass(config::ValidationConfig, passed::Bool)
    passed && return :PASS
    return config.rate_policy == :fail ? :FAIL : :WARN
end

function final_rows_by_order(rows::Vector{Dict{String, String}})
    orders = sort(unique(parse_required_int(row, "order") for row in rows))
    finals = Dict{Int, Dict{String, String}}()
    for order in orders
        subset = filter(row -> parse_required_int(row, "order") == order, rows)
        sort!(subset; by = row -> parse_required_int(row, "mesh_level"))
        finals[order] = subset[end]
    end
    return finals
end

function rate_specs(case::Symbol)
    if case in (:cavity_pec, :cavity_pmc)
        return [
            ("rate_electric", "aggregate electric L2", order -> order + 1.0),
            ("rate_magnetic", "aggregate magnetic L2", order -> Float64(order)),
        ]
    elseif case == :periodic
        return [
            ("rate_electric", "aggregate electric L2", order -> order + 1.0),
            ("rate_magnetic", "aggregate magnetic L2", order -> Float64(order)),
            ("rate_ez", "active Ez L2", order -> order + 1.0),
            ("rate_hy", "active Hy L2", order -> Float64(order)),
        ]
    end
    throw(ArgumentError("Unknown validation case $case."))
end

function add_rate_checks!(
    checks::Vector{ValidationCheck},
    config::ValidationConfig,
    case::Symbol,
    ranks::Int,
    csv_path::String,
)
    _header, rows = read_simple_csv(csv_path)
    isempty(rows) && error("No convergence rows in $csv_path.")
    finals = final_rows_by_order(rows)

    for (column, label, expected_fn) in rate_specs(case)
        for order in sort(collect(keys(finals)))
            row = finals[order]
            if !haskey(row, column)
                push_check!(
                    checks,
                    case_label(case),
                    "convergence-rate",
                    ranks,
                    "$label N=$order",
                    "missing",
                    "column $column",
                    "",
                    status_from_pass(config, false),
                    "Rate column $column is absent from $csv_path.",
                )
                continue
            end
            rate = parse_optional_float(row[column])
            expected = expected_fn(order)
            minimum = expected - config.rate_tolerance
            passed =
                !ismissing(rate) &&
                isfinite(rate) &&
                rate >= minimum
            push_check!(
                checks,
                case_label(case),
                "convergence-rate",
                ranks,
                "$label N=$order",
                ismissing(rate) ? "missing" : @sprintf("%.6e", rate),
                @sprintf(">= %.6e", minimum),
                @sprintf("expected %.2f, tolerance %.2f",
                         expected, config.rate_tolerance),
                status_from_pass(config, passed),
                passed ? "Observed rate meets the explicit target." :
                         "Observed rate is below the explicit target.",
            )
        end
    end
end

function is_numeric_string(value::AbstractString)
    !isempty(strip(value)) && tryparse(Float64, strip(value)) !== nothing
end

function values_match(
    left::AbstractString,
    right::AbstractString,
    config::ValidationConfig,
)
    if config.comparison == :bitwise
        return left == right, 0.0, 0.0
    end

    if is_numeric_string(left) && is_numeric_string(right)
        a = parse(Float64, left)
        b = parse(Float64, right)
        absdiff = abs(a - b)
        reldiff = absdiff / max(abs(a), abs(b), eps(Float64))
        return absdiff <= config.atol + config.rtol * max(abs(a), abs(b)),
               absdiff,
               reldiff
    end

    return left == right, left == right ? 0.0 : Inf, left == right ? 0.0 : Inf
end

function compare_csv_outputs!(
    checks::Vector{ValidationCheck},
    config::ValidationConfig,
    case::Symbol,
    category::String,
    reference_ranks::Int,
    candidate_ranks::Int,
    reference_path::String,
    candidate_path::String;
    exclude_columns = Set{String}(),
)
    reference_header, reference_rows = read_simple_csv(reference_path)
    candidate_header, candidate_rows = read_simple_csv(candidate_path)
    common_columns = [
        column
        for column in reference_header
        if column in candidate_header && !(column in exclude_columns)
    ]

    if length(reference_rows) != length(candidate_rows)
        push_check!(
            checks,
            case_label(case),
            category,
            "$reference_ranks:$candidate_ranks",
            "row-count",
            length(candidate_rows),
            length(reference_rows),
            "",
            :FAIL,
            "CSV row counts differ between rank counts.",
        )
        return
    end

    worst_column = ""
    worst_row = 0
    worst_abs = 0.0
    worst_rel = 0.0
    mismatches = 0
    for row_index in eachindex(reference_rows)
        reference = reference_rows[row_index]
        candidate = candidate_rows[row_index]
        for column in common_columns
            matched, absdiff, reldiff =
                values_match(reference[column], candidate[column], config)
            if !matched
                mismatches += 1
                if absdiff >= worst_abs
                    worst_column = column
                    worst_row = row_index
                    worst_abs = absdiff
                    worst_rel = reldiff
                end
            end
        end
    end

    passed = mismatches == 0
    push_check!(
        checks,
        case_label(case),
        category,
        "$reference_ranks:$candidate_ranks",
        "rank-equivalence",
        passed ? "0 mismatches" :
                 "$mismatches mismatches; worst $(worst_column) row $(worst_row)",
        config.comparison == :bitwise ? "identical text values" :
                                        "within atol/rtol",
        config.comparison == :bitwise ? "bitwise" :
            @sprintf("atol %.1e, rtol %.1e, worst abs %.3e, worst rel %.3e",
                     config.atol, config.rtol, worst_abs, worst_rel),
        passed ? :PASS : :FAIL,
        passed ? "Rank outputs agree." :
                 "Rank outputs differ beyond the selected comparison mode.",
    )
end

function add_invariant_schema_checks!(
    checks::Vector{ValidationCheck},
    config::ValidationConfig,
    case::Symbol,
    ranks::Int,
    csv_path::String,
)
    header, rows = read_simple_csv(csv_path)
    missing_columns = [
        column for column in INVARIANT_REQUIRED_COLUMNS if !(column in header)
    ]
    push_check!(
        checks,
        case_label(case),
        "invariant-schema",
        ranks,
        "required diagnostic columns",
        isempty(missing_columns) ? "present" : join(missing_columns, ";"),
        "all required columns present",
        "",
        isempty(missing_columns) ? :PASS : :FAIL,
        isempty(missing_columns) ?
        "Diagnostics contain E/H, energy, charge, momentum, and chirality columns." :
        "Diagnostics are missing required columns.",
    )
    isempty(missing_columns) || return

    finite_failure = ""
    for (row_index, row) in enumerate(rows)
        for column in INVARIANT_REQUIRED_COLUMNS
            column in ("step",) && continue
            value = parse_optional_float(row[column])
            if ismissing(value) || !isfinite(value)
                finite_failure = "row $row_index column $column"
                break
            end
        end
        isempty(finite_failure) || break
    end
    push_check!(
        checks,
        case_label(case),
        "invariant-schema",
        ranks,
        "finite diagnostic values",
        isempty(finite_failure) ? "finite" : finite_failure,
        "all finite",
        "",
        isempty(finite_failure) ? :PASS : :FAIL,
        isempty(finite_failure) ?
        "All required diagnostics are finite." :
        "A required diagnostic is missing or non-finite.",
    )

    for column in FIELD_ERROR_COLUMNS
        maximum_value = maximum(parse_required_float(row, column) for row in rows)
        passed = maximum_value <= config.field_error_max
        push_check!(
            checks,
            case_label(case),
            "field-error",
            ranks,
            column,
            @sprintf("%.6e", maximum_value),
            isfinite(config.field_error_max) ?
            @sprintf("<= %.6e", config.field_error_max) :
            "finite",
            "",
            passed ? :PASS : :FAIL,
            passed ? "Field diagnostic is finite and within the configured bound." :
                     "Field diagnostic exceeds the configured bound.",
        )
    end

    if case_has_exact_invariants(case)
        for (value_column, exact_column) in EXACT_PAIR_COLUMNS
            maximum_error = maximum(
                abs(
                    parse_required_float(row, value_column) -
                    parse_required_float(row, exact_column),
                )
                for row in rows
            )
            passed = maximum_error <= config.diagnostic_error_max
            push_check!(
                checks,
                case_label(case),
                "diagnostic-error",
                ranks,
                value_column,
                @sprintf("%.6e", maximum_error),
                @sprintf("<= %.6e", config.diagnostic_error_max),
                "",
                passed ? :PASS : :FAIL,
                passed ? "Numerical diagnostic matches its analytical reference." :
                         "Numerical diagnostic differs from its analytical reference.",
            )
        end
    else
        push_check!(
            checks,
            case_label(case),
            "diagnostic-error",
            ranks,
            "exact invariant targets",
            "skipped",
            "not meaningful for dissipative PML",
            "",
            :SKIP,
            "PML damping intentionally changes the conservative exact invariants.",
        )
    end
end

function add_energy_history_checks!(
    checks::Vector{ValidationCheck},
    config::ValidationConfig,
    case::Symbol,
    ranks::Int,
    csv_path::String,
)
    header, rows = read_simple_csv(csv_path)
    for column in ("step", "time", "total", "relative_drift")
        column in header ||
            push_check!(
                checks,
                case_label(case),
                "energy-history",
                ranks,
                "required energy columns",
                "missing $column",
                "present",
                "",
                :FAIL,
                "Energy history is missing a required column.",
            )
    end
    all(column -> column in header, ("step", "time", "total", "relative_drift")) ||
        return
    isempty(rows) &&
        push_check!(
            checks,
            case_label(case),
            "energy-history",
            ranks,
            "row-count",
            0,
            "> 0",
            "",
            :FAIL,
            "Energy history contains no rows.",
        )
    isempty(rows) && return

    totals = [parse_required_float(row, "total") for row in rows]
    drifts = [parse_required_float(row, "relative_drift") for row in rows]
    finite = all(isfinite, totals) && all(isfinite, drifts)
    push_check!(
        checks,
        case_label(case),
        "energy-history",
        ranks,
        "finite energy values",
        finite ? "finite" : "non-finite",
        "all finite",
        "",
        finite ? :PASS : :FAIL,
        finite ? "Energy history values are finite." :
                 "Energy history contains non-finite values.",
    )
    finite || return

    if case_is_pml(case)
        initial = first(totals)
        final = last(totals)
        tolerance = config.energy_drift_max * max(abs(initial), eps(Float64))
        passed = final <= initial + tolerance
        push_check!(
            checks,
            case_label(case),
            "energy-dissipation",
            ranks,
            "final total energy",
            @sprintf("%.6e", final),
            @sprintf("<= %.6e", initial + tolerance),
            @sprintf("absolute tolerance %.3e", tolerance),
            passed ? :PASS : :FAIL,
            passed ? "PML total energy is non-increasing." :
                     "PML total energy increased beyond tolerance.",
        )
    else
        maximum_drift = maximum(abs, drifts)
        passed = maximum_drift <= config.energy_drift_max
        push_check!(
            checks,
            case_label(case),
            "energy-invariant",
            ranks,
            "max relative energy drift",
            @sprintf("%.6e", maximum_drift),
            @sprintf("<= %.6e", config.energy_drift_max),
            "",
            passed ? :PASS : :FAIL,
            passed ? "Conservative energy drift is within tolerance." :
                     "Conservative energy drift exceeds tolerance.",
        )
    end
end

function csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end


function json_escape(value)
    text = string(value)
    text = replace(text, '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return "\"" * text * "\""
end

function validation_counts(checks::Vector{ValidationCheck})
    return Dict(status => count(check -> check.status == status, checks)
                for status in (:PASS, :WARN, :FAIL, :SKIP))
end

function validation_overall_status(checks::Vector{ValidationCheck})
    any(check -> check.status == :FAIL, checks) && return "FAIL"
    any(check -> check.status == :WARN, checks) && return "WARN"
    return "PASS"
end

function write_validation_summary_json(
    path::String,
    config::ValidationConfig,
    checks::Vector{ValidationCheck};
    matrix_path::String,
)
    mkpath(dirname(path))
    counts = validation_counts(checks)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"disco-gmpi-validation-summary/v1\",")
        println(io, "  \"generated_at\": ", json_escape(string(Dates.now())), ",")
        println(io, "  \"overall_status\": ", json_escape(validation_overall_status(checks)), ",")
        println(io, "  \"profile\": ", json_escape(config.profile), ",")
        println(io, "  \"cases\": [", join(json_escape.(case_label.(config.cases)), ", "), "],")
        println(io, "  \"ranks\": [", join(config.ranks, ", "), "],")
        println(io, "  \"orders\": [", join(config.orders, ", "), "],")
        println(io, "  \"matrix_csv\": ", json_escape(matrix_path), ",")
        println(io, "  \"counts\": {")
        for (index, status) in enumerate((:PASS, :WARN, :FAIL, :SKIP))
            comma = index == 4 ? "" : ","
            println(io, "    ", json_escape(status), ": ", get(counts, status, 0), comma)
        end
        println(io, "  },")
        println(io, "  \"checks\": [")
        for (index, check) in enumerate(checks)
            comma = index == length(checks) ? "" : ","
            println(io, "    {")
            println(io, "      \"case\": ", json_escape(check.case_name), ",")
            println(io, "      \"category\": ", json_escape(check.category), ",")
            println(io, "      \"ranks\": ", json_escape(check.ranks), ",")
            println(io, "      \"metric\": ", json_escape(check.metric), ",")
            println(io, "      \"value\": ", json_escape(check.value), ",")
            println(io, "      \"expected\": ", json_escape(check.expected), ",")
            println(io, "      \"tolerance\": ", json_escape(check.tolerance), ",")
            println(io, "      \"status\": ", json_escape(check.status), ",")
            println(io, "      \"message\": ", json_escape(check.message))
            println(io, "    }", comma)
        end
        println(io, "  ]")
        println(io, "}")
    end
    return path
end

function write_validation_matrix(path::String, checks::Vector{ValidationCheck})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "case,category,ranks,metric,value,expected,tolerance,status,message")
        for check in checks
            println(
                io,
                join(
                    csv_escape.(
                        (
                            check.case_name,
                            check.category,
                            check.ranks,
                            check.metric,
                            check.value,
                            check.expected,
                            check.tolerance,
                            check.status,
                            check.message,
                        ),
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function print_validation_summary(checks::Vector{ValidationCheck})
    counts = Dict(status => count(check -> check.status == status, checks)
                  for status in (:PASS, :WARN, :FAIL, :SKIP))
    println()
    println("Validation matrix summary")
    println("-------------------------")
    for status in (:PASS, :WARN, :FAIL, :SKIP)
        println(rpad(string(status), 7), get(counts, status, 0))
    end
    failures = filter(check -> check.status == :FAIL, checks)
    if !isempty(failures)
        println()
        println("Failures")
        println("--------")
        for check in failures
            println(
                check.case_name,
                " / ",
                check.category,
                " / ",
                check.ranks,
                " / ",
                check.metric,
                ": ",
                check.message,
                " value=",
                check.value,
                " expected=",
                check.expected,
            )
        end
    end
end

function validate_case!(checks::Vector{ValidationCheck}, config::ValidationConfig, case::Symbol)
    if config.run_convergence
        if !case_has_convergence(case)
            push_check!(
                checks,
                case_label(case),
                "convergence-rate",
                comma_join(config.ranks),
                "expected convergence rates",
                "skipped",
                "not meaningful for this smoke case",
                "",
                :SKIP,
                "No convergence target is defined for the PML smoke case.",
            )
        else
            successful_convergence_ranks = Int[]
            for ranks in config.ranks
                command = convergence_command(config, case, ranks)
                passed = run_or_print(
                    config,
                    command,
                    "Running convergence validation for $(case_label(case)) on $ranks rank(s)",
                )
                push_command_check!(
                    checks,
                    case_label(case),
                    "convergence-run",
                    ranks,
                    "driver command",
                    passed,
                    command,
                )
                if !config.dry_run && passed
                    push!(successful_convergence_ranks, ranks)
                    add_rate_checks!(
                        checks,
                        config,
                        case,
                        ranks,
                        convergence_output_path(config, case, ranks),
                    )
                end
            end

            if !config.dry_run && first(config.ranks) in successful_convergence_ranks
                reference = first(config.ranks)
                for ranks in Iterators.drop(config.ranks, 1)
                    ranks in successful_convergence_ranks || continue
                    compare_csv_outputs!(
                        checks,
                        config,
                        case,
                        "convergence-rank-comparison",
                        reference,
                        ranks,
                        convergence_output_path(config, case, reference),
                        convergence_output_path(config, case, ranks);
                        exclude_columns = Set([
                            "mpi_ranks",
                            "min_owned_elements",
                            "max_owned_elements",
                            "elapsed_seconds",
                        ]),
                    )
                end
            end
        end
    end

    if config.run_invariants
        successful_invariant_ranks = Int[]
        for ranks in config.ranks
            command = invariant_command(config, case, ranks)
            passed = run_or_print(
                config,
                command,
                "Running invariant validation for $(case_label(case)) on $ranks rank(s)",
            )
            push_command_check!(
                checks,
                case_label(case),
                "invariant-run",
                ranks,
                "driver command",
                passed,
                command,
            )
            if !config.dry_run && passed
                push!(successful_invariant_ranks, ranks)
                diagnostics_path =
                    joinpath(invariant_output_dir(config, case, ranks),
                             "quadrature_diagnostics.csv")
                energy_path =
                    joinpath(invariant_output_dir(config, case, ranks),
                             "energy.csv")
                add_invariant_schema_checks!(
                    checks,
                    config,
                    case,
                    ranks,
                    diagnostics_path,
                )
                add_energy_history_checks!(
                    checks,
                    config,
                    case,
                    ranks,
                    energy_path,
                )
            end
        end

        if !config.dry_run && first(config.ranks) in successful_invariant_ranks
            reference = first(config.ranks)
            for ranks in Iterators.drop(config.ranks, 1)
                ranks in successful_invariant_ranks || continue
                compare_csv_outputs!(
                    checks,
                    config,
                    case,
                    "invariant-rank-comparison",
                    reference,
                    ranks,
                    joinpath(invariant_output_dir(config, case, reference),
                             "quadrature_diagnostics.csv"),
                    joinpath(invariant_output_dir(config, case, ranks),
                             "quadrature_diagnostics.csv"),
                )
                compare_csv_outputs!(
                    checks,
                    config,
                    case,
                    "energy-rank-comparison",
                    reference,
                    ranks,
                    joinpath(invariant_output_dir(config, case, reference),
                             "energy.csv"),
                    joinpath(invariant_output_dir(config, case, ranks),
                             "energy.csv"),
                )
            end
        end
    end
end

function mpi_test_command(config::ValidationConfig, script::String, ranks::Int)
    return validation_command(config, ranks, script, String[])
end

function run_mpi_tests!(checks::Vector{ValidationCheck}, config::ValidationConfig)
    config.run_mpi_tests || return nothing
    for spec in MPI_TEST_SPECS
        for ranks in spec.ranks
            command = mpi_test_command(config, spec.script, ranks)
            passed = run_or_print(
                config,
                command,
                "Running MPI regression $(spec.name) on $ranks rank(s)",
            )
            push_command_check!(
                checks,
                "mpi-tests",
                "mpi-test",
                ranks,
                spec.name,
                passed,
                command,
            )
        end
    end
    return nothing
end

function run_validation_matrix(config::ValidationConfig)
    println("DiscoGMPI distributed Maxwell validation matrix")
    println("-----------------------------------------------")
    println("profile:          ", config.profile)
    println("cases:            ", join(case_label.(config.cases), ", "))
    println("ranks:            ", config.ranks)
    println("orders:           ", config.orders)
    println("rate targets:     cavity E=N+1, cavity H=N; periodic E=N+1, H=N, Ez=N+1, Hy=N")
    println("rate policy:      ", config.rate_policy)
    println("comparison:       ", config.comparison)
    println("MPI tests:        ", config.run_mpi_tests ? "enabled" : "disabled")
    println("output directory: ", config.output_dir)
    config.dry_run && println("dry run:          true")

    checks = ValidationCheck[]
    if config.dry_run
        for case in config.cases
            if config.run_convergence && case_has_convergence(case)
                for ranks in config.ranks
                    command = convergence_command(config, case, ranks)
                    run_or_print(
                        config,
                        command,
                        "Planned convergence validation for $(case_label(case)) on $ranks rank(s)",
                    )
                end
            elseif config.run_convergence
                push_check!(
                    checks,
                    case_label(case),
                    "convergence-rate",
                    comma_join(config.ranks),
                    "expected convergence rates",
                    "skipped",
                    "not meaningful for this smoke case",
                    "",
                    :SKIP,
                    "No convergence target is defined for the PML smoke case.",
                )
            end
            if config.run_invariants
                for ranks in config.ranks
                    command = invariant_command(config, case, ranks)
                    run_or_print(
                        config,
                        command,
                        "Planned invariant validation for $(case_label(case)) on $ranks rank(s)",
                    )
                end
            end
        end
        run_mpi_tests!(checks, config)
        return checks
    end

    mkpath(config.output_dir)
    for case in config.cases
        validate_case!(checks, config, case)
    end
    run_mpi_tests!(checks, config)

    matrix_path = write_validation_matrix(
        joinpath(config.output_dir, "validation_matrix.csv"),
        checks,
    )
    summary_path = write_validation_summary_json(
        joinpath(config.output_dir, "validation_summary.json"),
        config,
        checks;
        matrix_path = matrix_path,
    )
    print_validation_summary(checks)
    println()
    println("Wrote validation matrix: ", matrix_path)
    println("Wrote validation summary: ", summary_path)

    any(check -> check.status == :FAIL, checks) &&
        error("Validation matrix contains failing checks.")
    return checks
end

function main(args::Vector{String})
    config = parse_validation_arguments(args)
    if config === nothing
        print_validation_usage()
        return nothing
    end
    run_validation_matrix(config)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch error
        println(stderr, "ERROR: ", sprint(showerror, error))
        rethrow()
    end
end
