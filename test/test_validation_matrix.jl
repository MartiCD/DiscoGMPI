using Test

include(joinpath(@__DIR__, "..", "examples", "validate_distributed_maxwell_matrix.jl"))

function write_csv(path::AbstractString, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    return path
end

@testset "Validation matrix argument parsing" begin
    config = parse_validation_arguments([
        "--profile=standard",
        "--cases=cavity-pec,periodic,pml",
        "--ranks=1,2,4",
        "--orders=2,3",
        "--rate-policy=fail",
        "--comparison=tolerance",
        "--energy-drift-max=1e-8",
        "--run-invariants=false",
        "--run-mpi-tests=false",
    ])

    @test config.profile == :standard
    @test config.cases == [:cavity_pec, :periodic, :pml]
    @test config.ranks == [1, 2, 4]
    @test config.orders == [2, 3]
    @test config.rate_policy == :fail
    @test config.comparison == :tolerance
    @test config.energy_drift_max == 1e-8
    @test !config.run_invariants
    @test !config.run_mpi_tests
end

@testset "Validation matrix convergence-rate checks" begin
    directory = mktempdir()
    path = joinpath(directory, "convergence.csv")
    write_csv(
        path,
        [
            "mpi_ranks",
            "boundary_condition",
            "order",
            "mesh_level",
            "rate_electric",
            "rate_magnetic",
            "rate_total",
        ],
        [
            ["2", "pec", "2", "1", "", "", ""],
            ["2", "pec", "2", "2", "2.75", "1.75", "2.1"],
        ],
    )

    checks = ValidationCheck[]
    config = ValidationConfig(rate_policy = :fail)
    add_rate_checks!(checks, config, :cavity_pec, 2, path)

    @test length(checks) == 2
    @test all(check.status == :PASS for check in checks)
end

@testset "Validation matrix CSV rank comparison" begin
    directory = mktempdir()
    reference = joinpath(directory, "rank1.csv")
    candidate = joinpath(directory, "rank2.csv")
    header = ["mpi_ranks", "order", "mesh_level", "l2_electric_error"]
    write_csv(reference, header, [["1", "2", "1", "1.000000000000"]])
    write_csv(candidate, header, [["2", "2", "1", "1.000000000001"]])

    checks = ValidationCheck[]
    config = ValidationConfig(rtol = 1e-9, atol = 1e-12)
    compare_csv_outputs!(
        checks,
        config,
        :cavity_pec,
        "convergence-rank-comparison",
        1,
        2,
        reference,
        candidate;
        exclude_columns = Set(["mpi_ranks"]),
    )

    @test length(checks) == 1
    @test checks[1].status == :PASS
end

@testset "Validation matrix invariant diagnostics schema" begin
    directory = mktempdir()
    path = joinpath(directory, "quadrature_diagnostics.csv")
    row = [
        column == "step" ? "0" :
        column == "time" ? "0.0" :
        "0.0"
        for column in INVARIANT_REQUIRED_COLUMNS
    ]
    write_csv(path, INVARIANT_REQUIRED_COLUMNS, [row])

    checks = ValidationCheck[]
    config = ValidationConfig()
    add_invariant_schema_checks!(checks, config, :periodic, 2, path)

    @test !isempty(checks)
    @test all(check.status == :PASS for check in checks)
end


@testset "Validation matrix energy-history checks" begin
    directory = mktempdir()
    conservative = joinpath(directory, "energy_conservative.csv")
    write_csv(
        conservative,
        ["step", "time", "electric", "magnetic", "total", "relative_drift"],
        [
            ["0", "0.0", "0.5", "0.5", "1.0", "0.0"],
            ["1", "0.1", "0.5", "0.5", "1.0", "5e-9"],
        ],
    )
    checks = ValidationCheck[]
    config = ValidationConfig(energy_drift_max = 1e-8)
    add_energy_history_checks!(checks, config, :cavity_pec, 2, conservative)
    @test any(check -> check.category == "energy-invariant" && check.status == :PASS, checks)

    pml = joinpath(directory, "energy_pml.csv")
    write_csv(
        pml,
        ["step", "time", "electric", "magnetic", "total", "relative_drift"],
        [
            ["0", "0.0", "0.5", "0.5", "1.0", "0.0"],
            ["1", "0.1", "0.4", "0.4", "0.8", "-0.2"],
        ],
    )
    pml_checks = ValidationCheck[]
    add_energy_history_checks!(pml_checks, config, :pml, 2, pml)
    @test any(check -> check.category == "energy-dissipation" && check.status == :PASS, pml_checks)
end

@testset "Validation matrix JSON summary" begin
    directory = mktempdir()
    checks = [
        ValidationCheck(
            "cavity-pec",
            "energy-invariant",
            "2",
            "max relative energy drift",
            "1e-9",
            "<= 1e-8",
            "",
            :PASS,
            "ok",
        ),
    ]
    path = write_validation_summary_json(
        joinpath(directory, "validation_summary.json"),
        ValidationConfig(run_mpi_tests = false),
        checks;
        matrix_path = joinpath(directory, "validation_matrix.csv"),
    )
    text = read(path, String)
    @test occursin("\"overall_status\": \"PASS\"", text)
    @test occursin("disco-gmpi-validation-summary/v1", text)
end
