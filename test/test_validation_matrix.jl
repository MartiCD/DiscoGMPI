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
        "--cases=cavity-pec,periodic",
        "--ranks=1,2",
        "--orders=2,3",
        "--rate-policy=fail",
        "--comparison=tolerance",
        "--run-invariants=false",
    ])

    @test config.profile == :standard
    @test config.cases == [:cavity_pec, :periodic]
    @test config.ranks == [1, 2]
    @test config.orders == [2, 3]
    @test config.rate_policy == :fail
    @test config.comparison == :tolerance
    @test !config.run_invariants
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
