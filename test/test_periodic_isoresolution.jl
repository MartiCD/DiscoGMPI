using Test

include(
    joinpath(
        @__DIR__,
        "..",
        "examples",
        "convergence_distributed_periodic_isoresolution.jl",
    ),
)

function dummy_iso_result(case::PeriodicIsoresolutionCase, errors)
    return PeriodicIsoresolutionResult(
        case,
        1,
        100,
        100,
        100,
        num_tet_nodes(case.order),
        isoresolution_work_units(100, case.order),
        max(2, 2 * case.order + 4),
        case.h_target,
        case.h_target,
        case.h_target,
        case.h_target,
        1.0,
        1.0,
        0.01,
        0.01,
        1.0,
        0.7,
        0.8,
        0.001,
        10,
        0.1,
        errors,
        errors,
        ISO_EMPTY_RATES,
        1.0,
        1.0,
        0.0,
    )
end

@testset "Periodic isoresolution case planner" begin
    config = parse_periodic_isoresolution_arguments(["--dry-run"])
    cases = periodic_isoresolution_cases(config)
    @test length(cases) == 12
    @test [
        periodic_isoresolution_case(order, 0).intervals_per_wavelength
        for order in (4, 2, 1)
    ] == [4, 8, 16]
    @test [
        periodic_isoresolution_case(4, level).intervals_per_wavelength
        for level in 0:3
    ] == [4, 8, 16, 32]
    @test periodic_isoresolution_unique_resolutions(cases) ==
          [4, 8, 16, 32, 64, 128]
    @test periodic_isoresolution_case_key(periodic_isoresolution_case(2, 3)) ==
          "P2M3"

    smoke_config = parse_periodic_isoresolution_arguments([
        "--smoke",
        "--dry-run",
    ])
    @test [
        periodic_isoresolution_case_key(case)
        for case in periodic_isoresolution_cases(smoke_config)
    ] == ["P4M0", "P4M1"]
    @test [
        case.intervals_per_wavelength
        for case in periodic_isoresolution_cases(smoke_config)
    ] == [2, 4]
end

@testset "Periodic isoresolution CLI" begin
    config = parse_periodic_isoresolution_arguments([
        "--orders=1,4",
        "--levels=2",
        "--periods=0.5",
        "--cfl=0.1",
        "--cfl-divisor=8",
        "--target-work-per-rank=1000",
        "--min-elements-per-rank=10",
        "--max-active-ranks=3",
        "--output=output/custom.csv",
        "--dry-run",
    ])
    @test config.orders == [1, 4]
    @test config.levels == [0, 1]
    @test config.final_time ≈ 0.5 *
                              periodic_wave_period(
                                  config.epsilon,
                                  config.mu,
                                  config.wave_number,
                              )
    @test config.cfl == 0.1
    @test config.cfl_divisor == 8
    @test config.target_work_per_rank == 1000
    @test config.min_elements_per_rank == 10
    @test config.max_active_ranks == 3
    @test config.latex_output == "output/custom.tex"
    @test_throws ArgumentError parse_periodic_isoresolution_arguments([
        "--orders=3",
        "--dry-run",
    ])
end

@testset "Periodic isoresolution rank heuristic" begin
    config = parse_periodic_isoresolution_arguments([
        "--target-work-per-rank=1000",
        "--min-elements-per-rank=10",
        "--max-active-ranks=4",
        "--dry-run",
    ])
    @test periodic_isoresolution_rank_count(config, 100, 1, 8) == 2
    @test periodic_isoresolution_rank_count(config, 100, 4, 8) == 4
    @test periodic_isoresolution_rank_count(config, 20, 4, 8) == 2
end

@testset "Periodic Linf-time L2-space rates and verdict" begin
    coarse_case = periodic_isoresolution_case(2, 0)
    fine_case = periodic_isoresolution_case(2, 1)
    coarse_errors = (
        1.0,
        1.0,
        1.0,
        1e-14,
        1e-14,
        1.0,
        1e-14,
        1.0,
        1e-14,
    )
    fine_errors = (
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1e-14,
        1e-14,
        1.0 / 8.0,
        1e-14,
        1.0 / 4.0,
        1e-14,
    )
    results = [
        dummy_iso_result(coarse_case, coarse_errors),
        dummy_iso_result(fine_case, fine_errors),
    ]
    add_periodic_isoresolution_rates!(results)

    @test results[2].rates_time_linf_space_l2[1] ≈ 3.0
    @test results[2].rates_time_linf_space_l2[2] ≈ 2.0
    @test results[2].rates_time_linf_space_l2[6] ≈ 3.0
    @test results[2].rates_time_linf_space_l2[8] ≈ 2.0
    pass, messages = periodic_isoresolution_verdict(results, 0.0)
    @test pass
    @test occursin("P2M1", only(messages))
end

@testset "Periodic isoresolution CSV and LaTeX output" begin
    header = periodic_isoresolution_csv_header()
    @test occursin("linf_time_l2_space_ez_error", header)
    @test occursin("rate_linf_time_l2_space_hy", header)
    @test occursin("physical_rate_verdict", header)

    config = parse_periodic_isoresolution_arguments([
        "--output=output/custom.csv",
        "--dry-run",
    ])
    coarse_case = periodic_isoresolution_case(1, 0)
    fine_case = periodic_isoresolution_case(1, 1)
    results = [
        dummy_iso_result(coarse_case, ntuple(_ -> 1.0, 9)),
        dummy_iso_result(fine_case, ntuple(_ -> 0.5, 9)),
    ]
    add_periodic_isoresolution_rates!(results)
    directory = mktempdir()
    path = joinpath(directory, "tables.tex")
    write_periodic_isoresolution_latex(path, results, config)
    text = read(path, String)
    @test occursin("Periodic Poisson-bracket", text)
    @test occursin("Transverse component leakage", text)
end
