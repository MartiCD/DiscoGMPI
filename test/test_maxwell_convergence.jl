using Test
using DiscoGMPI

function synthetic_convergence_row(level; nodal_set = :equispaced, scale = 1.0)
    h = 0.5^level
    errors = (h^3, h^2, h^2, 1e-14, 1e-14, h^3, 1e-14, h^2, 1e-14)
    return maxwell_convergence_row((order = 2, mesh_level = level,
        boundary_condition = :periodic, nodal_set = nodal_set,
        h_max = h, final_time = 0.5, effective_cfl = 0.05,
        rate_tolerance = 0.0), scale .* errors, ntuple(_ -> 1e-6, 9))
end

@testset "Maximum-in-time errors, schema, and rate policy" begin
    rows = rate_maxwell_convergence_rows([synthetic_convergence_row(0), synthetic_convergence_row(1)])
    @test rows[2].rate_electric == 3.0
    @test rows[2].rate_hy == 2.0
    @test rows[2].rate_hx === missing
    @test rows[2].status_hx == "SKIP_ROUNDOFF"
    @test rows[2].rate_verdict == "PASS"
    @test maxwell_convergence_verdict(rows)
    @test !maxwell_convergence_verdict(rows[1:1])
    @test rows[2].final_l2_space_electric_error == 1e-6
    @test rows[2].linf_time_l2_space_electric_error == 0.125
    @test length(split(maxwell_convergence_csv_header(), ',')) == length(keys(rows[1]))
    @test parse_convergence_nodal_set("warp-blend") == :warp_blend
    @test parse_convergence_nodal_set("warp_blend") == :warp_blend
    @test_throws ArgumentError parse_convergence_nodal_set("unknown")
    @test_throws ArgumentError validate_convergence_policy(:unknown, 0.5)
    @test_throws ArgumentError validate_convergence_policy(:fail, NaN)
    @test maxwell_error_maxima(ntuple(_ -> 2.0, 9), ntuple(_ -> 1.0, 9)) == ntuple(_ -> 2.0, 9)
    @test_throws ErrorException maxwell_error_maxima(ntuple(_ -> 0.0, 9), ntuple(_ -> NaN, 9))
    @test maxwell_convergence_rate_check(1e-14, 1.0, 1.0, 0.5, 3.0).status == "FAIL"
    @test maxwell_convergence_rate_check(1.0, 0.0, 1.0, 0.5, 3.0).status == "PASS_FLOOR"
    @test maxwell_convergence_rate_check(NaN, 1.0, 1.0, 0.5, 3.0).status == "FAIL"
    @test maxwell_convergence_rate_check(1.0, 0.1, 1.0, 1.0, 3.0).status == "FAIL"
    @test maxwell_expected_rate(:ey, 2, :pec) === missing
    @test maxwell_expected_rate(:hy, 2, :pmc) === missing
    @test maxwell_expected_rate(:ey, 2, :pmc) == 3.0

    mixed = [synthetic_convergence_row(l; nodal_set=n, scale=n == :warp_blend ? 0.1 : 1.0)
             for n in (:warp_blend, :equispaced) for l in 0:1]
    rated = rate_maxwell_convergence_rows(mixed)
    @test count(r -> r.rate_verdict == "PASS", rated) == 2
    @test all(r -> ismissing(r.rate_electric) || r.rate_electric ≈ 3.0, rated)

    mixed_initializations = [merge(synthetic_convergence_row(l; scale=scale),
        (magnetic_initialization=mode,)) for (mode,scale) in
        (("interpolate",1.0),("curl_potential",0.1)) for l in 0:1]
    checked = rate_maxwell_convergence_rows(mixed_initializations)
    @test count(r -> r.rate_verdict == "PASS", checked) == 2
    @test all(r -> ismissing(r.rate_electric) || r.rate_electric ≈ 3.0, checked)
    @test maxwell_convergence_verdict(checked)

    mixed_electric = [merge(synthetic_convergence_row(l;scale=scale),
        (electric_initialization=mode,)) for (mode,scale) in
        (("interpolate",1.0),("curl_potential",0.1)) for l in 0:1]
    checked_electric = rate_maxwell_convergence_rows(mixed_electric)
    @test count(r -> r.rate_verdict == "PASS", checked_electric) == 2
    @test all(r -> ismissing(r.rate_electric) || r.rate_electric ≈ 3.0, checked_electric)
    @test maxwell_convergence_verdict(checked_electric)
    @test !maxwell_convergence_verdict(vcat(checked_electric[1:1],checked_electric[2:2]))

    mktempdir() do directory
        path = joinpath(directory, "convergence.csv")
        write_maxwell_convergence_csv(path, rows)
        @test first(readlines(path)) == maxwell_convergence_csv_header()
        @test length(readlines(path)) == 3
    end
end
