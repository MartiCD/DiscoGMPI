using Test

module CoarseSphereRCSGateHarness
include(joinpath(@__DIR__, "..", "examples", "validate_coarse_sphere_rcs.jl"))
end

@testset "Coarse sphere RCS gate parsing" begin
    mesh_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "meshes",
        "metallic_sphere_scattering_validation.vtk",
    )
    config =
        CoarseSphereRCSGateHarness.parse_coarse_sphere_rcs_gate_config([
            "--mesh=$(mesh_path)",
            "--final-time=0.125",
            "--rcs-theta-count=9",
            "--rcs-phi-degrees=0,90",
            "--min-max-rcs=1e-10",
            "--max-absolute-normalized-error=0.5",
            "--max-rms-relative-error=0.25",
            "--max-db-error=3.0",
            "--mie-terms=12",
        ])

    @test config.min_max_rcs == 1.0e-10
    @test config.max_absolute_normalized_error == 0.5
    @test config.max_rms_relative_error == 0.25
    @test config.max_db_error == 3.0
    @test config.mie_terms == 12
    @test config.scattering_config.mesh == abspath(mesh_path)
    @test config.scattering_config.final_time == 0.125
    @test config.scattering_config.rcs_enabled
    @test config.scattering_config.rcs_theta_count == 9
    @test config.scattering_config.rcs_phi_degrees == [0.0, 90.0]
end

@testset "Coarse sphere RCS Mie comparison" begin
    radius = 1.0
    wavelength = 1.0
    wavenumber = 2.0 * pi / wavelength
    size_parameter = wavenumber * radius
    terms = CoarseSphereRCSGateHarness.automatic_mie_terms(size_parameter)
    electric, magnetic =
        CoarseSphereRCSGateHarness.pec_mie_coefficients(size_parameter, terms)

    forward_rcs = CoarseSphereRCSGateHarness.exact_pec_sphere_rcs(
        0.0,
        0.0,
        wavenumber,
        electric,
        magnetic,
    )
    side_rcs = CoarseSphereRCSGateHarness.exact_pec_sphere_rcs(
        90.0,
        0.0,
        wavenumber,
        electric,
        magnetic,
    )

    @test isfinite(forward_rcs)
    @test isfinite(side_rcs)
    @test forward_rcs > 0.0
    @test side_rcs > 0.0
end

@testset "Coarse sphere RCS gate summary" begin
    directory = mktempdir()
    diagnostics_dir = joinpath(directory, "diagnostics")
    mkpath(diagnostics_dir)

    radius = 1.0
    wavelength = 1.0
    wavenumber = 2.0 * pi / wavelength
    terms = CoarseSphereRCSGateHarness.automatic_mie_terms(wavenumber * radius)
    electric, magnetic =
        CoarseSphereRCSGateHarness.pec_mie_coefficients(wavenumber * radius, terms)
    theta_values = (0.0, 45.0, 90.0)

    open(joinpath(diagnostics_dir, "rcs.csv"), "w") do io
        println(
            io,
            "theta_degrees,phi_degrees,dir_x,dir_y,dir_z," *
            "abs_E_infinity,abs_Etheta,abs_Ephi,rcs,rcs_db," *
            "real_Etheta,imag_Etheta,real_Ephi,imag_Ephi,samples," *
            "window_sum,rcs_start_time,rcs_end_time",
        )
        for theta in theta_values
            rcs = CoarseSphereRCSGateHarness.exact_pec_sphere_rcs(
                theta,
                0.0,
                wavenumber,
                electric,
                magnetic,
            )
            rcs_db = CoarseSphereRCSGateHarness.db(rcs)
            println(
                io,
                join(
                    (
                        theta,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        sqrt(rcs / (4.0 * pi)),
                        sqrt(rcs / (4.0 * pi)),
                        0.0,
                        rcs,
                        rcs_db,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4,
                        2.0,
                        0.0,
                        1.0,
                    ),
                    ',',
                ),
            )
        end
    end

    config =
        CoarseSphereRCSGateHarness.CoarseSphereRCSGateConfig(
            scattering_config =
                CoarseSphereRCSGateHarness.MetallicSphereScatteringConfig(
                    output_dir = directory,
                    radius = radius,
                    wavelength = wavelength,
                ),
            min_max_rcs = 0.0,
            max_absolute_normalized_error = 1.0e-12,
            max_rms_relative_error = 1.0e-12,
            max_db_error = 1.0e-12,
            mie_terms = terms,
        )
    summary = (
        output_dir = directory,
        final_time = 1.0,
        steps = 10,
        dt = 0.1,
        final_scattered_energy = 0.5,
    )

    record = CoarseSphereRCSGateHarness.coarse_sphere_rcs_record(
        summary,
        config,
    )
    comparison_path =
        CoarseSphereRCSGateHarness.write_coarse_rcs_comparison_csv(
            record.comparison_path,
            record.comparison,
        )
    csv_path =
        CoarseSphereRCSGateHarness.write_coarse_sphere_rcs_summary_csv(
            joinpath(directory, "coarse_sphere_rcs_summary.csv"),
            record,
        )
    json_path =
        CoarseSphereRCSGateHarness.write_coarse_sphere_rcs_summary_json(
            joinpath(directory, "coarse_sphere_rcs_summary.json"),
            record,
        )

    @test record.status == "PASS"
    @test record.metrics.rows == length(theta_values)
    @test record.metrics.positive_rows == length(theta_values)
    @test record.metrics.rms_relative_error <= 1.0e-12
    @test isfile(comparison_path)
    @test isfile(csv_path)
    @test isfile(json_path)
    @test occursin("\"overall_status\": \"PASS\"", read(json_path, String))
end
