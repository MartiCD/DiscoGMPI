using Test

module PECAlmondRCSSmokeGateHarness
include(joinpath(@__DIR__, "..", "examples", "validate_pec_almond_rcs_smoke.jl"))
end

function write_fake_almond_rcs_csv(path, observations; rcs_value = 1.0)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "theta_degrees,phi_degrees,dir_x,dir_y,dir_z," *
            "abs_E_infinity,abs_Etheta,abs_Ephi,rcs,rcs_db," *
            "real_Etheta,imag_Etheta,real_Ephi,imag_Ephi,samples," *
            "window_sum,rcs_start_time,rcs_end_time",
        )
        for obs in observations
            rcs_db = rcs_value > 0.0 ? 10.0 * log10(rcs_value) : -Inf
            println(
                io,
                join(
                    (
                        obs.theta_degrees,
                        obs.phi_degrees,
                        obs.direction[1],
                        obs.direction[2],
                        obs.direction[3],
                        sqrt(abs(rcs_value) / (4.0 * pi)),
                        sqrt(abs(rcs_value) / (4.0 * pi)),
                        0.0,
                        rcs_value,
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
    return path
end

@testset "PEC almond RCS smoke gate parsing" begin
    mesh_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "meshes",
        "pec_almond_scattering_step_fx1_epw4.vtk",
    )
    config =
        PECAlmondRCSSmokeGateHarness.parse_pec_almond_rcs_smoke_gate_config([
            "--mesh=$(mesh_path)",
            "--final-time=2e-6",
            "--rcs-theta-count=5",
            "--rcs-phi-degrees=0,90",
            "--frequency-hz=7.0e9",
            "--min-max-rcs=1e-12",
            "--min-samples=2",
            "--min-window-sum=0.5",
            "--angle-tolerance-degrees=1e-7",
        ])

    @test config.min_max_rcs == 1.0e-12
    @test config.min_samples == 2
    @test config.min_window_sum == 0.5
    @test config.angle_tolerance_degrees == 1.0e-7
    @test config.scattering_config.mesh == abspath(mesh_path)
    @test config.scattering_config.final_time == 2.0e-6
    @test config.scattering_config.rcs_enabled
    @test config.scattering_config.rcs_theta_count == 5
    @test config.scattering_config.rcs_phi_degrees == [0.0, 90.0]
    @test config.scattering_config.radius == 0.0
    @test config.scattering_config.scatterer_name == "PEC almond"
    @test config.scattering_config.pec_boundary_id == 10
    @test config.scattering_config.frequency == 0.0
    @test config.scattering_config.wavelength ≈
          PECAlmondRCSSmokeGateHarness.PEC_ALMOND_C0 / 7.0e9
end

@testset "PEC almond RCS smoke gate summary" begin
    directory = mktempdir()
    scattering_config =
        PECAlmondRCSSmokeGateHarness.MetallicSphereScatteringConfig(
            output_dir = directory,
            radius = 0.0,
            scatterer_name = "PEC almond",
            pec_boundary_id = 10,
            rcs_enabled = true,
            rcs_theta_count = 3,
            rcs_phi_degrees = [0.0, 90.0],
        )
    observations =
        PECAlmondRCSSmokeGateHarness.rcs_observation_directions(scattering_config)
    write_fake_almond_rcs_csv(
        joinpath(directory, "diagnostics", "rcs.csv"),
        observations;
        rcs_value = 2.0,
    )

    config =
        PECAlmondRCSSmokeGateHarness.PECAlmondRCSSmokeGateConfig(
            scattering_config = scattering_config,
            min_max_rcs = 0.0,
            min_samples = 1,
            min_window_sum = 0.0,
            angle_tolerance_degrees = 1.0e-8,
        )
    summary = (
        output_dir = directory,
        final_time = 1.0,
        steps = 10,
        dt = 0.1,
        final_scattered_energy = 0.5,
    )

    record =
        PECAlmondRCSSmokeGateHarness.pec_almond_rcs_smoke_record(
            summary,
            config,
        )
    csv_path =
        PECAlmondRCSSmokeGateHarness.write_pec_almond_rcs_smoke_summary_csv(
            joinpath(directory, "pec_almond_rcs_smoke_summary.csv"),
            record,
        )
    json_path =
        PECAlmondRCSSmokeGateHarness.write_pec_almond_rcs_smoke_summary_json(
            joinpath(directory, "pec_almond_rcs_smoke_summary.json"),
            record,
        )

    @test record.status == "PASS"
    @test record.metrics.rows == length(observations)
    @test record.metrics.expected_rows == length(observations)
    @test record.metrics.finite_rows == length(observations)
    @test record.metrics.positive_rows == length(observations)
    @test record.metrics.missing_angle_count == 0
    @test isfile(csv_path)
    @test isfile(json_path)
    @test occursin("\"overall_status\": \"PASS\"", read(json_path, String))
end

@testset "PEC almond RCS smoke gate failures" begin
    directory = mktempdir()
    scattering_config =
        PECAlmondRCSSmokeGateHarness.MetallicSphereScatteringConfig(
            output_dir = directory,
            radius = 0.0,
            scatterer_name = "PEC almond",
            pec_boundary_id = 10,
            rcs_enabled = true,
            rcs_theta_count = 3,
            rcs_phi_degrees = [0.0],
        )
    observations =
        PECAlmondRCSSmokeGateHarness.rcs_observation_directions(scattering_config)
    write_fake_almond_rcs_csv(
        joinpath(directory, "diagnostics", "rcs.csv"),
        observations[1:(end - 1)];
        rcs_value = 2.0,
    )

    config =
        PECAlmondRCSSmokeGateHarness.PECAlmondRCSSmokeGateConfig(
            scattering_config = scattering_config,
            min_max_rcs = 0.0,
            min_samples = 1,
            min_window_sum = 0.0,
            angle_tolerance_degrees = 1.0e-8,
        )
    summary = (
        output_dir = directory,
        final_time = 1.0,
        steps = 10,
        dt = 0.1,
        final_scattered_energy = 0.5,
    )

    missing_record =
        PECAlmondRCSSmokeGateHarness.pec_almond_rcs_smoke_record(
            summary,
            config,
        )
    @test missing_record.status == "FAIL"
    @test missing_record.message ==
          "RCS CSV row count does not match requested angles"

    write_fake_almond_rcs_csv(
        joinpath(directory, "diagnostics", "rcs.csv"),
        observations;
        rcs_value = 0.0,
    )
    zero_record =
        PECAlmondRCSSmokeGateHarness.pec_almond_rcs_smoke_record(
            summary,
            config,
        )
    @test zero_record.status == "FAIL"
    @test zero_record.message == "RCS CSV contains non-finite values"
end
