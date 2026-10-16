using Test

module PECAlmondBoundaryResidualGateHarness
include(joinpath(@__DIR__, "..", "examples", "validate_pec_almond_boundary_residual.jl"))
end

@testset "PEC almond boundary-residual gate parsing" begin
    mesh_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "meshes",
        "pec_almond_scattering_step_fx1_epw4.vtk",
    )
    config =
        PECAlmondBoundaryResidualGateHarness.parse_pec_almond_boundary_residual_gate_config([
            "--mesh=$(mesh_path)",
            "--final-time=0.1",
            "--diagnostics-every=1",
            "--paraview-every=25",
            "--frequency-hz=7.0e9",
            "--max-final-relative-rms=0.25",
            "--max-final-l2=10.0",
            "--max-final-relative-max=0.5",
            "--max-sampled-relative-rms=1.0",
        ])

    @test config.max_final_relative_rms == 0.25
    @test config.max_final_l2 == 10.0
    @test config.max_final_relative_max == 0.5
    @test config.max_sampled_relative_rms == 1.0
    @test config.scattering_config.mesh == abspath(mesh_path)
    @test config.scattering_config.final_time == 0.1
    @test config.scattering_config.diagnostics_every == 1
    @test config.scattering_config.paraview_every == 25
    @test !config.scattering_config.rcs_enabled
    @test config.scattering_config.radius == 0.0
    @test config.scattering_config.scatterer_name == "PEC almond"
    @test config.scattering_config.pec_boundary_id == 10
    @test config.scattering_config.frequency == 0.0
    @test config.scattering_config.wavelength ≈
          PECAlmondBoundaryResidualGateHarness.PEC_ALMOND_C0 / 7.0e9
end

@testset "PEC almond boundary-residual gate thresholds" begin
    config =
        PECAlmondBoundaryResidualGateHarness.PECAlmondBoundaryResidualGateConfig(
            scattering_config =
                PECAlmondBoundaryResidualGateHarness.MetallicSphereScatteringConfig(
                    radius = 0.0,
                    scatterer_name = "PEC almond",
                    pec_boundary_id = 10,
                ),
            max_final_relative_rms = 0.1,
            max_final_l2 = 2.0,
            max_final_relative_max = 0.25,
            max_sampled_relative_rms = 1.0,
        )
    passing = (
        output_dir = "/tmp/output",
        diagnostics_path = "/tmp/output/diagnostics/scattering_diagnostics.csv",
        pec_residual_path = "/tmp/output/diagnostics/pec_boundary_residual.csv",
        final_time = 1.0,
        steps = 10,
        dt = 0.1,
        final_scattered_energy = 0.5,
        final_pec_boundary_residual_l2 = 1.0,
        final_pec_boundary_residual_rms = 0.05,
        final_pec_boundary_residual_relative_rms = 0.05,
        final_pec_boundary_residual_max_pointwise = 0.2,
        final_pec_boundary_residual_relative_max_pointwise = 0.2,
        max_sampled_pec_boundary_residual_l2 = 1.2,
        max_sampled_pec_boundary_residual_relative_rms = 0.8,
        max_sampled_pec_boundary_residual_pointwise = 0.3,
    )
    failing = merge(
        passing,
        (final_pec_boundary_residual_relative_rms = 0.2,),
    )

    pass_record =
        PECAlmondBoundaryResidualGateHarness.pec_almond_boundary_residual_record(
            passing,
            config,
        )
    fail_record =
        PECAlmondBoundaryResidualGateHarness.pec_almond_boundary_residual_record(
            failing,
            config,
        )

    @test pass_record.status == "PASS"
    @test pass_record.message == "PEC almond boundary residual gate passed"
    @test fail_record.status == "FAIL"
    @test fail_record.message ==
          "final relative RMS residual exceeds threshold"
end
