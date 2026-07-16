using Test

module PECSphereBoundaryResidualGateHarness
include(joinpath(@__DIR__, "..", "examples", "validate_pec_sphere_boundary_residual.jl"))
end

@testset "PEC sphere boundary-residual gate parsing" begin
    mesh_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "meshes",
        "metallic_sphere_scattering.vtk",
    )
    config =
        PECSphereBoundaryResidualGateHarness.parse_pec_sphere_boundary_residual_gate_config([
            "--mesh=$(mesh_path)",
            "--final-time=0.1",
            "--diagnostics-every=1",
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
    @test !config.scattering_config.rcs_enabled
    @test config.scattering_config.paraview_every == 0
end

@testset "PEC sphere boundary-residual gate thresholds" begin
    config =
        PECSphereBoundaryResidualGateHarness.PECSphereBoundaryResidualGateConfig(
            scattering_config =
                PECSphereBoundaryResidualGateHarness.MetallicSphereScatteringConfig(),
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
        PECSphereBoundaryResidualGateHarness.pec_sphere_boundary_residual_record(
            passing,
            config,
        )
    fail_record =
        PECSphereBoundaryResidualGateHarness.pec_sphere_boundary_residual_record(
            failing,
            config,
        )

    @test pass_record.status == "PASS"
    @test fail_record.status == "FAIL"
    @test fail_record.message ==
          "final relative RMS residual exceeds threshold"
end
