using Test

module EmptyDomainIncidentPMLGateHarness
include(joinpath(@__DIR__, "..", "examples", "validate_empty_domain_incident_pml.jl"))
end

@testset "Empty-domain incident PML gate parsing" begin
    mesh_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "meshes",
        "periodic_box_structured_nx4_ny2_nz2.vtk",
    )
    config =
        EmptyDomainIncidentPMLGateHarness.parse_empty_domain_incident_pml_config([
            "--mesh=$(mesh_path)",
            "--fluxes=centered,alternating",
            "--paraview-every=0",
            "--max-reflection-ratio=0.2",
            "--max-final-energy-ratio=0.8",
            "--max-energy-growth=1e-8",
        ])

    @test config.max_reflection_ratio == 0.2
    @test config.max_final_energy_ratio == 0.8
    @test config.max_energy_growth == 1e-8
    @test config.pml_config.mesh_paths == [abspath(mesh_path)]
    @test length(config.pml_config.fluxes) == 2
    @test config.pml_config.paraview_every == 0
end

@testset "Empty-domain incident PML gate thresholds" begin
    config =
        EmptyDomainIncidentPMLGateHarness.EmptyDomainIncidentPMLGateConfig(
            pml_config =
                EmptyDomainIncidentPMLGateHarness.PMLValidationConfig(),
            max_reflection_ratio = 1.0e-2,
            max_final_energy_ratio = 2.5e-1,
            max_energy_growth = 1.0e-10,
        )
    passing = (
        case = "passing",
        mesh = "mesh",
        mesh_path = "/tmp/mesh.vtk",
        flux = "centered",
        pml_width = 0.5,
        sigma_max = 12.0,
        sigma_degree = 2,
        central_frequency = 2.0,
        central_wavelength = 0.5,
        central_wavenumber = 4.0 * pi,
        central_elements_per_wavelength = 4.0,
        final_energy_ratio = 1.0e-1,
        reflection_ratio = 5.0e-3,
        max_left_monitor_after_reflection = 5.0e-3,
        final_total_energy = 1.0e-1,
        initial_total_energy = 1.0,
        dt = 1.0e-3,
        steps = 100,
        output_dir = "/tmp/output",
    )
    failing = merge(passing, (case = "failing", reflection_ratio = 2.0e-2))

    records =
        EmptyDomainIncidentPMLGateHarness.empty_domain_incident_pml_records(
            [passing, failing],
            config,
        )

    @test records[1].status == "PASS"
    @test records[2].status == "FAIL"
    @test records[2].message == "reflection ratio exceeds threshold"
end
