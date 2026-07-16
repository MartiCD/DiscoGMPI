using Test

module ScatteredExHyFieldsGateHarness
include(joinpath(@__DIR__, "..", "examples", "validate_scattered_ex_hy_fields.jl"))
end

@testset "Scattered Ex/Hy field gate parsing" begin
    mesh_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "meshes",
        "metallic_sphere_scattering_validation.vtk",
    )
    config =
        ScatteredExHyFieldsGateHarness.parse_scattered_ex_hy_fields_gate_config([
            "--mesh=$(mesh_path)",
            "--final-time=0.125",
            "--paraview-every=7",
            "--min-final-scattered-energy=1e-9",
            "--allow-missing-paraview",
        ])

    @test config.min_final_scattered_energy == 1e-9
    @test !config.require_paraview
    @test config.scattering_config.mesh == abspath(mesh_path)
    @test config.scattering_config.final_time == 0.125
    @test config.scattering_config.paraview_every == 7
end

@testset "Scattered Ex/Hy field gate PVD checks" begin
    directory = mktempdir()
    fields_dir = joinpath(directory, "fields")
    mkpath(fields_dir)
    write(
        joinpath(directory, "fields.pvd"),
        """
        <VTKFile type="Collection">
          <Collection>
            <DataSet timestep="0.0" file="fields/fields_step00000000.pvtu"/>
            <DataSet timestep="1.0" file="fields/fields_step00000001.pvtu"/>
          </Collection>
        </VTKFile>
        """,
    )
    write(
        joinpath(fields_dir, "fields_step00000001.pvtu"),
        """
        <VTKFile type="PUnstructuredGrid">
          <PUnstructuredGrid>
            <PPointData>
              <PDataArray type="Float64" Name="E_scat" NumberOfComponents="3"/>
              <PDataArray type="Float64" Name="H_scat" NumberOfComponents="3"/>
            </PPointData>
          </PUnstructuredGrid>
        </VTKFile>
        """,
    )

    summary = (
        output_dir = directory,
        diagnostics_path = joinpath(directory, "diagnostics", "scattering.csv"),
        pec_residual_path = joinpath(directory, "diagnostics", "pec.csv"),
        final_time = 1.0,
        steps = 10,
        dt = 0.1,
        final_scattered_energy = 1.0e-6,
    )
    config =
        ScatteredExHyFieldsGateHarness.ScatteredExHyFieldsGateConfig(
            scattering_config =
                ScatteredExHyFieldsGateHarness.MetallicSphereScatteringConfig(),
            min_final_scattered_energy = 1.0e-12,
            require_paraview = true,
        )
    record =
        ScatteredExHyFieldsGateHarness.scattered_ex_hy_fields_record(
            summary,
            config,
        )

    @test record.status == "PASS"
    @test record.final_dataset == "fields/fields_step00000001.pvtu"
    @test record.has_e_scat
    @test record.has_h_scat
    @test record.recommended_ex_component == "E_scat_x"
    @test record.recommended_hy_component == "H_scat_y"
end
