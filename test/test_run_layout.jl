using Test
using DiscoGMPI

@testset "Run layout utilities" begin
    root = mktempdir()
    paths = prepare_run_directory(root)

    @test paths.root == abspath(root)
    @test isdir(paths.config_dir)
    @test isdir(paths.input_dir)
    @test isdir(paths.diagnostics_dir)
    @test isdir(paths.fields_dir)
    @test isdir(paths.checkpoints_dir)
    @test isdir(paths.logs_dir)
    @test isdir(paths.plots_dir)
    @test isdir(paths.tables_dir)

    @test isfile(run_config_path(paths, "run_layout.toml"))
    @test isfile(joinpath(paths.root, "status.toml"))
    @test islink(joinpath(paths.root, "energy.csv"))
    @test islink(joinpath(paths.root, "quadrature_diagnostics.csv"))
    @test islink(joinpath(paths.root, "run_metadata.toml"))
    @test islink(joinpath(paths.root, "partition_metadata.csv"))

    write_resolved_run_config(paths, Dict("order" => 4, "flux" => "centered"))
    write_run_input_manifest(paths, Dict("mesh_path" => "mesh.vtk"))
    write_run_status(paths, "complete"; values = Dict("final_time" => 1.0))

    @test isfile(run_config_path(paths, "resolved_config.toml"))
    @test isfile(run_input_path(paths, "input_manifest.toml"))
    @test occursin("complete", read(joinpath(paths.root, "status.toml"), String))

    @test sanitize_run_name("Periodic Cuboid AF p=4") == "periodic-cuboid-af-p=4"
    @test isolated_run_directory(
        "/tmp/discogmpi-runs",
        "Periodic Cuboid AF p=4";
        stamp = "20260707T170000",
    ) == "/tmp/discogmpi-runs/20260707T170000_periodic-cuboid-af-p=4"
end
