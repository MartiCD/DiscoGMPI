using Test, MPI, DiscoGMPI

module MPIFFDRCSHarness
include(joinpath(@__DIR__, "..", "examples", "distributed_metallic_sphere_scattering.jl"))
end

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)

try
    @testset "Collective RCS/FFD output" begin
        H = MPIFFDRCSHarness
        directory = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root = 0)
        paths = prepare_run_directory(directory, comm)
        config = H.MetallicSphereScatteringConfig(; rcs_enabled = true, ffd_enabled = true,
            amplitude = -2.0, rcs_theta_count = 3, rcs_phi_degrees = [0.0, 90.0])
        wave, wave_values = H.incident_wave(config)
        observations = H.rcs_observation_directions(config)
        # Owned contributions sum to the same phasor for any rank count.
        global_current = ComplexF64[complex(c + 0.2i, 0.13c - i) for c in 1:3, i in 1:6]
        local_accum = global_current ./ nranks
        workspace = H.RCSWorkspace(observations, H.RCSFaceSample[], local_accum,
            similar(local_accum), 8, 4.0, 0.0, 1.0, 2pi, 2pi, 1.0, 2.0)
        csv_path = H.write_rcs_results(paths, comm, workspace; config, wave, wave_values)
        @test isfile(csv_path)
        if rank == 0
            file = read_ffd(run_diagnostics_path(paths, "far_field.ffd"))
            expected = H.finalize_rcs_results(workspace, global_current)
            points = collect(each_ffd_point(only(file.blocks)))
            @test length(points) == 6
            @test all(points[i].Etheta ≈ conj(expected.Etheta[i] / -2.0) for i in 1:6)
            @test all(points[i].Ephi ≈ conj(expected.Ephi[i] / -2.0) for i in 1:6)
            @test length(readdir(paths.diagnostics_dir)) == 2
        end
        MPI.Barrier(comm)
        csv_only = H.MetallicSphereScatteringConfig(; rcs_enabled = true, rcs_theta_count = 3)
        csv_paths = prepare_run_directory(joinpath(directory, "csv_only"), comm)
        H.write_rcs_results(csv_paths, comm, workspace; config = csv_only)
        @test isfile(run_diagnostics_path(csv_paths, "rcs.csv"))
        @test !isfile(run_diagnostics_path(csv_paths, "far_field.ffd"))
        # A root writer failure must propagate to every rank without a deadlock.
        bad_paths = prepare_run_directory(joinpath(directory, "bad"), comm)
        bad_config = H.MetallicSphereScatteringConfig(; rcs_enabled = true,
            ffd_enabled = true, rcs_theta_count = 2)
        @test_throws ErrorException H.write_rcs_results(bad_paths, comm, workspace;
                                                       config = bad_config, wave, wave_values)
        @test !isfile(run_diagnostics_path(bad_paths, "far_field.ffd"))
        mismatched_wave = IncidentPlaneWaveParameters(; amplitude = -2.0, wavelength = 2.0)
        @test_throws ErrorException H.write_rcs_results(bad_paths, comm, workspace;
            config, wave = mismatched_wave, wave_values)
        @test !isfile(run_diagnostics_path(bad_paths, "far_field.ffd"))
    end
finally
    MPI.Finalize()
end
