using Test
using DiscoGMPI

module FFDRCSHarness
include(joinpath(@__DIR__, "..", "examples", "distributed_metallic_sphere_scattering.jl"))
end

@testset "FFD scattering configuration" begin
    H = FFDRCSHarness
    @test !H.parse_config(String[]).ffd_enabled
    @test_throws ErrorException H.parse_config(["--enable-ffd"])
    config = H.parse_config(["--enable-rcs", "--enable-ffd",
                             "--ffd-freq-unit=solver_frequency", "--ffd-field-unit=solver_length"])
    @test config.ffd_enabled
    @test config.ffd_freq_unit == "solver_frequency"
    @test config.ffd_field_unit == "solver_length"
    @test !H.parse_config(["--enable-rcs", "--enable-ffd", "--disable-ffd"]).ffd_enabled
    slanted = ["--enable-rcs", "--incident-direction=1,0,0", "--incident-polarization=0,1,1"]
    @test H.parse_config(slanted).rcs_enabled
    @test_throws ArgumentError H.parse_config(vcat(slanted, ["--enable-ffd"]))
    @test_throws ArgumentError H.parse_config(["--enable-rcs", "--enable-ffd", "--ffd-field-unit=m\ninjected true"])
    @test_throws ErrorException H.parse_config(["--amplitude=NaN"])
end

@testset "RCS complex result to CSV and FFD" begin
    H = FFDRCSHarness
    config = H.MetallicSphereScatteringConfig(;
        rcs_enabled = true, ffd_enabled = true, amplitude = -2.0,
        rcs_theta_count = 3, rcs_theta_min_degrees = 0.0,
        rcs_theta_max_degrees = 180.0, rcs_phi_degrees = [0.0, 90.0],
    )
    observations = H.rcs_observation_directions(config)
    wave = IncidentPlaneWaveParameters(; amplitude = config.amplitude, phase_shift = 0.47)
    wave_values = H.resolved_wave_values(config)
    accum = ComplexF64[complex(0.1c + 0.2i, 0.3c - 0.07i) for c in 1:3, i in 1:6]
    workspace = H.RCSWorkspace(observations, H.RCSFaceSample[], accum, similar(accum),
                               12, 4.2, 0.2, 1.2, 2pi, 2pi, 1.0, abs(config.amplitude))
    result = H.finalize_rcs_results(workspace, accum)
    @test length(result.Etheta) == 6
    @test length(result.Ephi) == 6
    @test_throws DimensionMismatch H.finalize_rcs_results(workspace, zeros(ComplexF64, 3, 5))

    mktempdir() do directory
        paths = prepare_run_directory(directory)
        csv_path = H.write_rcs_csv(run_diagnostics_path(paths, "rcs.csv"), result)
        path = H.write_rcs_ffd(run_diagnostics_path(paths, "far_field.ffd"), result,
                               config, wave, wave_values; date = "2026-10-22T10:00:00Z")
        file = read_ffd(path)
        @test file.header.date == "2026-10-22T10:00:00Z"
        @test file.header.freq_unit == "Hz"
        @test file.header.angle_unit == "deg"
        @test file.header.field_unit == "m"
        @test file.header.solver == "DiscoGMPI " * string(pkgversion(DiscoGMPI))
        @test file.header.extras["time_convention"] == "exp(+i omega t)"
        block = only(file.blocks)
        @test block.sweep_order == [:freq, :pol, :theta_inc, :phi_inc, :phi_obs, :theta_obs]
        @test block.columns == [:Etheta_re, :Etheta_im, :Ephi_re, :Ephi_im]
        @test block.n_points == 6
        points = collect(each_ffd_point(block))
        csv = split.(readlines(csv_path), ',')
        @test length(first(csv)) == 18
        @test length(csv) == 7
        @test all(length(row) == length(first(csv)) for row in csv)
        incident = config.amplitude * cis(wave.phase_shift)
        for (i, sample) in enumerate(points)
            @test sample.freq == wave_values.frequency
            @test sample.pol == FFD_V
            @test sample.theta_inc == 180.0
            @test sample.phi_inc == 180.0
            @test sample.theta_obs == observations[i].theta_degrees
            @test sample.phi_obs == observations[i].phi_degrees
            @test sample.Etheta ≈ conj(result.Etheta[i] / incident) rtol = 2e-15
            @test sample.Ephi ≈ conj(result.Ephi[i] / incident) rtol = 2e-15
            row = Dict(zip(first(csv), csv[i+1]))
            expected_rcs = 4pi * (abs2(sample.Etheta) + abs2(sample.Ephi))
            @test parse(Float64, row["rcs"]) ≈ expected_rcs rtol = 3e-15
            @test parse(Float64, row["real_Etheta"]) == real(result.Etheta[i])
            @test parse(Float64, row["imag_Ephi"]) == imag(result.Ephi[i])
            @test parse(Float64, row["rcs_start_time"]) == workspace.start_time
            @test parse(Float64, row["rcs_end_time"]) == workspace.end_time
        end

        # Removing the numerical epsilon floor preserves physical source scaling.
        tiny = H.RCSFarFieldResults(observations,
                                   result.Etheta .* 1e-250, result.Ephi .* 1e-250,
                                   12, 4.2, 0.2, 1.2, 2e-250)
        tiny_path = H.write_rcs_csv(joinpath(directory, "tiny.csv"), tiny)
        tiny_csv = split.(readlines(tiny_path), ',')
        @test parse.(Float64, getindex.(tiny_csv[2:end], 9)) ≈
              parse.(Float64, getindex.(csv[2:end], 9)) rtol = 3e-15

        empty_result = H.finalize_rcs_results(workspace, accum; sample_count = 0, window_sum = 0.0)
        @test_logs (:warn, r"no nonzero windowed samples") begin
            H.write_rcs_csv(joinpath(directory, "empty.csv"), empty_result)
        end
        @test length(readlines(joinpath(directory, "empty.csv"))) == 1
        missing_path = joinpath(directory, "empty.ffd")
        @test_throws ErrorException H.write_rcs_ffd(missing_path, empty_result, config, wave, wave_values)
        @test !isfile(missing_path)
        @test_throws DimensionMismatch H.write_rcs_ffd(joinpath(directory, "wrong.ffd"), result,
            H.MetallicSphereScatteringConfig(; rcs_enabled = true, ffd_enabled = true,
                                               rcs_theta_count = 2), wave, wave_values)
        @test_throws ErrorException H.write_rcs_ffd(joinpath(directory, "bad_amplitude.ffd"),
            result, config, IncidentPlaneWaveParameters(; amplitude = 3.0), wave_values)
        @test_throws ErrorException H.write_rcs_ffd(joinpath(directory, "bad_frequency.ffd"),
            result, config, wave, (; frequency = 3.0))
        invalid_result = H.RCSFarFieldResults(observations, result.Etheta,
            result.Ephi[1:end-1], 12, 4.2, 0.2, 1.2, 2.0)
        invalid_path = joinpath(directory, "invalid.csv")
        @test_throws DimensionMismatch H.write_rcs_csv(invalid_path, invalid_result)
        @test !isfile(invalid_path)
        @test_throws DimensionMismatch H.write_rcs_ffd(joinpath(directory, "invalid.ffd"),
            invalid_result, config, wave, wave_values)

        metadata = H.config_dictionary(config; wave, wave_values,
            mesh_lower = (-1., -1., -1.), mesh_upper = (1., 1., 1.),
            rcs_start_time = 0.2, dt = 0.01, estimated_dt = 0.02,
            steps = 120, global_hmin = 0.1, nranks = 2)
        @test metadata["ffd_enabled"]
        @test metadata["ffd_freq_unit"] == "Hz"
        @test metadata["ffd_angle_unit"] == "deg"
        @test metadata["ffd_field_unit"] == "m"
    end
end
