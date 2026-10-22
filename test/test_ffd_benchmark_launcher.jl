using Test
using TOML

include(joinpath(@__DIR__, "..", "tools", "ffd", "launch_sphere_benchmark.jl"))

@testset "FFD sphere benchmark preparation" begin
    mktempdir() do directory
        reference = joinpath(directory, "reference")
        for n in 1:3
            mkpath(joinpath(reference, "freq$n"))
            for prefix in ("mie", "numeric"), cut in ("phi", "theta")
                touch(joinpath(reference, "freq$n", "$(prefix)_$cut.ffd"))
            end
        end
        output = joinpath(directory, "campaign")
        arguments = ["--output-dir", output, "--emagpie-dir", reference, "--dry-run", "--no-plots"]
        campaign = FFDSphereBenchmark.main(arguments; io=IOBuffer())
        @test isfile(joinpath(output, "plan.toml"))
        manifest = TOML.parsefile(joinpath(output, "campaign.toml"))
        @test manifest["comparison"]["required_sources"] == ["Mie", "EMAGPIE", "DiscoGMPI"]
        @test manifest["comparison"]["output_dir"] == joinpath(output, "comparison")
        @test manifest["sources"][3]["frequency_scale"] == 299_792_458.0
        @test manifest["sources"][3]["expected_transform"] == "identity"
        @test length(manifest["sources"][3]["files"]) == 6
        @test !isdir(joinpath(output, "meshes"))
        @test !isdir(joinpath(output, "freq1"))
        @test length(campaign.meshes) == 3
        @test length(campaign.runs) == 6
        @test length(campaign.plan["input_sha256"]) == 12
        @test length(campaign.plan["source_sha256"]) == 7
        @test all(length(value) == 64 for value in values(campaign.plan["input_sha256"]))
        @test all(length(value) == 64 for value in values(campaign.plan["source_sha256"]))
        @test haskey(campaign.plan["source_sha256"], abspath(joinpath(@__DIR__, "..", "tools", "ffd", "launch_sphere_benchmark.jl")))
        for (n, wavelength) in enumerate(FFDSphereBenchmark.WAVELENGTHS)
            phi_args = campaign.runs[2n-1].command.exec
            theta_args = campaign.runs[2n].command.exec
            @test "--amplitude=1" in phi_args
            @test "--incident-direction=1,0,0" in phi_args
            @test "--incident-polarization=0,0,1" in phi_args
            @test "--wavelength=$wavelength" in phi_args
            @test "--rcs-theta-count=1" in phi_args
            @test "--rcs-theta-count=250" in theta_args
            @test "--rcs-phi-degrees=0,180" in theta_args
            phi_list = only(filter(arg -> startswith(arg, "--rcs-phi-degrees="), phi_args))
            angles = parse.(Float64, split(split(phi_list, '='; limit=2)[2], ','))
            @test length(angles) == 500
            @test first(angles) == 0.0 && last(angles) == 180.0
            theta_min = parse(Float64, split(only(filter(arg -> startswith(arg, "--rcs-theta-min-degrees="), theta_args)), '='; limit=2)[2])
            theta = collect(range(theta_min, 90.0; length=250))
            expected = collect(range(-90.0, 90.0; length=500))
            @test [-reverse(theta); theta] ≈ expected
        end
        @test FFDSphereBenchmark.main(arguments; io=IOBuffer()).plan == campaign.plan
        changed_output = joinpath(directory, "changed_inputs")
        changed_arguments = ["--output-dir=$changed_output", "--emagpie-dir=$reference", "--no-plots"]
        FFDSphereBenchmark.main(changed_arguments; io=IOBuffer())
        open(joinpath(reference, "freq1", "mie_phi.ffd"), "w") do io
            write(io, "changed reference contents")
        end
        @test_throws ErrorException FFDSphereBenchmark.main(changed_arguments; io=IOBuffer())
        # Restore the original fixture so the existing-output check below tests
        # protection of solver output rather than the input fingerprint guard.
        open(joinpath(reference, "freq1", "mie_phi.ffd"), "w") do io
        end
        mkpath(campaign.runs[1].path)
        @test_throws ErrorException FFDSphereBenchmark.main(arguments; io=IOBuffer())
        @test_throws ErrorException FFDSphereBenchmark.parse_arguments(["--output-dir=$output", "--emagpie-dir=$reference", "--periods=3", "--accumulation-periods=4"])
        @test_throws ErrorException FFDSphereBenchmark.parse_arguments(["--output-dir=$output", "--emagpie-dir=$reference", "--cfl=NaN"])
    end
end
