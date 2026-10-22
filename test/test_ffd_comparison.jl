using Test
using TOML
using LinearAlgebra

include(joinpath(@__DIR__, "..", "tools", "ffd", "compare_far_fields.jl"))
const FFDC = FFDComparison

function comparison_sample(theta, phi; frequency=1.0e8, Etheta=1 + 2im,
                           Ephi=0.3 - 0.4im, theta_inc=pi / 2, phi_inc=pi,
                           pol=FFDC.FFD_V)
    return FFDC.FFDSample(frequency, pol, theta_inc, phi_inc, theta, phi, Etheta, Ephi)
end

function comparison_write(path, points; angle_unit="rad", freq_unit="Hz", field_unit="m")
    header = FFDC.FFDHeader(; angle_unit, freq_unit, field_unit)
    FFDC.write_ffd(path, header, [FFDC.FFDExplicitBlock(points)])
    return path
end

function comparison_source(path; name="synthetic", kwargs...)
    return merge(Dict{String,Any}("name" => name, "files" => [basename(path)]),
                 Dict{String,Any}(string(key) => value for (key, value) in kwargs))
end

function comparison_load(path; kwargs...)
    provenance = Dict{String,Any}[]
    points = FFDC.source_samples(comparison_source(path; kwargs...), dirname(path), provenance)
    return points, provenance
end

function comparison_error(f)
    return try
        f()
        nothing
    catch exception
        exception
    end
end

function comparison_physical_points(; frequency=1.0e8, field_transform=identity,
                                    theta_angles=(-pi / 2, 0.0, pi / 2),
                                    phi_angles=(0.0, pi / 2, pi))
    points = FFDC.FFDSample[]
    # Project a fixed, genuinely complex Cartesian vector onto each tangent
    # plane. Shared cut endpoints consequently have identical Cartesian fields,
    # even when their signed spherical components differ.
    seed = ComplexF64[1 + 0.3im, 0.4 - 0.7im, 0.2 + 0.8im]
    for (theta, phi) in vcat([(pi / 2, p) for p in phi_angles],
                              [(t, 0.0) for t in theta_angles])
        direction, etheta, ephi = FFDC.spherical_basis(theta, phi)
        field = field_transform(seed .- dot(direction, seed) .* direction)
        push!(points, comparison_sample(theta, phi; frequency,
            Etheta=dot(etheta, field), Ephi=dot(ephi, field)))
    end
    return points
end

function comparison_fixture(directory; candidate_transform=identity)
    comparison_write(joinpath(directory, "mie.ffd"), comparison_physical_points())
    comparison_write(joinpath(directory, "emagpie.ffd"), comparison_physical_points())
    comparison_write(joinpath(directory, "disco.ffd"),
                     comparison_physical_points(; field_transform=candidate_transform))
    config = Dict{String,Any}(
        "comparison" => Dict{String,Any}("reference" => "Mie", "max_gap_rad" => 2.0,
            "output_dir" => "output", "max_complex_l2" => 1e-12, "max_rcs_l2" => 1e-12),
        "cases" => [Dict{String,Any}("name" => "freq1", "frequency_hz" => 1.0e8)],
        "sources" => [comparison_source(joinpath(directory, "mie.ffd"); name="Mie"),
                      comparison_source(joinpath(directory, "emagpie.ffd"); name="EMAGPIE"),
                      comparison_source(joinpath(directory, "disco.ffd"); name="DiscoGMPI")])
    return config
end

function comparison_config(directory, config; name="campaign.toml")
    path = joinpath(directory, name)
    open(path, "w") do io
        TOML.print(io, config; sorted=true)
    end
    return path
end

@testset "FFD comparison Cartesian geometry and explicit adaptations" begin
    mktempdir() do directory
        theta, phi = -0.7, 0.0
        signed = comparison_sample(theta, phi)
        canonical = comparison_sample(-theta, pi; Etheta=-signed.Etheta, Ephi=-signed.Ephi)
        a, _ = comparison_load(comparison_write(joinpath(directory, "signed.ffd"), [signed]))
        b, _ = comparison_load(comparison_write(joinpath(directory, "canonical.ffd"), [canonical]))
        @test only(a).direction ≈ only(b).direction atol=1e-14
        @test only(a).field ≈ only(b).field atol=1e-14
        @test FFDC.cut_angle(only(a).direction, "theta", 1e-12) ≈ theta

        # Unit conversion and periodic azimuths must preserve physical vectors.
        rad = comparison_sample(0.8, 0.4)
        deg = FFDC.FFDSample(rad.freq, rad.pol, rad2deg(rad.theta_inc), rad2deg(rad.phi_inc) + 360,
            rad2deg(rad.theta_obs), rad2deg(rad.phi_obs) - 720, rad.Etheta, rad.Ephi)
        r, _ = comparison_load(comparison_write(joinpath(directory, "radians.ffd"), [rad]))
        d, _ = comparison_load(comparison_write(joinpath(directory, "degrees.ffd"), [deg]; angle_unit="deg"))
        @test only(r).direction ≈ only(d).direction atol=1e-14
        @test only(r).field ≈ only(d).field atol=1e-14
        @test only(r).emitter ≈ only(d).emitter atol=1e-14
        @test only(r).incident_pol ≈ only(d).incident_pol atol=1e-14

        legacy_path = comparison_write(joinpath(directory, "legacy.ffd"), [rad])
        legacy_text = replace(read(legacy_path, String), "freq pol theta_inc" => "freq polarization theta_inc")
        write(legacy_path, legacy_text)
        @test_throws ArgumentError FFDC.read_source_ffd(legacy_path)
        parsed, replacements = FFDC.read_source_ffd(legacy_path; legacy_polarization_column=true)
        @test replacements == 1
        @test only(collect(FFDC.each_ffd_point(only(parsed.blocks)))).Etheta == rad.Etheta
        @test read(legacy_path, String) == legacy_text
        legacy_points, provenance = comparison_load(legacy_path; legacy_polarization_column=true)
        @test length(legacy_points) == 1
        @test only(provenance)["legacy_column_replacements"] == 1
        @test read(legacy_path, String) == legacy_text

        mislabeled = comparison_write(joinpath(directory, "legacy_units.ffd"), [rad]; field_unit="V/m")
        @test occursin("not a normalized scattering length", sprint(showerror,
            comparison_error(() -> comparison_load(mislabeled))))
        adapted, provenance = comparison_load(mislabeled; field_unit_override="m")
        @test only(adapted).field ≈ only(r).field
        @test only(provenance)["original_field_unit"] == "V/m"
        @test only(provenance)["interpreted_field_unit"] == "m"

        mhz = comparison_sample(rad.theta_obs, rad.phi_obs; frequency=100.0)
        frequency_path = comparison_write(joinpath(directory, "frequency.ffd"), [mhz]; freq_unit="MHz")
        scaled, _ = comparison_load(frequency_path)
        @test only(scaled).frequency == 1.0e8
        # An explicit scale is authoritative even for solver-defined units.
        custom = comparison_write(joinpath(directory, "solver_units.ffd"), [mhz]; freq_unit="inverse_solver_time")
        @test_throws ErrorException comparison_load(custom)
        scaled, provenance = comparison_load(custom; frequency_scale=2.0e6)
        @test only(scaled).frequency == 2.0e8
        @test only(provenance)["frequency_scale_to_hz"] == 2.0e6
        for bad in (0.0, -1.0, NaN, Inf)
            @test_throws ErrorException comparison_load(custom; frequency_scale=bad)
        end
        cm = comparison_write(joinpath(directory, "centimetres.ffd"), [rad]; field_unit="cm")
        centimetres, _ = comparison_load(cm; field_scale=[0.0, 2.0])
        @test only(centimetres).field ≈ 0.02im .* only(r).field
    end
end

@testset "FFD comparison cut coverage, matching, and interpolation" begin
    mktempdir() do directory
        path = comparison_write(joinpath(directory, "both_cuts.ffd"), comparison_physical_points())
        samples, _ = comparison_load(path)
        settings = Dict{String,Any}("max_gap_rad" => 2.0)
        incidence = first(samples)
        phi = FFDC.extract_cut(samples, 1.0e8, "phi", settings, incidence)
        theta = FFDC.extract_cut(samples, 1.0e8, "theta", settings, incidence)
        @test getproperty.(phi, :angle) ≈ [0.0, pi / 2, pi]
        @test getproperty.(theta, :angle) ≈ [-pi / 2, 0.0, pi / 2]
        @test length(phi) == length(theta) == 3 # Shared endpoints deduplicate.
        aligned, count = FFDC.align_cut(phi, phi, settings)
        @test count == 0
        @test all(norm(a.field - b.field) < 1e-14 for (a, b) in zip(aligned, phi))
        @test_throws ErrorException FFDC.extract_cut(samples, 2.0e8, "phi", settings, incidence)
        @test_throws ErrorException FFDC.extract_cut(samples, 1.0e8, "phi", Dict{String,Any}(), incidence)

        bad_incidence = FFDC.CartesianSample(incidence.frequency, incidence.direction, incidence.field,
            -incidence.emitter, incidence.incident_pol)
        @test occursin("Incidence direction mismatch", sprint(showerror,
            comparison_error(() -> FFDC.extract_cut([bad_incidence; samples], 1.0e8, "phi", settings, incidence))))
        bad_polarization = FFDC.CartesianSample(incidence.frequency, incidence.direction, incidence.field,
            incidence.emitter, -incidence.incident_pol)
        @test occursin("Incident V/H basis mismatch", sprint(showerror,
            comparison_error(() -> FFDC.extract_cut([bad_polarization; samples], 1.0e8, "phi", settings, incidence))))
        conflict = FFDC.CartesianSample(incidence.frequency, incidence.direction, 2 .* incidence.field,
            incidence.emitter, incidence.incident_pol)
        @test occursin("Conflicting duplicate directions", sprint(showerror,
            comparison_error(() -> FFDC.extract_cut([samples; conflict], 1.0e8, "phi", settings, incidence))))

        # A common Cartesian z field makes linear interpolation independently
        # calculable without spherical-basis interpolation or radial leakage.
        zfield(value) = ComplexF64[0, 0, value]
        cutpoint(angle, value) = FFDC.CutPoint(angle, [cos(angle), sin(angle), 0.0], zfield(value))
        endpoints = [cutpoint(0.0, 1 + im), cutpoint(pi / 2, 3 - im)]
        target = [cutpoint(pi / 4, 2.0)]
        @test_throws ErrorException FFDC.align_cut(endpoints, target, settings)
        linear = merge(settings, Dict("alignment" => "linear"))
        interpolated, count = FFDC.align_cut(endpoints, target, linear)
        @test count == 1
        @test only(interpolated).field ≈ zfield(2.0)
        @test abs(dot(only(interpolated).direction, only(interpolated).field)) < 1e-14
        @test occursin("extrapolate", sprint(showerror,
            comparison_error(() -> FFDC.align_cut(endpoints, [cutpoint(-0.2, 0.0)], linear))))
        @test occursin("gap exceeds", sprint(showerror,
            comparison_error(() -> FFDC.align_cut(endpoints, target, merge(linear, Dict("max_gap_rad" => 0.1))))))
        @test_throws ErrorException FFDC.align_cut(endpoints, target, Dict("alignment" => "nearest"))
    end
end

@testset "FFD comparison complex transforms and polarization invariants" begin
    direction, etheta, ephi = FFDC.spherical_basis(0.7, 0.4)
    field = (1 + 2im) .* etheta .+ (3 - 4im) .* ephi
    expected = Dict(
        "identity" => (1 + 2im, 3 - 4im),
        "conjugate" => (1 - 2im, 3 + 4im),
        "negative_identity" => (-1 - 2im, -3 + 4im),
        "negative_conjugate" => (-1 + 2im, -3 - 4im),
        "flip_theta" => (-1 - 2im, 3 - 4im),
        "flip_phi" => (1 + 2im, -3 + 4im),
        "conjugate_flip_theta" => (-1 + 2im, 3 + 4im),
        "conjugate_flip_phi" => (1 - 2im, -3 - 4im))
    for transformation in FFDC.TRANSFORMS
        transformed = FFDC.transform_field(field, direction, transformation)
        a, b = expected[transformation]
        @test transformed ≈ a .* etheta .+ b .* ephi atol=1e-14
        @test norm(transformed) ≈ norm(field)
    end
    @test_throws ErrorException FFDC.transform_field(field, direction, "guess")
    point = FFDC.CutPoint(0.0, direction, field)
    et, ep, co, cross = FFDC.field_components(point, [0.0, 0.0, 1.0])
    @test abs2(et) + abs2(ep) ≈ abs2(co) + abs2(cross)
    row = FFDC.sample_row("test", "freq1", "theta", 1e8, point, [0.0, 0.0, 1.0], -100.0)
    @test length(row) == length(FFDC.SAMPLE_COLUMNS)
    @test row[19] ≈ row[20] + row[21]
    @test row[19] ≈ row[22] + row[23]
    # The signed meridional plotting basis remains continuous through +z,
    # where projection of the incident polarization becomes singular.
    for signed_theta in (-1e-4, 0.0, 1e-4)
        pole_direction, continuous_theta, _ = FFDC.spherical_basis(signed_theta, 0.0)
        pole_point = FFDC.CutPoint(signed_theta, pole_direction, (1 + 2im) .* continuous_theta)
        _, _, pole_co, pole_cross = FFDC.field_components(pole_point, [0.0, 0.0, -1.0]; cut="theta")
        @test pole_co ≈ 1 + 2im atol=1e-13
        @test abs(pole_cross) < 1e-13
    end
end

@testset "FFD comparison campaign regression gates and reports" begin
    mktempdir() do directory
        config = comparison_fixture(directory)
        path = comparison_config(directory, config)
        report = FFDC.compare(path; plots=false)
        @test report["passed"]
        @test isempty(report["failures"])
        @test report["sources_present"] == ["Mie", "EMAGPIE", "DiscoGMPI"]
        @test length(report["comparisons"]) == 4
        @test all(result["complex_l2"] < 1e-14 && result["rcs_l2"] < 1e-14 for result in report["comparisons"])
        for name in ("samples.csv", "pairs.csv", "transformations.csv", "summary.csv", "report.toml")
            @test isfile(joinpath(directory, "output", name))
        end
        @test length(readlines(joinpath(directory, "output", "samples.csv"))) == 19
        @test length(readlines(joinpath(directory, "output", "transformations.csv"))) == 33
        @test TOML.parsefile(joinpath(directory, "output", "report.toml"))["passed"]

        defaults = deepcopy(config)
        delete!(defaults["comparison"], "max_complex_l2")
        delete!(defaults["comparison"], "max_rcs_l2")
        default_report = FFDC.compare(comparison_config(directory, defaults; name="defaults.toml");
                                      output_dir=joinpath(directory, "defaults"), plots=false)
        @test all(result["limit_complex_l2"] == 0.03 && result["limit_rcs_l2"] == 0.04 for result in default_report["comparisons"])

        # Conjugation is diagnosed, but a diagnostic must never silently change
        # the transform selected for the actual regression gate.
        comparison_write(joinpath(directory, "disco.ffd"), comparison_physical_points(; field_transform=conj))
        @test_throws ErrorException FFDC.compare(path; output_dir=joinpath(directory, "wrong_transform"), plots=false)
        failed = TOML.parsefile(joinpath(directory, "wrong_transform", "report.toml"))
        disco = filter(result -> result["source"] == "DiscoGMPI", failed["comparisons"])
        @test !failed["passed"]
        @test all(result["best_diagnostic_transform"] == "conjugate" for result in disco)
        @test all(result["expected_transform"] == "identity" && !result["passed"] for result in disco)
        @test all(result["rcs_l2"] < 1e-14 for result in disco)
        adapted = deepcopy(config)
        adapted["sources"][3]["expected_transform"] = "conjugate"
        @test FFDC.compare(comparison_config(directory, adapted; name="adapted.toml");
            output_dir=joinpath(directory, "adapted"), plots=false)["passed"]

        for transformation in ("negative_identity", "flip_theta")
            transformed_points = FFDC.FFDSample[]
            for p in comparison_physical_points()
                direction, etheta, ephi = FFDC.spherical_basis(p.theta_obs, p.phi_obs)
                field = p.Etheta .* etheta .+ p.Ephi .* ephi
                transformed = FFDC.transform_field(field, direction, transformation)
                push!(transformed_points, comparison_sample(p.theta_obs, p.phi_obs;
                    Etheta=dot(etheta, transformed), Ephi=dot(ephi, transformed)))
            end
            comparison_write(joinpath(directory, "disco.ffd"), transformed_points)
            selected = deepcopy(config)
            selected["sources"][3]["expected_transform"] = transformation
            selected_report = FFDC.compare(comparison_config(directory, selected; name="selected.toml");
                output_dir=joinpath(directory, transformation), plots=false)
            @test selected_report["passed"]
            selected_disco = filter(result -> result["source"] == "DiscoGMPI", selected_report["comparisons"])
            @test all(result["best_diagnostic_transform"] == transformation for result in selected_disco)
        end

        # A pure phase corruption preserves RCS exactly and still fails the
        # complex gate. Every report must be written before that failure.
        comparison_write(joinpath(directory, "disco.ffd"),
                         comparison_physical_points(; field_transform=field -> cis(0.3) .* field))
        failure_dir = joinpath(directory, "phase_failure")
        @test_throws ErrorException FFDC.compare(path; output_dir=failure_dir, plots=false)
        phase_report = TOML.parsefile(joinpath(failure_dir, "report.toml"))
        @test !phase_report["passed"]
        phase_disco = filter(result -> result["source"] == "DiscoGMPI", phase_report["comparisons"])
        @test all(result["rcs_l2"] < 1e-14 && result["complex_l2"] > 0.1 for result in phase_disco)
        @test all(isfile(joinpath(failure_dir, name)) for name in
                  ("samples.csv", "pairs.csv", "transformations.csv", "summary.csv", "report.toml"))
        script = joinpath(@__DIR__, "..", "tools", "ffd", "compare_far_fields.jl")
        cli_dir = joinpath(directory, "cli_failure")
        command = `$(Base.julia_cmd()) --startup-file=no $script --config $path --no-plots --output-dir $cli_dir`
        @test !success(pipeline(command; stdout=devnull, stderr=devnull))
        @test isfile(joinpath(cli_dir, "report.toml"))
        @test !TOML.parsefile(joinpath(cli_dir, "report.toml"))["passed"]
    end
end

@testset "FFD comparison rejects incomplete or invalid campaigns" begin
    mktempdir() do directory
        config = comparison_fixture(directory)
        function rejects(mutator; message=nothing)
            candidate = deepcopy(config)
            mutator(candidate)
            exception = comparison_error(() -> FFDC.compare(comparison_config(directory, candidate); plots=false))
            @test exception isa ErrorException
            isnothing(message) || @test occursin(message, sprint(showerror, exception))
        end
        rejects(candidate -> pop!(candidate["sources"]); message="Required source DiscoGMPI")
        rejects(candidate -> (candidate["sources"][3]["files"] = ["missing.ffd"]); message="Missing FFD file")
        rejects(candidate -> (candidate["cases"][1]["frequency_hz"] = 2.0e8); message="Reference is missing")
        rejects(candidate -> push!(candidate["cases"], Dict("name" => "nearby", "frequency_hz" => 1e8 + 1));
                message="Ambiguous overlapping frequency cases")
        for bad in (-1.0, NaN, Inf)
            rejects(candidate -> (candidate["comparison"]["max_complex_l2"] = bad); message="must be finite and nonnegative")
            rejects(candidate -> (candidate["comparison"]["duplicate_field_rtol"] = bad);
                    message="duplicate_field_rtol must be finite and nonnegative")
        end
        rejects(candidate -> (candidate["comparison"]["max_complex_error"] = 1e-12);
                message="Unknown global tolerance")
        rejects(candidate -> (candidate["comparison"]["max_gap_rad"] = 0.0); message="max_gap_rad must be finite and positive")
        rejects(candidate -> (candidate["comparison"]["db_floor"] = NaN); message="db_floor must be finite")
        rejects(candidate -> (candidate["sources"][3]["tolerances"] = Dict("max_rcs_l2" => -1.0));
                message="max_rcs_l2 must be finite and nonnegative")
        rejects(candidate -> (candidate["sources"][3]["tolerances"] = Dict("max_complex_error" => 1.0));
                message="Unknown source tolerance")
        rejects(candidate -> (candidate["comparison"]["max_cross_power_fraction"] = 0.0);
                message="cross_power_fraction")

        comparison_write(joinpath(directory, "disco.ffd"), comparison_physical_points(; frequency=2e8))
        rejects(identity; message="Insufficient phi cut coverage")

        # Presence of one complete cut cannot compensate for a missing second
        # cut. Phi contributes only the two shared theta endpoints here.
        comparison_write(joinpath(directory, "disco.ffd"), comparison_physical_points(; theta_angles=()))
        rejects(identity; message="Insufficient theta cut coverage")
        comparison_write(joinpath(directory, "disco.ffd"),
                         comparison_physical_points(; phi_angles=(0.1, pi / 2, pi - 0.1), theta_angles=(0.0,)))
        rejects(identity; message="Incomplete phi cut endpoints")
    end
    @test FFDC.positive_setting(Dict("limit" => 0.0), "limit", 1.0; zero=true) == 0.0
    @test_throws ErrorException FFDC.positive_setting(Dict("limit" => 0.0), "limit", 1.0)
    @test_throws ErrorException FFDC.main(["--unknown"])
    @test_throws ErrorException FFDC.main(["--config"])
    @test_throws ErrorException FFDC.main(String[])
end
