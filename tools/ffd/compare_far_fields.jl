#!/usr/bin/env julia

"""Solver-independent FFD comparison. Run with --help for the command-line interface."""
module FFDComparison

using Dates, LinearAlgebra, Printf, SHA, TOML

# Load exactly the production codec, without loading the DG solver or MPI.
include(joinpath(@__DIR__, "..", "..", "src", "FFDFormat.jl"))

const TRANSFORMS = ("identity", "conjugate", "negative_identity", "negative_conjugate",
                    "flip_theta", "flip_phi", "conjugate_flip_theta", "conjugate_flip_phi")
const LIMIT_KEYS = ("max_complex_l2", "max_rcs_l2", "max_normalized_rcs_error", "max_db_error", "max_cross_power_fraction")

struct CartesianSample
    frequency::Float64
    direction::Vector{Float64}
    field::Vector{ComplexF64}
    emitter::Vector{Float64}
    incident_pol::Vector{Float64}
end

struct CutPoint
    angle::Float64
    direction::Vector{Float64}
    field::Vector{ComplexF64}
end

"""Spherical basis evaluated at the original angles, including signed theta."""
function spherical_basis(theta, phi)
    st, ct = sincos(theta)
    sp, cp = sincos(phi)
    return ([st * cp, st * sp, ct], [ct * cp, ct * sp, -st], [-sp, cp, 0.0])
end

function canonical_angles(direction)
    theta = atan(hypot(direction[1], direction[2]), direction[3])
    phi = hypot(direction[1], direction[2]) < 1e-14 ? 0.0 : mod(atan(direction[2], direction[1]), 2pi)
    return theta, phi
end

"""Apply a named diagnostic transform in the common canonical spherical basis."""
function transform_field(field, direction, name)
    name in TRANSFORMS || error("Unknown complex transform: $name")
    theta, phi = canonical_angles(direction)
    _, etheta, ephi = spherical_basis(theta, phi)
    a, b = dot(etheta, field), dot(ephi, field)
    if name in ("conjugate", "negative_conjugate", "conjugate_flip_theta", "conjugate_flip_phi")
        a, b = conj(a), conj(b)
    end
    name in ("negative_identity", "negative_conjugate") && ((a, b) = (-a, -b))
    name in ("flip_theta", "conjugate_flip_theta") && (a = -a)
    name in ("flip_phi", "conjugate_flip_phi") && (b = -b)
    return a .* etheta .+ b .* ephi
end

function positive_setting(dict, key, default; zero=false)
    value = Float64(get(dict, key, default))
    isfinite(value) && (zero ? value >= 0 : value > 0) || error("$key must be finite and $(zero ? "nonnegative" : "positive")")
    return value
end

"""Adapt only the explicitly requested legacy column name; always parse with read_ffd."""
function read_source_ffd(path; legacy_polarization_column=false)
    legacy_polarization_column || return read_ffd(path; lazy=false), 0
    return mktemp() do temporary, io
        in_data = false
        table_header = false
        changed = 0
        for line in eachline(path)
            stripped = strip(line)
            if stripped == "[DATA]"
                in_data = true
                table_header = true
            elseif stripped == "[END_DATA]"
                in_data = false
            elseif in_data && table_header && !isempty(stripped) && !startswith(stripped, "#")
                tokens = split(stripped)
                count(==("polarization"), tokens) <= 1 || error("Duplicate legacy polarization column in $path")
                changed += count(==("polarization"), tokens)
                line = join(replace(tokens, "polarization" => "pol"), " ")
                table_header = false
            end
            println(io, line)
        end
        flush(io)
        return read_ffd(temporary; lazy=false), changed
    end
end

function source_samples(source, base, provenance)
    name = String(source["name"])
    raw_paths = get(source, "files", String[])
    isempty(raw_paths) && error("Source $name requires a nonempty files array")
    angle_factors = Dict("rad" => 1.0, "deg" => pi / 180)
    frequency_factors = Dict("Hz" => 1.0, "kHz" => 1e3, "MHz" => 1e6, "GHz" => 1e9)
    length_factors = Dict("m" => 1.0, "cm" => 1e-2, "mm" => 1e-3)
    scale_parts = get(source, "field_scale", [1.0, 0.0])
    length(scale_parts) == 2 || error("Source $name field_scale must be [real, imaginary]")
    scale = ComplexF64(scale_parts...)
    isfinite(scale) && !iszero(scale) || error("Source $name field_scale must be finite and nonzero")
    samples = CartesianSample[]
    for raw_path in raw_paths
        path = abspath(joinpath(base, raw_path))
        isfile(path) || error("Missing FFD file for $name: $path")
        ffd, aliases = read_source_ffd(path; legacy_polarization_column=get(source, "legacy_polarization_column", false))
        header = ffd.header
        haskey(angle_factors, header.angle_unit) || error("Unsupported angle_unit $(header.angle_unit) in $path")
        angular_scale = angle_factors[header.angle_unit]
        frequency_scale = if haskey(source, "frequency_scale")
            positive_setting(source, "frequency_scale", 1.0)
        else
            haskey(frequency_factors, header.freq_unit) || error("$name needs explicit frequency_scale for unit $(header.freq_unit)")
            frequency_factors[header.freq_unit]
        end
        unit = String(get(source, "field_unit_override", header.field_unit))
        haskey(length_factors, unit) || error("$name field_unit $(header.field_unit) is not a normalized scattering length; supply a justified field_unit_override")
        field_scale = scale * length_factors[unit]
        count_before = length(samples)
        for block in ffd.blocks, point in each_ffd_point(block)
            theta, phi = angular_scale * point.theta_obs, angular_scale * point.phi_obs
            direction, etheta, ephi = spherical_basis(theta, phi)
            emitter, ei_theta, ei_phi = spherical_basis(angular_scale * point.theta_inc, angular_scale * point.phi_inc)
            incident_pol = point.pol == FFD_V ? ei_theta : ei_phi
            field = field_scale .* (point.Etheta .* etheta .+ point.Ephi .* ephi)
            frequency = point.freq * frequency_scale
            frequency > 0 && isfinite(frequency) && all(isfinite, field) || error("Invalid scaled sample in $path")
            push!(samples, CartesianSample(frequency, direction, field, emitter, incident_pol))
        end
        push!(provenance, Dict("source" => name, "path" => path, "sha256" => bytes2hex(open(sha256, path)),
            "points" => length(samples) - count_before, "legacy_column_replacements" => aliases,
            "original_freq_unit" => header.freq_unit, "frequency_scale_to_hz" => frequency_scale,
            "original_field_unit" => header.field_unit, "interpreted_field_unit" => unit,
            "field_scale_real" => real(field_scale), "field_scale_imag" => imag(field_scale),
            "time_convention_metadata" => get(header.extras, "time_convention", "unspecified")))
    end
    return samples
end

function cut_angle(direction, cut, tolerance)
    x, y, z = direction
    if cut == "phi"
        abs(z) <= tolerance && y >= -tolerance || return nothing
        return clamp(atan(max(y, 0.0), x), 0.0, pi)
    elseif cut == "theta"
        abs(y) <= tolerance && z >= -tolerance || return nothing
        return clamp(atan(x, max(z, 0.0)), -pi / 2, pi / 2)
    end
    error("Unknown cut $cut; expected phi or theta")
end

"""Select a cut geometrically, then deduplicate shared endpoints with consistent fields."""
function extract_cut(samples, frequency, cut, settings, incidence)
    tolerance = positive_setting(settings, "direction_atol", 2e-6)
    freq_rtol = positive_setting(settings, "frequency_rtol", 1e-6)
    duplicate_tolerance = positive_setting(settings, "duplicate_field_rtol", 1e-4; zero=true)
    points = CutPoint[]
    for sample in samples
        isapprox(sample.frequency, frequency; rtol=freq_rtol, atol=0.0) || continue
        norm(sample.emitter - incidence.emitter) <= tolerance || error("Incidence direction mismatch at frequency $frequency")
        norm(sample.incident_pol - incidence.incident_pol) <= tolerance || error("Incident V/H basis mismatch at frequency $frequency; normalize source incidence explicitly")
        angle = cut_angle(sample.direction, cut, tolerance)
        isnothing(angle) || push!(points, CutPoint(angle, sample.direction, sample.field))
    end
    sort!(points; by=p -> p.angle)
    unique_points = CutPoint[]
    for point in points
        if !isempty(unique_points) && abs(point.angle - unique_points[end].angle) <= tolerance
            previous = unique_points[end]
            scale = max(norm(previous.field), norm(point.field), eps(Float64))
            norm(point.field - previous.field) <= duplicate_tolerance * scale ||
                error("Conflicting duplicate directions in $cut cut at angle $(point.angle)")
        else
            push!(unique_points, point)
        end
    end
    length(unique_points) >= 3 || error("Insufficient $cut cut coverage at frequency $frequency: $(length(unique_points)) points")
    low, high = cut == "phi" ? (0.0, pi) : (-pi / 2, pi / 2)
    abs(first(unique_points).angle - low) <= tolerance && abs(last(unique_points).angle - high) <= tolerance ||
        error("Incomplete $cut cut endpoints at frequency $frequency")
    maximum(diff([p.angle for p in unique_points])) <= positive_setting(settings, "max_gap_rad", 0.05) ||
        error("$cut cut sampling exceeds max_gap_rad at frequency $frequency")
    return unique_points
end

function align_cut(candidate, reference, settings)
    mode = String(get(settings, "alignment", "exact"))
    mode in ("exact", "linear") || error("alignment must be exact or linear")
    tolerance = positive_setting(settings, "direction_atol", 2e-6)
    max_gap = positive_setting(settings, "max_gap_rad", 0.05)
    angles = [p.angle for p in candidate]
    aligned = CutPoint[]
    interpolated = 0
    for target in reference
        j = searchsortedfirst(angles, target.angle)
        nearby = clamp.([j - 1, j], 1, length(candidate))
        distances = [norm(candidate[k].direction - target.direction) for k in nearby]
        best = nearby[argmin(distances)]
        field = if minimum(distances) <= tolerance
            candidate[best].field
        else
            mode == "linear" || error("Unmatched Cartesian direction at angle $(target.angle); use matching samples or explicitly enable linear alignment")
            1 < j <= length(candidate) || error("Interpolation would extrapolate at angle $(target.angle)")
            a, b = candidate[j - 1], candidate[j]
            b.angle - a.angle <= max_gap || error("Interpolation gap exceeds max_gap_rad")
            weight = (target.angle - a.angle) / (b.angle - a.angle)
            interpolated += 1
            (1 - weight) .* a.field .+ weight .* b.field
        end
        # Suppress the radial interpolation/rounding residual in the common basis.
        field = field .- dot(target.direction, field) .* target.direction
        push!(aligned, CutPoint(target.angle, target.direction, field))
    end
    return aligned, interpolated
end

function field_components(point, incident_pol; cut=nothing)
    theta, phi = canonical_angles(point.direction)
    _, etheta, ephi = spherical_basis(theta, phi)
    anchor = if cut == "theta"
        spherical_basis(point.angle, 0.0)[2]
    elseif cut == "phi"
        spherical_basis(pi / 2, point.angle)[2]
    else
        etheta
    end
    anchor = anchor .- dot(point.direction, anchor) .* point.direction
    anchor ./= norm(anchor)
    co = incident_pol .- dot(point.direction, incident_pol) .* point.direction
    if norm(co) < 1e-6
        # At a projection singularity use the cut's continuous meridional gauge.
        co = anchor
    else
        co ./= norm(co)
        # Keep the complex plotting gauge continuous across signed theta=0.
        dot(co, anchor) < 0 && (co = -co)
    end
    cross_pol = cross(point.direction, co)
    return dot(etheta, point.field), dot(ephi, point.field), dot(co, point.field), dot(cross_pol, point.field)
end

function regression_metrics(candidate, reference, settings, radius)
    a = [p.field for p in candidate]
    b = [p.field for p in reference]
    denominator = sum(sum(abs2, v) for v in b)
    denominator > 0 || error("Reference complex field is identically zero")
    sigma = [4pi * sum(abs2, v) for v in a]
    exact = [4pi * sum(abs2, v) for v in b]
    scale = pi * radius^2
    floor_power = 10.0^(Float64(get(settings, "db_floor", -100.0)) / 10)
    null_floor = positive_setting(settings, "null_relative_floor", 1e-6; zero=true) * maximum(exact)
    active = findall(>(max(null_floor, floor_power)), exact)
    isempty(active) && error("No reference samples above the configured null floor")
    return Dict{String,Any}(
        "complex_l2" => sqrt(sum(sum(abs2, a[i] - b[i]) for i in eachindex(a)) / denominator),
        "rcs_l2" => norm(sigma - exact) / norm(exact),
        "max_normalized_rcs_error" => maximum(abs.(sigma - exact)) / scale,
        "max_db_error" => maximum(abs(10log10(max(sigma[i], floor_power) / max(exact[i], floor_power))) for i in active),
        "db_active_points" => length(active),
        "reference_points" => length(reference))
end

function write_csv(path, columns, rows)
    quote_cell(value) = begin
        string_value = value isa AbstractFloat ? @sprintf("%.16e", value) : string(value)
        occursin(r"[,\"\r\n]", string_value) ? "\"" * replace(string_value, "\"" => "\"\"") * "\"" : string_value
    end
    open(path, "w") do io
        println(io, join(columns, ','))
        for row in rows
            println(io, join(quote_cell.(row), ','))
        end
    end
end

function sample_row(source, case, cut, frequency, point, incident_pol, db_floor)
    theta, phi = canonical_angles(point.direction)
    et, ep, co, cr = field_components(point, incident_pol; cut=cut)
    st, sp, sc, sx = 4pi .* abs2.((et, ep, co, cr))
    db(sigma) = max(db_floor, 10log10(max(sigma, floatmin(Float64))))
    return (source, case, cut, frequency, point.angle, theta, phi, point.direction...,
            real(et), imag(et), real(ep), imag(ep), real(co), imag(co), real(cr), imag(cr),
            st + sp, st, sp, sc, sx, db(st + sp), db(sc), db(sx))
end

const SAMPLE_COLUMNS = ["source", "case", "cut", "freq_hz", "angle_rad", "theta_rad", "phi_rad",
    "rx", "ry", "rz", "Etheta_re", "Etheta_im", "Ephi_re", "Ephi_im", "Eco_re", "Eco_im",
    "Ecross_re", "Ecross_im", "sigma_total_m2", "sigma_theta_m2", "sigma_phi_m2", "sigma_co_m2", "sigma_cross_m2",
    "total_dbsm", "co_dbsm", "cross_dbsm"]

"""Run a TOML campaign and write all diagnostics before evaluating its regression gates."""
function compare(config_path; output_dir=nothing, plots=true, python="python3")
    config_path = abspath(config_path)
    config = TOML.parsefile(config_path)
    settings = get(config, "comparison", Dict{String,Any}())
    base = dirname(config_path)
    output = isnothing(output_dir) ? abspath(joinpath(base, get(settings, "output_dir", "comparison_output"))) : abspath(output_dir)
    mkpath(output)
    radius = positive_setting(settings, "radius_m", 1.0)
    reference_name = String(get(settings, "reference", "Mie"))
    sources = get(config, "sources", Any[])
    names = String[String(source["name"]) for source in sources]
    length(unique(names)) == length(names) || error("Source names must be unique")
    reference_name in names || error("Reference $reference_name is missing")
    length(names) >= 2 || error("Comparison needs at least two sources")
    for required in get(settings, "required_sources", ["Mie", "EMAGPIE", "DiscoGMPI"])
        required in names || error("Required source $required is missing; partial campaigns must explicitly set required_sources")
    end
    cases = get(config, "cases", Any[])
    isempty(cases) && error("A nonempty cases array is required")
    case_names = String[String(case["name"]) for case in cases]
    length(unique(case_names)) == length(case_names) || error("Case names must be unique")
    frequencies = [positive_setting(case, "frequency_hz", 0.0) for case in cases]
    frequency_rtol = positive_setting(settings, "frequency_rtol", 1e-6)
    for i in eachindex(frequencies), j in 1:i-1
        isapprox(frequencies[i], frequencies[j]; rtol=2frequency_rtol, atol=0.0) && error("Ambiguous overlapping frequency cases")
    end
    for key in LIMIT_KEYS
        haskey(settings, key) && positive_setting(settings, key, 0.0; zero=true)
    end
    for key in keys(settings)
        startswith(key, "max_") && key != "max_gap_rad" && !(key in LIMIT_KEYS) && error("Unknown global tolerance $key")
    end
    positive_setting(settings, "duplicate_field_rtol", 1e-4; zero=true)
    for source in sources, (key, value) in get(source, "tolerances", Dict{String,Any}())
        key in LIMIT_KEYS || error("Unknown source tolerance $key")
        positive_setting(Dict(key => value), key, 0.0; zero=true)
    end
    db_floor = Float64(get(settings, "db_floor", -100.0))
    isfinite(db_floor) || error("db_floor must be finite")
    provenance = Dict{String,Any}[]
    datasets = Dict(String(source["name"]) => source_samples(source, base, provenance) for source in sources)
    summaries, diagnostics = Dict{String,Any}[], Any[]
    samples_rows, pair_rows = Any[], Any[]
    failures = String[]
    for (case, frequency) in zip(cases, frequencies)
        case_name = String(case["name"])
        reference_data = datasets[reference_name]
        incident_index = findfirst(p -> isapprox(p.frequency, frequency; rtol=frequency_rtol, atol=0.0), reference_data)
        isnothing(incident_index) && error("Reference is missing case $case_name at $frequency Hz")
        incidence = reference_data[incident_index]
        for cut in ("phi", "theta")
            reference = extract_cut(reference_data, frequency, cut, settings, incidence)
            ref_source = sources[findfirst(==(reference_name), names)]
            get(ref_source, "expected_transform", "identity") == "identity" || error("Reference expected_transform must be identity; use field_scale for documented source normalization")
            append!(samples_rows, [sample_row(reference_name, case_name, cut, frequency, p, incidence.incident_pol, db_floor) for p in reference])
            for source in sources
                name = String(source["name"])
                name == reference_name && continue
                selected = String(get(source, "expected_transform", "identity"))
                selected in TRANSFORMS || error("Invalid expected_transform $selected for $name")
                candidate = extract_cut(datasets[name], frequency, cut, settings, incidence)
                aligned, interpolated = align_cut(candidate, reference, settings)
                errors = Float64[]
                for transformation in TRANSFORMS
                    transformed = [CutPoint(p.angle, p.direction, transform_field(p.field, p.direction, transformation)) for p in aligned]
                    metrics = regression_metrics(transformed, reference, settings, radius)
                    push!(errors, metrics["complex_l2"])
                    push!(diagnostics, (name, reference_name, case_name, cut, transformation, transformation == selected,
                        metrics["complex_l2"], metrics["rcs_l2"], metrics["max_normalized_rcs_error"], metrics["max_db_error"]))
                end
                # A null reference component cannot distinguish its sign. Report that
                # degeneracy and prefer the first equivalent transform deterministically.
                smallest_error = minimum(errors)
                equivalent = [TRANSFORMS[i] for i in eachindex(errors) if errors[i] <= smallest_error + max(1e-12, 1e-10 * smallest_error)]
                best = first(equivalent)
                transformed = [CutPoint(p.angle, p.direction, transform_field(p.field, p.direction, selected)) for p in aligned]
                metrics = regression_metrics(transformed, reference, settings, radius)
                cross_power = sum(abs2(field_components(p, incidence.incident_pol; cut=cut)[4]) for p in transformed)
                total_power = sum(sum(abs2, p.field) for p in transformed)
                metrics["cross_power_fraction"] = total_power == 0 ? 0.0 : cross_power / total_power
                merge!(metrics, Dict("source" => name, "reference" => reference_name, "case" => case_name, "cut" => cut,
                    "frequency_hz" => frequency, "expected_transform" => selected, "best_diagnostic_transform" => best,
                    "equivalent_diagnostic_transforms" => equivalent,
                    "source_cut_points" => length(candidate), "interpolated_points" => interpolated))
                thresholds = merge(copy(settings), get(source, "tolerances", Dict{String,Any}()))
                # At least the two global measures always gate; other limits are opt-in.
                get!(thresholds, "max_complex_l2", 0.03)
                get!(thresholds, "max_rcs_l2", 0.04)
                before = length(failures)
                for metric in ("complex_l2", "rcs_l2", "max_normalized_rcs_error", "max_db_error", "cross_power_fraction")
                    threshold_key = startswith(metric, "max_") ? metric : "max_" * metric
                    haskey(thresholds, threshold_key) || continue
                    limit = positive_setting(thresholds, threshold_key, 0.0; zero=true)
                    metrics["limit_" * metric] = limit
                    metrics[metric] <= limit || push!(failures, "$name/$case_name/$cut: $metric=$(metrics[metric]) exceeds $limit (selected=$selected, best diagnostic=$best)")
                end
                metrics["passed"] = before == length(failures)
                push!(summaries, metrics)
                append!(samples_rows, [sample_row(name, case_name, cut, frequency, p, incidence.incident_pol, db_floor) for p in transformed])
                reference_peak = maximum(norm(p.field) for p in reference)
                for (p, r) in zip(transformed, reference)
                    sigma, exact = 4pi * sum(abs2, p.field), 4pi * sum(abs2, r.field)
                    floor_power = 10.0^(db_floor / 10)
                    push!(pair_rows, (name, reference_name, case_name, cut, r.angle, (sigma - exact) / (pi * radius^2),
                        10log10(max(sigma, floor_power) / max(exact, floor_power)), norm(p.field - r.field) / reference_peak))
                end
                @printf("%s / %s / %s: complex L2 %.5g, RCS L2 %.5g; selected %s, diagnostic %s%s\n",
                    name, case_name, cut, metrics["complex_l2"], metrics["rcs_l2"], selected, best, metrics["passed"] ? " [PASS]" : " [FAIL]")
            end
        end
    end
    write_csv(joinpath(output, "samples.csv"), SAMPLE_COLUMNS, samples_rows)
    write_csv(joinpath(output, "pairs.csv"), ["source", "reference", "case", "cut", "angle_rad", "sigma_error_normalized", "db_error", "complex_error_normalized"], pair_rows)
    write_csv(joinpath(output, "transformations.csv"), ["source", "reference", "case", "cut", "transform", "selected", "complex_l2", "rcs_l2", "max_normalized_rcs_error", "max_db_error"], diagnostics)
    report = Dict("created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z",
        "config_path" => config_path, "config_sha256" => bytes2hex(open(sha256, config_path)), "configuration" => config,
        "sources_present" => names, "reference" => reference_name, "passed" => isempty(failures), "failures" => failures,
        "files" => provenance, "comparisons" => summaries)
    open(joinpath(output, "report.toml"), "w") do io
        TOML.print(io, report; sorted=true)
    end
    summary_columns = ["source", "reference", "case", "cut", "frequency_hz", "expected_transform", "best_diagnostic_transform",
        "complex_l2", "rcs_l2", "max_normalized_rcs_error", "max_db_error", "cross_power_fraction", "reference_points", "source_cut_points", "interpolated_points", "passed"]
    write_csv(joinpath(output, "summary.csv"), summary_columns, [[r[key] for key in summary_columns] for r in summaries])
    if plots
        command = `$python $(joinpath(@__DIR__, "plot_comparison.py")) --samples $(joinpath(output, "samples.csv")) --pairs $(joinpath(output, "pairs.csv")) --output-dir $(joinpath(output, "plots")) --db-floor $db_floor`
        run(command)
    end
    isempty(failures) || error("FFD regression failed:\n" * join(failures, '\n') * "\nDiagnostics: " * output)
    println("FFD comparison passed. Sources: ", join(names, ", "), ". Results: ", output)
    return report
end

function main(args=ARGS)
    if "--help" in args || "-h" in args
        println("""
        Usage: julia tools/ffd/compare_far_fields.jl --config campaign.toml [options]
          --output-dir PATH  Override the campaign output directory
          --no-plots         Write CSV/TOML only (no Python dependency)
          --python PATH      Python interpreter with matplotlib (default python3)
        All field, angle and frequency adaptations must be declared in the TOML.
        Regression failures return a nonzero exit code after writing diagnostics.
        See tools/ffd/README.md and emagpie_reference.toml for the schema.
        """)
        return
    end
    options = Dict{String,String}()
    plots = true
    i = 1
    while i <= length(args)
        argument = args[i]
        if argument == "--no-plots"
            plots = false
        else
            tokens = split(argument, '='; limit=2)
            key = tokens[1]
            key in ("--config", "--output-dir", "--python") || error("Unknown option $key")
            if length(tokens) == 2
                options[key] = tokens[2]
            else
                i += 1
                i <= length(args) || error("Missing value for $key")
                options[key] = args[i]
            end
        end
        i += 1
    end
    haskey(options, "--config") || error("--config is required")
    compare(options["--config"]; output_dir=get(options, "--output-dir", nothing), plots=plots, python=get(options, "--python", "python3"))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    try
        FFDComparison.main()
    catch exception
        showerror(stderr, exception)
        println(stderr)
        exit(1)
    end
end
