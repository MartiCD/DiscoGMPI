"""Shared error/CSV contract for the distributed Maxwell convergence studies."""
const MAXWELL_CONVERGENCE_QUANTITIES =
    (:electric, :magnetic, :total, :ex, :ey, :ez, :hx, :hy, :hz)

parse_convergence_nodal_set(value::AbstractString) =
    canonical_tet_nodal_set(Symbol(replace(lowercase(value), '-' => '_')))

function validate_convergence_policy(policy::Symbol, tolerance::Real)
    policy in (:fail, :report) ||
        throw(ArgumentError("--rate-policy must be fail or report."))
    isfinite(tolerance) && tolerance >= 0 ||
        throw(ArgumentError("--rate-tolerance must be finite and non-negative."))
    return nothing
end

function maxwell_error_maxima(previous, current)
    length(previous) == length(current) == 9 ||
        throw(ArgumentError("Expected E,H,total and six component errors."))
    all(x -> isfinite(x) && x >= 0, current) ||
        error("Non-finite or negative Maxwell convergence error.")
    return ntuple(i -> max(previous[i], current[i]), 9)
end

const MAXWELL_CONVERGENCE_METADATA = (
    schema_version = 2,
    error_norm = "linf_time_l2_space",
    rate_h = "h_max",
    case = "",
    boundary_condition = "",
    flux = "centered",
    nodal_set = "equispaced",
    magnetic_initialization = "interpolate",
    electric_initialization = "interpolate",
    mesh_family = "",
    mesh_source = "",
    mesh_parameter = missing,
    mesh_level = missing,
    order = missing,
    esprk_order = missing,
    cubature_order = missing,
    mpi_ranks = missing,
    nelements = missing,
    nodes_per_element = missing,
    min_owned_elements = missing,
    max_owned_elements = missing,
    final_time = missing,
    epsilon = missing,
    mu = missing,
    wave_number = missing,
    jitter = missing,
    seed = missing,
    effective_cfl = missing,
    dt = missing,
    nsteps = missing,
    error_samples = missing,
    error_mode = "max-time",
    error_every = 1,
    elapsed_seconds = missing,
    target_h = missing,
    characteristic_h = missing,
    h_min = missing,
    h_max = missing,
    h_ratio = missing,
    volume_total = missing,
    volume_min = missing,
    volume_max = missing,
    volume_ratio = missing,
    edge_min = missing,
    edge_max = missing,
    edge_ratio = missing,
    mean_ratio_min = missing,
    mean_ratio_avg = missing,
    initial_energy = missing,
    final_energy = missing,
    relative_energy_drift = missing,
    initial_l2_space_electric_error = missing,
    initial_l2_space_magnetic_error = missing,
    initial_l2_space_total_error = missing,
    initial_l2_space_ex_error = missing,
    initial_l2_space_ey_error = missing,
    initial_l2_space_ez_error = missing,
    initial_l2_space_hx_error = missing,
    initial_l2_space_hy_error = missing,
    initial_l2_space_hz_error = missing,
    rate_tolerance = DEFAULT_MAXWELL_CONVERGENCE_RATE_TOLERANCE,
    rate_error_floor = DEFAULT_COMPONENT_RATE_ERROR_FLOOR,
    rate_policy = "fail",
)

function maxwell_convergence_row(metadata::NamedTuple, maxima, final_errors)
    maxwell_error_maxima(ntuple(_ -> 0.0, 9), maxima)
    maxwell_error_maxima(ntuple(_ -> 0.0, 9), final_errors)
    unknown = setdiff(keys(metadata), keys(MAXWELL_CONVERGENCE_METADATA))
    isempty(unknown) || throw(ArgumentError("Unknown convergence metadata: $unknown"))
    fields = Pair{Symbol, Any}[]
    for (i, name) in enumerate(MAXWELL_CONVERGENCE_QUANTITIES)
        push!(fields, Symbol("linf_time_l2_space_$(name)_error") => maxima[i])
        push!(fields, Symbol("final_l2_space_$(name)_error") => final_errors[i])
        push!(fields, Symbol("rate_$(name)") => missing)
        push!(fields, Symbol("expected_rate_$(name)") => missing)
        push!(fields, Symbol("status_$(name)") => "NO_PAIR")
    end
    return merge(MAXWELL_CONVERGENCE_METADATA, metadata, (; fields...),
                 (rate_verdict = "NO_PAIR",))
end

"""Copy the common measured metadata from a driver's result record."""
function maxwell_convergence_metadata(result)
    return (; (name => getproperty(result, name)
               for name in keys(MAXWELL_CONVERGENCE_METADATA)
               if hasproperty(result, name))...)
end

maxwell_convergence_csv_header() = join(
    string.(keys(maxwell_convergence_row((;), ntuple(_ -> 0.0, 9),
                                        ntuple(_ -> 0.0, 9)))), ',')

function maxwell_expected_rate(name::Symbol, order::Int, boundary)
    name == :electric && return Float64(order + 1)
    name in (:magnetic, :total) && return Float64(order)
    active = Symbol(boundary) == :periodic ? (:ez, :hy) :
             Symbol(boundary) == :pec ? (:ex, :ez, :hx, :hy, :hz) :
             Symbol(boundary) == :pmc ? (:ex, :ey, :ez, :hx, :hz) :
             throw(ArgumentError("Unknown convergence boundary: $boundary"))
    name in active || return missing
    return name in (:ex, :ey, :ez) ? Float64(order + 1) : Float64(order)
end

"""Classify one measured refinement pair; zero-solution components are informational."""
function maxwell_convergence_rate_check(coarse, fine, coarse_h, fine_h, expected;
                                       tolerance = 0.5, floor = 1e-12)
    valid = all(isfinite, (coarse, fine, coarse_h, fine_h)) &&
            coarse >= 0 && fine >= 0 && coarse_h > fine_h > 0
    valid || return (rate = missing, status = "FAIL")
    rate = coarse > floor && fine > 0 ?
           observed_convergence_rate(coarse, fine, coarse_h, fine_h) : missing
    status = if coarse <= floor && fine <= floor
        "SKIP_ROUNDOFF"
    elseif ismissing(expected)
        "INFO"
    elseif coarse <= floor
        "FAIL"
    elseif fine <= floor
        "PASS_FLOOR"
    elseif !ismissing(rate) && isfinite(rate) && rate >= expected - tolerance
        "PASS"
    else
        "FAIL"
    end
    return (; rate, status)
end

function rate_maxwell_convergence_rows(rows)
    rated = NamedTuple[]
    previous = Dict{Tuple, NamedTuple}()
    for row in sort(collect(rows); by = r -> (r.order, r.mesh_level))
        key = (row.boundary_condition, row.flux, row.nodal_set, row.magnetic_initialization, row.electric_initialization, row.mesh_family,
               row.order, row.final_time, row.effective_cfl, row.epsilon, row.mu,
               row.wave_number, row.jitter, row.seed)
        coarse = get(previous, key, nothing)
        fields = Pair{Symbol, Any}[]
        statuses = String[]
        for name in MAXWELL_CONVERGENCE_QUANTITIES
            expected = maxwell_expected_rate(name, row.order, row.boundary_condition)
            error_key = Symbol("linf_time_l2_space_$(name)_error")
            check = coarse === nothing ? (rate = missing, status = "NO_PAIR") :
                maxwell_convergence_rate_check(
                    coarse[error_key], row[error_key], coarse.h_max, row.h_max, expected;
                    tolerance = row.rate_tolerance, floor = row.rate_error_floor)
            push!(fields, Symbol("rate_$(name)") => check.rate)
            push!(fields, Symbol("expected_rate_$(name)") => expected)
            push!(fields, Symbol("status_$(name)") => check.status)
            push!(statuses, check.status)
        end
        verdict = coarse === nothing ? "NO_PAIR" :
                  "FAIL" in statuses ? "FAIL" : "PASS"
        push!(rated, merge(row, (; fields...), (rate_verdict = verdict,)))
        previous[key] = row
    end
    return rated
end

"""The finest pair per degree determines the study verdict; all pairs remain in CSV."""
function maxwell_convergence_verdict(rows)
    isempty(rows) && return false
    finals = Dict{Tuple, NamedTuple}()
    for row in sort(collect(rows); by = r -> r.mesh_level)
        finals[(row.boundary_condition, row.flux, row.nodal_set, row.magnetic_initialization, row.electric_initialization, row.mesh_family,
                row.order)] = row
    end
    return all(row -> row.rate_verdict == "PASS", values(finals))
end

function _convergence_csv_value(value)
    ismissing(value) && return ""
    text = string(value)
    return occursin(r"[,\"\n\r]", text) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end

function write_maxwell_convergence_csv(path::AbstractString, rows)
    header = maxwell_convergence_csv_header()
    for row in rows
        join(string.(keys(row)), ',') == header ||
            error("Convergence row does not match schema v2.")
    end
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, join(_convergence_csv_value.(values(row)), ','))
        end
    end
    return path
end

function print_maxwell_convergence_rows(rows)
    modes = unique(string(getproperty(row, :error_mode)) for row in rows)
    error_description = modes == ["final"] ? "final-time spatial L2" :
                        "sampled max-in-time spatial L2"
    println("Errors: $error_description; rates: measured h_max")
    println(" p level nodes         h_max        dt           E error      H error      rate E  rate H  verdict")
    for r in rows
        @printf("%2d %5d %-12s %.4e   %.4e   %.4e   %.4e   %6s  %6s  %s\n",
                r.order, r.mesh_level, r.nodal_set, r.h_max, r.dt,
                r.linf_time_l2_space_electric_error, r.linf_time_l2_space_magnetic_error,
                formatted_optional_rate(r.rate_electric), formatted_optional_rate(r.rate_magnetic),
                r.rate_verdict)
        r.rate_verdict == "FAIL" && println("  Failing quantities: ", join(
            (string(n) for n in MAXWELL_CONVERGENCE_QUANTITIES
             if r[Symbol("status_$(n)")] == "FAIL"), ", "))
    end
end

"""Write results before collectively enforcing failure, including root I/O errors."""
function finish_maxwell_convergence(rows, path, comm; policy::Symbol = :fail,
                                    after_write = _ -> nothing)
    root_error = nothing
    passed = false
    rated = NamedTuple[]
    if MPI.Comm_rank(comm) == 0
        try
            rated = rate_maxwell_convergence_rows(rows)
            write_maxwell_convergence_csv(path, rated)
            after_write(rated)
            print_maxwell_convergence_rows(rated)
            passed = maxwell_convergence_verdict(rated)
            println("Overall expected-rate convergence: ", passed ? "PASS" : "FAIL",
                    " (rate-policy=", policy, ")")
            println("Wrote CSV: ", path)
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, comm; root = 0)
    root_error === nothing || error(root_error)
    passed = MPI.bcast(passed, comm; root = 0)
    policy == :fail && !passed && error("Maxwell convergence rate checks failed; results written to $path.")
    return rated
end
