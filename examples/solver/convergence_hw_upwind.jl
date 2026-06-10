#!/usr/bin/env julia

using DiscoGMPI
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "..", "src", "solver", "kernels", "JaskowiecSukumar.jl"))

const CONV_PEC_BOUNDARY_ID = 10
const CONV_TET_FACES = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)
const REF_TET_VOLUME_LOCAL = 4.0 / 3.0

struct ConvergenceConfig
    cells_per_axis::Vector{Int}
    orders::Vector{Int}
    final_time::Float64
    cfl::Float64
    rk_order::Int
    backend::Symbol
    jitter::Float64
    seed::Int
    output::String
    eps::Float64
    mu::Float64
end

struct ConvergenceResult
    order::Int
    mesh_level::Int
    cells_per_axis::Int
    nelements::Int
    h::Float64
    dt::Float64
    nsteps::Int
    l2_E::Float64
    l2_H::Float64
    l2_total::Float64
    rel_total::Float64
    rate_E::Union{Missing, Float64}
    rate_H::Union{Missing, Float64}
    rate::Union{Missing, Float64}
end

struct MaxwellContinuousError
    cubature_order::Int
    l2_E::Float64
    l2_H::Float64
    l2_total::Float64
    rel_total::Float64
    linf_E::Float64
    linf_H::Float64
    linf_total::Float64
end

function parse_int_list(value::AbstractString)
    entries = split(value, ",")
    parsed = [parse(Int, strip(v)) for v in entries if !isempty(strip(v))]

    if isempty(parsed)
        error("Expected a comma-separated integer list, got '$value'.")
    end

    return parsed
end

function parse_convergence_args(args)
    config = ConvergenceConfig(
        [2, 3, 4, 5],
        [1, 2, 4],
        0.005,
        0.1,
        5,
        :serial,
        0.08,
        1234,
        joinpath("output", "convergence_hw_upwind.csv"),
        1.0,
        1.0,
    )

    cells_per_axis = config.cells_per_axis
    orders = config.orders
    final_time = config.final_time
    cfl = config.cfl
    rk_order = config.rk_order
    backend = config.backend
    jitter = config.jitter
    seed = config.seed
    output = config.output
    eps = config.eps
    mu = config.mu

    for arg in args
        if arg == "--help" || arg == "-h"
            print_convergence_usage()
            exit(0)
        elseif startswith(arg, "--cells=")
            cells_per_axis = parse_int_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--orders=")
            orders = parse_int_list(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--time=") || startswith(arg, "--final-time=")
            final_time = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--rk=")
            rk_order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--backend=")
            backend = parse_backend(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--jitter=")
            jitter = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--eps=")
            eps = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--mu=")
            mu = parse(Float64, split(arg, "=", limit = 2)[2])
        else
            error("Unknown argument: $arg. Run with --help for usage.")
        end
    end

    if length(cells_per_axis) != 4
        error(
            "This convergence driver expects exactly four mesh levels. " *
            "Use --cells=a,b,c,d."
        )
    end

    if any(n -> n < 1, cells_per_axis)
        error("All entries in --cells must be positive.")
    end

    if any(N -> N < 1, orders)
        error("All entries in --orders must be positive.")
    end

    if final_time <= 0.0
        error("--time must be positive.")
    end

    if cfl <= 0.0
        error("--cfl must be positive.")
    end

    if jitter < 0.0
        error("--jitter must be nonnegative.")
    end

    if eps <= 0.0 || mu <= 0.0
        error("--eps and --mu must be positive.")
    end

    return ConvergenceConfig(
        cells_per_axis,
        orders,
        final_time,
        cfl,
        rk_order,
        backend,
        jitter,
        seed,
        output,
        eps,
        mu,
    )
end

function parse_backend(value::AbstractString)
    normalized = lowercase(strip(value))

    if normalized == "serial"
        return :serial
    elseif normalized == "threaded" || normalized == "threads"
        return :threaded
    else
        error("Unsupported --backend=$value. Expected serial or threaded.")
    end
end

function build_backend(kind::Symbol)
    if kind == :serial
        return SerialBackend()
    elseif kind == :threaded
        return ThreadedBackend()
    else
        error("Unsupported backend kind: $kind.")
    end
end

function backend_description(kind::Symbol)
    if kind == :serial
        return "serial"
    elseif kind == :threaded
        return "threaded ($(Base.Threads.nthreads()) Julia threads)"
    else
        error("Unsupported backend kind: $kind.")
    end
end

function print_convergence_usage()
    println("Hesthaven-Warburton Maxwell convergence driver with upwind flux")
    println()
    println("Usage:")
    println("  julia --project=. examples/solver/convergence_hw_upwind.jl [options]")
    println()
    println("Options:")
    println("  --cells=a,b,c,d     Four jittered tet cube mesh levels. Default: 2,3,4,5")
    println("  --orders=a,b,c      DG polynomial orders. Default: 1,2,4")
    println("  --time=T            Final time. Default: 0.005")
    println("  --cfl=C             CFL factor for dt estimate. Default: 0.1")
    println("  --rk=N              Explicit RK order. Default: 5")
    println("  --backend=B         Backend: serial or threaded. Default: serial")
    println("  --jitter=J          Interior-node jitter fraction of grid spacing. Default: 0.08")
    println("  --seed=N            Random seed for deterministic meshes. Default: 1234")
    println("  --eps=X             Electric permittivity. Default: 1.0")
    println("  --mu=X              Magnetic permeability. Default: 1.0")
    println("  --output=PATH       CSV output path. Default: output/convergence_hw_upwind.csv")
    println()
    println("For threaded runs, launch Julia with JULIA_NUM_THREADS=N.")
end

function node_id(i::Int, j::Int, k::Int, cells_per_axis::Int)
    n = cells_per_axis + 1
    return 1 + i + n * (j + n * k)
end

function build_jittered_cube_points(
    cells_per_axis::Int;
    jitter::Float64,
    rng::AbstractRNG,
)
    npoints_1d = cells_per_axis + 1
    coords = collect(range(-1.0, 1.0; length = npoints_1d))
    h = 2.0 / cells_per_axis

    points = zeros(Float64, 3, npoints_1d^3)

    for k in 0:cells_per_axis
        for j in 0:cells_per_axis
            for i in 0:cells_per_axis
                id = node_id(i, j, k, cells_per_axis)
                x = coords[i + 1]
                y = coords[j + 1]
                z = coords[k + 1]

                if 0 < i < cells_per_axis &&
                   0 < j < cells_per_axis &&
                   0 < k < cells_per_axis
                    scale = jitter * h
                    x += scale * (2.0 * rand(rng) - 1.0)
                    y += scale * (2.0 * rand(rng) - 1.0)
                    z += scale * (2.0 * rand(rng) - 1.0)
                end

                points[1, id] = x
                points[2, id] = y
                points[3, id] = z
            end
        end
    end

    return points
end

function build_cube_tets(cells_per_axis::Int)
    tets = NTuple{4, Int}[]

    for k in 0:(cells_per_axis - 1)
        for j in 0:(cells_per_axis - 1)
            for i in 0:(cells_per_axis - 1)
                v000 = node_id(i,     j,     k,     cells_per_axis)
                v100 = node_id(i + 1, j,     k,     cells_per_axis)
                v010 = node_id(i,     j + 1, k,     cells_per_axis)
                v110 = node_id(i + 1, j + 1, k,     cells_per_axis)
                v001 = node_id(i,     j,     k + 1, cells_per_axis)
                v101 = node_id(i + 1, j,     k + 1, cells_per_axis)
                v011 = node_id(i,     j + 1, k + 1, cells_per_axis)
                v111 = node_id(i + 1, j + 1, k + 1, cells_per_axis)

                push!(tets, (v000, v100, v110, v111))
                push!(tets, (v000, v110, v010, v111))
                push!(tets, (v000, v010, v011, v111))
                push!(tets, (v000, v011, v001, v111))
                push!(tets, (v000, v001, v101, v111))
                push!(tets, (v000, v101, v100, v111))
            end
        end
    end

    return reduce(hcat, collect.(tets))
end

function sorted_face_key(nodes::NTuple{3, Int})
    s = sort(collect(nodes))
    return (s[1], s[2], s[3])
end

function build_boundary_tris(tets::Matrix{Int})
    counts = Dict{NTuple{3, Int}, Int}()
    oriented_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for e in axes(tets, 2)
        tet = tets[:, e]

        for local_face in 1:4
            ids = CONV_TET_FACES[local_face]
            nodes = (tet[ids[1]], tet[ids[2]], tet[ids[3]])
            key = sorted_face_key(nodes)

            counts[key] = get(counts, key, 0) + 1
            oriented_nodes[key] = nodes
        end
    end

    tris = NTuple{3, Int}[]

    for (key, count) in counts
        if count == 1
            push!(tris, oriented_nodes[key])
        elseif count != 2
            error("Non-manifold face $key appears $count times.")
        end
    end

    return reduce(hcat, collect.(tris))
end

function build_jittered_pec_cube_mesh(
    cells_per_axis::Int;
    jitter::Float64,
    seed::Int,
)
    rng = MersenneTwister(seed + cells_per_axis)

    points = build_jittered_cube_points(
        cells_per_axis;
        jitter = jitter,
        rng = rng,
    )
    tets = build_cube_tets(cells_per_axis)
    tris = build_boundary_tris(tets)

    ntets = size(tets, 2)
    ntris = size(tris, 2)

    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))

    boundary_id = zeros(Int, ntets + ntris)
    boundary_id[tri_cell_ids] .= CONV_PEC_BOUNDARY_ID

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_id),
    )

    check_mesh_consistency(mesh)

    return mesh
end

function exact_maxwell_functions(t::Float64; eps::Float64, mu::Float64)
    alpha = pi / 2.0
    omega = sqrt(2.0) * alpha / sqrt(eps * mu)
    hscale = alpha / (mu * omega)

    Efun = function (x, y, z)
        sx = sin(alpha * (x + 1.0))
        sy = sin(alpha * (y + 1.0))
        Ez = sx * sy * cos(omega * t)
        return 0.0, 0.0, Ez
    end

    Hfun = function (x, y, z)
        sx = sin(alpha * (x + 1.0))
        cx = cos(alpha * (x + 1.0))
        sy = sin(alpha * (y + 1.0))
        cy = cos(alpha * (y + 1.0))

        Hx = -hscale * sx * cy * sin(omega * t)
        Hy =  hscale * cx * sy * sin(omega * t)

        return Hx, Hy, 0.0
    end

    return Efun, Hfun
end

function exact_maxwell_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    t::Float64;
    eps::Float64,
    mu::Float64,
)
    Efun, Hfun = exact_maxwell_functions(t; eps = eps, mu = mu)
    return interpolate_maxwell_field(mesh, ref, Efun, Hfun)
end

function field_l2_components(
    U::MaxwellField,
    ref::ReferenceTet,
    mappings::DGReferenceMapping,
)
    ne = size(U.Ex, 2)
    M = ref.M

    e2 = 0.0
    h2 = 0.0

    for e in 1:ne
        J = mappings.tet_mappings[e].absdetJ

        e2 += J * dot(U.Ex[:, e], M * U.Ex[:, e])
        e2 += J * dot(U.Ey[:, e], M * U.Ey[:, e])
        e2 += J * dot(U.Ez[:, e], M * U.Ez[:, e])

        h2 += J * dot(U.Hx[:, e], M * U.Hx[:, e])
        h2 += J * dot(U.Hy[:, e], M * U.Hy[:, e])
        h2 += J * dot(U.Hz[:, e], M * U.Hz[:, e])
    end

    return sqrt(max(e2, 0.0)), sqrt(max(h2, 0.0))
end

function field_difference(U::MaxwellField, V::MaxwellField)
    return MaxwellField(
        U.Ex .- V.Ex,
        U.Ey .- V.Ey,
        U.Ez .- V.Ez,
        U.Hx .- V.Hx,
        U.Hy .- V.Hy,
        U.Hz .- V.Hz,
    )
end

function maxwell_l2_error(
    U::MaxwellField,
    exact::MaxwellField,
    ref::ReferenceTet,
    mappings::DGReferenceMapping,
)
    dU = field_difference(U, exact)
    err_E, err_H = field_l2_components(dU, ref, mappings)
    norm_E, norm_H = field_l2_components(exact, ref, mappings)

    err_total = sqrt(err_E^2 + err_H^2)
    norm_total = sqrt(norm_E^2 + norm_H^2)
    rel_total = err_total / max(norm_total, eps(Float64))

    return err_E, err_H, err_total, rel_total
end

function continuous_error_cubature_order(spatial_order::Int)
    cubature_order = max(2, 2 * spatial_order + 4)

    if cubature_order > 20
        error(
            "Requested strict continuous error cubature order $cubature_order, " *
            "but JaskowiecSukumar.jl currently provides orders up to 20."
        )
    end

    return cubature_order
end

function reference_interpolation_matrix(
    ref::ReferenceTet,
    cubature_points::Matrix{Float64},
)
    rq = collect(cubature_points[:, 1])
    sq = collect(cubature_points[:, 2])
    tq = collect(cubature_points[:, 3])

    Vq_modal = DiscoGMPI.orthonormal_vandermonde_tet(rq, sq, tq, ref.basis)

    return Vq_modal * ref.invV
end

function maxwell_continuous_error(
    U::MaxwellField,
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    mappings::DGReferenceMapping,
    t::Float64;
    eps::Float64,
    mu::Float64,
    cubature_order::Int = continuous_error_cubature_order(ref.N),
)
    cubature_points, cubature_weights, nq =
        get_JaskowiecSukumar_cubature(cubature_order)
    Vq = reference_interpolation_matrix(ref, cubature_points)
    Efun, Hfun = exact_maxwell_functions(t; eps = eps, mu = mu)

    e2 = 0.0
    h2 = 0.0
    exact_e2 = 0.0
    exact_h2 = 0.0

    linf_E = 0.0
    linf_H = 0.0
    linf_total = 0.0

    ne = size(U.Ex, 2)

    for elem in 1:ne
        tet_nodes = mesh.tets[:, elem]
        J = mappings.tet_mappings[elem].absdetJ

        @views begin
            Ex = U.Ex[:, elem]
            Ey = U.Ey[:, elem]
            Ez = U.Ez[:, elem]
            Hx = U.Hx[:, elem]
            Hy = U.Hy[:, elem]
            Hz = U.Hz[:, elem]

            for q in 1:nq
                r = cubature_points[q, 1]
                s = cubature_points[q, 2]
                tref = cubature_points[q, 3]
                x, y, z = DiscoGMPI.map_to_physical(
                    mesh.points,
                    tet_nodes,
                    r,
                    s,
                    tref,
                )

                exact_Ex, exact_Ey, exact_Ez = Efun(x, y, z)
                exact_Hx, exact_Hy, exact_Hz = Hfun(x, y, z)

                uh_Ex = dot(Vq[q, :], Ex)
                uh_Ey = dot(Vq[q, :], Ey)
                uh_Ez = dot(Vq[q, :], Ez)

                uh_Hx = dot(Vq[q, :], Hx)
                uh_Hy = dot(Vq[q, :], Hy)
                uh_Hz = dot(Vq[q, :], Hz)

                dEx = uh_Ex - exact_Ex
                dEy = uh_Ey - exact_Ey
                dEz = uh_Ez - exact_Ez

                dHx = uh_Hx - exact_Hx
                dHy = uh_Hy - exact_Hy
                dHz = uh_Hz - exact_Hz

                point_e2 = dEx^2 + dEy^2 + dEz^2
                point_h2 = dHx^2 + dHy^2 + dHz^2
                point_total2 = point_e2 + point_h2

                weight = J * cubature_weights[q]

                e2 += weight * point_e2
                h2 += weight * point_h2
                exact_e2 += weight * (exact_Ex^2 + exact_Ey^2 + exact_Ez^2)
                exact_h2 += weight * (exact_Hx^2 + exact_Hy^2 + exact_Hz^2)

                linf_E = max(linf_E, sqrt(point_e2))
                linf_H = max(linf_H, sqrt(point_h2))
                linf_total = max(linf_total, sqrt(point_total2))
            end
        end
    end

    l2_E = sqrt(max(e2, 0.0))
    l2_H = sqrt(max(h2, 0.0))
    l2_total = sqrt(max(e2 + h2, 0.0))
    norm_total = sqrt(max(exact_e2 + exact_h2, 0.0))
    rel_total = l2_total / max(norm_total, Base.eps(Float64))

    return MaxwellContinuousError(
        cubature_order,
        l2_E,
        l2_H,
        l2_total,
        rel_total,
        linf_E,
        linf_H,
        linf_total,
    )
end

function characteristic_h(dg::DGDiscretization)
    ne = length(dg.mappings.tet_mappings)
    volume = 0.0

    for mapping in dg.mappings.tet_mappings
        volume += REF_TET_VOLUME_LOCAL * mapping.absdetJ
    end

    return (volume / ne)^(1.0 / 3.0)
end

function advance_maxwell!(
    U::MaxwellField,
    dg::DGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    rk_order::Int,
    dt::Float64,
    nsteps::Int,
    eps::Float64,
    mu::Float64,
)
    scheme = explicit_rk_scheme(rk_order)
    work = DiscoGMPI.MaxwellRKWorkspace(U, scheme)
    rhs_function! = DiscoGMPI.make_maxwell_rhs_function(
        dg,
        registry,
        formulation;
        ε = eps,
        μ = mu,
    )

    for _ in 1:nsteps
        DiscoGMPI.rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )
    end

    return U
end

function run_case(
    cells_per_axis::Int,
    order::Int,
    level::Int,
    config::ConvergenceConfig,
)
    mesh = build_jittered_pec_cube_mesh(
        cells_per_axis;
        jitter = config.jitter,
        seed = config.seed,
    )
    dg = DGDiscretization(mesh, order; backend = build_backend(config.backend))

    registry = MaxwellBoundaryRegistry(
        Dict(CONV_PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )
    formulation = HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

    U = exact_maxwell_field(
        mesh,
        dg.ref,
        0.0;
        eps = config.eps,
        mu = config.mu,
    )

    dt_est, _ = estimate_maxwell_dt(
        mesh,
        dg.geometry,
        dg.ref;
        CFL = config.cfl,
        ε = config.eps,
        μ = config.mu,
    )

    nsteps = max(1, ceil(Int, config.final_time / dt_est))
    dt = config.final_time / nsteps

    advance_maxwell!(
        U,
        dg,
        registry,
        formulation;
        rk_order = config.rk_order,
        dt = dt,
        nsteps = nsteps,
        eps = config.eps,
        mu = config.mu,
    )

    exact = exact_maxwell_field(
        mesh,
        dg.ref,
        config.final_time;
        eps = config.eps,
        mu = config.mu,
    )

    l2_E, l2_H, l2_total, rel_total = maxwell_l2_error(
        U,
        exact,
        dg.ref,
        dg.mappings,
    )

    return ConvergenceResult(
        order,
        level,
        cells_per_axis,
        size(mesh.tets, 2),
        characteristic_h(dg),
        dt,
        nsteps,
        l2_E,
        l2_H,
        l2_total,
        rel_total,
        missing,
        missing,
        missing,
    )
end

function convergence_rate(
    error::Float64,
    previous_error::Float64,
    h::Float64,
    previous_h::Float64,
)
    if error <= 0.0 || previous_error <= 0.0 || h <= 0.0 || previous_h <= 0.0
        return missing
    end

    return log(error / previous_error) / log(h / previous_h)
end

function with_rates(results::Vector{ConvergenceResult})
    out = ConvergenceResult[]

    for order in sort(unique(r.order for r in results))
        subset = sort(
            filter(r -> r.order == order, results);
            by = r -> r.mesh_level,
        )

        previous = nothing

        for result in subset
            rate_E = missing
            rate_H = missing
            rate = missing

            if previous !== nothing
                rate_E = convergence_rate(
                    result.l2_E,
                    previous.l2_E,
                    result.h,
                    previous.h,
                )
                rate_H = convergence_rate(
                    result.l2_H,
                    previous.l2_H,
                    result.h,
                    previous.h,
                )
                rate = convergence_rate(
                    result.l2_total,
                    previous.l2_total,
                    result.h,
                    previous.h,
                )
            end

            push!(
                out,
                ConvergenceResult(
                    result.order,
                    result.mesh_level,
                    result.cells_per_axis,
                    result.nelements,
                    result.h,
                    result.dt,
                    result.nsteps,
                    result.l2_E,
                    result.l2_H,
                    result.l2_total,
                    result.rel_total,
                    rate_E,
                    rate_H,
                    rate,
                ),
            )

            previous = result
        end
    end

    return out
end

function print_results(results::Vector{ConvergenceResult})
    println()
    println("HW upwind Maxwell convergence")
    println("-----------------------------")
    println(
        rpad("N", 4),
        rpad("level", 8),
        rpad("cells", 8),
        rpad("Ne", 10),
        rpad("h", 14),
        rpad("dt", 14),
        rpad("steps", 8),
        rpad("L2 E", 16),
        rpad("L2 H", 16),
        rpad("L2 total", 16),
        rpad("rel total", 16),
        rpad("rate E", 10),
        rpad("rate H", 10),
        "rate total",
    )

    for result in results
        rate_E_string = ismissing(result.rate_E) ? "-" : @sprintf("%.4f", result.rate_E)
        rate_H_string = ismissing(result.rate_H) ? "-" : @sprintf("%.4f", result.rate_H)
        rate_string = ismissing(result.rate) ? "-" : @sprintf("%.4f", result.rate)
        println(
            rpad(string(result.order), 4),
            rpad(string(result.mesh_level), 8),
            rpad(string(result.cells_per_axis), 8),
            rpad(string(result.nelements), 10),
            rpad(@sprintf("%.6e", result.h), 14),
            rpad(@sprintf("%.6e", result.dt), 14),
            rpad(string(result.nsteps), 8),
            rpad(@sprintf("%.6e", result.l2_E), 16),
            rpad(@sprintf("%.6e", result.l2_H), 16),
            rpad(@sprintf("%.6e", result.l2_total), 16),
            rpad(@sprintf("%.6e", result.rel_total), 16),
            rpad(rate_E_string, 10),
            rpad(rate_H_string, 10),
            rate_string,
        )
    end
end

function ensure_parent_dir(path::AbstractString)
    dir = dirname(path)

    if !isempty(dir) && dir != "."
        mkpath(dir)
    end
end

function write_results_csv(path::AbstractString, results::Vector{ConvergenceResult})
    ensure_parent_dir(path)

    open(path, "w") do io
        println(
            io,
            "order,mesh_level,cells_per_axis,nelements,h,dt,nsteps," *
            "l2_E,l2_H,l2_total,rel_total,rate_E,rate_H,rate_total",
        )

        for result in results
            rate_E_string = ismissing(result.rate_E) ? "" : string(result.rate_E)
            rate_H_string = ismissing(result.rate_H) ? "" : string(result.rate_H)
            rate_string = ismissing(result.rate) ? "" : string(result.rate)
            println(
                io,
                join(
                    (
                        result.order,
                        result.mesh_level,
                        result.cells_per_axis,
                        result.nelements,
                        result.h,
                        result.dt,
                        result.nsteps,
                        result.l2_E,
                        result.l2_H,
                        result.l2_total,
                        result.rel_total,
                        rate_E_string,
                        rate_H_string,
                        rate_string,
                    ),
                    ",",
                ),
            )
        end
    end
end

function main(args = ARGS)
    config = parse_convergence_args(args)

    println("Hesthaven-Warburton Maxwell convergence driver")
    println("----------------------------------------------")
    println("Formulation:        HesthavenWarburtonFormulation")
    println("Numerical flux:     upwind")
    println("Time integrator:    explicit RK", config.rk_order)
    println("Backend:            ", backend_description(config.backend))
    println("Orders:             ", config.orders)
    println("Mesh levels:        ", config.cells_per_axis)
    println("Final time:         ", config.final_time)
    println("CFL:                ", config.cfl)
    println("Interior jitter:    ", config.jitter)
    println("PEC boundary id:    ", CONV_PEC_BOUNDARY_ID)
    println()

    results = ConvergenceResult[]

    for order in config.orders
        for (level, cells_per_axis) in enumerate(config.cells_per_axis)
            @printf(
                "Running order N=%d, mesh level %d, cells_per_axis=%d\n",
                order,
                level,
                cells_per_axis,
            )

            result = run_case(cells_per_axis, order, level, config)
            push!(results, result)
        end
    end

    rated_results = with_rates(results)

    print_results(rated_results)
    write_results_csv(config.output, rated_results)

    println()
    println("Wrote CSV: ", config.output)

    return rated_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
