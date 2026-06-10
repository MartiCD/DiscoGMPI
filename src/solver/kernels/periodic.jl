# -------------------------------------------------------------------------
# Periodic boundary trace maps
# -------------------------------------------------------------------------

function build_periodic_flux_face_colors(faces::Vector{PeriodicFluxFace})
    return greedy_flux_face_coloring(
        (
            (ff.trace.minus_elem, ff.trace.plus_elem)
            for ff in faces
        ),
    )
end

function DGPeriodicFluxFaces(faces::Vector{PeriodicFluxFace})
    return DGPeriodicFluxFaces(
        faces,
        build_periodic_flux_face_colors(faces),
    )
end


# Default periodic specs for the unit box
function default_unit_box_periodic_specs()
    return (
        PeriodicBoundarySpec(1, 2, (-1.0, 0.0, 0.0), :x_periodic),
        PeriodicBoundarySpec(3, 4, (0.0, -1.0, 0.0), :y_periodic),
        PeriodicBoundarySpec(5, 6, (0.0, 0.0, -1.0), :z_periodic),
    )
end

function box_periodic_specs(Lx::Float64, Ly::Float64, Lz::Float64)
    return (
        PeriodicBoundarySpec(1, 2, (-Lx, 0.0, 0.0), :x_periodic),
        PeriodicBoundarySpec(3, 4, (0.0, -Ly, 0.0), :y_periodic),
        PeriodicBoundarySpec(5, 6, (0.0, 0.0, -Lz), :z_periodic),
    )
end

# Small vector helps
function shift_point(
    p::NTuple{3, Float64},
    shift::NTuple{3, Float64},
)
    return (
        p[1] + shift[1],
        p[2] + shift[2],
        p[3] + shift[3],
    )
end

# Group boundary fluxes by boundary id 
function boundary_flux_faces_by_id(flux_faces::DGFluxFaces)
    groups = Dict{Int, Vector{Int}}()

    for i in eachindex(flux_faces.boundary)
        bid = flux_faces.boundary[i].boundary_id

        if !haskey(groups, bid)
            groups[bid] = Int[]
        end

        push!(groups[bid], i)
    end

    return groups
end

# Match one minus boundary face to one plus boundary face
function match_periodic_partner(
    minus_face::BoundaryFluxFace,
    plus_faces::Vector{BoundaryFluxFace},
    used_plus::AbstractVector{Bool},
    shift::NTuple{3, Float64};
    tol::Float64 = 1e-10,
)
    best_j = 0
    best_d2 = Inf

    for j in eachindex(plus_faces)
        if used_plus[j]
            continue
        end

        shifted_plus_centroid = shift_point(plus_faces[j].centroid, shift)
        d2 = squared_distance(minus_face.centroid, shifted_plus_centroid)

        if d2 < best_d2
            best_d2 = d2
            best_j = j
        end
    end

    if best_j == 0 || best_d2 > tol^2
        error(
            "Could not find periodic partner for boundary face with centroid " *
            "$(minus_face.centroid). Best squared distance = $best_d2, " *
            "tolerance squared = $(tol^2)."
        )
    end

    used_plus[best_j] = true

    return best_j
end

# Match face-node permutation with periodic shift 
function match_periodic_face_node_permutation(
    minus_points::Vector{NTuple{3, Float64}},
    plus_points::Vector{NTuple{3, Float64}},
    shift::NTuple{3, Float64};
    tol::Float64 = 1e-10,
)
    shifted_plus_points = [
        shift_point(p, shift) for p in plus_points
    ]

    return match_face_node_permutation(
        minus_points,
        shifted_plus_points;
        tol = tol,
    )
end

# Build periodic fluxes 
function build_periodic_flux_faces(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    flux_faces::DGFluxFaces,
    specs;
    centroid_tol::Float64 = 1e-9,
    node_tol::Float64 = 1e-9,
    area_rtol::Float64 = 1e-10,
)
    groups = boundary_flux_faces_by_id(flux_faces)

    periodic_faces = PeriodicFluxFace[]

    for spec in specs
        minus_ids = get(groups, spec.minus_boundary_id, Int[])
        plus_ids = get(groups, spec.plus_boundary_id, Int[])

        if length(minus_ids) != length(plus_ids)
            error(
                "Periodic pair $(spec.name) has incompatible face counts: " *
                "boundary_id $(spec.minus_boundary_id) has $(length(minus_ids)) faces, " *
                "boundary_id $(spec.plus_boundary_id) has $(length(plus_ids)) faces. " *
                "The mesh is not periodic-compatible for this pair."
            )
        end

        minus_faces = flux_faces.boundary[minus_ids]
        plus_faces = flux_faces.boundary[plus_ids]

        used_plus = falses(length(plus_faces))

        for i in eachindex(minus_faces)
            mf = minus_faces[i]

            j = match_periodic_partner(
                mf,
                plus_faces,
                used_plus,
                spec.plus_to_minus_shift;
                tol = centroid_tol,
            )

            pf = plus_faces[j]

            # Area compatibility check.
            area_error = abs(mf.area - pf.area)
            area_scale = max(abs(mf.area), abs(pf.area), eps(Float64))

            if area_error / area_scale > area_rtol
                error(
                    "Periodic face area mismatch for $(spec.name): " *
                    "minus area = $(mf.area), plus area = $(pf.area), " *
                    "relative error = $(area_error / area_scale)."
                )
            end

            minus_points = physical_face_points(
                mesh,
                ref,
                mf.trace.elem,
                mf.trace.nodes,
            )

            plus_points = physical_face_points(
                mesh,
                ref,
                pf.trace.elem,
                pf.trace.nodes,
            )

            perm = match_periodic_face_node_permutation(
                minus_points,
                plus_points,
                spec.plus_to_minus_shift;
                tol = node_tol,
            )

            tr = PeriodicTraceMap(
                mf.trace.elem,
                mf.trace.face,
                mf.boundary_id,
                copy(mf.trace.nodes),

                pf.trace.elem,
                pf.trace.face,
                pf.boundary_id,
                copy(pf.trace.nodes),

                perm,
                spec.plus_to_minus_shift,
            )

            push!(
                periodic_faces,
                PeriodicFluxFace(
                    tr,
                    mf.normal,
                    mf.area,
                    mf.centroid,
                    spec.name,
                ),
            )
        end
    end

    return DGPeriodicFluxFaces(periodic_faces)
end

# Periodic face summary 
function print_periodic_flux_face_summary(periodic::DGPeriodicFluxFaces)
    println("DG periodic flux faces")
    println("----------------------")
    println("Number of periodic faces: ", length(periodic.faces))
    println("Periodic face colors:     ", length(periodic.colors))

    if isempty(periodic.faces)
        return nothing
    end

    names = sort(unique(f.name for f in periodic.faces))

    println()
    println("Periodic face counts")
    println("--------------------")

    for name in names
        n = count(f -> f.name == name, periodic.faces)
        area = sum(f.area for f in periodic.faces if f.name == name)

        println("  ", name, ": faces = ", n, ", area = ", area)
    end

    println()
    println("Permutation examples")
    println("--------------------")

    nexamples = min(5, length(periodic.faces))

    for i in 1:nexamples
        f = periodic.faces[i]
        tr = f.trace

        println(
            "periodic face ", i,
            ": ", f.name,
            ", elem ", tr.minus_elem, " face ", tr.minus_face,
            " ↔ elem ", tr.plus_elem, " face ", tr.plus_face,
            ", perm = ", tr.plus_to_minus_perm,
        )
    end

    return nothing
end

# Geometry validation for periodic maps 
function test_periodic_flux_face_geometry(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    periodic::DGPeriodicFluxFaces,
)
    max_distance = 0.0
    worst_face = 0

    for i in eachindex(periodic.faces)
        f = periodic.faces[i]
        tr = f.trace

        minus_points = physical_face_points(
            mesh,
            ref,
            tr.minus_elem,
            tr.minus_nodes,
        )

        plus_points = physical_face_points(
            mesh,
            ref,
            tr.plus_elem,
            tr.plus_nodes,
        )

        plus_points_aligned = plus_points[tr.plus_to_minus_perm]

        for q in eachindex(minus_points)
            shifted_plus = shift_point(
                plus_points_aligned[q],
                tr.plus_to_minus_shift,
            )

            d = sqrt(squared_distance(minus_points[q], shifted_plus))

            if d > max_distance
                max_distance = d
                worst_face = i
            end
        end
    end

    println("Periodic flux-face geometry test")
    println("--------------------------------")
    println("max matched-node distance: ", max_distance)
    println("worst periodic face id:    ", worst_face)

    if max_distance < 1e-10
        println("✓ periodic matched nodes are geometrically consistent")
    else
        println("⚠ periodic matched nodes are not geometrically consistent")
    end

    return nothing
end

# Periodic trace continuity test 
function periodic_test_scalar_field(x::Float64, y::Float64, z::Float64)
    return sin(2.0 * pi * x) +
           0.3 * cos(2.0 * pi * y) +
           0.2 * sin(2.0 * pi * z)
end


function test_periodic_trace_maps_scalar_function(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    periodic::DGPeriodicFluxFaces,
)
    max_jump = 0.0
    worst_face = 0

    for i in eachindex(periodic.faces)
        f = periodic.faces[i]
        tr = f.trace

        minus_points = physical_face_points(
            mesh,
            ref,
            tr.minus_elem,
            tr.minus_nodes,
        )

        plus_points = physical_face_points(
            mesh,
            ref,
            tr.plus_elem,
            tr.plus_nodes,
        )

        plus_points_aligned = plus_points[tr.plus_to_minus_perm]

        for q in eachindex(minus_points)
            xm, ym, zm = minus_points[q]

            shifted_plus = shift_point(
                plus_points_aligned[q],
                tr.plus_to_minus_shift,
            )

            xp, yp, zp = shifted_plus

            uM = periodic_test_scalar_field(xm, ym, zm)
            uP = periodic_test_scalar_field(xp, yp, zp)

            jump = abs(uM - uP)

            if jump > max_jump
                max_jump = jump
                worst_face = i
            end
        end
    end

    println("Periodic scalar trace continuity test")
    println("-------------------------------------")
    println("max periodic jump:       ", max_jump)
    println("worst periodic face id:  ", worst_face)

    if max_jump < 1e-10
        println("✓ periodic trace pairing is consistent for periodic scalar field")
    else
        println("⚠ periodic trace pairing failed scalar periodicity test")
    end

    return nothing
end

# Maxwell periodic surface RHS 

# Periodic Maxwell RHS wrapper 

# Important: registry for periodic runs 
function periodic_box_with_pec_sphere_registry()
    return MaxwellBoundaryRegistry(
        Dict(
            1  => MaxwellBC_None,
            2  => MaxwellBC_None,
            3  => MaxwellBC_None,
            4  => MaxwellBC_None,
            5  => MaxwellBC_None,
            6  => MaxwellBC_None,
            10 => MaxwellBC_PEC,
        ),
    )
end


function test_maxwell_periodic_rhs_zero_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    registry = periodic_box_with_pec_sphere_registry()

    maxwell_rhs_periodic!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell periodic RHS zero-field test")
    println("------------------------------------")
    println("max |rhsEx|: ", errEx)
    println("max |rhsEy|: ", errEy)
    println("max |rhsEz|: ", errEz)
    println("max |rhsHx|: ", errHx)
    println("max |rhsHy|: ", errHy)
    println("max |rhsHz|: ", errHz)
    println("max error:   ", maxerr)

    if maxerr < 1e-12
        println("✓ periodic Maxwell RHS vanishes for zero field")
    else
        println("⚠ periodic Maxwell RHS zero-field test failed")
    end

    return nothing
end

function test_rk_periodic_zero_field_step(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces;
    rk_order::Int = 4,
    dt::Float64 = 1e-4,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    registry = periodic_box_with_pec_sphere_registry()

    rhs_function! = make_maxwell_periodic_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    max_field = max_abs_maxwell_field(U)

    println("RK periodic zero-field step test")
    println("--------------------------------")
    println("RK scheme:          ", scheme.name)
    println("dt:                 ", dt)
    println("max |U| after step: ", max_field)

    if max_field < 1e-14
        println("✓ RK periodic step preserves the zero Maxwell field")
    else
        println("⚠ RK periodic zero-field test failed")
    end

    return nothing
end

function test_rk_periodic_one_step_energy_diagnostic(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces;
    rk_order::Int = 4,
    dt::Float64 = 1e-5,
)
    Efun = (x, y, z) -> (
        sin(2.0 * pi * x) * cos(2.0 * pi * y),
        sin(2.0 * pi * y) * cos(2.0 * pi * z),
        sin(2.0 * pi * z) * cos(2.0 * pi * x),
    )

    Hfun = (x, y, z) -> (
        cos(2.0 * pi * x) * sin(2.0 * pi * z),
        cos(2.0 * pi * y) * sin(2.0 * pi * x),
        cos(2.0 * pi * z) * sin(2.0 * pi * y),
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    energy0 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    registry = periodic_box_with_pec_sphere_registry()

    rhs_function! = make_maxwell_periodic_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
        flux_kind = MaxwellFlux_Upwind
    )

    rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    energy1 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    ΔE = energy1.total - energy0.total
    relΔE = ΔE / max(energy0.total, eps(Float64))

    println("RK periodic one-step energy diagnostic")
    println("--------------------------------------")
    println("RK scheme:        ", scheme.name)
    println("dt:               ", dt)
    println("energy before:    ", energy0.total)
    println("energy after:     ", energy1.total)
    println("ΔE:               ", ΔE)
    println("relative ΔE:      ", relΔE)

    return nothing
end


function run_maxwell_periodic_time_steps!(
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces,
    registry::MaxwellBoundaryRegistry;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Upwind,
)
    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_periodic_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry;
        ε = ε,
        μ = μ,
        flux_kind = flux_kind
    )

    println("Periodic Maxwell time marching")
    println("------------------------------")
    println("RK scheme:       ", scheme.name)
    println("dt:              ", dt)
    println("nsteps:          ", nsteps)

    energy0 = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
    println("initial energy:  ", energy0.total)

    for step in 1:nsteps
        rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end

function triangle_area_from_nodes(
    points::Matrix{Float64},
    nodes::NTuple{3, Int},
)
    x1 = point3(points, nodes[1])
    x2 = point3(points, nodes[2])
    x3 = point3(points, nodes[3])

    a = vsub(x2, x1)
    b = vsub(x3, x1)

    return 0.5 * norm3(cross3(a, b))
end


function tet_total_surface_area(
    mesh::RawVTUMesh,
    elem::Int,
)
    tet = mesh.tets[:, elem]

    area = 0.0

    for lf in 1:4
        local_nodes = TET_FACES[lf]

        face_nodes = (
            tet[local_nodes[1]],
            tet[local_nodes[2]],
            tet[local_nodes[3]],
        )

        area += triangle_area_from_nodes(mesh.points, face_nodes)
    end

    return area
end


function element_size_diagnostics(
    mesh::RawVTUMesh,
    geometry::DGGeometry,
)
    ne = length(geometry.cells)

    h = Vector{Float64}(undef, ne)
    volumes = Vector{Float64}(undef, ne)
    areas = Vector{Float64}(undef, ne)

    for e in 1:ne
        V = geometry.cells[e].volume
        A = tet_total_surface_area(mesh, e)

        if V <= 0.0
            error("Element $e has non-positive volume $V.")
        end

        if A <= 0.0
            error("Element $e has non-positive surface area $A.")
        end

        # Characteristic length based on volume-to-surface ratio.
        h[e] = 3.0 * V / A

        volumes[e] = V
        areas[e] = A
    end

    hmin, worst_elem = findmin(h)

    return ElementSizeDiagnostics(
        hmin,
        maximum(h),
        sum(h) / length(h),
        minimum(volumes),
        maximum(volumes),
        minimum(areas),
        maximum(areas),
        worst_elem,
    )
end

function maxwell_wave_speed(; ε::Float64 = 1.0, μ::Float64 = 1.0)
    if ε <= 0.0
        error("ε must be positive.")
    end

    if μ <= 0.0
        error("μ must be positive.")
    end

    return 1.0 / sqrt(ε * μ)
end

function estimate_maxwell_dt(
    mesh::RawVTUMesh,
    geometry::DGGeometry,
    ref::ReferenceTet;
    CFL::Float64 = 0.15,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    if CFL <= 0.0
        error("CFL must be positive.")
    end

    sizes = element_size_diagnostics(mesh, geometry)

    c = maxwell_wave_speed(; ε = ε, μ = μ)

    dt = CFL * sizes.hmin / ((2.0 * ref.N + 1.0) * c)

    return dt, sizes
end

function print_element_size_diagnostics(sizes::ElementSizeDiagnostics)
    println("Element-size diagnostics")
    println("------------------------")
    println("hmin:              ", sizes.hmin)
    println("hmax:              ", sizes.hmax)
    println("hmean:             ", sizes.hmean)
    println("worst element:     ", sizes.worst_elem)

    println()
    println("Volume range")
    println("------------")
    println("vmin:              ", sizes.vmin)
    println("vmax:              ", sizes.vmax)

    println()
    println("Surface-area range")
    println("------------------")
    println("amin:              ", sizes.amin)
    println("amax:              ", sizes.amax)

    return nothing
end

function print_maxwell_dt_estimate(
    dt::Float64,
    sizes::ElementSizeDiagnostics,
    ref::ReferenceTet;
    CFL::Float64,
    ε::Float64,
    μ::Float64,
)
    c = maxwell_wave_speed(; ε = ε, μ = μ)

    println("Maxwell CFL time-step estimate")
    println("------------------------------")
    println("Polynomial order N:       ", ref.N)
    println("CFL:                      ", CFL)
    println("ε:                        ", ε)
    println("μ:                        ", μ)
    println("wave speed c:             ", c)
    println("hmin:                     ", sizes.hmin)
    println("DG denominator 2N + 1:    ", 2 * ref.N + 1)
    println("estimated dt:             ", dt)

    return nothing
end

function test_periodic_time_marching_with_cfl_dt(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    geometry::DGGeometry,
    flux_faces::DGFluxFaces,
    periodic_faces::DGPeriodicFluxFaces;
    rk_order::Int = 4,
    CFL::Float64 = 0.05,
    nsteps::Int = 5,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    dt, sizes = estimate_maxwell_dt(
        mesh,
        geometry,
        ref;
        CFL = CFL,
        ε = ε,
        μ = μ,
    )

    Efun = (x, y, z) -> (
        sin(2.0 * pi * x) * cos(2.0 * pi * y),
        sin(2.0 * pi * y) * cos(2.0 * pi * z),
        sin(2.0 * pi * z) * cos(2.0 * pi * x),
    )

    Hfun = (x, y, z) -> (
        cos(2.0 * pi * x) * sin(2.0 * pi * z),
        cos(2.0 * pi * y) * sin(2.0 * pi * x),
        cos(2.0 * pi * z) * sin(2.0 * pi * y),
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    registry = periodic_box_with_pec_sphere_registry()

    energy0 = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)

    run_maxwell_periodic_time_steps!(
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
        registry;
        rk_order = rk_order,
        dt = dt,
        nsteps = nsteps,
        ε = ε,
        μ = μ,
        energy_every = 1,
    )

    energy1 = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)

    ΔE = energy1.total - energy0.total
    relΔE = ΔE / max(energy0.total, eps(Float64))

    println()
    println("Periodic CFL time-marching smoke test")
    println("-------------------------------------")
    println("RK order:        ", rk_order)
    println("CFL:             ", CFL)
    println("dt:              ", dt)
    println("nsteps:          ", nsteps)
    println("initial energy:  ", energy0.total)
    println("final energy:    ", energy1.total)
    println("relative ΔE:     ", relΔE)
    println("max |U|:         ", max_abs_maxwell_field(U))

    if isfinite(energy1.total) && isfinite(max_abs_maxwell_field(U))
        println("✓ time marching completed without NaNs/Infs")
    else
        println("⚠ time marching produced NaNs/Infs")
    end

    return nothing
end
