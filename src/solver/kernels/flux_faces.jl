# -------------------------------------------------------------------------
# Flux-ready DG faces
# -------------------------------------------------------------------------

function greedy_flux_face_coloring(face_elements)
    colors = Vector{Vector{Int}}()
    color_elements = Vector{Set{Int}}()

    for (i, elems) in enumerate(face_elements)
        assigned = false

        for color_id in eachindex(colors)
            used = color_elements[color_id]

            if all(elem -> !(elem in used), elems)
                push!(colors[color_id], i)

                for elem in elems
                    push!(used, elem)
                end

                assigned = true
                break
            end
        end

        if !assigned
            push!(colors, [i])
            push!(color_elements, Set{Int}(elems))
        end
    end

    return colors
end

function build_interior_flux_face_colors(interior::Vector{InteriorFluxFace})
    return greedy_flux_face_coloring(
        (
            (ff.trace.minus_elem, ff.trace.plus_elem)
            for ff in interior
        ),
    )
end

function build_boundary_flux_face_colors(boundary::Vector{BoundaryFluxFace})
    return greedy_flux_face_coloring(
        (
            (ff.trace.elem,)
            for ff in boundary
        ),
    )
end

function DGFluxFaces(
    interior::Vector{InteriorFluxFace},
    boundary::Vector{BoundaryFluxFace},
)
    return DGFluxFaces(
        interior,
        boundary,
        build_interior_flux_face_colors(interior),
        build_boundary_flux_face_colors(boundary),
    )
end

function build_dg_flux_faces(
    trace_maps::DGTraceMaps,
    geometry::DGGeometry,
)
    if length(trace_maps.interior) != length(geometry.interior_faces)
        error(
            "Mismatch between interior trace maps and interior face geometry: " *
            "$(length(trace_maps.interior)) vs $(length(geometry.interior_faces))."
        )
    end

    if length(trace_maps.boundary) != length(geometry.boundary_faces)
        error(
            "Mismatch between boundary trace maps and boundary face geometry: " *
            "$(length(trace_maps.boundary)) vs $(length(geometry.boundary_faces))."
        )
    end

    interior = Vector{InteriorFluxFace}(undef, length(trace_maps.interior))

    for i in eachindex(trace_maps.interior)
        tr = trace_maps.interior[i]
        fg = geometry.interior_faces[i]

        interior[i] = InteriorFluxFace(
            tr,
            fg.normal,
            fg.area,
            fg.centroid,
        )
    end

    boundary = Vector{BoundaryFluxFace}(undef, length(trace_maps.boundary))

    for i in eachindex(trace_maps.boundary)
        tr = trace_maps.boundary[i]
        fg = geometry.boundary_faces[i]

        boundary[i] = BoundaryFluxFace(
            tr,
            fg.normal,
            fg.area,
            fg.centroid,
            tr.boundary_id,
        )
    end

    return DGFluxFaces(interior, boundary)
end

function print_flux_face_summary(flux_faces::DGFluxFaces)
    println("DG flux faces")
    println("-------------")
    println("Number of interior flux faces: ", length(flux_faces.interior))
    println("Number of boundary flux faces: ", length(flux_faces.boundary))
    println("Interior face colors:          ", length(flux_faces.interior_colors))
    println("Boundary face colors:          ", length(flux_faces.boundary_colors))

    interior_areas = [f.area for f in flux_faces.interior]
    boundary_areas = [f.area for f in flux_faces.boundary]

    println()
    println("Interior face areas")
    println("-------------------")
    println("min area: ", minimum(interior_areas))
    println("max area: ", maximum(interior_areas))
    println("sum area: ", sum(interior_areas))

    println()
    println("Boundary face areas")
    println("-------------------")
    println("min area: ", minimum(boundary_areas))
    println("max area: ", maximum(boundary_areas))
    println("sum area: ", sum(boundary_areas))

    boundary_ids = sort(unique(f.boundary_id for f in flux_faces.boundary))

    println()
    println("Boundary area by boundary_id")
    println("----------------------------")

    for bid in boundary_ids
        area_bid = sum(f.area for f in flux_faces.boundary if f.boundary_id == bid)
        count_bid = count(f -> f.boundary_id == bid, flux_faces.boundary)

        println(
            "  boundary_id = ", bid,
            " : faces = ", count_bid,
            ", area = ", area_bid,
        )
    end

    return nothing
end

function test_interior_flux_face_normals(
    geometry::DGGeometry,
    flux_faces::DGFluxFaces,
)
    min_dot = Inf
    max_bad = 0
    worst_face = 0

    for i in eachindex(flux_faces.interior)
        f = flux_faces.interior[i]
        tr = f.trace

        c_minus = geometry.cells[tr.minus_elem].centroid
        c_plus = geometry.cells[tr.plus_elem].centroid

        minus_to_plus = vsub(c_plus, c_minus)

        d = dot3(f.normal, minus_to_plus)

        if d < min_dot
            min_dot = d
            worst_face = i
        end

        if d <= 0.0
            max_bad += 1
        end
    end

    println("Interior flux-face normal test")
    println("------------------------------")
    println("minimum n · (c_plus - c_minus): ", min_dot)
    println("number of non-positive cases:   ", max_bad)
    println("worst face id:                  ", worst_face)

    if max_bad == 0
        println("✓ all interior normals point from minus to plus")
    else
        println("⚠ some interior normals may have wrong orientation")
    end

    return nothing
end

function expected_box_normal(boundary_id::Int)
    if boundary_id == 1
        return (-1.0, 0.0, 0.0)
    elseif boundary_id == 2
        return (1.0, 0.0, 0.0)
    elseif boundary_id == 3
        return (0.0, -1.0, 0.0)
    elseif boundary_id == 4
        return (0.0, 1.0, 0.0)
    elseif boundary_id == 5
        return (0.0, 0.0, -1.0)
    elseif boundary_id == 6
        return (0.0, 0.0, 1.0)
    else
        error("No expected box normal for boundary_id = $boundary_id.")
    end
end

function test_boundary_box_normals(flux_faces::DGFluxFaces; tol::Float64 = 1e-10)
    box_ids = Set([1, 2, 3, 4, 5, 6])

    max_error = 0.0
    worst_face = 0
    bad_count = 0

    for i in eachindex(flux_faces.boundary)
        f = flux_faces.boundary[i]

        if !(f.boundary_id in box_ids)
            continue
        end

        expected = expected_box_normal(f.boundary_id)

        err = norm3(vsub(f.normal, expected))

        if err > max_error
            max_error = err
            worst_face = i
        end

        if err > tol
            bad_count += 1
        end
    end

    println("Boundary box-normal test")
    println("------------------------")
    println("max normal error:      ", max_error)
    println("bad count:             ", bad_count)
    println("worst boundary face:   ", worst_face)

    if bad_count == 0
        println("✓ all box-boundary normals match expected directions")
    else
        println("⚠ some box-boundary normals differ from expected directions")
    end

    return nothing
end

function test_pec_sphere_normals(
    flux_faces::DGFluxFaces;
    sphere_center::NTuple{3, Float64} = (0.5, 0.5, 0.5),
    pec_boundary_id::Int = 10,
)
    min_dot = Inf
    bad_count = 0
    checked = 0
    worst_face = 0

    for i in eachindex(flux_faces.boundary)
        f = flux_faces.boundary[i]

        if f.boundary_id != pec_boundary_id
            continue
        end

        checked += 1

        center_direction = vsub(sphere_center, f.centroid)
        d = dot3(f.normal, center_direction)

        if d < min_dot
            min_dot = d
            worst_face = i
        end

        if d <= 0.0
            bad_count += 1
        end
    end

    println("PEC sphere normal test")
    println("----------------------")
    println("checked faces:                 ", checked)
    println("minimum n · (center-centroid): ", min_dot)
    println("bad count:                     ", bad_count)
    println("worst boundary face:           ", worst_face)

    if checked == 0
        println("⚠ no PEC sphere faces were found")
    elseif bad_count == 0
        println("✓ PEC sphere normals point outward from domain into spherical hole")
    else
        println("⚠ some PEC sphere normals may have wrong orientation")
    end

    return nothing
end

function print_boundary_area_reference_check(
    flux_faces::DGFluxFaces;
    sphere_radius::Float64 = 0.2,
)
    total_boundary_area = sum(f.area for f in flux_faces.boundary)

    expected_outer_box_area = 6.0
    expected_sphere_area = 4.0 * pi * sphere_radius^2
    expected_total = expected_outer_box_area + expected_sphere_area

    println("Boundary area reference check")
    println("-----------------------------")
    println("computed total boundary area: ", total_boundary_area)
    println("expected outer box area:      ", expected_outer_box_area)
    println("expected sphere area:         ", expected_sphere_area)
    println("expected total area:          ", expected_total)
    println("absolute difference:          ", abs(total_boundary_area - expected_total))

    return nothing
end
