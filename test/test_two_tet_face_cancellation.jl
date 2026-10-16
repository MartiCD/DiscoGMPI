using Test
using LinearAlgebra
using Random
using DiscoGMPI

const FACE_CANCELLATION_TET_FACES = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)

function face_cancellation_tet(face, face_vertices, apex)
    vertices = zeros(Int, 4)
    local_face = FACE_CANCELLATION_TET_FACES[face]
    for i in 1:3
        vertices[local_face[i]] = face_vertices[i]
    end
    vertices[only(setdiff(1:4, collect(local_face)))] = apex
    return vertices
end

function face_cancellation_mesh(minus_face, plus_face, orientation)
    common = (1, 2, 3)
    vertex_permutation = TRI_PERMUTATIONS[orientation]
    plus_order = zeros(Int, 3)
    for i in 1:3
        plus_order[vertex_permutation[i]] = common[i]
    end

    minus_tet = face_cancellation_tet(minus_face, common, 4)
    plus_tet = face_cancellation_tet(plus_face, Tuple(plus_order), 5)
    tets = hcat(minus_tet, plus_tet)
    points = [
        0.0  1.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0 -1.0  1.0
    ]

    boundary = NTuple{3, Int}[]
    shared_key = (1, 2, 3)
    for elem in 1:2, face in 1:4
        local_face = FACE_CANCELLATION_TET_FACES[face]
        vertices = Tuple(tets[collect(local_face), elem])
        Tuple(sort(collect(vertices))) == shared_key && continue
        push!(boundary, vertices)
    end
    tris = reduce(hcat, (collect(face) for face in boundary))
    return RawVTUMesh(
        points,
        tets,
        tris,
        [1, 2],
        collect(3:(2 + length(boundary))),
        Dict{String, Any}(
            "boundary_id" => vcat([0, 0], fill(10, length(boundary))),
        ),
    )
end

function random_vector_polynomial(rng, degree)
    exponents = NTuple{3, Int}[]
    for total in 0:degree, i in 0:total, j in 0:(total - i)
        push!(exponents, (i, j, total - i - j))
    end
    coefficients = randn(rng, 6, length(exponents))
    component(c, x, y, z) = sum(
        coefficients[c, q] * x^a * y^b * z^d
        for (q, (a, b, d)) in enumerate(exponents)
    )
    electric = (x, y, z) -> (
        component(1, x, y, z), component(2, x, y, z), component(3, x, y, z))
    magnetic = (x, y, z) -> (
        component(4, x, y, z), component(5, x, y, z), component(6, x, y, z))
    return electric, magnetic
end

function geometric_outward_face_normal(mesh, elem, face)
    local_face = FACE_CANCELLATION_TET_FACES[face]
    face_points = [mesh.points[:, mesh.tets[local_face[i], elem]] for i in 1:3]
    opposite = only(setdiff(1:4, collect(local_face)))
    opposite_point = mesh.points[:, mesh.tets[opposite, elem]]
    normal = cross(face_points[2] - face_points[1], face_points[3] - face_points[1])
    normal ./= norm(normal)
    face_centroid = (face_points[1] + face_points[2] + face_points[3]) / 3
    dot(normal, opposite_point - face_centroid) < 0 || (normal .*= -1)
    return normal
end

face_components(field) =
    (field.Ex, field.Ey, field.Ez, field.Hx, field.Hy, field.Hz)

face_components(field::MaxwellRHS) =
    (field.rhsEx, field.rhsEy, field.rhsEz,
     field.rhsHx, field.rhsHy, field.rhsHz)

flux_components(flux) =
    (flux.fluxEx, flux.fluxEy, flux.fluxEz,
     flux.fluxHx, flux.fluxHy, flux.fluxHz)

function max_face_trace_jump(U, trace)
    maximum(
        abs(U_component[trace.minus_nodes[q], trace.minus_elem] -
            U_component[trace.plus_nodes[trace.plus_to_minus_perm[q]], trace.plus_elem])
        for U_component in face_components(U), q in eachindex(trace.minus_nodes)
    )
end

function weak_face_pairing(rhs, test_field, dg, component)
    values = face_components(test_field)[component]
    residual = face_components(rhs)[component]
    total = 0.0
    for elem in 1:2
        mass = dg.mappings.tet_mappings[elem].absdetJ .* dg.ref.M
        total += dot(view(values, :, elem), mass * view(residual, :, elem))
    end
    return total
end

function run_two_tet_face_cancellation_test()
    maximum_trace_jump = 0.0
    maximum_normal_defect = 0.0
    maximum_solver_normal_defect = 0.0
    maximum_flux_defect = Dict(
        MaxwellFlux_Central => 0.0,
        MaxwellFlux_Alternating => 0.0,
    )
    maximum_weak_defect = Dict(
        MaxwellFlux_Central => 0.0,
        MaxwellFlux_Alternating => 0.0,
    )
    configurations = 0
    weak_tests = 0

    for degree in 1:4, minus_face in 1:4, plus_face in 1:4, orientation in 1:6
        mesh = face_cancellation_mesh(minus_face, plus_face, orientation)
        dg = DGDiscretization(mesh, degree; nodal_set=:equispaced)
        @test length(dg.flux_faces.interior) == 1
        ff = only(dg.flux_faces.interior)
        trace = ff.trace
        @test trace.minus_face == minus_face
        @test trace.plus_face == plus_face

        minus_vertices = Tuple(mesh.tets[collect(FACE_CANCELLATION_TET_FACES[minus_face]), 1])
        plus_vertices = Tuple(mesh.tets[collect(FACE_CANCELLATION_TET_FACES[plus_face]), 2])
        @test DiscoGMPI.DistributedMesh3D.triangle_orientation_code(
            minus_vertices, plus_vertices) == orientation

        rng = MersenneTwister(10_000degree + 1_000minus_face + 100plus_face + orientation)
        Efun, Hfun = random_vector_polynomial(rng, degree)
        U = interpolate_maxwell_field(mesh, dg.ref, Efun, Hfun)
        trace_jump = max_face_trace_jump(U, trace)
        maximum_trace_jump = max(maximum_trace_jump, trace_jump)
        @test trace_jump <= 2.0e-11

        geometric_nminus = geometric_outward_face_normal(mesh, 1, minus_face)
        geometric_nplus = geometric_outward_face_normal(mesh, 2, plus_face)
        maximum_normal_defect = max(
            maximum_normal_defect,
            maximum(abs.(geometric_nminus .+ geometric_nplus)),
        )
        @test geometric_nplus ≈ -geometric_nminus atol=2.0e-14 rtol=0.0

        nminus = ff.normal
        nplus = (-nminus[1], -nminus[2], -nminus[3])
        maximum_solver_normal_defect = max(
            maximum_solver_normal_defect,
            maximum(abs(nminus[i] - geometric_nminus[i]) for i in 1:3),
        )
        @test collect(nminus) ≈ geometric_nminus atol=2.0e-14 rtol=0.0

        workspace = DiscoGMPI.MaxwellSurfaceWorkspace(dg.ref, dg.fops)
        minus, plus = DiscoGMPI.interior_face_traces!(workspace, U, trace)
        for flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating)
            minus_flux = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
                minus, plus, nminus; flux_kind=flux_kind, ε=1.0, μ=1.0)
            plus_flux = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
                plus, minus, nplus; flux_kind=flux_kind, ε=1.0, μ=1.0)
            flux_defect = maximum(
                maximum(abs.(minus_values .+ plus_values))
                for (minus_values, plus_values) in
                    zip(flux_components(minus_flux), flux_components(plus_flux))
            )
            maximum_flux_defect[flux_kind] = max(
                maximum_flux_defect[flux_kind], flux_defect)
            @test flux_defect <= 2.0e-11

            rhs = DiscoGMPI.similar_maxwell_rhs(U)
            for component_values in face_components(rhs)
                fill!(component_values, 0.0)
            end
            DiscoGMPI.maxwell_interior_surface_face_rhs!(
                rhs, U, dg.ref, dg.fops, dg.mappings, ff,
                PoissonBracketFormulation(flux_kind);
                ε=1.0, μ=1.0,
            )
            # Three independent continuous polynomial test vectors exercise
            # the lifted plus-face values after the face-node permutation.
            for trial in 1:3
                V_E, V_H = random_vector_polynomial(rng, degree)
                V = interpolate_maxwell_field(mesh, dg.ref, V_E, V_H)
                for component in 1:6
                    defect = abs(weak_face_pairing(rhs, V, dg, component))
                    maximum_weak_defect[flux_kind] = max(
                        maximum_weak_defect[flux_kind], defect)
                    @test defect <= 2.0e-10
                    weak_tests += 1
                end
            end
        end
        configurations += 1
    end

    return (
        configurations=configurations,
        weak_tests=weak_tests,
        maximum_trace_jump=maximum_trace_jump,
        maximum_normal_defect=maximum_normal_defect,
        maximum_solver_normal_defect=maximum_solver_normal_defect,
        maximum_centered_flux_defect=maximum_flux_defect[MaxwellFlux_Central],
        maximum_alternating_flux_defect=maximum_flux_defect[MaxwellFlux_Alternating],
        maximum_centered_weak_defect=maximum_weak_defect[MaxwellFlux_Central],
        maximum_alternating_weak_defect=maximum_weak_defect[MaxwellFlux_Alternating],
    )
end

@testset "Two-tetrahedron Maxwell face cancellation" begin
    summary = run_two_tet_face_cancellation_test()
    println("Two-tetrahedron face-cancellation summary: ", summary)
end
