using Test
using LinearAlgebra
using DiscoGMPI

function reference_tet_barycentric(r::Float64, s::Float64, t::Float64)
    lambda2 = 0.5 * (r + 1.0)
    lambda3 = 0.5 * (s + 1.0)
    lambda4 = 0.5 * (t + 1.0)
    lambda1 = 1.0 - lambda2 - lambda3 - lambda4
    return (lambda1, lambda2, lambda3, lambda4)
end

function single_reference_tet_boundary_mesh()
    points = [
        -1.0  1.0 -1.0 -1.0
        -1.0 -1.0  1.0 -1.0
        -1.0 -1.0 -1.0  1.0
    ]

    tets = reshape([1, 2, 3, 4], 4, 1)
    tris = [
        2 1 1 1
        3 4 2 3
        4 3 4 2
    ]

    return RawVTUMesh(
        points,
        tets,
        tris,
        [1],
        collect(2:5),
        Dict{String, Any}("boundary_id" => [0, 10, 10, 10, 10]),
    )
end

@testset "Hesthaven-Warburton tetrahedral warp-blend nodes" begin
    for N in 0:6
        r, s, t = warp_blend_tet_nodes(N)

        @test length(r) == num_tet_nodes(N)
        @test length(s) == num_tet_nodes(N)
        @test length(t) == num_tet_nodes(N)
        @test all(isfinite, r)
        @test all(isfinite, s)
        @test all(isfinite, t)

        for node in eachindex(r)
            lambda = reference_tet_barycentric(r[node], s[node], t[node])
            @test abs(sum(lambda) - 1.0) <= 1e-12
            @test minimum(lambda) >= -1e-11
            @test maximum(lambda) <= 1.0 + 1e-11
        end
    end

    @test_throws ArgumentError build_reference_tet(2; nodal_set = :unknown)
end

@testset "Warp-blend reference matrices and face operators" begin
    for N in 1:5
        ref = build_reference_tet(N; nodal_set = :warp_blend)
        face_ops = build_reference_face_operators(ref)
        expected_nfp = (N + 1) * (N + 2) ÷ 2

        @test ref.Np == num_tet_nodes(N)
        @test all(length(nodes) == expected_nfp for nodes in face_ops.face_nodes)
        @test size(face_ops.LIFT) == (ref.Np, ref.Np)

        ones_vec = ones(ref.Np)
        @test isapprox(dot(ones_vec, ref.M * ones_vec), 4.0 / 3.0; atol = 1e-10)
        @test isapprox(
            dot(ones_vec, face_ops.Emat * ones_vec),
            6.0 + 2.0 * sqrt(3.0);
            atol = 1e-10,
        )

        @test norm(ref.Dr * ones_vec) <= 1e-10
        @test norm(ref.Ds * ones_vec) <= 1e-10
        @test norm(ref.Dt * ones_vec) <= 1e-10
        @test norm(ref.Dr * ref.r .- 1.0) <= 1e-10
        @test norm(ref.Ds * ref.r) <= 1e-10
        @test norm(ref.Dt * ref.r) <= 1e-10
        @test norm(ref.Dr * ref.s) <= 1e-10
        @test norm(ref.Ds * ref.s .- 1.0) <= 1e-10
        @test norm(ref.Dt * ref.s) <= 1e-10
        @test norm(ref.Dr * ref.t) <= 1e-10
        @test norm(ref.Ds * ref.t) <= 1e-10
        @test norm(ref.Dt * ref.t .- 1.0) <= 1e-10
    end
end

@testset "Warp-blend nodal set is available through DGDiscretization" begin
    mesh = single_reference_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 4; nodal_set = :warp_blend)

    @test dg.ref.N == 4
    @test dg.ref.Np == num_tet_nodes(4)
    @test length(dg.fops.face_nodes[1]) == 15
end
