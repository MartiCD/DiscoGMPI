using Test, LinearAlgebra, DiscoGMPI

@testset "Mixed-order reference operators" begin
    @test_throws ArgumentError build_mixed_reference_tet(1; p_H=1)
    @test_throws ArgumentError build_mixed_reference_tet(0)
    for nodes in (:equispaced, :warp_blend), p in 1:4
        op = build_mixed_reference_tet(p; nodal_set=nodes)
        E,H = op.ref_E,op.ref_H
        @test E.Np == binomial(p+3,3)
        @test H.Np == binomial(p+4,3)
        @test size(op.S_EH[1]) == (E.Np,H.Np)
        @test isposdef(Symmetric(op.M_E))
        @test isposdef(Symmetric(op.M_H))
        @test sum(op.volume_weights) ≈ 4/3
        # Independent analytic monomial moments in unit-simplex coordinates.
        xyz = (op.volume_points .+ 1) ./ 2
        for (a,b,c) in DiscoGMPI.monomial_exponents_3d(2p+2)
            exact = 8DiscoGMPI.unit_simplex_monomial_integral(a,b,c)
            @test sum(op.volume_weights .* xyz[:,1].^a .* xyz[:,2].^b .* xyz[:,3].^c) ≈ exact rtol=2e-12 atol=2e-15
        end
        normals = ((1.,1.,1.)./sqrt(3),(-1.,0.,0.),(0.,-1.,0.),(0.,0.,-1.))
        for a in 1:3
            boundary = sum(normals[f][a] * op.face_E[f]' *
                           Diagonal(op.face_weights[f]) * op.face_H[f] for f in 1:4)
            @test norm(op.S_EH[a] + op.S_HE[a]' - boundary, Inf) < 2e-11
        end
        for (ref,V,M) in ((E,op.V_E,op.M_E),(H,op.V_H,op.M_H))
            for (a,b,c) in ref.basis.exponents
                values = ref.r.^a .* ref.s.^b .* ref.t.^c
                exact = op.volume_points[:,1].^a .* op.volume_points[:,2].^b .* op.volume_points[:,3].^c
                @test V*values ≈ exact atol=2e-11 rtol=2e-11
                @test M \ (V' * (op.volume_weights .* exact)) ≈ values atol=2e-10 rtol=2e-10
            end
        end
    end
end
